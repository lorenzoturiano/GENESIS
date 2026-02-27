import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np

# A simple fully-connected residual block.
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=False)
        # Adjust skip connection if dimensions differ.
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

# Permutation-invariant pooling over genes:
# Given expression x (B, G) and gene embeddings E (G, D),
# returns pooled cell embedding (B, D) = normalized(x) @ E.
class GeneExpressionWeightedPool(nn.Module):
    def __init__(self, n_genes: int, emb_dim: int, eps: float = 1e-8, init_weight=None):
        super().__init__()
        self.eps = eps
        self.embedding = nn.Embedding(n_genes, emb_dim)
        if init_weight is not None:
            if init_weight.shape != (n_genes, emb_dim):
                raise ValueError(f"init_weight must have shape {(n_genes, emb_dim)}, got {tuple(init_weight.shape)}")
            with torch.no_grad():
                self.embedding.weight.copy_(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, G) non-negative gene expression values
        w = x / (x.sum(dim=1, keepdim=True) + self.eps)   # (B, G)
        E = self.embedding.weight                         # (G, D)
        return w @ E                                      # (B, D)

# Encoder with residual blocks and storing skip connections.
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            self.use_skips = False
            self.model = nn.Sequential(
                nn.Linear(input_dim, latent_dim * 2),
                nn.ReLU(inplace=False)
            )
            self.fc_mu    = nn.Linear(latent_dim * 2, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        else:
            self.use_skips = True
            self.layers = nn.ModuleList()
            # First layer.
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0]),
                    nn.ReLU(inplace=False)
                )
            )
            # Subsequent residual blocks.
            for i in range(1, len(hidden_dims)):
                self.layers.append(ResidualBlock(hidden_dims[i-1], hidden_dims[i]))
            self.fc_mu    = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
            
    def forward(self, x):
        if not self.use_skips:
            x = self.model(x)
            return self.fc_mu(x), self.fc_logvar(x), None
        else:
            skip_connections = []
            for layer in self.layers:
                x = layer(x)
                skip_connections.append(x)
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar, skip_connections

# Decoder with UNet-style skip connections using reversed_hidden and ReLU.
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, reversed_hidden=None, normalized=False):
        super(Decoder, self).__init__()
        self.normalized = normalized
        # Final activation: if normalized, use Sigmoid; else ReLU.
        self.last_activation = nn.Sigmoid() if normalized else nn.ReLU(inplace=False)
        
        if reversed_hidden is None:
            self.use_skips = False
            self.model = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(inplace=False),
                nn.Linear(latent_dim * 2, output_dim),
                self.last_activation
            )
        else:
            self.use_skips = True

            # Project the latent vector into the bottleneck feature space.
            # Note: reversed_hidden[0] is equivalent to hidden_dims[-1].
            self.fc_initial = nn.Linear(latent_dim, reversed_hidden[0])
            self.decoder_blocks = nn.ModuleList()
            # For each block, the input dimension is the sum of the previous block output and
            # the corresponding skip connection from the encoder.
            # For example, if hidden_dims = [H1, H2, H3] then reversed_hidden = [H3, H2, H1].
            # Block 0: input = H3 (from fc_initial) concatenated with encoder skip (H2) -> output = H2.
            # Block 1: input = H2 concatenated with encoder skip (H1) -> output = H1.
            for i in range(1, len(reversed_hidden)):
                in_features = reversed_hidden[i-1] + reversed_hidden[i]
                out_features = reversed_hidden[i]
                # self.decoder_blocks.append(nn.Linear(in_features, out_features))
                self.decoder_blocks.append(nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=False)))
            self.final_layer = nn.Linear(reversed_hidden[-1], output_dim)
            self.activation = nn.ReLU(inplace=False)
            
    def forward(self, z, skip_connections=None):
        if not self.use_skips or skip_connections is None:
            return self.model(z)
        # Project latent vector into the bottleneck space.
        x = self.fc_initial(z)
        # Use the reversed order for skip connections, excluding the bottleneck.
        skips = list(reversed(skip_connections[:-1]))
        for block, skip_feat in zip(self.decoder_blocks, skips):
            x = torch.cat([x, skip_feat], dim=-1)
            x = self.activation(block(x))
        x = self.final_layer(x)
        x = self.last_activation(x)
        return x

# RNA_VAE combining the encoder and decoder.
class RNA_VAE_UNET(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_dims=[2048, 1024, 512],
                 normalized=False, recon_loss=None, device=None,
                 use_gene_pool=True, gene_emb_dim=512, pool_eps=1e-8, gene_emb_init=None):
        super(RNA_VAE_UNET, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recon_loss = recon_loss if recon_loss is not None else nn.MSELoss()

        self.use_gene_pool = use_gene_pool
        if self.use_gene_pool:
            # Learnable gene embeddings + weighted-mean pooling (perm-invariant to gene order)
            self.gene_pool = GeneExpressionWeightedPool(
                input_dim, gene_emb_dim, eps=pool_eps, init_weight=gene_emb_init
            ).to(self.device)
            encoder_input_dim = gene_emb_dim
        else:
            self.gene_pool = None
            encoder_input_dim = input_dim

        self.encoder = Encoder(encoder_input_dim, latent_dim, hidden_dims=hidden_dims).to(self.device)

        reversed_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(latent_dim, output_dim, reversed_hidden=reversed_hidden, normalized=normalized).to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if getattr(self, "use_gene_pool", False):
            x = self.gene_pool(x)   # (B, 512) if gene_emb_dim=512

        mu, logvar, skip_connections = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, skip_connections)
        return recon_x, mu, logvar

    def train(self, dataloader, n_epochs=10, optimizer=None, beta=0.1, lr=1e-4, threshold=False, name_weight="VAE_UNET_weights"):
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            # Resample targets-within-cell-type once per epoch (if supported)
            ds = getattr(dataloader, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            r_loss, kl_loss, total_loss = [], [], []
            for batch_idx, (gene_exp, real_rna, cell_types) in enumerate(dataloader):
                optimizer.zero_grad()
                gene_exp = gene_exp.to(self.device)
                real_rna = real_rna.to(self.device)
                
                recon_x, mu, logvar = self.forward(gene_exp)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                if threshold:
                    zero_fraction = (real_rna == 0).float().mean()
                    th = torch.quantile(recon_x, zero_fraction)
                    recon_x = torch.where(recon_x < th, torch.tensor(0.0, device=recon_x.device), recon_x)
                
                rec_loss = self.recon_loss(recon_x, real_rna)
                loss = rec_loss + beta * KLD
                loss.backward()
                optimizer.step()
                
                total_loss.append(loss.item())
                kl_loss.append(KLD.item())
                r_loss.append(rec_loss.item())
            print(f"[Epoch {epoch+1}/{n_epochs}]: [Total loss: {np.mean(total_loss):.4f}] "
                  f"[Rec loss: {np.mean(r_loss):.4f}] [KLD: {np.mean(kl_loss):.4f}]")
        
        os.makedirs("weights", exist_ok=True)
        torch.save(self.state_dict(), f'weights/{name_weight}.pth')

    def generate_anndata(self, gene_exp, rna, threshold=False):
        model_cpu = self.to("cpu")
        fake_rna, _, _ = model_cpu(torch.tensor(gene_exp.X))
        fake_rna = fake_rna.detach().numpy()
        if threshold:
            zero_fraction = (rna.X == 0).mean()
            th = np.quantile(fake_rna, zero_fraction)
            fake_rna = np.where(fake_rna < th, 0, fake_rna)
        gene_exp_fake = rna.copy()
        gene_exp_fake.X = fake_rna
        return gene_exp_fake