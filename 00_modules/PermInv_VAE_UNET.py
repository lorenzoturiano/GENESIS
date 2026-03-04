# PermInv_VAE_UNET.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def build_ensg_to_idx(gene_ids_inorder):
    """
    Build ENSG_to_idx mapping from a gene list in the EXACT order used as input columns.
    This is the metadata you want to ship with the trained model.

    Parameters
    ----------
    gene_ids_inorder : sequence of str
        Gene identifiers (e.g., ENSG...) in the same order as the columns of the training GEX matrix.

    Returns
    -------
    dict[str, int]
        Mapping ENSG -> column/embedding row index.
    """
    if gene_ids_inorder is None:
        raise ValueError("gene_ids_inorder is None. Provide the input gene list (ENSG) in column order.")

    gene_ids_inorder = [str(g) for g in gene_ids_inorder]
    if len(set(gene_ids_inorder)) != len(gene_ids_inorder):
        # Duplicate gene IDs would make a 1:1 mapping ambiguous.
        # Better to fail fast than ship incorrect metadata.
        raise ValueError("Duplicate gene IDs detected in gene_ids_inorder. Cannot build a 1:1 ENSG_to_idx mapping.")

    return {g: int(i) for i, g in enumerate(gene_ids_inorder)}


def save_gex_vocab_metadata(out_dir, name_prefix, gene_ids_inorder, extra_meta=None):
    """
    Save the mapping + gene list as model metadata for downstream inference use.
    This does NOT implement any reorder logic; it just records what training used.

    Files written:
      - {name_prefix}_ENSG_to_idx.json
      - {name_prefix}_gex_genes_inorder.txt
      - {name_prefix}_gex_vocab_meta.json
    """
    os.makedirs(out_dir, exist_ok=True)

    ENSG_to_idx = build_ensg_to_idx(gene_ids_inorder)

    map_path = os.path.join(out_dir, f"{name_prefix}_ENSG_to_idx.json")
    genes_path = os.path.join(out_dir, f"{name_prefix}_gex_genes_inorder.txt")
    meta_path = os.path.join(out_dir, f"{name_prefix}_gex_vocab_meta.json")

    with open(map_path, "w") as f:
        json.dump(ENSG_to_idx, f)

    with open(genes_path, "w") as f:
        for g in gene_ids_inorder:
            f.write(str(g) + "\n")

    meta = {
        "n_genes": int(len(gene_ids_inorder)),
        "mapping_definition": "ENSG_to_idx maps gene_id (ENSG) to the input column index / embedding row index used at training time.",
        "notes": "To apply the model to a new dataset with different genes/order, you will later need to align/reorder or build gene_idx using this mapping.",
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {"map": map_path, "genes": genes_path, "meta": meta_path}


# A simple fully-connected residual block.
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=False)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity


# Weighted-mean pooling over genes (assumes fixed input gene order).
# NOTE: True permutation invariance across datasets requires gene_idx/reorder logic,
# which you explicitly do NOT want to implement right now. So we keep the simple version.
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
        # x: (B, G) non-negative gene expression values, columns must match embedding row order
        w = x / (x.sum(dim=1, keepdim=True) + self.eps)   # (B, G)
        E = self.embedding.weight                         # (G, D)
        return w @ E                                      # (B, D)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            self.use_skips = False
            self.model = nn.Sequential(
                nn.Linear(input_dim, latent_dim * 2),
                nn.ReLU(inplace=False),
            )
            self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        else:
            self.use_skips = True
            self.layers = nn.ModuleList()
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0]),
                    nn.ReLU(inplace=False),
                )
            )
            for i in range(1, len(hidden_dims)):
                self.layers.append(ResidualBlock(hidden_dims[i - 1], hidden_dims[i]))
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        if not self.use_skips:
            x = self.model(x)
            return self.fc_mu(x), self.fc_logvar(x), None

        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, skip_connections


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, reversed_hidden=None, normalized=False):
        super().__init__()
        self.normalized = normalized
        self.last_activation = nn.Sigmoid() if normalized else nn.ReLU(inplace=False)

        if reversed_hidden is None:
            self.use_skips = False
            self.model = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(inplace=False),
                nn.Linear(latent_dim * 2, output_dim),
                self.last_activation,
            )
        else:
            self.use_skips = True
            self.fc_initial = nn.Linear(latent_dim, reversed_hidden[0])
            self.decoder_blocks = nn.ModuleList()
            for i in range(1, len(reversed_hidden)):
                in_features = reversed_hidden[i - 1] + reversed_hidden[i]
                out_features = reversed_hidden[i]
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU(inplace=False),
                    )
                )
            self.final_layer = nn.Linear(reversed_hidden[-1], output_dim)

    def forward(self, z, skip_connections=None):
        if not self.use_skips or skip_connections is None:
            return self.model(z)

        x = self.fc_initial(z)
        skips = list(reversed(skip_connections[:-1]))
        for block, skip_feat in zip(self.decoder_blocks, skips):
            x = torch.cat([x, skip_feat], dim=-1)
            x = block(x)
        x = self.final_layer(x)
        x = self.last_activation(x)
        return x


class RNA_VAE_UNET(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        output_dim,
        hidden_dims=(2048, 1024, 512),
        normalized=False,
        recon_loss=None,
        device=None,
        use_gene_pool=True,
        gene_emb_dim=512,
        pool_eps=1e-8,
        gene_emb_init=None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recon_loss = recon_loss if recon_loss is not None else nn.MSELoss()

        self.use_gene_pool = bool(use_gene_pool)
        if self.use_gene_pool:
            # IMPORTANT: Here the embedding rows are assumed to match the training GEX column order.
            self.gene_pool = GeneExpressionWeightedPool(
                n_genes=input_dim,
                emb_dim=gene_emb_dim,
                eps=pool_eps,
                init_weight=gene_emb_init,
            ).to(self.device)
            encoder_input_dim = gene_emb_dim
        else:
            self.gene_pool = None
            encoder_input_dim = input_dim

        self.encoder = Encoder(encoder_input_dim, latent_dim, hidden_dims=list(hidden_dims)).to(self.device)
        reversed_hidden = list(reversed(list(hidden_dims)))
        self.decoder = Decoder(latent_dim, output_dim, reversed_hidden=reversed_hidden, normalized=normalized).to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.use_gene_pool:
            x = self.gene_pool(x)  # (B, gene_emb_dim)

        mu, logvar, skip_connections = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, skip_connections)
        return recon_x, mu, logvar

    def fit(
        self,
        dataloader,
        n_epochs=10,
        optimizer=None,
        beta=0.1,
        lr=1e-4,
        threshold=False,
        name_prefix="PermInv_VAE_UNET",
        out_dir="weights",
        # === metadata saving (no reorder logic) ===
        gex_gene_ids_inorder=None,   # pass gex.var["gene_id"] (or var_names) here if you want to save mapping
        extra_vocab_meta=None,       # optional dict to add to vocab meta json
    ):
        """
        Train the model and save:
          - weights/{name_prefix}.pth
          - (optional) weights/{name_prefix}_ENSG_to_idx.json + gene list + meta
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train(True)  # set module mode

        for epoch in range(n_epochs):
            ds = getattr(dataloader, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            r_loss, kl_loss, total_loss = [], [], []
            for batch in dataloader:
                # Support both old batches (x,y,ct) and newer variants.
                # Training logic expects: gene_exp, real_rna, cell_types
                if len(batch) == 3:
                    gene_exp, real_rna, cell_types = batch
                else:
                    # If user later returns additional fields, keep first/third/last structure sane.
                    # Example: (gene_exp, gene_idx, real_rna, cell_types) -> ignore gene_idx for now.
                    gene_exp, _, real_rna, cell_types = batch

                optimizer.zero_grad()
                gene_exp = gene_exp.to(self.device)
                real_rna = real_rna.to(self.device)

                recon_x, mu, logvar = self.forward(gene_exp)
                kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
                KLD = kld_per_sample.mean()

                if threshold:
                    zero_fraction = (real_rna == 0).float().mean()
                    th = torch.quantile(recon_x, zero_fraction)
                    recon_x = torch.where(recon_x < th, torch.tensor(0.0, device=recon_x.device), recon_x)

                rec_loss = self.recon_loss(recon_x, real_rna)
                
                # warm up
                beta_target = beta
                warmup_epochs = max(1, int(0.3 * n_epochs))  # e.g. 30% of epochs
                beta_t = beta_target * min(1.0, (epoch + 1) / warmup_epochs)

                loss = rec_loss + beta_t * KLD
                # loss = rec_loss + beta * KLD
                loss.backward()
                optimizer.step()

                total_loss.append(float(loss.item()))
                kl_loss.append(float(KLD.item()))
                r_loss.append(float(rec_loss.item()))

            print(
                f"[Epoch {epoch+1}/{n_epochs}]: "
                f"[Total loss: {np.mean(total_loss):.4f}] "
                f"[Rec loss: {np.mean(r_loss):.4f}] "
                f"[KLD: {np.mean(kl_loss):.4f}]"
            )

        os.makedirs(out_dir, exist_ok=True)
        weight_path = os.path.join(out_dir, f"{name_prefix}.pth")
        torch.save(self.state_dict(), weight_path)

        # Optional: save ENSG_to_idx mapping used for the GEX input order at training
        saved_vocab = None
        if gex_gene_ids_inorder is not None:
            saved_vocab = save_gex_vocab_metadata(
                out_dir=out_dir,
                name_prefix=name_prefix,
                gene_ids_inorder=gex_gene_ids_inorder,
                extra_meta=extra_vocab_meta,
            )

        return {"weights": weight_path, "vocab": saved_vocab}

    @torch.no_grad()
    def generate_anndata(self, gene_exp, rna, threshold=False):
        """
        NOTE: This uses the same assumption as training: gene_exp.X columns match training GEX order.
        (No reorder logic implemented here by design.)
        """
        model_cpu = self.to("cpu")
        model_cpu.eval()

        X = gene_exp.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        x = torch.tensor(X, dtype=torch.float32)

        fake_rna, _, _ = model_cpu(x)
        fake_rna = fake_rna.detach().numpy()

        if threshold:
            zero_fraction = (rna.X == 0).mean()
            th = np.quantile(fake_rna, zero_fraction)
            fake_rna = np.where(fake_rna < th, 0, fake_rna)

        gene_exp_fake = rna.copy()
        gene_exp_fake.X = fake_rna
        return gene_exp_fake