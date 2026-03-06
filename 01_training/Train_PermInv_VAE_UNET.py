#!/usr/bin/env python
# coding: utf-8

import torch, sys, time
import scanpy as sc
import numpy as np
import json
from pathlib import Path

sys.path.append('00_modules/')
from PermInv_VAE_UNET import RNA_VAE_UNET
from data_loader import CustomDataloader

data = Path("/project/Wellcome_Discovery/datashare/lturiano/data/")
home = Path("/project/Wellcome_Discovery/lturiano/GENESIS/")
sriva = Path("/project/Wellcome_Discovery/datashare/sriva/GENESIS/")

rna = sc.read_h5ad(sriva / "RNA_filt_log_subset.h5ad")
gex = sc.read_h5ad(sriva / "GEX_filt_log_subset.h5ad")

def downsample_anndata(adata, max_cells_per_type=7000, seed=42):
    rng = np.random.default_rng(seed)
    keep = []
    for ct in adata.obs["cell_type"].unique():
        idx = adata.obs_names[adata.obs["cell_type"] == ct].to_numpy()
        if len(idx) > max_cells_per_type:
            keep.extend(rng.choice(idx, max_cells_per_type, replace=False))
        else:
            keep.extend(idx)
    keep = np.array(keep)
    return adata[keep].copy()

rna = downsample_anndata(rna)
gex = downsample_anndata(gex)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom = CustomDataloader(gex, rna, batch_size=64, seed=42)
dataloader = custom.get_dataloader()

gene_emb_dim = 2048
hidden_dims  = [1024, 512, 256]
latent_dim   = 64

vae = RNA_VAE_UNET(
    input_dim=gex.n_vars,
    output_dim=rna.n_vars,
    latent_dim=latent_dim,
    hidden_dims=hidden_dims,
    use_gene_pool=True,
    gene_emb_dim=gene_emb_dim,
)

t0 = time.perf_counter()
vae.fit(
        dataloader,
        n_epochs=100,
        beta=0.2,
        lr=1e-4,
        threshold=False,
        name_prefix="PermInv_VAE_UNET",
        out_dir=home / "02_weights/",
        gex_gene_ids_inorder=gex.var_names.tolist(),
        extra_vocab_meta=None,
    )
t1 = time.perf_counter()

elapsed_s = t1 - t0
elapsed_min = elapsed_s / 60
if elapsed_min >= 60:
    elapsed_hr = elapsed_min / 60
    print(f"Training time: {elapsed_hr:.2f} hours ({elapsed_min:.1f} min)")
else:
    print(f"Training time: {elapsed_min:.2f} minutes")

rna_fake = vae.generate_anndata(gex, rna, threshold=False)

print("Max value gene_exp:", gex.X.max())
print("Max value rna_exp :", rna.X.max())
print("Max value rna_fake:", rna_fake.X.max())

file_name = "correct_ds_breast_nonzeroMSE_fake_PermInv_VAE_UNET.h5ad"
rna_fake.write(data / file_name, compression="gzip")
print(f"Object: {file_name} saved!")