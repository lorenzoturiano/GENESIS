#!/usr/bin/env python
# coding: utf-8

import torch, sys, time
import scanpy as sc
from pathlib import Path

sys.path.append('modules/')

from PermInv_VAE_UNET import RNA_VAE_UNET
from data_loader import CustomDataloader

BASE = Path("/project/Wellcome_Discovery/datashare/lturiano/data")
sriva = Path("/project/Wellcome_Discovery/datashare/sriva/GENESIS/")

rna = sc.read_h5ad(sriva / "RNA_filt_log_subset.h5ad")
gex = sc.read_h5ad(sriva / "GEX_filt_log_subset.h5ad")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom = CustomDataloader(gex, rna, batch_size=64, seed=42)
dataloader = custom.get_dataloader()

vae = RNA_VAE_UNET(
    input_dim=gex.n_vars,
    output_dim=rna.n_vars,
    latent_dim=64,
    hidden_dims=[1024, 512, 256],   # halves starting from 2048
    use_gene_pool=True,
    gene_emb_dim=2048
)

t0 = time.perf_counter()
vae.train(dataloader, n_epochs=100, beta=0.1, threshold=False, name_weight="PermInv_VAE_UNET_log")
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

rna_fake.write(BASE / "breast_fake_RNA_PermInv_VAE_UNET.h5ad", compression="gzip")
print("Object saved")