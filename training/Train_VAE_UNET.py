#!/usr/bin/env python
# coding: utf-8

import torch, sys, time
import scanpy as sc
from pathlib import Path

sys.path.append('modules/')

from VAE_UNET import RNA_VAE_UNET
from data_loader import CustomDataloader

BASE = Path("/project/Wellcome_Discovery/datashare/lturiano/data")

rna_sub      = sc.read_h5ad(BASE / "rna_filt_aligned.h5ad")
gene_exp_sub = sc.read_h5ad(BASE / "gex_filt_aligned.h5ad")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom = CustomDataloader(gene_exp_sub, rna_sub, batch_size=64, seed=42)
dataloader = custom.get_dataloader()

vae = RNA_VAE_UNET(input_dim=gene_exp_sub.n_vars, output_dim=rna_sub.n_vars, latent_dim=32, normalized=False, device=device)

t0 = time.perf_counter()
vae.train(dataloader, n_epochs=100, beta=0.1, threshold=False, name_weight="VAE_UNET_log")
t1 = time.perf_counter()

elapsed_s = t1 - t0
elapsed_min = elapsed_s / 60
if elapsed_min >= 60:
    elapsed_hr = elapsed_min / 60
    print(f"Training time: {elapsed_hr:.2f} hours ({elapsed_min:.1f} min)")
else:
    print(f"Training time: {elapsed_min:.2f} minutes")

rna_fake = vae.generate_anndata(gene_exp_sub, rna_sub, threshold=False)

print("Max value gene_exp:", gene_exp_sub.X.max())
print("Max value rna_exp :", rna_sub.X.max())
print("Max value rna_fake:", rna_fake.X.max())

rna_fake.write(BASE / "fake_RNA_VAE_UNET_log.h5ad", compression="gzip")
print("Object saved")