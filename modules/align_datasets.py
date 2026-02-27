import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp
from pathlib import Path

BASE = Path("/project/Wellcome_Discovery/datashare/lturiano/GENESIS/data")

def make_paired_by_cell_type(
    sn: ad.AnnData,
    sc: ad.AnnData,
    cell_type_key: str = "cell_type",
    cap_per_ct: int | None = 10_000,
    min_samples: int = 1,          # NEW: drop ct if SN or SC has < min_samples
    seed: int = 0,
    sample: bool = True,           # True: random sample within ct; False: take first k deterministically
    gene_policy: str = "intersection",
):
    # --- Step 1: shared cell types ---
    sn_ct = sn.obs[cell_type_key].astype(str)
    sc_ct = sc.obs[cell_type_key].astype(str)
    shared_cts = sorted(set(sn_ct) & set(sc_ct))
    if len(shared_cts) == 0:
        raise ValueError("No shared cell types between SN and SC.")

    sn_sub = sn[sn_ct.isin(shared_cts)].copy()
    sc_sub = sc[sc_ct.isin(shared_cts)].copy()

    # --- Step 1b: drop cell types with too few samples in either modality (NEW) ---
    sn_counts = sn_sub.obs[cell_type_key].astype(str).value_counts()
    sc_counts = sc_sub.obs[cell_type_key].astype(str).value_counts()

    eligible_cts = []
    dropped_cts = []
    for ct in shared_cts:
        n_sn = int(sn_counts.get(ct, 0))
        n_sc = int(sc_counts.get(ct, 0))
        if min(n_sn, n_sc) >= min_samples:
            eligible_cts.append(ct)
        else:
            dropped_cts.append((ct, n_sn, n_sc))

    if len(eligible_cts) == 0:
        msg = "No cell types meet min_samples in both SN and SC."
        if dropped_cts:
            msg += " Dropped (ct, n_sn, n_sc): " + ", ".join([f"{ct}:{n_sn}/{n_sc}" for ct, n_sn, n_sc in dropped_cts[:20]])
            if len(dropped_cts) > 20:
                msg += f", ... (+{len(dropped_cts)-20} more)"
        raise ValueError(msg)

    sn_sub = sn_sub[sn_sub.obs[cell_type_key].astype(str).isin(eligible_cts)].copy()
    sc_sub = sc_sub[sc_sub.obs[cell_type_key].astype(str).isin(eligible_cts)].copy()

    # --- Step 2: build row-wise pairs by cell type ---
    rng = np.random.default_rng(seed)
    sn_take, sc_take = [], []

    for ct in eligible_cts:
        sn_idx = np.where(sn_sub.obs[cell_type_key].astype(str).to_numpy() == ct)[0]
        sc_idx = np.where(sc_sub.obs[cell_type_key].astype(str).to_numpy() == ct)[0]

        k = min(len(sn_idx), len(sc_idx))
        if cap_per_ct is not None:
            k = min(k, cap_per_ct)

        # min_samples already enforced, but keep guard anyway
        if k == 0:
            continue

        if sample:
            sn_sel = rng.choice(sn_idx, size=k, replace=False)
            sc_sel = rng.choice(sc_idx, size=k, replace=False)
        else:
            sn_sel = sn_idx[:k]
            sc_sel = sc_idx[:k]

        # deterministic order within each ct block
        sn_sel = np.sort(sn_sel)
        sc_sel = np.sort(sc_sel)

        sn_take.append(sn_sel)
        sc_take.append(sc_sel)

    if len(sn_take) == 0:
        raise ValueError("After balancing, no pairs were created (all k=0).")

    sn_take = np.concatenate(sn_take)
    sc_take = np.concatenate(sc_take)

    sn_paired = sn_sub[sn_take].copy()
    sc_paired = sc_sub[sc_take].copy()

    # --- Step 3: align genes ---
    if gene_policy == "intersection":
        genes = sn_paired.var_names.intersection(sc_paired.var_names)
        sn_paired = sn_paired[:, genes].copy()
        sc_paired = sc_paired[:, genes].copy()
        sc_paired = sc_paired[:, sn_paired.var_names].copy()
    else:
        raise ValueError("Only gene_policy='intersection' is implemented safely here.")

    # --- Step 4: safety checks ---
    if sn_paired.n_obs != sc_paired.n_obs:
        raise AssertionError("SN and SC have different n_obs after pairing.")
    if sn_paired.n_vars != sc_paired.n_vars:
        raise AssertionError("SN and SC have different n_vars after gene alignment.")
    if not (sn_paired.var_names == sc_paired.var_names).all():
        raise AssertionError("SN and SC var_names are not identical / ordered the same.")
    if not (sn_paired.obs[cell_type_key].astype(str).to_numpy()
            == sc_paired.obs[cell_type_key].astype(str).to_numpy()).all():
        raise AssertionError("Row-wise cell_type mismatch between SN and SC.")

    # Optional: add a pair_id
    sn_paired.obs["pair_id"] = np.arange(sn_paired.n_obs).astype(str)
    sc_paired.obs["pair_id"] = sn_paired.obs["pair_id"].values

    return sn_paired, sc_paired, eligible_cts, dropped_cts


# main

sn_paired, sc_paired, shared_cts, drop_cts = make_paired_by_cell_type(
    all_organs_sn_filt_1,
    all_organs_sc_filt_1,
    cell_type_key = "cell_type",
    cap_per_ct = 20_000,
    min_samples = 1000,          # drop ct if SN or SC has < min_samples
    seed = 42,
    sample = True,           # True: random sample within ct; False: take first k deterministically
    gene_policy = "intersection",
)

print(sn_paired.shape, sc_paired.shape)
print("Shared cell types:", len(shared_cts))
print("Dropped cell types:", drop_cts)

sn_paired.write_h5ad(BASE / "sn_paired.h5ad", compression="gzip")
sc_paired.write_h5ad(BASE / "sc_paired.h5ad", compression="gzip")