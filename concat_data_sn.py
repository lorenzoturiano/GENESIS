from pathlib import Path
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

path = "/project/Wellcome_Discovery/datashare/lturiano/data/"

def cap_data(adata, cap=10_000, cell_type_key="cell_type", seed=0):
    rng = np.random.default_rng(seed)

    ct = adata.obs[cell_type_key].to_numpy()
    keep_mask = np.zeros(adata.n_obs, dtype=bool)

    for c in np.unique(ct):
        idx = np.where(ct == c)[0]
        if idx.size <= cap:
            keep_mask[idx] = True
        else:
            keep_idx = rng.choice(idx, size=cap, replace=False)
            keep_mask[keep_idx] = True

    keep_idx = np.sort(np.where(keep_mask)[0])  # preserve original order
    return adata[keep_idx].copy()

def drop_rare_cell_types(
    adata,
    cell_type_key: str = "cell_type",
    min_count: int = 1000,
    copy: bool = True,
    verbose: bool = True,
):
    """
    Drop all cells belonging to cell types that have < min_count cells.

    Returns:
      adata_filt: filtered AnnData
      kept_cell_types: list[str]
      dropped_cell_types: list[tuple[str, int]]  # (cell_type, count)
    """
    if cell_type_key not in adata.obs:
        raise KeyError(f"'{cell_type_key}' not found in adata.obs")

    ct = adata.obs[cell_type_key].astype(str)
    counts = ct.value_counts()

    kept = counts[counts >= min_count].index.tolist()
    dropped = counts[counts < min_count]
    dropped_list = [(k, int(v)) for k, v in dropped.items()]

    mask = ct.isin(kept).to_numpy()
    adata_filt = adata[mask].copy() if copy else adata[mask]

    if verbose:
        print(f"Total cells before: {adata.n_obs:,}")
        print(f"Total cells after : {adata_filt.n_obs:,}")
        print(f"Kept cell types   : {len(kept)}")
        print(f"Dropped cell types: {len(dropped_list)}")
        if dropped_list:
            # show up to first 20
            preview = ", ".join([f"{k}({v})" for k, v in dropped_list[:20]])
            more = "" if len(dropped_list) <= 20 else f", ... (+{len(dropped_list)-20} more)"
            print("Dropped:", preview + more)

    return adata_filt, kept, dropped_list


heart_gex = sc.read_h5ad(path + "gex/heart_10x_filt.h5ad")
endoderm_gex = sc.read_h5ad(path + "gex/endoderm_10x_filt.h5ad")
lymph_gex = sc.read_h5ad(path + "gex/kidney_10x_filt.h5ad")
kidney_gex = sc.read_h5ad(path + "gex/lymph_10x_normal_filt.h5ad")

data_gex = [heart_gex, endoderm_gex, lymph_gex, kidney_gex]
    
# concatenate:
data_gex_concat = ad.concat(
    data_gex,          # dict or list
    join="outer",
    axis=0,
    index_unique=None,  # <- keeps original obs_names unchanged
    merge="first",
    fill_value=0.0
)

data_gex_concat = cap_data(data_gex_concat, cap=15_000, cell_type_key="cell_type", seed=42)

data_gex_concat, _, _ = drop_rare_cell_types(data_gex_concat, min_count=1_500)

print(data_gex_concat.obs["cell_type"].value_counts())

if sp.issparse(data_gex_concat.X):
    data_gex_concat.X.data = data_gex_concat.X.data.astype(np.float32, copy=False)
else:
    data_gex_concat.X = data_gex_concat.X.astype(np.float32, copy=False)

data_gex_concat.write_h5ad(path + "gex_filt_try.h5ad", compression="gzip")