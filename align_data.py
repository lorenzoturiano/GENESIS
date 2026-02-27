import numpy as np
import scanpy as sc
import anndata as ad

path = "/project/Wellcome_Discovery/datashare/lturiano/data/"
sriva = "/project/Wellcome_Discovery/datashare/sriva/GENESIS/"

def align_anndata_by_cell_type(
    rna: ad.AnnData,
    gex: ad.AnnData,
    cell_type_key: str = "cell_type",
    seed: int = 0,
    copy: bool = True,
):
    """
    Align two AnnData objects by cell type so that:
      1) They contain only the intersection of cell types.
      2) For each shared cell type, both are downsampled to the same count
         (the min count across the two datasets for that cell type).
      3) The resulting AnnData objects have the same number of cells and
         are ordered so that for every index i, rna_aligned.obs[cell_type_key][i]
         == gex_aligned.obs[cell_type_key][i].

    Returns
    -------
    rna_aligned, gex_aligned, info_dict
    """
    if cell_type_key not in rna.obs or cell_type_key not in gex.obs:
        raise KeyError(f"'{cell_type_key}' must be present in both .obs")

    # Work on views or copies
    rna0 = rna.copy() if copy else rna
    gex0 = gex.copy() if copy else gex

    # Ensure strings (avoid categorical mismatch headaches)
    rna0.obs[cell_type_key] = rna0.obs[cell_type_key].astype(str)
    gex0.obs[cell_type_key] = gex0.obs[cell_type_key].astype(str)

    # Shared cell types
    ct_rna = set(rna0.obs[cell_type_key].unique())
    ct_gex = set(gex0.obs[cell_type_key].unique())
    shared_ct = sorted(ct_rna.intersection(ct_gex))

    if len(shared_ct) == 0:
        raise ValueError("No shared cell types between the two AnnData objects.")

    # Subset to shared cell types only
    rna0 = rna0[rna0.obs[cell_type_key].isin(shared_ct)].copy()
    gex0 = gex0[gex0.obs[cell_type_key].isin(shared_ct)].copy()

    # Counts per cell type
    rna_counts = rna0.obs[cell_type_key].value_counts()
    gex_counts = gex0.obs[cell_type_key].value_counts()

    # Determine per-cell-type target (min across datasets)
    target = {ct: int(min(rna_counts.get(ct, 0), gex_counts.get(ct, 0))) for ct in shared_ct}
    # Drop any ct that ends up with 0 (paranoia / safety)
    target = {ct: n for ct, n in target.items() if n > 0}
    shared_ct = sorted(target.keys())

    if len(shared_ct) == 0:
        raise ValueError("After computing min counts, no cell types have >0 cells in both datasets.")

    rng = np.random.default_rng(seed)

    def _sample_indices(adata_obj: ad.AnnData, ct: str, n: int):
        idx = np.where(adata_obj.obs[cell_type_key].values == ct)[0]
        if len(idx) < n:
            raise ValueError(f"Not enough cells for cell type '{ct}' (needed {n}, found {len(idx)}).")
        return rng.choice(idx, size=n, replace=False)

    # Sample indices per CT for each dataset
    rna_sel = np.concatenate([_sample_indices(rna0, ct, target[ct]) for ct in shared_ct])
    gex_sel = np.concatenate([_sample_indices(gex0, ct, target[ct]) for ct in shared_ct])

    # Reorder within each CT so alignment is consistent/deterministic:
    # we build a final "ct-ordered" list where all cells of ct1 come first, then ct2, ...
    # and within each CT we keep a stable order by sorting by the selected indices.
    def _ct_ordered(adata_obj: ad.AnnData, selected_idx: np.ndarray):
        adata_sub = adata_obj[selected_idx].copy()
        # stable sort by (cell_type, original selection index order)
        # We'll sort by cell_type then by obs_names to keep it deterministic.
        order = np.lexsort((adata_sub.obs_names.values, adata_sub.obs[cell_type_key].values))
        return adata_sub[order].copy()

    rna_aligned = _ct_ordered(rna0, rna_sel)
    gex_aligned = _ct_ordered(gex0, gex_sel)

    # Final sanity check: aligned cell types at every position
    if not np.array_equal(
        rna_aligned.obs[cell_type_key].values,
        gex_aligned.obs[cell_type_key].values
    ):
        raise RuntimeError("Alignment failed: cell types are not identical row-by-row after ordering.")

    info = {
        "shared_cell_types": shared_ct,
        "target_per_cell_type": target,
        "n_cells_final": int(rna_aligned.n_obs),
    }
    return rna_aligned, gex_aligned, info


rna = sc.read_h5ad(sriva + "rna_filt.h5ad")
gex = sc.read_h5ad(path + "gex_filt.h5ad")

rna_aligned, gex_aligned, info = align_anndata_by_cell_type(rna, gex, cell_type_key="cell_type", seed=42)
print(info)
assert (rna_aligned.obs["cell_type"].values == gex_aligned.obs["cell_type"].values).all()

rna_aligned.write_h5ad(path + "rna_filt_aligned.h5ad", compression="gzip")
gex_aligned.write_h5ad(path + "gex_filt_aligned.h5ad", compression="gzip")

