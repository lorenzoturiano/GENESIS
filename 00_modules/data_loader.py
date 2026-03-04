# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class DualAnnDataDataset(Dataset):
    def __init__(self, adata1, adata2, cell_type_key, base_seed: int = 0):
        """
        adata1 = input (e.g., multiome GEX)
        adata2 = target (e.g., scRNA)
        They do NOT need to be paired by barcode. Only cell_type_key is required.

        IMPORTANT (for your current training stage):
        - This loader assumes adata1.X columns are already aligned to the model's expected input gene order.
        - No gene reordering / ENSG-based masking happens here (by design, per your request).
        """
        self.adata1 = adata1
        self.adata2 = adata2
        self.cell_type_key = cell_type_key
        self.base_seed = int(base_seed)

        # cell types for each source cell (adata1)
        self.src_ct = np.asarray(self.adata1.obs[self.cell_type_key].values)
        self.cell_types = self.src_ct  # for easy access by batch sampler

        # Build: cell_type -> list of indices in adata2
        tgt_ct = np.asarray(self.adata2.obs[self.cell_type_key].values)
        self.ct_to_tgt_indices = {}
        for ct in np.unique(tgt_ct):
            self.ct_to_tgt_indices[ct] = np.where(tgt_ct == ct)[0]

        # Validate: every source ct exists in target
        missing = sorted(set(np.unique(self.src_ct)) - set(self.ct_to_tgt_indices.keys()))
        if missing:
            raise ValueError(f"Some cell types exist in adata1 but not in adata2: {missing}")

        # Will be filled by set_epoch()
        self.target_map = np.empty(self.adata1.n_obs, dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        """
        Call this at the start of each epoch to resample targets within each cell type.
        Mapping is fixed during the epoch.
        """
        rng = np.random.default_rng(self.base_seed + int(epoch))

        # For each cell type: sample target indices (with replacement) for each source cell in that ct
        for ct in np.unique(self.src_ct):
            src_idx = np.where(self.src_ct == ct)[0]
            tgt_pool = self.ct_to_tgt_indices[ct]
            sampled_tgt = rng.choice(tgt_pool, size=len(src_idx), replace=True)
            self.target_map[src_idx] = sampled_tgt

    def __len__(self):
        return self.adata1.n_obs

    def __getitem__(self, idx):
        j = int(self.target_map[idx])

        x = self.adata1.X[idx]
        y = self.adata2.X[j]

        # convert to dense if needed (common with sparse AnnData)
        if not isinstance(x, np.ndarray):
            x = x.toarray().ravel()
        else:
            x = np.asarray(x).ravel()

        if not isinstance(y, np.ndarray):
            y = y.toarray().ravel()
        else:
            y = np.asarray(y).ravel()

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        ct = self.src_ct[idx]
        return x, y, ct


class CellTypeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Groups indices by cell type so that each batch is composed
        entirely of cells from one cell population.

        Parameters:
         - dataset: Instance of DualAnnDataDataset.
         - batch_size: Desired batch size.
        """
        self.batch_size = int(batch_size)

        # Build a dictionary mapping cell type to list of indices.
        self.cell_type_to_indices = {}
        for idx, ct in enumerate(dataset.cell_types):
            self.cell_type_to_indices.setdefault(ct, []).append(idx)

        # Shuffle indices within each cell type.
        for ct in self.cell_type_to_indices:
            np.random.shuffle(self.cell_type_to_indices[ct])

        # Shuffle the order of cell types.
        self.groups = list(self.cell_type_to_indices.items())
        np.random.shuffle(self.groups)

        # Pre-calculate the batches per group.
        self.batches = []
        for ct, indices in self.groups:
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i : i + self.batch_size])

        # Shuffle the order of batches so that cell type groups are not strictly sequential.
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class CustomDataloader(object):
    def __init__(self, adata1, adata2, batch_size=32, cell_type_key="cell_type", seed=None, base_seed: int = 0):
        if seed is not None:
            np.random.seed(int(seed))

        dataset = DualAnnDataDataset(adata1, adata2, cell_type_key=cell_type_key, base_seed=base_seed)

        sampler = CellTypeBatchSampler(dataset, batch_size=batch_size)
        self.dataloader = DataLoader(dataset, batch_sampler=sampler)

    def get_dataloader(self):
        return self.dataloader