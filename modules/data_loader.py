import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

class DualAnnDataDataset(Dataset):
    def __init__(self, adata1, adata2, cell_type_key):
        """
        Parameters:
         - adata1, adata2: AnnData objects with the same number of cells.
         - cell_type_key: key in adata.obs that identifies cell populations.
        """
        if adata1.shape[0] != adata2.shape[0]:
            raise ValueError("Both AnnData objects must have the same number of cells.")
        
        self.adata1 = adata1
        self.adata2 = adata2
        # Assumes both adata have the same cell order for the given cell_type_key.
        self.cell_types = adata1.obs[cell_type_key].values
        
    def __len__(self):
        return self.adata1.shape[0]
    
    def __getitem__(self, idx):
        # Get expression data for cell at index idx.
        x1 = self.adata1.X[idx]
        x2 = self.adata2.X[idx]
        
        # If stored as sparse, convert to dense.
        if hasattr(x1, "toarray"):
            x1 = x1.toarray().squeeze()
        if hasattr(x2, "toarray"):
            x2 = x2.toarray().squeeze()
        
        # Convert to torch tensors.
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        cell_type = self.cell_types[idx]
        return x1, x2, cell_type


class CellTypeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Groups indices by cell type so that each batch is composed
        entirely of cells from one cell population.
        
        Parameters:
         - dataset: Instance of DualAnnDataDataset.
         - batch_size: Desired batch size.
        """
        self.batch_size = batch_size
        
        # Build a dictionary mapping cell type to list of indices.
        self.cell_type_to_indices = {}
        for idx, ct in enumerate(dataset.cell_types):
            self.cell_type_to_indices.setdefault(ct, []).append(idx)
        
        # Optionally, shuffle indices within each cell type.
        for ct in self.cell_type_to_indices:
            np.random.shuffle(self.cell_type_to_indices[ct])
        
        # Create a list of (cell type, indices) pairs and shuffle the order of cell types.
        self.groups = list(self.cell_type_to_indices.items())
        np.random.shuffle(self.groups)
        
        # Pre-calculate the batches per group.
        self.batches = []
        for ct, indices in self.groups:
            # Split indices for this cell type into chunks.
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                self.batches.append(batch)
                
        # Shuffle the order of batches so that cell type groups are not strictly sequential.
        np.random.shuffle(self.batches)
    
    def __iter__(self):
        # Yield one batch (list of indices) at a time.
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


class CustomDataloader(object):
    def __init__(self, adata1, adata2, batch_size=32, cell_type_key = "cell_type", seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.adata1 = adata1
        self.adata2 = adata2
        
        
        # Assuming adata1 and adata2 are already loaded and have an observation column "cell_type".
        dataset = DualAnnDataDataset(adata1, adata2, cell_type_key=cell_type_key)

        # Use the custom batch sampler in the DataLoader.
        sampler    = CellTypeBatchSampler(dataset, batch_size=batch_size)
        self.dataloader = DataLoader(dataset, batch_sampler=sampler)
    
    def get_dataloader(self):
        return self.dataloader
    
    