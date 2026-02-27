import numpy as np
import pandas as pd
import anndata
from scipy.special import kl_div

MARKERS = {
    "heart": {
        "endothelial cell": ["PECAM1", "VWF", "KDR", "CLDN5", "ENG"],
        "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
        "lymphocyte": ["PTPRC", "CD3D", "CD3E", "IL7R", "CD2"],
        "mural_cell": ["RGS5", "PDGFRB", "ACTA2", "TAGLN", "MYH11"],
        "myeloid cell": ["LYZ", "CD68", "CSF1R", "HLA-DRA", "ITGAM"]
    },
    
    "lung": {
        "endothelial_cell": ["PECAM1", "VWF", "KDR", "CLDN5", "ENG"],
        "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    },
    
    "kidney": {
        "endothelial_cell": ["PECAM1", "VWF", "KDR"],
    },
}

def compute_gene_distribution(X, aggregate, epsilon):
    """
    Computes a probability distribution over genes.
    
    Parameters:
    -----------
    X : array-like
        Gene expression matrix (cells x genes)
    aggregate : bool
        Whether to aggregate across cells
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    P : array-like
        Probability distribution
    """
    
    # Sum expression values for each gene across all cells.
    gene_sums = X.sum(axis=0) + epsilon
    
    if aggregate and X.shape[1] > 1:
        # Normalize to create a probability distribution.
        P = gene_sums / gene_sums.sum()
    else:
        # This creates, for each gene, a probability distribution over cells.
        P = X/gene_sums
    
    return P

# Good handling of different input types
def get_array(data):
    """
    Converts different input types to numpy array.
    
    Parameters:
    -----------
    data : AnnData, DataFrame, or array-like
        Input data to convert
    
    Returns:
    --------
    X : numpy.ndarray
        Converted array
    """
    
    if type(data) is anndata._core.anndata.AnnData:
        X = data.X.toarray() if hasattr(data.X, "toarray") else data.X
    elif type(data) is pd.core.frame.DataFrame:
        X = data.values
    else:
        X = np.array(data)
    
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    return X

def kl_divergence(P, Q, epsilon):
    """
    Computes the Kullback-Leibler divergence D_KL(P || Q).
    
    Parameters:
    -----------
    P, Q : array-like
        Input distributions
    epsilon : float
        Small value to avoid numerical issues
    
    Returns:
    --------
    float
        KL divergence value
    """
    KL = kl_div(P+epsilon, Q+epsilon)
    return KL.sum(0)        

# Good implementation of JS divergence
def jensen_shannon_divergence(P, Q, epsilon):
    """
    Computes the Jensen-Shannon Divergence between distributions P and Q.
    JSD(P||Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), where M = 0.5 * (P + Q).
    """
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M, epsilon) + 0.5 * kl_divergence(Q, M, epsilon)

def calculate_distance(data1, data2, distance="JS", aggregate=True, reduction='mean', epsilon=1e-10):
    """
    Calculate distance between two gene expression matrices.
    
    Parameters:
    -----------
    data1, data2 : array-like
        Input gene expression matrices
    distance : str
        Type of distance metric ('KL' or 'JS')
    aggregate : bool
        Whether to aggregate across cells
    reduction : str
        How to reduce the final metric ('mean', 'sum', or None) if not aggregated across cells
    epsilon : float
        Small value to avoid numerical issues
    
    Returns:
    --------
    float or array
        Computed distance metric
    """
    
    P = get_array(data1)
    Q = get_array(data2)
    
    if P.shape[1] != Q.shape[1]:
        raise ValueError("The two matrices must have the same number of genes (columns)")
    
    if (P < 0).any() or (Q < 0).any():
        raise ValueError("The two matrices must contain non-negative values")
    
    P = compute_gene_distribution(P, aggregate, epsilon)
    Q = compute_gene_distribution(Q, aggregate, epsilon)
            
    if distance == "KL":
        metric = kl_divergence(P, Q, epsilon)
    elif distance == "JS":
        metric = jensen_shannon_divergence(P, Q, epsilon)
    else:
        raise ValueError("Only Kullback-Leibler divergence and Jensen-Shannon divergence are supported (provide 'KL' or 'JS')")
    
    if reduction == 'mean':
        return np.mean(metric)
    elif reduction == 'sum':
        return np.sum(metric)
    else:
        return metric 