import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from pathlib import Path

np.random.seed(42)

BASE = Path("/project/Wellcome_Discovery/datashare/lturiano/data")

rna = sc.read_h5ad(BASE / "rna_filt_aligned.h5ad")
gex  = sc.read_h5ad(BASE / "gex_filt_aligned.h5ad")
fake = sc.read_h5ad(BASE / "fake_RNA_VAE_UNET_log.h5ad")

marker_genes = {
    "T cell": [
        "TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E", "CD2", "LCK", "IL7R", "CCR7",
    ],

    "lymphocyte": [
        # broad lymphoid bucket (T/NK/B markers + pan-lymphoid)
        "PTPRC", "CD74", "MS4A1", "CD79A", "CD79B", "CD3D", "TRAC", "NKG7", "GNLY",
    ],

    "myeloid cell": [
        # broad myeloid bucket
        "LYZ", "S100A8", "S100A9", "FCN1", "LGALS3", "CTSS", "TYMP", "VCAN",
    ],

    "macrophage": [
        "LYZ", "LST1", "CST3", "CTSS", "FCER1G", "TYROBP", "C1QA", "C1QB", "C1QC",
        "APOE", "MSR1", "MARCO",
    ],

    "dendritic cell": [
        "FCER1A", "CST3", "CD74", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CLEC10A",
        "ITGAX",
    ],

    "endothelial cell": [
        "PECAM1", "VWF", "KDR", "ESAM", "CDH5", "RAMP2", "ENG", "CLDN5",
    ],

    "mural cell": [
        # pericyte / vascular smooth muscle-like
        "RGS5", "PDGFRB", "CSPG4", "MCAM", "DES", "TAGLN", "ACTA2", "MYH11",
    ],

    "fibroblast": [
        "COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "FBLN1", "C7", "PDGFRA",
    ],

    "epithelial cell": [
        # broad epithelial
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "TACSTD2",
    ],

    "duct cell": [
        # ductal / epithelial-ductal
        "KRT19", "KRT8", "KRT18", "KRT7", "MUC1", "KRT17", "SLC4A4", "CFTR",
        "SOX9",
    ],

    "myocyte": [
        # muscle (striated / smooth-ish mix for robustness)
        "ACTA1", "ACTN2", "TTN", "MYH1", "MYH2", "TNNT3", "CKM",
        "DES", "TAGLN",
    ],

    "neural cell": [
        "SOX2", "NES", "TUBB3", "RBFOX3", "SLC1A3", "PLP1", "MBP", "GFAP",
    ],
}

rna_gene_names = set(rna.var["feature_name"])
genes = []
for row in list(marker_genes.values()):
    for item in row:
        if item not in genes and item in rna_gene_names:
            genes.append(item)
            
np.random.shuffle(genes)

df_rna = pd.DataFrame(index   = rna.obs.index,
                      columns = genes,
                      data    = rna.X[:, rna.var.feature_name.isin(genes)])

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(rna.obs["cell_type"].values.tolist())
label_encoder.classes_

df_rna["cell_type"]  = label_encoder.transform(rna.obs["cell_type"].values.tolist())

# Create train-test split (final evaluation set)
X_train_rna, X_test_rna, y_train_rna, y_test_rna = train_test_split(df_rna[df_rna.columns[:-1]],
                                                                    df_rna[df_rna.columns[-1]],
                                                                    test_size=0.15,
                                                                    random_state=42,
                                                                    stratify=df_rna[df_rna.columns[-1]]
                                                                    )

# Training and hyperparameter setting xgboost on scRNA
# Define XGBoost classifier
xgb_clf = xgboost.XGBClassifier(objective='multi:softprob',
                             num_class=len(label_encoder.classes_),
                             device="cpu",
                             random_state=42)

# Hyperparameter grid
param_grid = {'booster': ['gbtree', 'dart'],
              'max_depth': [2, 4, 6],
              'learning_rate': [0.01, 0.05, 0.1],
              'n_estimators': [50, 100, 150],
              'subsample': [0.6, 0.8, 1.0],
              }

# Cross-validation and hyperparameter tuning
clf_cv = GridSearchCV(estimator=xgb_clf,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=5,
                      verbose=1,
                      n_jobs=8)

clf_cv.fit(X_train_rna, y_train_rna)

# Best hyperparameters
print("Best hyperparameters:")
print(clf_cv.best_params_)

# Evaluate on final test set
best_model = clf_cv.best_estimator_
best_model.save_model("best_xgb_model.json")