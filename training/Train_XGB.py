import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from pathlib import Path

np.random.seed(42)

data = "/project/Wellcome_Discovery/datashare/lturiano/data/"
home = "/project/Wellcome_Discovery/lturiano/GENESIS/"
sriva = "/project/Wellcome_Discovery/datashare/sriva/GENESIS/"

# Load breast data
rna = sc.read_h5ad(sriva+"RNA_filt_log_subset.h5ad")
gex = sc.read_h5ad(sriva+"GEX_filt_log_subset.h5ad")
fake = sc.read_h5ad(data+"breast_fake_RNA_PermInv_VAE_UNET.h5ad")

marker_genes = {'T cell': ['IL7R', 'CCL5', 'PTPRC', 'CXCR4', 'GNLY', 'CD2', 'SRGN'],
                'basal cell': ['KRT14', 'KRT17', 'DST', 'KRT5', 'SAA1', 'ACTA2', 'SFN'],
                'endothelial cell': ['SELE', 'ACKR1', 'FABP4', 'STC1', 'ANGPT2', 'CSF3'],
                'fibroblast': ['DCN', 'APOD', 'CFD', 'TNFAIP6', 'LUM', 'COL1A2', 'COL1A1'],
                'macrophage': ['HLA-DRA', 'IL1B', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRB1', 'CD74', 'CCL3']
              }

genes = []
for row in list(marker_genes.values()):
    for item in row:
        if item not in genes:
            genes.append(item)  
# np.random.shuffle(genes)

df_rna = pd.DataFrame(index   = rna.obs.index,
                      columns = genes,
                      data    = rna.X[:, rna.var.feature_name.isin(genes)])

df_gex = pd.DataFrame(index   = gex.obs.index,
                      columns = genes,
                      data    = gex.X[:, gex.var.feature_name.isin(genes)])

df_fake = pd.DataFrame(index   = fake.obs.index,
                       columns = genes,
                       data    = fake.X[:, fake.var.feature_name.isin(genes)])
# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(rna.obs["cell_type"].values.tolist())

df_rna["cell_type"]  = label_encoder.transform(rna.obs["cell_type"].values.tolist())
df_gex["cell_type"]  = label_encoder.transform(gex.obs["cell_type"].values.tolist())
df_fake["cell_type"] = label_encoder.transform(rna.obs["cell_type"].values.tolist())

# Create train-test split (final evaluation set)
X_train_rna, X_test_rna, y_train_rna, y_test_rna = train_test_split(df_rna[df_rna.columns[:-1]],
                                                                    df_rna[df_rna.columns[-1]],
                                                                    test_size=0.15,
                                                                    random_state=42,
                                                                    stratify=df_rna[df_rna.columns[-1]]
                                                                    )
# Create train-test split (final evaluation set)
X_train_gex, X_test_gex, y_train_gex, y_test_gex = train_test_split(df_gex[df_gex.columns[:-1]],
                                                                    df_gex[df_gex.columns[-1]],
                                                                    test_size=0.15,
                                                                    random_state=42,
                                                                    stratify=df_gex[df_gex.columns[-1]]
                                                                    )
# Create train-test split (final evaluation set)
X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(df_fake[df_fake.columns[:-1]],
                                                                        df_fake[df_fake.columns[-1]],
                                                                        test_size=0.15,
                                                                        random_state=42,
                                                                        stratify=df_fake[df_fake.columns[-1]]
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
best_model.save_model(home + "weights/breast_best_xgb_model.json")

y_pred_rna = best_model.predict(X_test_rna)
y_pred_gex = best_model.predict(X_test_gex)
y_pred_fake = best_model.predict(X_test_fake)

# Classification report
print("Classification Report:")
print(classification_report(y_test_rna.values, y_pred_rna, target_names=label_encoder.classes_))

# Classification report
print("Classification Report for GEX:")
print(classification_report(y_test_gex.values, y_pred_gex, target_names=label_encoder.classes_))

# Classification report
print("Classification Report for GEX:")
print(classification_report(y_test_fake.values, y_pred_fake, target_names=label_encoder.classes_))
