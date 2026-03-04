import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from pathlib import Path
import joblib
import os

np.random.seed(42)

home = Path("/project/Wellcome_Discovery/lturiano/GENESIS")
data = Path("/project/Wellcome_Discovery/datashare/lturiano/data")

print("Everything ready!\n")
directory = home / "02_weights"
if os.path.isdir(directory):
    print(f"Directory found: {directory}")
else:
    print(f"Directory not found: {directory}")

rna = sc.read_h5ad(data / "rna_filt_aligned.h5ad")
gex  = sc.read_h5ad(data / "gex_filt_aligned.h5ad")
fake = sc.read_h5ad(data / "fake_PermInv_VAE_UNET.h5ad")

marker_genes = {'T cell': ['ENSG00000156482',
  'ENSG00000134419',
  'ENSG00000171858',
  'ENSG00000112306',
  'ENSG00000149273',
  'ENSG00000177954',
  'ENSG00000108107',
  'ENSG00000143947',
  'ENSG00000198918',
  'ENSG00000167526'],
 'dendritic cell': ['ENSG00000019582',
  'ENSG00000204287',
  'ENSG00000223865',
  'ENSG00000231389',
  'ENSG00000138326',
  'ENSG00000196126',
  'ENSG00000179344',
  'ENSG00000196735',
  'ENSG00000265681',
  'ENSG00000161970'],
 'duct cell': ['ENSG00000143153',
  'ENSG00000119888',
  'ENSG00000272398',
  'ENSG00000101443',
  'ENSG00000240972',
  'ENSG00000185883',
  'ENSG00000219200',
  'ENSG00000175061',
  'ENSG00000111057',
  'ENSG00000181885'],
 'endothelial cell': ['ENSG00000172889',
  'ENSG00000065054',
  'ENSG00000127920',
  'ENSG00000261371',
  'ENSG00000165949',
  'ENSG00000129538',
  'ENSG00000164035',
  'ENSG00000102755',
  'ENSG00000131477',
  'ENSG00000175899'],
 'epithelial cell': ['ENSG00000272398',
  'ENSG00000143153',
  'ENSG00000119888',
  'ENSG00000167642',
  'ENSG00000111057',
  'ENSG00000240972',
  'ENSG00000166347',
  'ENSG00000170421',
  'ENSG00000120306',
  'ENSG00000163399'],
 'fibroblast': ['ENSG00000011465',
  'ENSG00000111341',
  'ENSG00000139329',
  'ENSG00000159403',
  'ENSG00000132386',
  'ENSG00000182326',
  'ENSG00000077942',
  'ENSG00000142173',
  'ENSG00000148180',
  'ENSG00000164692'],
 'lymphocyte': ['ENSG00000166710',
  'ENSG00000105374',
  'ENSG00000234745',
  'ENSG00000206503',
  'ENSG00000110848',
  'ENSG00000204525',
  'ENSG00000140988',
  'ENSG00000271503',
  'ENSG00000142541',
  'ENSG00000077984'],
 'macrophage': ['ENSG00000019582',
  'ENSG00000204287',
  'ENSG00000196126',
  'ENSG00000011600',
  'ENSG00000231389',
  'ENSG00000223865',
  'ENSG00000173372',
  'ENSG00000110077',
  'ENSG00000204472',
  'ENSG00000196735'],
 'mural cell': ['ENSG00000101335',
  'ENSG00000122786',
  'ENSG00000163453',
  'ENSG00000143248',
  'ENSG00000185633',
  'ENSG00000148671',
  'ENSG00000152583',
  'ENSG00000107796',
  'ENSG00000140545',
  'ENSG00000135744'],
 'myeloid cell': ['ENSG00000087086',
  'ENSG00000158869',
  'ENSG00000204472',
  'ENSG00000011600',
  'ENSG00000163220',
  'ENSG00000167996',
  'ENSG00000196154',
  'ENSG00000130066',
  'ENSG00000122862',
  'ENSG00000051523'],
 'myocyte': ['ENSG00000118194',
  'ENSG00000129991',
  'ENSG00000175084',
  'ENSG00000198125',
  'ENSG00000173991',
  'ENSG00000159251',
  'ENSG00000109846',
  'ENSG00000140416',
  'ENSG00000114854',
  'ENSG00000156885'],
 'neural cell': ['ENSG00000078328',
  'ENSG00000198763',
  'ENSG00000198840',
  'ENSG00000228253',
  'ENSG00000212907',
  'ENSG00000174469',
  'ENSG00000198786',
  'ENSG00000198712',
  'ENSG00000198899',
  'ENSG00000198886']}

rna_gene_names = set(rna.var_names)
genes = []
for row in list(marker_genes.values()):
    for item in row:
        if item not in genes and item in rna_gene_names:
            genes.append(item)
            
np.random.shuffle(genes)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(rna.obs["cell_type"].values.tolist())

# save genes + label mapping
meta = {"genes": genes, "label_classes": label_encoder.classes_}
joblib.dump(meta, home / "02_weights" / "top10_xgb_meta.joblib")

df_rna = pd.DataFrame(index   = rna.obs.index,
                      columns = genes,
                      data    = rna.X[:, rna.var_names.isin(genes)])

df_gex = pd.DataFrame(index   = gex.obs.index,
                      columns = genes,
                      data    = gex.X[:, gex.var_names.isin(genes)])

df_fake = pd.DataFrame(index   = fake.obs.index,
                       columns = genes,
                       data    = fake.X[:, fake.var_names.isin(genes)])

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
best_model.save_model(home / "02_weights" / "top10_xgb.json")

model_rna = best_model

# # Performance evaluation on scRNA, scGEX, generated scGEX
y_pred_rna = model_rna.predict(X_test_rna)
print(f'Accuracy: {accuracy_score(y_test_rna.values, y_pred_rna):.4f}')

y_pred_gex = model_rna.predict(X_test_gex)
print(f'Accuracy: {accuracy_score(y_test_gex.values, y_pred_gex):.4f}')

y_pred_fake = model_rna.predict(X_test_fake)
print(f'Accuracy: {accuracy_score(y_test_fake, y_pred_fake):.4f}')