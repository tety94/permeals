"""
Time series clustering with tslearn (Soft-DTW)
1 paziente = 1 serie temporale multivariata
Imputazione verticale per timepoint
"""

# =========================
# IMPORT
# =========================
from functions_clustering import *
from get_data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.manifold import TSNE


# =========================
# PARAMETRI
# =========================
RESULT_FOLDER = 'results/test6/'
seed = 42

timepoints = ['T0', 'T6', 'T12']


# =========================
# CARICAMENTO DATI
# =========================
df = get_sheet(sheet_name='clinic_EP')

clinical_columns = get_clinical_data()
plasma_columns = get_plasma_columns()
liquor_columns = get_liquor_columns()

cluster_features = plasma_columns
relevant_features = ['pt_code', 'time_point'] + cluster_features + clinical_columns + liquor_columns

dataset = df[relevant_features].copy()
dataset = dataset[dataset['time_point'].isin(timepoints)]

# =========================
# CAST NUMERICO
# =========================
for col in relevant_features[2:]:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')


# =========================
# PULIZIA RIGHE TROPPO VUOTE
# =========================
dataset = dataset[dataset.isna().sum(axis=1) < 38].copy()

# =========================
# TIENI SOLO PAZIENTI COMPLETI
# =========================
tp_count = dataset.groupby('pt_code')['time_point'].nunique()
valid_patients = tp_count[tp_count == len(timepoints)].index
dataset = dataset[dataset['pt_code'].isin(valid_patients)].copy()

print("Numero pazienti validi:", len(valid_patients))


# =========================
# IMPUTAZIONE VERTICALE (PER TIMEPOINT)
# =========================
for tp in timepoints:
    mask = dataset['time_point'] == tp
    for col in cluster_features:
        mean_tp = dataset.loc[mask, col].mean()
        dataset.loc[mask, col] = dataset.loc[mask, col].fillna(mean_tp)


# =========================
# COSTRUZIONE SERIE TEMPORALI
# =========================
patients = dataset['pt_code'].unique()
X_all = []

for pt in patients:
    patient_ts = (
        dataset[dataset['pt_code'] == pt]
        .set_index('time_point')
        .loc[timepoints, cluster_features]
        .astype(float)
    )

    # sicurezza: niente righe completamente NaN
    if patient_ts.isna().all(axis=1).any():
        continue

    X_all.append(patient_ts.values)

X_all = np.array(X_all)

print("Shape X_all:", X_all.shape)


# =========================
# SCALING
# =========================
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_all)

print("Shape X_scaled:", X_scaled.shape)
print("NaN count:", np.isnan(X_scaled).sum())
print("Min/max:", X_scaled.min(), X_scaled.max())


# =========================
# SCELTA NUMERO CLUSTER (SILHOUETTE)
# =========================
best_k = None
best_score = -np.inf

silhouettes = []
inertias = []

Nclusters = range(2, 4)

for k in Nclusters:
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric='softdtw',
        random_state=seed
    )

    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)

    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_scaled, labels, metric='softdtw')
    else:
        score = np.nan

    silhouettes.append(score)

    print(f"k={k} | inertia={model.inertia_:.2f} | silhouette={score}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest number of clusters: {best_k} (silhouette={best_score})")

best_k = 3

# =========================
# PLOT INERTIA / SILHOUETTE
# =========================
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(list(Nclusters), inertias, 'o-')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Inertia')

ax[1].plot(list(Nclusters), silhouettes, 'o-')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette')

plt.tight_layout()
plt.savefig(f"{RESULT_FOLDER}/inertia_silhouette.png")
plt.show()


# =========================
# CLUSTERING FINALE
# =========================
model = TimeSeriesKMeans(
    n_clusters=best_k,
    metric='softdtw',
    random_state=seed
)

labels = model.fit_predict(X_scaled)


# =========================
# ANALISI CLUSTER PER TIMEPOINT
# =========================
for tp in ['T0', 'T6', 'T12']:
    df_tp = dataset[dataset['time_point'] == tp].copy()
    df_tp['cluster'] = labels
    print(f"\n--- {tp} ---")
    analyze_cluster_differences(df_tp, 'cluster', relevant_features[2:])


# =========================
# t-SNE VISUALIZZAZIONE
# =========================
X_flat = X_scaled.reshape(X_scaled.shape[0], -1)

X_tsne2 = TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(X_flat)
X_tsne3 = TSNE(n_components=3, perplexity=30, random_state=seed).fit_transform(X_flat)

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
ax1.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=labels, cmap='jet')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_tsne3[:, 0], X_tsne3[:, 1], X_tsne3[:, 2], c=labels, cmap='jet')

plt.show()
