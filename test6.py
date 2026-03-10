"""
Clustering di pazienti ALS basato su biomarcatori biologici (plasma, liquor, dati clinici)
per identificare sottogruppi di pazienti con profili temporali simili.

Pipeline generale:

1. Caricamento del dataset da Excel tramite funzioni personalizzate.
2. Selezione delle feature biologiche e cliniche usate per il clustering.
3. Pulizia e pre-processing dei dati:
      - conversione a valori numerici
      - rimozione righe troppo incomplete
      - selezione solo dei pazienti con tutti i timepoint disponibili
      - imputazione dei valori mancanti verticalmente per ogni timepoint
4. Costruzione delle serie temporali multivariate per ogni paziente.
5. Scaling delle serie temporali con media=0 e varianza=1.
6. Clustering usando TimeSeriesKMeans con metrica Soft-DTW.
7. Selezione del numero ottimale di cluster basata sul silhouette score.
8. Visualizzazione dei risultati:
      - grafici inertia/silhouette
      - t-SNE 2D/3D per proiezione e visualizzazione dei cluster
9. Analisi delle differenze tra cluster sui biomarcatori e dati clinici.
10. Salvataggio dei risultati e dei grafici nella cartella designata.

Nota:
Solo un sottoinsieme delle feature (es. biomarcatori) viene usato per costruire i cluster.
Le altre feature (es. cliniche) vengono utilizzate per interpretare e confrontare i cluster ottenuti.
"""

# =========================
# IMPORT
# =========================
from functions_clustering import *   # funzioni personalizzate per clustering e analisi
from get_data import *               # funzioni per caricare i dati dai fogli Excel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance  # scaling per serie temporali
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.manifold import TSNE  # riduzione dimensionale per visualizzazione

# =========================
# PARAMETRI
# =========================
RESULT_FOLDER = 'results/test6/'  # cartella dove salvare risultati e grafici
seed = 42                         # seme per riproducibilità
timepoints = ['T0', 'T6', 'T12'] # timepoint considerati

# =========================
# CARICAMENTO DATI
# =========================
df = get_sheet(sheet_name='clinic_EP')  # carica il foglio clinico

# colonne per clustering e analisi clinica
clinical_columns = get_clinical_data()
plasma_columns = get_plasma_columns()
liquor_columns = get_liquor_columns()

cluster_features = plasma_columns  # usate per il clustering
relevant_features = ['pt_code', 'time_point'] + cluster_features + clinical_columns + liquor_columns

dataset = df[relevant_features].copy()
dataset = dataset[dataset['time_point'].isin(timepoints)]  # filtra solo i timepoint considerati

# =========================
# CAST NUMERICO
# =========================
# conversione tutte le colonne di interesse in numerico
for col in relevant_features[2:]:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# =========================
# PULIZIA RIGHE TROPPO VUOTE
# =========================
# rimuove righe con troppi valori NaN (>38)
dataset = dataset[dataset.isna().sum(axis=1) < 38].copy()

# =========================
# TIENI SOLO PAZIENTI COMPLETI
# =========================
# seleziona solo pazienti che hanno tutti i timepoint richiesti
tp_count = dataset.groupby('pt_code')['time_point'].nunique()
valid_patients = tp_count[tp_count == len(timepoints)].index
dataset = dataset[dataset['pt_code'].isin(valid_patients)].copy()

print("Numero pazienti validi:", len(valid_patients))

# =========================
# IMPUTAZIONE VERTICALE (PER TIMEPOINT)
# =========================
# per ogni timepoint, riempi i NaN con la media della colonna del timepoint stesso
for tp in timepoints:
    mask = dataset['time_point'] == tp
    for col in cluster_features:
        mean_tp = dataset.loc[mask, col].mean()
        dataset.loc[mask, col] = dataset.loc[mask, col].fillna(mean_tp)

# =========================
# COSTRUZIONE SERIE TEMPORALI
# =========================
# crea una lista di array 2D: un array per paziente (timepoints x features)
patients = dataset['pt_code'].unique()
X_all = []

for pt in patients:
    patient_ts = (
        dataset[dataset['pt_code'] == pt]
        .set_index('time_point')
        .loc[timepoints, cluster_features]
        .astype(float)
    )

    # sicurezza: salta pazienti con righe completamente NaN
    if patient_ts.isna().all(axis=1).any():
        continue

    X_all.append(patient_ts.values)

X_all = np.array(X_all)
print("Shape X_all:", X_all.shape)

# =========================
# SCALING
# =========================
# media=0, varianza=1 per ogni feature
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
Nclusters = range(2, 4)  # range di k da testare

for k in Nclusters:
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric='softdtw',   # metrica Soft Dynamic Time Warping
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

# per sicurezza, qui imposto manualmente il numero di cluster
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
# analizza differenze dei cluster per ogni timepoint
for tp in ['T0', 'T6', 'T12']:
    df_tp = dataset[dataset['time_point'] == tp].copy()
    df_tp['cluster'] = labels
    print(f"\n--- {tp} ---")
    analyze_cluster_differences(df_tp, 'cluster', relevant_features[2:], log_file=f"{RESULT_FOLDER}cluster_differences_{tp}.txt")

# =========================
# t-SNE VISUALIZZAZIONE
# =========================
# flatten per TSNE
X_flat = X_scaled.reshape(X_scaled.shape[0], -1)

X_tsne2 = TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(X_flat)
X_tsne3 = TSNE(n_components=3, perplexity=30, random_state=seed).fit_transform(X_flat)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax1.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=labels, cmap='jet')
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_tsne3[:, 0], X_tsne3[:, 1], X_tsne3[:, 2], c=labels, cmap='jet')
plt.show()