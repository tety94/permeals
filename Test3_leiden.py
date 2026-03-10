"""
Clustering di pazienti ALS basato su biomarcatori biologici (CSF, plasma, MacsPlex)
per identificare sottogruppi stabili di pazienti e analizzarne le differenze cliniche.

Pipeline generale:

1. Caricamento del dataset da file Excel.
2. Selezione delle feature biologiche usate per il clustering.
3. Pulizia dei dati:
      - conversione a numerico
      - imputazione dei valori mancanti con la media della colonna
4. Clustering con algoritmo Leiden variando il parametro di risoluzione.
5. Valutazione della stabilità dei clustering ottenuti usando
      Adjusted Rand Index (ARI) tra tutte le partizioni.
6. Costruzione della matrice di co-associazione tra pazienti
      (quanto spesso due pazienti finiscono nello stesso cluster).
7. Clustering gerarchico sulla matrice di co-associazione
      per identificare cluster stabili.
8. Scelta del numero finale di cluster dal dendrogramma.
9. Analisi delle differenze tra cluster su biomarcatori e dati clinici.
10. Salvataggio dei risultati e dei grafici.

Nota:
Solo un sottoinsieme delle feature viene usato per costruire i cluster.
Le altre feature (es. cliniche) vengono invece utilizzate per
interpretare e confrontare i cluster ottenuti.
"""

############################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# February 2026

#############################################################################################################

# ---------------------------
# LIBRERIE
# ---------------------------

import pandas as pd                     # gestione e manipolazione dataset tabellari
import numpy as np                      # operazioni numeriche e matriciali

from sklearn import preprocessing       # scaling delle variabili
import seaborn as sns                   # visualizzazione statistica
import matplotlib.pyplot as plt         # creazione grafici

# funzioni per clustering gerarchico
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster

# metrica per confrontare partizioni di clustering
from sklearn.metrics.cluster import adjusted_rand_score

# moduli custom del progetto
from get_data import *                  # funzioni per caricare e selezionare colonne
from utilities import *
from functions_clustering import *      # funzioni di clustering e plotting

import os

# cartella in cui salvare risultati e grafici
RESULT_FOLDER = 'results/test1/'

# seed per riproducibilità dei risultati
np.random.seed(42)

# limitazione dei thread per evitare conflitti con librerie numeriche
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

################################################################################################################

if __name__ == '__main__':

    # ---------------------------
    # CARICAMENTO DATI
    # ---------------------------

    file = "database.xlsx"

    # lettura dello sheet "CSF"
    dataset = get_sheet(file=file, sheet_name='CSF')

    # ---------------------------
    # PULIZIA DEL DATASET
    # ---------------------------

    # rimozione di alcuni pazienti problematici o outlier
    dataset = dataset[dataset['pt_code'] != 'MI-ALS-EP-A1481'].copy()
    dataset = dataset[dataset['pt_code'] != 'MI-ALS-EP-A1362'].copy()

    # ---------------------------
    # SELEZIONE DELLE FEATURE
    # ---------------------------

    # gruppi di variabili biologiche e cliniche
    macsplex_columns = get_macsplex_columns()
    liquor_columns = get_liquor_columns()
    clinical_columns = get_clinical_data()
    clinical_columns_categorical = get_clinical_parameters_categorical()
    plasma_columns = get_plasma_columns()

    # feature usate per il clustering
    features_clusters = macsplex_columns + liquor_columns + plasma_columns

    # dataset contenente solo le feature per clustering
    dataset_cluster = dataset[features_clusters].copy()

    # feature totali usate poi per interpretare i cluster
    relevant_features = features_clusters + clinical_columns

    # ---------------------------
    # GESTIONE VALORI MANCANTI
    # ---------------------------

    # conversione delle colonne a numerico
    for ii in features_clusters:
        dataset_cluster[ii] = pd.to_numeric(dataset_cluster[ii], errors="coerce")

    # imputazione dei valori mancanti con la media della colonna
    dataset_filled = dataset_cluster.fillna(dataset_cluster.mean()).astype(float).copy()

    # ---------------------------
    # PARAMETRI CLUSTERING
    # ---------------------------

    # il parametro "resolution" controlla la granularità del clustering Leiden
    lower_bound = 0.5
    upper_bound = 0.9

    # generazione dei valori di resolution da testare
    resolutions = np.arange(lower_bound, upper_bound + 0.05, 0.05)

    # numero di configurazioni di clustering
    M = len(resolutions)

    # numero di pazienti
    N = len(dataset_filled)

    # matrice che conterrà le assegnazioni dei cluster
    # dimensioni: (numero resolution) x (numero pazienti)
    CLUSTERS = np.zeros((M, N))

    # ---------------------------
    # CLUSTERING LEIDEN
    # ---------------------------

    # per ogni valore di resolution eseguiamo il clustering
    for ii, res in enumerate(resolutions):

        Leiden_clusters = compute_clustering(
            dataset_filled,
            scaler_method='StandardScaler',
            resolution_Leiden=res
        )

        # salvataggio delle assegnazioni dei cluster
        CLUSTERS[ii, :] = Leiden_clusters

    # ---------------------------
    # STABILITÀ DEL CLUSTERING
    # ---------------------------

    # confronto tra tutte le partizioni usando Adjusted Rand Index
    # ARI = misura di similarità tra due clusterizzazioni

    ad_rand_index = np.array([
        [adjusted_rand_score(CLUSTERS[i, :], CLUSTERS[j, :]) for i in range(M)]
        for j in range(M)
    ])

    # heatmap della similarità tra clustering
    sns.heatmap(ad_rand_index, annot=True, cmap="coolwarm")
    plt.title('Adjusted Rand index matrix')
    plt.savefig(f"{RESULT_FOLDER}img/ad_rand_index.png")
    plt.show()

    # ---------------------------
    # MATRICE DI CO-ASSOCIAZIONE
    # ---------------------------

    # misura quanto spesso due pazienti finiscono nello stesso cluster
    # attraverso tutte le resolution testate

    co_assoc = np.array([
        [sum(c[i] == c[j] for c in CLUSTERS) / M for j in range(N)]
        for i in range(N)
    ])

    # ---------------------------
    # CLUSTERING GERARCHICO
    # ---------------------------

    # clustering gerarchico sulla matrice di co-associazione
    # per identificare cluster stabili

    row_linkage = hierarchy.linkage(distance.pdist(co_assoc), method='average')
    col_linkage = hierarchy.linkage(distance.pdist(co_assoc.T), method='average')

    # visualizzazione dendrogramma con heatmap
    fig2 = sns.set(font_scale=0.6)

    cg2 = sns.clustermap(
        co_assoc,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        method='average',
        xticklabels=True,
        yticklabels=True,
        cmap="coolwarm",
        linewidths=1,
        linecolor='black'
    )

    fig2 = plt.savefig(f"{RESULT_FOLDER}img/den_co-assoc.png")
    plt.show()

    # ---------------------------
    # SCELTA NUMERO CLUSTER
    # ---------------------------

    # l'utente osserva il dendrogramma e inserisce il numero di cluster
    n_cluster = int(input(' Insert the number of clusters identified from the dendrogram = '))

    # assegnazione finale dei cluster
    stable_Leiden_clusters = fcluster(row_linkage, n_cluster, criterion='maxclust')

    # ---------------------------
    # ANALISI DEI CLUSTER
    # ---------------------------

    # visualizzazione distribuzione delle feature nei cluster
    plot_features_clusters(
        dataset[relevant_features],
        stable_Leiden_clusters,
        RESULT_FOLDER
    )

    # aggiunta etichetta cluster al dataset originale
    dataset.loc[:, 'Leiden_clusters'] = stable_Leiden_clusters

    # analisi statistica delle differenze tra cluster
    analyze_cluster_differences(
        dataset,
        'Leiden_clusters',
        relevant_features
    )

    # ---------------------------
    # SALVATAGGIO RISULTATI
    # ---------------------------

    # dataset con cluster
    dataset.to_excel(
        RESULT_FOLDER + file + '_CLUSTERS_LEIDEN_stable.xlsx'
    )

    # dataset ordinato per cluster
    dataset_sort = dataset.sort_values('Leiden_clusters')

    dataset_sort.to_excel(
        RESULT_FOLDER + file + '_CLUSTERS_LEIDEN_stable_sorted.xlsx'
    )