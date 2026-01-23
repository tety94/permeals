"""
Clustering con dati di biomarcatori per valutare differenze tra gruppi clinici.

Esegue:
1. Pre-processing e scaling
2. Confronto di tre algoritmi di clustering (KMeans, Spectral, Gaussian Mixture)
   tramite silhouette score
3. Analisi statistica delle differenze tra cluster
4. Salvataggio dei risultati (CSV e heatmap dei centroidi per KMeans/GMM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway, kruskal, chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.manifold import TSNE

from get_data import *   # funzioni di caricamento dati definite altrove
from utilities import *

# ---------------------------------------------------------------------
# 1. Confronto metodi di clustering
# ---------------------------------------------------------------------

RESULT_FOLDER = 'results/test3/'

def compare_clustering_methods(X_scaled, max_k=10, time=0, plot=True):
    """
    Confronta KMeans, Spectral e GaussianMixture in base al silhouette score.
    Ritorna il numero ottimale di cluster per ciascun algoritmo.
    """
    K = range(2, max_k + 1)
    scores = {name: [] for name in ('kmeans', 'spectral', 'gmm')}

    for k in K:
        # KMeans
        km_labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X_scaled)
        scores['kmeans'].append(silhouette_score(X_scaled, km_labels))

        # Spectral Clustering
        sp_labels = SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors',
            n_neighbors=5, random_state=42
        ).fit_predict(X_scaled)
        scores['spectral'].append(silhouette_score(X_scaled, sp_labels))

        # Gaussian Mixture
        gmm_labels = GaussianMixture(
            n_components=k, covariance_type='full', random_state=42
        ).fit_predict(X_scaled)
        scores['gmm'].append(silhouette_score(X_scaled, gmm_labels))

    if plot:
        plt.figure(figsize=(8, 6))
        for method, vals in scores.items():
            plt.plot(K, vals, marker='o', label=method)
        plt.title('Silhouette score per algoritmo')
        plt.xlabel('Numero di cluster')
        plt.ylabel('Silhouette score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULT_FOLDER}img/compare_silhouette_{time}.png')

    best_k = {method: K[np.argmax(vals)] for method, vals in scores.items()}
    print("Numero ottimale di cluster (silhouette):", best_k)
    return best_k, scores



def plot_tsne_clusters(X_scaled, cluster_labels_dict, time=0, perplexity=30, random_state=42):
    """
    Visualizza i cluster su una mappa 2D ottenuta con t-SNE.
    """

    # t-SNE 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    for method_name, labels in cluster_labels_dict.items():
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
        sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            hue=labels,
            palette=palette,
            s=60,
            alpha=0.8,
            legend='full'
        )
        plt.title(f"t-SNE clusters - {method_name.upper()} (time={time})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{RESULT_FOLDER}img/tsne_{method_name}_{time}.png")
        plt.show()


# ---------------------------------------------------------------------
# 3. Pipeline principale
# ---------------------------------------------------------------------
def run_clustering_analysis(df, cluster_features, other_features,
                            max_k=10, time=0):
    """
    Preprocessing, clustering con KMeans/Spectral/Gaussian,
    analisi differenze tra cluster e salvataggio dei risultati.
    """
    # --- Preprocessing
    X_scaled, df_clean, used_features = preprocess_data(df, cluster_features)

    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(X_scaled)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    # plt.show()

    # --- Confronto metodi
    best_k_all, _ = compare_clustering_methods(X_scaled, max_k=max_k, time=time)

    # --- Fit modelli con rispettivo k ottimale
    kmeans = KMeans(n_clusters=best_k_all['kmeans'], random_state=42, n_init='auto')
    df_clean['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

    spectral = SpectralClustering(
        n_clusters=best_k_all['spectral'], random_state=42, affinity='rbf'
    )
    df_clean['cluster_spectral'] = spectral.fit_predict(X_scaled)

    gmm = GaussianMixture(
        n_components=best_k_all['gmm'], covariance_type='full', random_state=42
    )
    df_clean['cluster_gmm'] = gmm.fit_predict(X_scaled)

    cluster_labels = {
        'kmeans': ('cluster_kmeans', kmeans),
        'spectral': ('cluster_spectral', None),
        'gmm': ('cluster_gmm', gmm)
    }
    cluster_labels_clean = {
        method: df_clean[col].values
        for method, (col, model) in cluster_labels.items()
    }

    plot_tsne_clusters(X_scaled, cluster_labels_clean, time=t)

    # --- Analisi e salvataggio per ciascun algoritmo
    for method, (col, model) in cluster_labels.items():
        print(f"\n=== Analisi per {method.upper()} ===")

        df_tmp = df_clean.copy()
        df_tmp['cluster'] = df_tmp[col]

        # Rimuovi cluster con ≤2 elementi
        # small_clusters = df_tmp['cluster'].value_counts()[lambda x: x <= 2].index
        # df_tmp = df_tmp[~df_tmp['cluster'].isin(small_clusters)].copy()

        true_k = df_tmp['cluster'].nunique()
        print(f"True best K ({method}): {true_k}")

        analyze_cluster_differences(df_tmp, 'cluster',
                                    cluster_features + other_features)

        # Salva CSV
        out_csv = f"clustered_data_{method}_{time}.csv"
        df_tmp.to_csv(f"{RESULT_FOLDER}{out_csv}", index=False, sep=';', decimal=',')
        print(f"✅ Risultati salvati in: {out_csv}")

        # Heatmap centroidi (solo KMeans e GMM)
        if method in ['kmeans', 'gmm']:
            centroids = (model.cluster_centers_
                         if method == 'kmeans' else model.means_)
            centroids_df = pd.DataFrame(centroids, columns=used_features)
            plt.figure(figsize=(10, 6))
            sns.heatmap(centroids_df, annot=True, cmap="coolwarm",
                        fmt=".2f", cbar_kws={'label': 'Valore normalizzato'})
            plt.title(f"Centroidi dei cluster - {method.upper()}")
            plt.xlabel("Feature")
            plt.ylabel("Cluster")
            plt.tight_layout()
            out_img = f"{RESULT_FOLDER}img/centroid_cluster_{method}_{time}.png"
            plt.savefig(out_img)
            plt.close()
            print(f"🖼  Heatmap salvata in: {out_img}")
        else:
            print("⚠️  Spectral Clustering non fornisce centroidi.")

    return df_clean


# ---------------------------------------------------------------------
# 4. Esecuzione
# ---------------------------------------------------------------------
if __name__ == "__main__":
    plasma_cols      = get_plasma_columns()
    liquor_cols      = get_liquor_columns()
    umns_cols        = get_umn_score_columns()
    alsfrs_cols      = get_alsfrs_columns()
    clinical_cols    = get_clinical_data()
    respiratory_cols = get_respiratory_columns()
    mrc_columns      = get_mrc_columns()
    macsplex_columns = get_macsplex_columns()
    als_no_als_colums= get_als_controls_data()

    cluster_features = macsplex_columns #+ liquor_cols
    other_features   = (umns_cols + alsfrs_cols + plasma_cols  +
                        clinical_cols + ['delta_bmi'] + mrc_columns + liquor_cols +
                        als_no_als_colums + respiratory_cols)

    df = get_sheet(sheet_name='CSF')

    df =df[df['CD13'].notna()]
    df = df[~df['pt_code'].isin(['MI-ALS-EP-A1362', 'MI-ALS-EP-A1354']) ]



    # csf_df = get_csf_sheet()
    # df = get_merged_csf_data(df, csf_df)
    # df = df[df['Sara'] == 'x']

    # for t in [0, 6, 12]:
    for t in [0]:
        print(f"\n########## Time {t} ##########")
        df_time = df[df['time_point'] == f'T{t}']
        run_clustering_analysis(df_time,
                                cluster_features=cluster_features,
                                other_features=other_features,
                                max_k=8,
                                time=t)
