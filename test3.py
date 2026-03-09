"""
Clustering con dati di biomarcatori per valutare differenze tra gruppi clinici.

Pipeline aggiornata:
1. Pre-processing e scaling delle feature
2. (Opzionale) PCA per riduzione rumore e collinearità
3. Confronto di tre algoritmi di clustering (KMeans, Spectral, Gaussian Mixture)
   usando silhouette score e cluster stability (bootstrap + ARI)
4. Clustering finale con il numero ottimale di cluster
5. Analisi statistica delle differenze tra cluster sulle feature originali
6. Salvataggio risultati (CSV e heatmap centroidi)
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import resample

from get_data import *
from utilities import *

RESULT_FOLDER = 'results/test3/'
CLUSTER_CLINICS = False
USE_PCA = True

def clustering_stability(X, model_class, k, n_boot=30):
    """
    Valuta la stabilità di un clustering tramite bootstrap (ARI medio).
    Gestisce KMeans, SpectralClustering e GaussianMixture.
    """
    labels_list = []
    for i in range(n_boot):
        X_boot = resample(X, random_state=i)

        # --- Gestione dei parametri diversi ---
        if model_class == GaussianMixture:
            model = model_class(n_components=k, covariance_type='full', random_state=i)
            labels = model.fit_predict(X_boot)
        elif model_class == SpectralClustering:
            model = model_class(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5, random_state=i)
            labels = model.fit_predict(X_boot)
        else:  # KMeans
            model = model_class(n_clusters=k, random_state=i, n_init='auto')
            labels = model.fit_predict(X_boot)

        labels_list.append(labels)

    scores = [adjusted_rand_score(labels_list[i], labels_list[i+1])
              for i in range(len(labels_list)-1)]
    return np.mean(scores)


def apply_pca(X_scaled, variance_threshold=0.9):
    """Applica PCA per ridurre rumore e collinearità mantenendo variance_threshold della varianza totale."""
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"PCA riduzione: {X_scaled.shape[1]} → {X_reduced.shape[1]} componenti")
    return X_reduced


def compare_clustering_methods(X_scaled, max_k=10, time=0, use_pca=False, plot=True):
    """
    Confronta KMeans, Spectral e GMM usando silhouette score e stability.
    Ritorna il numero ottimale di cluster per ciascun algoritmo.
    """
    if use_pca:
        X_scaled = apply_pca(X_scaled, variance_threshold=0.9)

    K = range(2, max_k + 1)
    scores = {name: [] for name in ('kmeans', 'spectral', 'gmm')}
    stability = {name: [] for name in ('kmeans', 'spectral', 'gmm')}

    for k in K:
        # ---------------- KMeans ----------------
        km_labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X_scaled)
        scores['kmeans'].append(silhouette_score(X_scaled, km_labels))
        stability['kmeans'].append(clustering_stability(X_scaled, KMeans, k))

        # ---------------- Spectral ----------------
        sp_labels = SpectralClustering(
            n_clusters=k,
            affinity='nearest_neighbors',
            n_neighbors=5,
            random_state=42
        ).fit_predict(X_scaled)
        scores['spectral'].append(silhouette_score(X_scaled, sp_labels))
        stability['spectral'].append(clustering_stability(X_scaled, SpectralClustering, k))

        # ---------------- GMM ----------------
        gmm_labels = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42
        ).fit_predict(X_scaled)
        scores['gmm'].append(silhouette_score(X_scaled, gmm_labels))
        stability['gmm'].append(clustering_stability(X_scaled, GaussianMixture, k))

    # --- Plot dei risultati ---
    if plot:
        plt.figure(figsize=(10, 6))
        for method in scores.keys():
            plt.plot(K, scores[method], marker='o', label=f'{method} silhouette')
            plt.plot(K, stability[method], marker='x', label=f'{method} stability')
        plt.xlabel('Numero di cluster')
        plt.ylabel('Score')
        plt.title('Silhouette e Stability per algoritmo')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULT_FOLDER}img/compare_silhouette_stability_{time}.png')
        plt.show()

    best_k = {method: K[np.argmax(vals)] for method, vals in scores.items()}
    print("Numero ottimale cluster (silhouette):", best_k)
    return best_k, scores, stability


def plot_tsne_clusters(X_scaled, cluster_labels_dict, time=0,
                       perplexity=30, random_state=42):
    """Visualizza i cluster usando t-SNE 2D."""
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


def run_clustering_analysis(df, cluster_features, other_features,
                            max_k=10, time=0, use_pca=False):
    """
    Pipeline completa:
    1. Preprocessing
    2. Confronto metodi di clustering con silhouette e stability
    3. Fit modelli finali
    4. Analisi differenze tra cluster sulle feature originali
    """
    # --- Preprocessing ---
    X_scaled, df_clean, used_features = preprocess_data(df, cluster_features)

    # --- Confronto metodi ---
    best_k_all, _, _ = compare_clustering_methods(
        X_scaled, max_k=max_k, time=time, use_pca=use_pca
    )

    # --- Fit finale dei modelli ---
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

    # --- Dizionario cluster ---
    cluster_labels = {
        'kmeans': ('cluster_kmeans', kmeans),
        'spectral': ('cluster_spectral', None),
        'gmm': ('cluster_gmm', gmm)
    }

    cluster_labels_clean = {m: df_clean[col].values for m, (col, _) in cluster_labels.items()}

    # --- t-SNE per visualizzare i cluster ---
    plot_tsne_clusters(X_scaled, cluster_labels_clean, time=time)

    # --- Analisi e salvataggio ---
    for method, (col, model) in cluster_labels.items():
        print(f"\n=== Analisi per {method.upper()} ===")
        df_tmp = df_clean.copy()
        df_tmp['cluster'] = df_tmp[col]

        true_k = df_tmp['cluster'].nunique()
        print(f"True best K ({method}): {true_k}")

        analyze_cluster_differences(df_tmp, 'cluster', cluster_features + other_features, log_file=f"{RESULT_FOLDER}cluster_differences_{method}_{time}.txt")

        out_csv = f"clustered_data_{method}_{time}.csv"
        df_tmp.to_csv(f"{RESULT_FOLDER}{out_csv}", index=False, sep=';', decimal=',')
        print(f"✅ Risultati salvati in: {out_csv}")

        # Heatmap centroidi (solo KMeans/GMM)
        if method in ['kmeans', 'gmm']:
            centroids = model.cluster_centers_ if method == 'kmeans' else model.means_
            centroids_df = pd.DataFrame(centroids, columns=used_features)
            plt.figure(figsize=(10, 6))
            sns.heatmap(centroids_df, annot=True, cmap="coolwarm", fmt=".2f",
                        cbar_kws={'label': 'Valore normalizzato'})
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
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Caricamento colonne
    plasma_cols      = get_plasma_columns()
    liquor_cols      = get_liquor_columns()
    umns_cols        = get_umn_score_columns()
    alsfrs_cols      = get_alsfrs_columns()
    clinical_cols    = get_clinical_data()
    respiratory_cols = get_respiratory_columns()
    mrc_columns      = get_mrc_columns()
    macsplex_columns = get_macsplex_columns()
    als_no_als_colums= get_als_controls_data()

    # Definizione feature
    cluster_features = macsplex_columns  # + liquor_cols (opzionale)
    other_features   = (umns_cols + alsfrs_cols + plasma_cols +
                        clinical_cols + ['delta_bmi'] + mrc_columns +
                        liquor_cols + als_no_als_colums + respiratory_cols)
    if CLUSTER_CLINICS:
        cluster_features = clinical_cols
        other_features = (umns_cols + alsfrs_cols + plasma_cols +
                          macsplex_columns + ['delta_bmi'] + mrc_columns +
                          liquor_cols + als_no_als_colums + respiratory_cols)

    # Caricamento dati
    df = get_sheet(sheet_name='CSF')
    df = df[df['CD13'].notna()]
    df = df[~df['pt_code'].isin(['MI-ALS-EP-A1362', 'MI-ALS-EP-A1354'])]

    # Loop su timepoint (qui solo T0 per esempio)
    for t in [0]:
        print(f"\n########## Time {t} ##########")
        df_time = df[df['time_point'] == f'T{t}']
        run_clustering_analysis(
            df_time,
            cluster_features=cluster_features,
            other_features=other_features,
            max_k=8,
            time=t,
            use_pca=USE_PCA
        )