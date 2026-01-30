#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains various functions useful for the experiments.

Author: Chiara Faccio, Università degli Studi di Padova
Email: chiara.faccio@unipd.it
Date: January 2026
"""

# ======================================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, kendalltau
from scipy.sparse import csr_matrix
from scipy.linalg import issymmetric

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from sknetwork.clustering import Leiden, get_modularity

from get_data import *
from utilities import *

# ======================================================================================================================
# Utility functions for selecting specific columns from a database



# ======================================================================================================================
# Function to compute clusters using Leiden algorithm


def compute_clustering(dataset_clusters, scaler_method='StandardScaler', resolution_Leiden=1):
    '''It applies Leiden algorithm:
        - dataset_clusters = dataset used to identify the clusters. Missing values have already been imputed.
        - dirpath, name = name to save
        - scaler = standardization (default 'StandardScaler)
        - resolution_Leiden = resolution parameter of Leiden (default 1)
    '''

    if scaler_method == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
        dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_clusters), columns=dataset_clusters.columns,
                                      index=dataset_clusters.index)
    else:
        dataset_scaled = dataset_clusters.copy()

    # Compute the graph (use k-NearestNeighbors with k = 4)
    n = np.shape(dataset_scaled)[0]
    neigh = NearestNeighbors(algorithm='brute', metric='euclidean', n_jobs=1).fit(dataset_scaled)
    # W = neigh.kneighbors_graph(dataset_scaled).toarray() - np.diag(np.ones(n))
    W = neigh.kneighbors_graph(dataset_scaled).toarray()
	np.fill_diagonal(W, 0)
    D = (W == 1)
    boolean_graph = csr_matrix(D)

    # Apply Leiden algorithm
    if issymmetric(W):
        formula_modularity = 'newman'
    else:
        formula_modularity = 'dugue'
    leiden = Leiden(resolution=resolution_Leiden, random_state=42, modularity=formula_modularity)
    Leiden_clusters = leiden.fit_predict(boolean_graph)

    # print(f'Resolution parameter = {resolution_Leiden}:')
    # print(f'       - the associated graph is {'
    # undirected
    # ' if issymmetric(W) else '
    # directed
    # '};')
    # print(f'       - with Leiden algorithm we obtain {len(set(Leiden_clusters))} clusters;')
    # print(f'       - the modularity is {float(np.round(get_modularity(boolean_graph, Leiden_clusters), 6))}. \n')

    return Leiden_clusters

# ======================================================================================================================
# Plot 2D / 3D clusters and feature distributions

def plot_features_clusters(
    dataset,
    clusters,
    dirpath,
    scaler_method="StandardScaler",
):
    """
    Plot cluster visualizations and feature distributions.

    Parameters
    ----------
    dataset : pd.DataFrame
        Original dataset.
    clusters : array-like
        Cluster labels.
    dirpath : str
        Directory path to save plots.
    scaler_method : str, optional
        Scaling method for visualization.
    """

    dataset_original = dataset.copy()
    n = len(dataset_original)

    # Convert to numeric and handle missing values
    for col in dataset_original.columns:
        dataset_original[col] = pd.to_numeric(dataset_original[col], errors="coerce")

    dataset_filled = dataset_original.fillna(dataset_original.mean()).astype(float)

    # ---------------- Scaling (for visualization only) ----------------
    if scaler_method == "StandardScaler":
        scaler = preprocessing.StandardScaler()
        dataset_scaled = pd.DataFrame(
            scaler.fit_transform(dataset_filled),
            columns=dataset_filled.columns,
            index=dataset_filled.index,
        )

    elif scaler_method == "RobustScaler":
        scaler = preprocessing.RobustScaler()
        dataset_scaled = pd.DataFrame(
            scaler.fit_transform(dataset_filled),
            columns=dataset_filled.columns,
            index=dataset_filled.index,
        )

    else:
        dataset_scaled = dataset_filled.copy()

    # ---------------- t-SNE ----------------
    X_tsne2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
        dataset_scaled.values.reshape((n, -1))
    )

    X_tsne3 = TSNE(n_components=3, perplexity=30, random_state=42).fit_transform(
        dataset_scaled.values.reshape((n, -1))
    )

    # ---------------- Scatter plots ----------------
    fig1, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=clusters, cmap="jet")
    ax[0].set_title("t-SNE 2D")

    fig1.delaxes(ax[1])
    ax[1] = fig1.add_subplot(1, 2, 2, projection="3d")
    ax[1].scatter(
        X_tsne3[:, 0],
        X_tsne3[:, 1],
        X_tsne3[:, 2],
        c=clusters,
        cmap="jet",
    )
    ax[1].set_title("t-SNE 3D")

    plt.show()

    # ---------------- Feature distributions ----------------
    dataset_original["group"] = clusters
    groups = [group for _, group in dataset_original.groupby("group")]

    for feat in dataset_original.columns.drop("group"):

        fig2 = plt.figure(figsize=(15, 5))

        # Box plot
        plt.subplot(1, 3, 1)
        sns.boxplot(
            x=feat,
            y="group",
            data=dataset_original,
            hue="group",
            palette="Set1",
            showmeans=True,
            orient="h",
            meanprops={
                "marker": "*",
                "markerfacecolor": "blue",
                "markeredgecolor": "k",
                "markersize": 10,
            },
        )
        plt.title(f"Box Plot of {feat}")

        # Violin plot
        plt.subplot(1, 3, 2)
        sns.violinplot(
            x=feat,
            y="group",
            data=dataset_original,
            hue="group",
            palette="Set1",
            orient="h",
            inner="points",
        )
        plt.title(f"Violin Plot of {feat}")

        # Histogram
        plt.subplot(1, 3, 3)
        for jj, group in enumerate(groups):
            sns.histplot(group[feat], alpha=0.5, label=str(jj), kde=True)

        plt.title(f"Distribution of {feat}")
        plt.legend()

        safe_feat = feat.replace("/", "-")
        fig2.savefig(f"{dirpath}{safe_feat}.png")

        plt.show()

    return
