"""
The script computes clusters using the Leiden algorithm while varying the resolution parameter, yielding stable clustering solutions.
A subset of features is used to perform the clustering, while the remaining features are exploited to evaluate how their values change within the resulting clusters.
"""

############################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# February 2026

#############################################################################################################

from functions_clustering import *                 # it contains utility functions
import pandas as pd                                # it permits data manipulation and analysis
import numpy as np                                 # it is a package for scientific computing in Python
from sklearn import preprocessing                  # used for standardization
import seaborn as sns                              # it contains tools for statistical data visualization
import matplotlib.pyplot as plt                    # it is a library for creating plots
from scipy.spatial import distance                 # to perform hierarchical clustering
from scipy.cluster import hierarchy, hierarchy
from scipy.cluster.hierarchy import fcluster

from sklearn.metrics.cluster import adjusted_rand_score  # to perform ARI


################################################################################################################

if __name__ == '__main__':  
    
    dirpath = "C:/Users/chiara/Documents/Post-Doc_Padova/LAVORO_ASL/liquor/Clustering_Leiden_DB_09_01_26/"
    dirpath_img = "C:/Users/chiara/Documents/Post-Doc_Padova/LAVORO_ASL/liquor/Clustering_Leiden_DB_09_01_26/img/"
    file = "Per analisi CSF_Db PERMEALS - EvTestinALS_verticaltimepoints_07.05_08.01"
    sheet_name = "CSF"

    # Read the Excel file and convert categorical features to numeric
    dataset = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name, converters={'el_escorial_criteria' : convert_el_escorial_criteria, 
                                                           'gold_coast_criteria': convert_gold_coast_criteria,
                                                           'genetic_WT_mut' : convert_genetic_WT_mut,
                                                           'Strong_CAT' : convert_Strong_CAT})
                                                           

    # Delete pz with 'el_escorial_criteria = PLS definite' (Sara)
    dataset = dataset[dataset['el_escorial_criteria'] != 'pls definite'].copy()
    dataset = dataset[dataset['pt_code'] != 'MI-ALS-EP-A1362'].copy()

    # Dataset to compute the clusters
    features_clusters = get_EV_markers_columns() + get_plasma() + get_biochemical_markers_columns() 
    dataset_cluster = dataset[features_clusters].copy()
    
    # Relevant features 
    relevant_features = features_clusters + get_clinical_parameters_columns()
    
    
    # Missing values are replaced with the column mean
    for ii in features_clusters:
        dataset_cluster[ii] = pd.to_numeric(dataset_cluster[ii], errors="coerce")
    dataset_filled = dataset_cluster.fillna(dataset_cluster.mean()).astype(float).copy()
    
    # Insert resolution parametr
    lower_bound = float(input(' lower bound for resolution = '))
    upper_bound = float(input(' upper bound for resolution = '))
    
    resolutions = np.arange(lower_bound, upper_bound+0.05, 0.05)
    
    M = len(resolutions)
    N = len(dataset_filled)

    CLUSTERS  = np.zeros((M, N))
    
    # For each resolution parameter, compute the Leiden algorithm
    for ii, res in enumerate(resolutions):        
        Leiden_clusters = compute_clustering(dataset_filled, dirpath_img, 'stable_Leiden_algorithm', scaler_method = 'StandardScaler', resolution_Leiden = res)
        CLUSTERS[ii, :] = Leiden_clusters
       
    
    # In order to compute the similarity between the partitions, we compute the Adjusted Rand Index1 pairwise and used the results to construct a similarity matrix
    ad_rand_index = np.array([[adjusted_rand_score(CLUSTERS[i,:], CLUSTERS[j,:]) for i in range(M) ]for j in range(M)])

    sns.heatmap(ad_rand_index, annot = True, cmap="coolwarm")
    plt.title('Adjusted Rand index matrix')
    plt.savefig(dirpath_img + "ad_rand_index.png")
    plt.show()
    
    # Compute the co-association matrix
    co_assoc = np.array([[sum(c[i] == c[j] for c in CLUSTERS)/M for j in range(N)] for i in range(N) ])
    
    # Apply hierarchical.clustering to the co-association matrix to find stable clusters
    row_linkage = hierarchy.linkage(distance.pdist(co_assoc), method='average')
    col_linkage = hierarchy.linkage(distance.pdist(co_assoc.T), method='average')

    fig2 = sns.set(font_scale=0.6)
    cg2 = sns.clustermap(co_assoc, row_linkage=row_linkage, col_linkage=col_linkage, method='average', xticklabels=True,yticklabels=True,cmap="coolwarm", linewidths=1,linecolor='black')
    fig2 = plt.savefig(dirpath_img + "den_co-assoc.png")
    plt.show()
    
    # Identify the number of clusters from the dendogram
    n_cluster = int(input(' Insert the number of clusters identified from the dendrogram = '))
    
    stable_Leiden_clusters = fcluster(row_linkage, n_cluster, criterion='maxclust')
    
    # Plot all the features in the clusters
    plot_features_clusters(dataset[relevant_features], stable_Leiden_clusters, dirpath_img)

    # Added the column with the identified clusters to the original datasets
    dataset.loc[:,'Leiden_clusters'] = stable_Leiden_clusters
    
    # Analyze difference in the clusters
    analyze_cluster_differences(dataset, 'Leiden_clusters', relevant_features)

    # Save file Excel
    dataset.to_excel(dirpath + file +'_CLUSTERS_LEIDEN_stable.xlsx')

    dataset_sort = dataset.sort_values('Leiden_clusters')
    dataset_sort.to_excel(dirpath + file +'_CLUSTERS_LEIDEN_stable_sorted.xlsx')
    
   
