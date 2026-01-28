"""
The script computes Time Series clustering using tslearn
"""

############################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# February 2026

#############################################################################################################

from functions_clustering import *                            # it contains utility functions
import pandas as pd                                # it permits data manipulation and analysis
import numpy as np                                 # it is a package for scientific computing in Python
import seaborn as sns                              # it contains tools for statistical data visualization
import matplotlib.pyplot as plt                    # it is a library for creating plots

#import tslearn
from tslearn.utils import to_time_series_dataset                  # tslearn performs time series clustering
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


################################################################################################################



if __name__ == '__main__':  

    dirpath_img = "..."
    file = "..."
    dirpath = "..."
	sheet_name = sheet_name

    # Read the Excel file 
    dataset = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name)
    
    relevant_features = ['time_point'] + [...]  # add features

    # Features used to compute clusters
    cluster_features = [...]  # add features


    dataset = dataset[relevant_features].copy()
    for ii in relevant_features[1:]:
        dataset[ii] = pd.to_numeric(dataset[ii], errors = "coerce")

    # Remove rows with 38 or more missing values
    dataset = dataset[dataset.isna().sum(axis=1) < 38].copy()                                                    

    # Extract data at T0, T6, T12 and impute the missing values
    data_T0 = dataset[dataset['time_point'] == 'T0'].copy()
    for col in relevant_features[1:]:
        average = (data_T0.loc[:,col]).mean()
        data_T0.loc[:,col] = (data_T0.loc[:,col]).astype(float).fillna(average)
        
    data_T6 = dataset[dataset['time_point'] == 'T6'].copy()
    for col in relevant_features[1:]:
        average = (data_T6.loc[:,col]).mean()
        data_T6.loc[:,col] = (data_T6.loc[:,col]).astype(float).fillna(average)
        
    data_T12 = dataset[dataset['time_point'] == 'T12'].copy()
    for col in relevant_features[1:]:
        average = (data_T12.loc[:,col]).mean()
        data_T12.loc[:,col] = (data_T12.loc[:,col]).astype(float).fillna(average)
        
        
    # Transform the dataset for tslearn
    X_all = to_time_series_dataset(np.stack([np.array(data_T0[cluster_features]), np.array(data_T6[cluster_features]), np.array(data_T12[cluster_features])], axis=1))
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_all)
        
        
    # Silhoutte score to find obtain number of clusters
    seed = 42
    best_config = 0
    best_score = float("-inf")
    silhouette = []
    inertias = []

    Nclusters = np.arange(2,10)

    for kk in Nclusters:
        model = TimeSeriesKMeans(n_clusters = kk, metric='softdtw', random_state= seed)
        y_pred = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)

        score = silhouette_score(X_scaled, y_pred, metric='softdtw')
        silhouette.append(score)

        if score > best_score:
            best_score = score
            best_config = kk

    print(f"The best number of clusters is {best_config} with Silhouette Score: {best_score}")
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(Nclusters, inertias, 'bo-')
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Inertia')

    ax[1].plot(Nclusters, silhouette, 'ro-')
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Silhouette')
    fig.savefig(dirpath_img +"_time_series_Inertia_Silhoutte.png")
    plt.show()
   
    
    # Ask optimal number of clusters
    n_clusters = int(input('Enter the number of clusters found = '))
    
    # Find the clusters
    model2 = TimeSeriesKMeans(n_clusters=n_clusters, metric='softdtw', random_state= seed)
    y_pred2 = model2.fit_predict(X_scaled)
    
    # Analysis clusters at T0
    original_data_T0 = dataset[dataset['time_point'] == 'T0'].copy()
    original_data_T0.loc[:,'cluster'] = y_pred2
    print('\n-----TIME T0------')
    analyze_cluster_differences(original_data_T0, 'cluster', relevant_features[1:])
    
    # Analysis clusters at T6
    original_data_T6 = dataset[dataset['time_point'] == 'T6'].copy()
    original_data_T6.loc[:,'cluster'] = y_pred2
    print('\n-----TIME T6------')
    analyze_cluster_differences(original_data_T6, 'cluster', relevant_features[1:])
    
    # Analysis clusters at T12
    original_data_T12 = dataset[dataset['time_point'] == 'T12'].copy()
    original_data_T12.loc[:,'cluster'] = y_pred2
    print('\n-----TIME T12------')
    analyze_cluster_differences(original_data_T12, 'cluster', relevant_features[1:])
   
    
    
    X_plt = np.stack([np.array(data_T0[relevant_features[1:]]), np.array(data_T6[relevant_features[1:]]), np.array(data_T12[relevant_features[1:]])], axis=1)
    X_plt_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_plt)
    
    X_tsne2 = TSNE(n_components=2,perplexity=30,random_state=seed).fit_transform(X_plt_scaled.reshape(X_plt_scaled.shape[0], -1))
    X_tsne3 = TSNE(n_components=3,perplexity=30,random_state=seed).fit_transform(X_plt_scaled.reshape(X_plt_scaled.shape[0], -1))

    fig1, ax = plt.subplots(1, 2)
    ax[0].scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=model2.labels_, cmap='jet')
    fig1.delaxes(ax[1])
    ax[1] = fig1.add_subplot(1, 2, 2, projection='3d')
    ax[1].scatter(X_tsne3[:, 0], X_tsne3[:, 1],X_tsne3[:, 2], c=model2.labels_, cmap='jet')
    plt.show()
   