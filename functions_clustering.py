# This script contains various functions useful for the experiments

###############################################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# January 2026

################################################################################################################################

from scipy.stats import spearmanr, kendalltau

from sklearn import preprocessing
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Leiden
from sknetwork.clustering import get_modularity
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.linalg import issymmetric

from scipy.stats import kruskal, chi2_contingency, ttest_ind, mannwhitneyu


###############################################################################################################################################################################

# Utility functions for selecting specific columns from a database

def get_EV_markers_columns():
    return  ['CD13', 'CD24', 'CD29', 'CD31', 'CD36', 'CD38', 'CD45',
       'CD49a', 'CD49e', 'CD49f', 'CD54',  'CD81',
        'CD106', 'CD133/1', 'CD140a', 'CD222 IGF2R',
       'CX3CR1', 'EGFR', 'Ganglioside GD2', 'Podoplanin', 
       'CD9', 'CD44', 'CD63', 'CD107a',
        'GLAST ACSA-1', 'O4',  'CD11b', 'CSPG4']
        
def get_biochemical_markers_columns():
    return ['CSF_GFAP', 'CSF_NFL', 'CSF_tau', 'CSF_UCHL1', 'CSF_MMP-9', 'CSF_MCP-1']

def get_clinical_parameters_columns():
	return ['sex', 'onset_age', 'dgn_age', 'dgn_delay', 'el_escorial_criteria', 'gold_coast_criteria', 'genetic_WT_mut', 'C9orf72',
		   'phenotype', 'site_of_onset', 'side_of_onset', 'pre-morbid_weight', 'weight_at_diagnosis', 'pre-morbid_BMI', 'dgn_BMI',
		   'delta_weight_pre_dgn', 'delta_BMI_pre_dgn', 'time_to_NIV', 'time_to_PEG', 'time_to_tracheo', 'time_to_death', 'riluzole',
		   'other_DM_therapies', 'trial', 'MRC_composite_score', 'ALSFRSr_TOT', 'delta_ALSFRS_onset', "King's", 'C_MiToS_TOT', 'MGH_TOT',
		   'Penn_TOT', 'Strong_CAT']

def get_plasma():
	return ['plasma_GFAP', 'plasma_NFL', 'plasma_tau', 'plasma_UCHL1', 'plasma_MMP-9', 'plasma_MCP-1']
    
    
def get_clinical_parameters_categorical():
	return ['sex', 'el_escorial_criteria', 'gold_coast_criteria', 'genetic_WT_mut', 'C9orf72', 'phenotype', 'site_of_onset', 'side_of_onset', 'riluzole',
		   'other_DM_therapies', 'trial', 'Strong_CAT']


##################################################################################################################################

# Functions for converting categorical features into numerical ones

def convert_el_escorial_criteria(x):
    x = x.lower()
    if x == 'definite':
        res = 5
    elif  x == 'probable':
        res = 4
    elif x == 'probable laboratory supported':
        res = 3
    elif x == 'possible':
        res = 2
    elif x == 'suspected':
        res = 1
    else:
        res = x
    return res

def convert_gold_coast_criteria(x):
    x = x.lower()
    if x == 'yes':
        res = 1
    elif  x == 'no':
        res = 0
    else:
        res = x
    return res

def convert_genetic_WT_mut(x):
	x = x.lower()
	if x == 'mut':
		res = 1
	elif x == 'wt':
		res = 0
	else:
		res = x
	return res

def convert_Strong_CAT(x):
	x = x.lower()
	if x == 'cn':
		res = 1
	else:
		res = x
	return res
    

######################################################################################################################################

# Stefano’s function to analyze whether the features are statistically significant within the identified clusters

def analyze_cluster_differences(df, cluster_col, other_features):
    """
    Analizza differenze tra i cluster per ciascuna variabile (numerica o categorica),
    mostrando sempre le IQR per le variabili numeriche.
    """
    
    categorical_features = get_clinical_parameters_categorical()
    
    print("\n== Analisi delle differenze tra cluster ==")
    num_clusters = df[cluster_col].nunique()

    for feature in other_features:
        print(f"\nVariabile: {feature}")
        try:
            groups = [group[feature].dropna() for _, group in df.groupby(cluster_col)]
            
            if feature in categorical_features:
                dtype = "object"
            else:
                dtype = df[feature].dtype

            if np.issubdtype(dtype, np.number):
                # ---- Stampo SEMPRE le IQR di base, per ogni cluster ----
                total_n = sum(len(g) for g in groups)
                for i, g in enumerate(groups):
                    if len(g) == 0:
                        continue
                    q1, q3 = np.percentile(g, [25, 75])
                    median = np.median(g)
                    mean = np.mean(g)
                    count = len(g)
                    perc = (count / total_n) * 100
                    print(
                        f"  Gruppo {i}: N={count}, "
                        f"Media={mean:.2f}, Mediana={median:.2f}, "
                        f"IQR=[{q1:.2f}, {q3:.2f}] "
                        f"({perc:.1f}% dei dati)"
                    )

                # ---- Test statistici in base al numero di cluster ----
                if num_clusters == 2:
                    group1, group2 = groups
                    unique_vals = df[feature].dropna().unique()
                    n1, n2 = len(group1), len(group2)

                    if len(unique_vals) == 2:
                        # t-test + percentuali
                        stat, p = ttest_ind(group1, group2, equal_var=False)
                        print(f"  T-test independent: t = {stat:.2f}, p = {p:.3f}")
                    else:
                        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
                        print(f"  Mann–Whitney U: U = {stat:.2f}, p = {p:.3f}")

                else:
                    if all(len(g) > 1 for g in groups):
                        stat, p = kruskal(*groups)
                        print(f"  Test Kruskal–Wallis (non param.): p = {p:.4f}")

            else:
                # Variabile categorica → Chi-quadro
                contingency = pd.crosstab(df[cluster_col], df[feature])
                stat, p, _, _ = chi2_contingency(contingency)
                print(f"  Test Chi-quadro: p = {p:.4f}")

        except Exception as e:
            print(f"  Errore con la variabile {feature}: {e}")

##########################################################################################################################################################################


# Function to compute clusters using Leiden algorithm

def compute_clustering(dataset_clusters, dirpath, name, scaler_method = 'StandardScaler', resolution_Leiden = 1):

	'''It applies Leiden algorithm:	
		- dataset_clusters = dataset used to identify the clusters. Missing values have already been imputed.
		- dirpath, name = name to save
		- scaler = standardization (default 'StandardScaler)
        - resolution_Leiden = resolution parameter of Leiden (default 1)
	'''


	if scaler_method == 'StandardScaler':
		scaler = preprocessing.StandardScaler()
		dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_clusters), columns=dataset_clusters.columns, index=dataset_clusters.index)
		
	elif scaler == 'RobustScaler':
		scaler = preprocessing.RobustScaler()
		dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_clusters), columns=dataset_clusters.columns, index=dataset_clusters.index)
		
	else:
		dataset_scaled = dataset_clusters.copy()

	# Compute the graph (use k-NearestNeighbors with k = 5)
	n = np.shape(dataset_scaled)[0]
	neigh = NearestNeighbors().fit(dataset_scaled)  
	W = neigh.kneighbors_graph(dataset_scaled).toarray() - np.diag(np.ones(n))
	D = (W == 1)
	boolean_graph = csr_matrix(D)

	# Apply Leiden algorithm
	if issymmetric(W):
		formula_modularity = 'newman'
	else:
		formula_modularity = 'dugue'
	leiden = Leiden(resolution = resolution_Leiden, random_state=42, modularity = formula_modularity)
	Leiden_clusters = leiden.fit_predict(boolean_graph)
	
	print(f'Resolution parameter = {resolution_Leiden}:')
	print(f'       - the associated graph is {'undirected' if issymmetric(W) else 'directed'};')
	print(f'       - with Leiden algorithm we obtain {len(set(Leiden_clusters))} clusters;')
	print(f'       - the modularity is {float(np.round(get_modularity(boolean_graph, Leiden_clusters), 6))}. \n')
    
	return Leiden_clusters
    
##########################################################################################################################################################################

# It plot a 2d and 3D clusters and the box plots, violin plots and distributions.

def plot_features_clusters(dataset, clusters, dirpath, scaler_method = 'StandardScaler'):
    
    '''
    Several plots:
     - dataset : the original dataset
     - clusters : label of the cluster
     - dirpath : to save
    
    '''

    dataset_original = dataset.copy()
    n = len(dataset_original)
    
    # For visualization only: missing values replaced with column mean and data scaled to visualize clusters.
    columns = dataset_original.columns
    for ii in columns:
        dataset_original[ii] = pd.to_numeric(dataset_original[ii], errors = "coerce")
    dataset_original_filled = dataset_original.fillna(dataset_original.mean()).astype(float).copy()
    
    if scaler_method == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
        dataset_original_scaled = pd.DataFrame(scaler.fit_transform(dataset_original_filled), columns=dataset_original_filled.columns, 
                                               index=dataset_original_filled.index)
    elif scaler_method == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
        dataset_original_scaled = pd.DataFrame(scaler.fit_transform(dataset_original_filled), columns=dataset_original_filled.columns, 
                                               index=dataset_original_filled.index)
    else:
        dataset_original_scaled = dataset_original_filled.copy()
    
    X_tsne2 = TSNE(n_components=2,perplexity=30,random_state=42).fit_transform(np.array(dataset_original_scaled).reshape((n,-1)))
    X_tsne3 = TSNE(n_components=3,perplexity=30,random_state=42).fit_transform(np.array(dataset_original_scaled).reshape((n,-1)))

    fig1, ax = plt.subplots(1, 2)
    ax[0].scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=clusters, cmap='jet')
    fig1.delaxes(ax[1])
    ax[1] = fig1.add_subplot(1, 2, 2, projection='3d')
    ax[1].scatter(X_tsne3[:, 0], X_tsne3[:, 1],X_tsne3[:, 2], c=clusters, cmap='jet')
    plt.show()
    
    # Added the column with the identified clusters to the original datasets
    dataset_original.loc[:,'group'] = clusters

    # Show how the features behave within the identified clusters
    groups = [group for _, group in dataset_original.groupby('group')]


    for feat in columns:
    
        fig2 = plt.figure(figsize=(15, 5))
    
        # 1. Box plots
        plt.subplot(1, 3, 1)
        ax = sns.boxplot(x=feat, y = 'group', data = dataset_original, hue='group', palette= "Set1",  showmeans=True,  
                         orient = 'h', meanprops=({"marker":"*", "markerfacecolor":"blue", "markeredgecolor": "k", "markersize":10}))
        plt.title(f'Box Plot of {feat}')
        plt.xlabel(feat)
    
        # 2. Violin plots
        plt.subplot(1, 3, 2)
        ax = sns.violinplot(x=feat, y = 'group', data = dataset_original,  hue = 'group', palette= "Set1", 
                            orient='h', inner='points')
        plt.title(f'Violin Plot of {feat}')
        plt.xlabel(feat)
    
        # 3. Histogram for each cluster
        plt.subplot(1, 3, 3)
        for jj in range(len(groups)):
            sns.histplot(groups[jj][feat], alpha=0.5, label= str(jj), kde=True)
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        plt.legend()
        if feat == 'CD133/1':
            feat = 'CD133-1'
        fig2.savefig(dirpath + str(feat) +".png")
        plt.show()
    return 
    
