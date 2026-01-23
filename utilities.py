import numpy as np
from scipy.stats import f_oneway, kruskal, chi2_contingency, ttest_ind, mannwhitneyu
import pandas as pd
from get_data import get_clinical_parameters_categorical

# ---------------------------------------------------------------------
# 2. Analisi differenze tra cluster
# ---------------------------------------------------------------------
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
