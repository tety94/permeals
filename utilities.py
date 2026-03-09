import numpy as np
from scipy.stats import f_oneway, kruskal, chi2_contingency, ttest_ind, mannwhitneyu
import pandas as pd
from get_data import get_clinical_parameters_categorical

# ---------------------------------------------------------------------
# 2. Analisi differenze tra cluster
# ---------------------------------------------------------------------
def analyze_cluster_differences(df, cluster_col, other_features, log_file=None):
    """
    Analizza differenze tra i cluster per ciascuna variabile (numerica o categorica),
    mostrando sempre le IQR per le variabili numeriche.

    Se log_file è specificato, salva tutto l'output su quel file.
    """
    categorical_features = get_clinical_parameters_categorical()
    num_clusters = df[cluster_col].nunique()

    # --- Gestione output su file o console ---
    if log_file:
        f = open(log_file, "w", encoding="utf-8")
        write = lambda s: f.write(s + "\n")
    else:
        write = print

    write("\n== Analisi delle differenze tra cluster ==")

    for feature in other_features:
        write(f"\nVariabile: {feature}")
        try:
            groups = [group[feature].dropna() for _, group in df.groupby(cluster_col)]

            dtype = "object" if feature in categorical_features else df[feature].dtype

            if np.issubdtype(dtype, np.number):
                total_n = sum(len(g) for g in groups)
                for i, g in enumerate(groups):
                    if len(g) < 1:
                        write(f"  Gruppo {i}: meno di 1 elemento, skip")
                        continue
                    q1, q3 = np.percentile(g, [25, 75])
                    median = np.median(g)
                    mean = np.mean(g)
                    count = len(g)
                    perc = (count / total_n) * 100
                    write(
                        f"  Gruppo {i}: N={count}, "
                        f"Media={mean:.2f}, Mediana={median:.2f}, "
                        f"IQR=[{q1:.2f}, {q3:.2f}] "
                        f"({perc:.1f}% dei dati)"
                    )

                # Test statistico
                if num_clusters == 2 and all(len(g) > 0 for g in groups):
                    group1, group2 = groups
                    unique_vals = df[feature].dropna().unique()

                    if len(unique_vals) == 2:
                        stat, p = ttest_ind(group1, group2, equal_var=False)
                        write(f"  T-test independent: t = {stat:.2f}, p = {p:.3f}")
                    else:
                        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
                        write(f"  Mann–Whitney U: U = {stat:.2f}, p = {p:.3f}")

                elif num_clusters > 2 and all(len(g) > 1 for g in groups):
                    stat, p = kruskal(*groups)
                    write(f"  Test Kruskal–Wallis (non param.): p = {p:.4f}")

            else:
                # Categoriale → Chi-quadro
                contingency = pd.crosstab(df[cluster_col], df[feature])
                stat, p, _, _ = chi2_contingency(contingency)
                write(f"  Test Chi-quadro: p = {p:.4f}")

        except Exception as e:
            write(f"  Errore con la variabile {feature}: {e}")

    if log_file:
        f.close()
        print(f"✅ Log salvato in: {log_file}")