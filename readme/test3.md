# Analisi Clustering su Dati di Biomarcatori

Questo progetto esegue un'analisi di **clustering** sui dati di biomarcatori per identificare gruppi clinici simili e studiare le differenze tra i gruppi.

Il codice è scritto in **Python** e utilizza librerie scientifiche come `pandas`, `numpy`, `scikit-learn` e `seaborn`.

---

## Cosa fa il codice

La pipeline completa include:

1. **Pre-processing e scaling**
   - Pulizia dei dati, gestione dei valori mancanti e normalizzazione delle feature selezionate per il clustering.
   
2. **(Opzionale) PCA**
   - Riduzione della dimensionalità per eliminare rumore e collinearità tra variabili, mantenendo la maggior parte della varianza.

3. **Confronto di algoritmi di clustering**
   - Si confrontano **KMeans**, **Spectral Clustering** e **Gaussian Mixture**.
   - Viene calcolato lo **silhouette score**, che misura quanto i cluster siano separati e coerenti internamente.
   - Viene calcolata la **stabilità del clustering** con bootstrap e ARI (Adjusted Rand Index), per capire quanto i cluster siano robusti a variazioni dei dati.

4. **Clustering finale**
   - Ogni algoritmo viene eseguito con il numero ottimale di cluster trovato dal confronto.
   - I cluster assegnati vengono aggiunti ai dati originali.

5. **Analisi delle differenze tra cluster**
   - Per ogni variabile, vengono calcolate **media, mediana e IQR** per ciascun cluster.
   - Vengono eseguiti test statistici:
     - Variabili numeriche:
       - 2 cluster: t-test o Mann–Whitney U
       - >2 cluster: Kruskal–Wallis
     - Variabili categoriali: Chi-quadro
   - Tutti i risultati dell’analisi vengono salvati in file di **log** (`.txt`).

6. **Salvataggio dei risultati**
   - I dati clusterizzati vengono salvati in CSV (`clustered_data_<algoritmo>_<time>.csv`).
   - Centroidi dei cluster per KMeans e GMM vengono salvati in **heatmap** (`.png`).
   - Visualizzazione dei cluster su **mappa 2D t-SNE** (`.png`) per facilitare l’interpretazione.

---

## Struttura dei file

- `test3.py` → Script principale con tutta la pipeline.
- `get_data.py` → Funzioni per caricare le varie colonne e dataset.
- `utilities.py` → Funzioni di supporto (preprocessing, ecc.)
- `results/` → Cartella dove vengono salvati CSV, heatmap e log.
  - `results/img/` → Grafici delle heatmap e t-SNE.

---

## Come eseguire

1. Assicurati di avere **Python 3.9+** installato.
2. Installa le librerie richieste:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib scipy
```

Apri il terminale nella cartella del progetto.

Esegui lo script:

```bash
python test3.py
```

Puoi modificare use_pca=True se vuoi ridurre dimensionalità.

I risultati saranno salvati nella cartella results/test3/.