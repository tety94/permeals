# Analisi Clustering Serie Temporali di Pazienti ALS

Questo progetto esegue un'analisi di **clustering** sui biomarcatori di pazienti ALS (plasma, liquor, dati clinici)
per identificare sottogruppi di pazienti con profili temporali simili e studiare le differenze tra i gruppi.

---

## Cosa fa il codice

La pipeline completa include:

1. **Pre-processing e pulizia dei dati**
   - Selezione delle feature per il clustering.
   - Conversione a valori numerici.
   - Rimozione di righe con troppi valori mancanti.
   - Selezione solo dei pazienti con tutti i timepoint disponibili.
   - Imputazione dei valori mancanti verticalmente (media per timepoint).

2. **Costruzione delle serie temporali multivariate**
   - Ogni paziente è rappresentato da una serie temporale con shape `(timepoints x features)`.
   - Le serie vengono scalate con media=0 e varianza=1.

3. **Selezione del numero di cluster**
   - Clustering preliminare con **TimeSeriesKMeans** usando metrica **Soft-DTW**.
   - Calcolo di **inertia** e **silhouette score** per valutare separazione e coerenza dei cluster.
   - Scelta del numero ottimale di cluster basata sul silhouette score.

4. **Clustering finale**
   - Esecuzione di TimeSeriesKMeans con il numero ottimale di cluster.
   - Etichette dei cluster assegnate ai pazienti.

5. **Analisi delle differenze tra cluster**
   - Per ogni timepoint e per ogni variabile selezionata:
     - Calcolo di medie, mediane, IQR.
     - Test statistici per identificare differenze significative tra cluster.
   - Risultati salvati in file di log (`.txt`).

6. **Visualizzazione dei cluster**
   - Grafici **inertia / silhouette** per valutare il numero di cluster.
   - **t-SNE 2D e 3D** per visualizzare la separazione dei cluster in spazio ridotto.

7. **Salvataggio dei risultati**
   - Serie temporali clusterizzate salvate in CSV.
   - Grafici e log salvati nella cartella `results/test6/`.

---

## Struttura dei file

- `test6.py` → Script principale con tutta la pipeline di clustering temporale.
- `get_data.py` → Funzioni per caricare dataset e colonne rilevanti.
- `functions_clustering.py` → Funzioni di supporto per analisi dei cluster e plotting.
- `results/` → Cartella dove vengono salvati CSV, heatmap e log.
  - `results/test6/` → Grafici inertia/silhouette, t-SNE e log delle analisi cluster.

---

## Come eseguire

1. Assicurati di avere **Python 3.9+** installato.
2. Installa le librerie richieste:

```bash
pip install numpy pandas matplotlib tslearn scikit-learn
```

## Esecuzione dello script

Apri il terminale nella cartella del progetto ed esegui:

```bash
python test6.py
```

I risultati saranno salvati nella cartella `results/test6/`.

Puoi modificare i parametri come:

- `timepoints` → selezione dei timepoint da analizzare
- `cluster_features` → colonne dei biomarcatori da usare per il clustering
- `best_k` → numero di cluster finale (opzionale, se vuoi forzarlo manualmente)