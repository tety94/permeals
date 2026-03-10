# 📘 Progetto **permeals** — Analisi di Dati Clinici e Biomarcatori ALS

Questo repository contiene una serie di pipeline Python per l’analisi di **dati clinici e biomarcatori** di pazienti con ALS, finalizzate a:

- Identificare **sottogruppi di pazienti** con profili biologici o clinici simili.  
- Analizzare **serie temporali** di biomarcatori.  
- Eseguire **classificazione predittiva** usando XGBoost.  

La cartella `readme/` ospita diversi script e documentazione associata, ognuno con un focus specifico.

---

## 🔹 Panoramica dei sotto-file e pipeline

### 1. Clustering su biomarcatori o dati clinici (`test3.py`)

- Analizza **cluster clinici** partendo da biomarcatori o dati clinici.
- **Pipeline principale:**
  1. Pre-processing e scaling dei dati.
  2. Riduzione dimensionale opzionale con PCA.
  3. Confronto di algoritmi di clustering: **KMeans, Spectral Clustering, Gaussian Mixture**.
  4. Valutazione della **stabilità dei cluster** con bootstrap e ARI.
  5. Analisi delle differenze tra cluster con test statistici.
  6. Salvataggio di CSV, heatmap e visualizzazioni t-SNE 2D.

- **Cartelle principali**:
  - `results/` → CSV, heatmap, log dei risultati.
  - `results/img/` → Grafici cluster.

---

### 2. Clustering non supervisionato con algoritmo Leiden (`Test3_leiden.py`)

- Identifica **sottogruppi stabili di pazienti ALS** basandosi sui biomarcatori.
- **Pipeline principale:**
  1. Selezione delle feature biologiche (MacsPlex, CSF, plasma).
  2. Pre-processing e imputazione dei dati mancanti.
  3. Clustering **Leiden** ripetuto con parametri diversi.
  4. Costruzione della **matrice di co-associazione** e clustering gerarchico.
  5. Identificazione dei cluster **stabili** e analisi statistica delle differenze.
  6. Salvataggio dei risultati in Excel e grafici in `results/`.

- **Cartelle principali**:
  - `results/test1/img/` → Grafici dendrogrammi e heatmap.

---

### 3. Classificazione XGBoost per ALS (`run_xgboost.py`)

- Implementa una pipeline di **classificazione binaria** per pazienti ALS.
- **Target predittivi:**
  - `delta ALSFRS` sopra/sotto mediana.
  - Sito di esordio bulbare vs spinale.
- **Pipeline principale:**
  1. Pre-processing e scaling dei dati.
  2. Grid Search per iperparametri XGBoost.
  3. Salvataggio feature importance e permutation importance.
  4. Interpretazione dei risultati con **SHAP**.
  5. Calcolo AUC-ROC sul test set.

- **Cartelle principali**:
  - `results/test4/shap_values/` → Valori e grafici SHAP.
  - `results/test4/img/` → ROC curve.
  - CSV con feature importance.

---

### 4. Clustering su serie temporali di pazienti ALS (`test6.py`)

- Analizza serie temporali di biomarcatori per identificare cluster di pazienti con **andamento simile nel tempo**.
- **Pipeline principale:**
  1. Pre-processing, pulizia e imputazione dei dati.
  2. Costruzione di **serie temporali multivariate**.
  3. Clustering preliminare con **TimeSeriesKMeans + Soft-DTW**.
  4. Selezione del numero ottimale di cluster tramite silhouette score.
  5. Clustering finale e assegnazione dei cluster ai pazienti.
  6. Analisi statistica delle differenze tra cluster.
  7. Visualizzazioni t-SNE 2D/3D e grafici inertia/silhouette.
  8. Salvataggio dei risultati in CSV e grafici in `results/test6/`.

---

## 📂 Struttura generale dei file

```text
readme/
  test3.py
  Test3_leiden.py
  run_xgboost.py
  test6.py
  get_data.py           # Funzioni comuni per caricare dataset e colonne
  utilities.py          # Funzioni di supporto (preprocessing, plotting)
  functions_clustering.py # Funzioni specifiche per clustering e grafici
results/                # Risultati di tutti gli script
```

## ⚙️ Come eseguire gli script

Assicurati di avere **Python 3.9+** installato.

Installa le librerie richieste:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy tslearn xgboost shap
```

Apri il terminale nella cartella del progetto ed esegui lo script desiderato:

```bash
python test3.py
python Test3_leiden.py
python run_xgboost.py
python test6.py
```
I risultati saranno salvati nelle rispettive cartelle results/.

## 🔹 Note aggiuntive

- Ogni script ha **parametri modificabili** come il numero di cluster, le feature da usare o il tipo di target.
- I file di supporto `get_data.py`, `utilities.py` e `functions_clustering.py` contengono funzioni riutilizzabili per caricamento dati, preprocessing e visualizzazioni.
- Le pipeline possono essere adattate ad altri dataset con struttura simile.