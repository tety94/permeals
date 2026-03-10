# Analisi di Clustering su Biomarcatori ALS

Questo progetto esegue un'analisi di **clustering non supervisionato** sui dati di biomarcatori per identificare possibili **sottogruppi di pazienti ALS** con profili biologici simili e studiare le differenze cliniche tra i gruppi.

L'approccio utilizzato cerca **cluster stabili**, cioè gruppi di pazienti che risultano simili anche quando si varia il parametro principale dell'algoritmo di clustering.

---

## Cosa fa il codice

La pipeline completa include:

1. **Caricamento e selezione dei dati**
   - I dati vengono letti da un file Excel (`database.xlsx`).
   - Vengono selezionate le feature biologiche utilizzate per il clustering:
     - biomarcatori **MacsPlex**
     - biomarcatori **CSF (liquor)**
     - biomarcatori **plasma**
   - Le variabili cliniche vengono invece utilizzate successivamente per interpretare i cluster.

2. **Pre-processing dei dati**
   - Conversione delle variabili a formato numerico.
   - Gestione dei valori mancanti tramite **imputazione con la media della colonna**.

3. **Clustering con algoritmo Leiden**
   - Il clustering viene eseguito più volte utilizzando l'algoritmo **Leiden**.
   - Il parametro `resolution`, che controlla la granularità dei cluster, viene fatto variare su un intervallo di valori.

4. **Valutazione della stabilità del clustering**
   - Le diverse clusterizzazioni ottenute vengono confrontate usando l'**Adjusted Rand Index (ARI)**.
   - Il risultato viene visualizzato in una **heatmap**, che mostra quanto le diverse partizioni siano simili tra loro.

5. **Costruzione della matrice di co-associazione**
   - Viene costruita una matrice che misura **quanto spesso due pazienti finiscono nello stesso cluster** nelle diverse analisi.

6. **Identificazione dei cluster stabili**
   - La matrice di co-associazione viene utilizzata per eseguire un **clustering gerarchico**.
   - Il dendrogramma risultante permette di identificare i **cluster finali stabili**.

7. **Scelta del numero finale di cluster**
   - L'utente osserva il dendrogramma e inserisce manualmente il numero di cluster desiderato.

8. **Analisi delle differenze tra cluster**
   - Per ogni variabile biologica e clinica vengono analizzate le differenze tra i cluster.
   - I risultati statistici vengono salvati nei file di output.

9. **Salvataggio dei risultati**
   - Il dataset finale con i cluster assegnati viene salvato in Excel.
   - Viene salvata anche una versione ordinata per cluster.
   - I grafici prodotti (heatmap e dendrogrammi) vengono salvati nella cartella `results/`.

---

## Struttura dei file

- `Test3_leiden.py` → Script principale con la pipeline di clustering.
- `get_data.py` → Funzioni per selezionare le colonne dei biomarcatori e dei dati clinici.
- `utilities.py` → Funzioni di supporto.
- `functions_clustering.py` → Funzioni per eseguire il clustering e generare i grafici.
- `results/` → Cartella dove vengono salvati i risultati.
  - `results/test1/img/` → Grafici generati durante l'analisi.

---

## Come eseguire

1. Assicurati di avere **Python 3.9+** installato.
2. Installa le librerie richieste:

```bash
pip install numpy pandas scikit-learn scipy seaborn matplotlib
```

Apri il terminale nella cartella del progetto ed esegui:
```bash
python test1.py
```

Durante l'esecuzione verrà mostrato il **dendrogramma dei cluster** e verrà richiesto di inserire il numero di cluster finali.

```text
Insert the number of clusters identified from the dendrogram
```
## Output

I risultati vengono salvati nella cartella:

```text
results/test1/
```
