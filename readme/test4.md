# XGBoost Classification Pipeline per ALS

Questo repository contiene uno script Python che implementa una pipeline di **classificazione** per pazienti con ALS, utilizzando **dati clinici, biomarcatori plasmatici e liquorali**.  

Il modello predice due possibili target binari:

- `delta ALSFRS` sopra/sotto mediana  
- `sito di esordio` bulbare vs spinale  

---

## 📋 Pipeline

La pipeline aggiornata include:

1. **Pre-processing e scaling** delle feature.  
2. Creazione della **colonna target binaria**.  
3. **Grid Search** su XGBoost per ottimizzare iperparametri.  
4. Salvataggio di:
   - **Feature importance** XGBoost
   - **Permutation importance**
5. Interpretazione con **SHAP**:
   - Valori SHAP per ogni paziente
   - Grafici `bar` e `beeswarm`  
6. Calcolo **AUC-ROC** sul test set.

---

## 📂 Struttura cartelle

- `results/test4/` → Cartella principale dei risultati
  - `shap_values/` → Valori SHAP e grafici
  - `img/` → Curva ROC
  - `feature_importance.csv` → Importanza features
  - `permutation_importance.csv` → Permutation importance

---

## ⚙️ Come lanciare lo script

```bash
python run_xgboost.py
```

### Parametri principali modificabili

| Parametro       | Descrizione                                                      | Default |
|-----------------|-----------------------------------------------------------------|--------|
| `TARGET_ALSFRS` | Se `True` predice delta ALSFRS, se `False` predice sito di esordio | `True` |

---

### Interpretazione dei risultati

- **Feature importance**: indicano quali variabili influenzano maggiormente la predizione.  
- **Permutation importance**: conferma la robustezza delle feature principali.  
- **SHAP values**: spiegano l'impatto di ciascuna feature per ogni paziente.  
- **AUC-ROC**: misura della performance complessiva del modello sul test set.  

---

### 🔧 Note tecniche

Lo script richiede:

```text
xgboost
scikit-learn
matplotlib
shap
pandas
```

Se non presenti installarle con 

```bash
pip install xgboost scikit-learn matplotlib shap pandas
```