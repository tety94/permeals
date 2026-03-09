"""
Questo script implementa un problema di classificazione usando XGBoost
per predire:
- delta ALSFRS sopra/sotto mediana
- esordio bulbare / spinale
utilizzando biomarcatori, dati clinici e liquor.

Pipeline aggiornata:
1. Pre-processing e scaling delle feature
2. Creazione target binario (delta ALSFRS sopra/sotto mediana)
3. Grid Search su XGBoost
4. Salvataggio feature importance e permutation importance
5. Interpretazione con SHAP
6. Salvataggio SHAP values per tutti i pazienti
8. Calcolo AUC-ROC su test set
"""

# -----------------------
# Import librerie necessarie
# -----------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap
import os

from get_data import *

# -----------------------
# Definizione costanti
# -----------------------

TARGET_ALSFRS = True

TARGET_COLUMN = 'delta_ALSFRS_onset'  # variabile target originale
if not TARGET_ALSFRS:
    TARGET_COLUMN = 'site_of_onset_BS'  # variabile target originale

RESULT_FOLDER = 'results/test4/'       # cartella dove salvare risultati e immagini
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -----------------------
# Recupero liste di colonne per tipo
# -----------------------
clinical_columns = get_clinical_data()   # dati clinici
plasma_columns = get_plasma_columns()    # biomarcatori plasmatici
liquor_columns = get_liquor_columns()    # biomarcatori liquorali
classification_features = liquor_columns + clinical_columns  # feature da usare per classificazione

# -----------------------
# Caricamento dati e filtri
# -----------------------
df = get_sheet(sheet_name='clinic_EP')   # recupero sheet principale
df = df[df['time_point'] == 'T0']        # consideriamo solo primo time point

df = df[df['pt_code'] != 'NEMO TN 24']

# -----------------------
# Creazione colonna target binaria
# -----------------------
df = add_median_split_column(df, TARGET_COLUMN)
TARGET_COLUMN = f'{TARGET_COLUMN}_median'  # 1 = sopra mediana, 0 = sotto

# -----------------------
# Preprocessing features
# -----------------------
X_scaled, df, used_features = preprocess_data(df, classification_features)

# -----------------------
# Funzioni di supporto
# -----------------------
def get_scale_pos_weight(y):
    """ Calcola rapporto negativi/positivi per gestire classi sbilanciate """
    n_pos = sum(y==1)
    n_neg = sum(y==0)
    return n_neg/n_pos if n_pos>0 else 1

def save_feature_importance(model, feature_names, output_path='feature_importance.csv'):
    """ Salva importanza feature XGBoost in CSV """
    importances = model.feature_importances_
    df_feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    df_feat_imp.to_csv(f"{RESULT_FOLDER}{output_path}", index=False, sep=';')
    print(f"📄 Feature importance salvata in: {output_path}")
    return df_feat_imp

def save_permutation_importance(model, X_test, y_test, feature_names,
                                output_path='permutation_importance.csv',
                                output_img='permutation_importance.png',
                                n_repeats=30, random_state=42, scoring='roc_auc', top_n=20):
    """ Calcola e salva permuation importance """
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=-1)
    df_perm_imp = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    df_perm_imp.to_csv(f"{RESULT_FOLDER}{output_path}", index=False, sep=';')
    print(f"📄 Permutation importance salvata in: {output_path}")

    # Plot top N
    df_plot = df_perm_imp.head(top_n)
    plt.figure(figsize=(8, max(4, 0.4*len(df_plot))))
    plt.barh(df_plot['feature'], df_plot['importance_mean'], xerr=df_plot['importance_std'])
    plt.xlabel('Decrease in ROC AUC')
    plt.title('Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULT_FOLDER}{output_img}", dpi=300)
    plt.close()
    print(f"🖼️ Immagine salvata in: {output_img}")
    return df_perm_imp

def plot_roc_auc(y_true, y_probs, output_path="img/roc_curve.png"):
    """ Disegna e salva curva ROC """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,label=f'AUC={auc_score:.2f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_FOLDER}{output_path}")
    plt.close()
    print(f"ROC curve salvata in: {output_path}")
    return auc_score

# -----------------------
# Funzione principale
# -----------------------
def run_xgboost_with_grid_search(X, y, classification_features, test_size=0.2, random_state=42):
    # Split train/test stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Modello base con random_state per riproducibilità
    model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=get_scale_pos_weight(y_train),
        random_state=random_state
    )

    # Griglia iperparametri
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'max_depth': [6, 7, 8],
        'n_estimators': [40, 50, 60, 100, 150, 200],
        'subsample': [0.7, 0.8, 0.9]
    }

    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Migliori parametri: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Valutazione su test
    y_pred = best_model.predict(X_test)
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # Salvataggio importanza features
    save_feature_importance(best_model, classification_features)
    save_permutation_importance(best_model, X_test, y_test, classification_features)

    # -----------------------
    # SHAP values — FIX compatibilità XGBoost >= 2.0
    # -----------------------
    shap_folder = os.path.join(RESULT_FOLDER, 'shap_values')
    os.makedirs(shap_folder, exist_ok=True)

    shap_folder = os.path.join(RESULT_FOLDER, 'shap_values')
    os.makedirs(shap_folder, exist_ok=True)

    # Fix base_score per compatibilità XGBoost >= 2.0 con SHAP
    best_model.base_score = 0.5
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)

    # Salvataggio valori SHAP per tutti i pazienti
    df_shap = pd.DataFrame(shap_values, columns=classification_features)
    df_shap['patient_id'] = df['pt_code'].values
    df_shap.to_csv(os.path.join(shap_folder, 'shap_values_per_patient.csv'), index=False, sep=';')
    print(f"📄 SHAP values salvati in: {shap_folder}")

    # Bar plot e beeswarm SHAP
    shap.summary_plot(shap_values, X, feature_names=classification_features, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder, 'shap_summary_bar.png'), dpi=300)
    plt.close()

    shap.summary_plot(shap_values, X, feature_names=classification_features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder, 'shap_summary_beeswarm.png'), dpi=300)
    plt.close()


    # -----------------------
    # ROC-AUC — fix: crea sottocartella img/
    # -----------------------
    img_folder = os.path.join(RESULT_FOLDER, 'img')
    os.makedirs(img_folder, exist_ok=True)

    y_probs = best_model.predict_proba(X_test)[:, 1]
    auc = plot_roc_auc(y_test, y_probs, output_path="img/roc_curve.png")
    print(f"AUC-ROC (test): {auc:.4f}")

    return best_model

# -----------------------
# Esecuzione pipeline
# -----------------------
model = run_xgboost_with_grid_search(X_scaled, df[TARGET_COLUMN], classification_features)