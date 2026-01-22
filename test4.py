#problema di classificazione per vedere se con biomarcatori + variabili cliniche
#si riesce a predirre delta alsfrs sopra / sotto mediana o esordio bulbare / spinale
#https://mljar.com/blog/feature-importance-xgboost/

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import shap

from get_data import *
from sklearn.inspection import permutation_importance

# 'delta_ALSFRS_onset'
# 'death/tracheo'
TARGET_COLUMN = 'delta_ALSFRS_onset'

clinical_columns = get_clinical_data()
plasma_columns = get_plasma_columns()
liquor_columns = get_liquor_columns()
cluster_features = liquor_columns + clinical_columns

df = get_sheet(sheet_name='clinic_EP')
df = df[df['time_point'] == 'T0']
# df = df[df['Sara'] == 'x']


#per debug
colonne_ok = ['COD', 'DRIVEALS', 'MYSTICALS', 'pt_code', 'Sara', 'PERMEALS', 'EvTestinALS']

# filtra solo le colonne non numeriche e non nella lista "ok"
colonne_problematiche = df.columns[
    (~df.dtypes.isin(['float64', 'float32', 'int64', 'int32'])) &
    (~df.columns.isin(colonne_ok)) &
(df.columns.isin(cluster_features))
]


df = add_median_split_column(df,TARGET_COLUMN )
TARGET_COLUMN = f'{TARGET_COLUMN}_median'

X_scaled, df, used_features = preprocess_data(df, cluster_features)

# -----------------------
# 2. XGBoost + Grid Search + Validation split
# -----------------------

def get_scale_pos_weight(y):
    # XGBoost (negativi / positivi)
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)
    return n_neg / n_pos if n_pos > 0 else 1

def save_feature_importance(model, feature_names, output_path='test4/feature_importance.csv'):
    importances = model.feature_importances_
    df_feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    df_feat_imp.to_csv(output_path, index=False, sep=';')
    print(f"Feature importance salvata in: {output_path}")
    return df_feat_imp


def save_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    output_path='test4/permutation_importance.csv',
    output_img='test4/permutation_importance.png',
    n_repeats=30,
    random_state=42,
    scoring='roc_auc',
    top_n=20
):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    df_perm_imp = (
        pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        .sort_values(by='importance_mean', ascending=False)
    )

    # --- Salva CSV
    df_perm_imp.to_csv(output_path, index=False, sep=';')
    print(f"📄 Permutation importance salvata in: {output_path}")

    # --- Plot (top N feature)
    df_plot = df_perm_imp.head(top_n)

    plt.figure(figsize=(8, max(4, 0.4 * len(df_plot))))
    plt.barh(
        df_plot['feature'],
        df_plot['importance_mean'],
        xerr=df_plot['importance_std']
    )
    plt.xlabel('Decrease in ROC AUC')
    plt.title('Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"🖼️  Immagine salvata in: {output_img}")

    return df_perm_imp

def plot_roc_auc(y_true, y_probs, output_path="img/test4/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve salvata in: {output_path}")
    return auc_score

def run_xgboost_with_grid_search(X, y, cluster_features, test_size=0.2, random_state=42):

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Modello base
    model = XGBClassifier( eval_metric='logloss', scale_pos_weight=get_scale_pos_weight(y_train))

    # Griglia di iperparametri
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'max_depth': [6, 7, 8],
        'n_estimators': [40, 50, 60, 100, 150, 200],
        'subsample': [0.7, 0.8, 0.9]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Migliori parametri: {grid_search.best_params_}")
    print(f"Accuracy (train): {grid_search.best_score_:.4f}")

    # Valutazione su test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    save_feature_importance(best_model, cluster_features)
    save_permutation_importance(best_model, X_test, y_test, cluster_features)

    # SHAP Importance
    explainer = shap.TreeExplainer(best_model)

    # Per XGBClassifier binario, usare .shap_values() restituisce array 2D
    shap_values = explainer.shap_values(X_scaled)

    # --- 1. Bar plot (importanza media)
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=cluster_features,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("img/test4/shap_summary_bar.png", dpi=300)
    plt.close()
    print("🖼️ SHAP bar plot salvato in img/test4/shap_summary_bar.png")

    # --- 2. Beeswarm plot (puntini rosso/blu)
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=cluster_features,
        show=False
    )
    plt.tight_layout()
    plt.savefig("img/test4/shap_summary_beeswarm.png", dpi=300)
    plt.close()
    print("🖼️ SHAP beeswarm plot salvato in img/test4/shap_summary_beeswarm.png")

    # --- 3. Force plot del primo campione
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_scaled[0, :],
        feature_names=cluster_features,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig("img/test4/shap_force_sample0.png", dpi=300)
    plt.close()
    print("🖼️ SHAP force plot salvato in img/test4/shap_force_sample0.png")


    # Calcola probabilità e AUC
    y_probs = best_model.predict_proba(X_test)[:, 1]
    auc = plot_roc_auc(y_test, y_probs)
    print(f"AUC-ROC (test): {auc:.4f}")

    return best_model

model = run_xgboost_with_grid_search(X_scaled,  df[TARGET_COLUMN], cluster_features)

