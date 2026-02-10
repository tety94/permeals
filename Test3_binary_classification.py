"""
The script performes binary classification 
"""

############################################################################################################

#!/usr/bin/env python3

#############################################################################################################

import pandas as pd                                # it permits data manipulation and analysis
import numpy as np                                 # it is a package for scientific computing in Python
from sklearn import preprocessing                  # used for standardization
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
import seaborn as sns                              # it contains tools for statistical data visualization
import matplotlib.pyplot as plt                    # it is a library for creating plots

from xgboost import XGBClassifier

import smote_variants as sv

import time
import shap

import lime
import lime.lime_tabular

################################################################################################################

if __name__ == '__main__':  
    
    columns = ...  # insert columns to be considered
    
    dirpath_img = "..."

    dirpath = "..."
    
    file = "database"
    sheet_name = 0


    # Upload dataset 
    dataset = pd.read_excel(dirpath + file+'.xlsx', sheet_name=sheet_name)

    # Delete Nan rows
    dataset = dataset[dataset.isna().sum(axis=1) < len(dataset.columns)].copy()  

    # 0-class
    dataset0 = dataset[dataset['Unnamed: 0'] == '0-class'].copy()
    dataset0['patient'] = 0

    # 1-class
    dataset1 = dataset[dataset['Unnamed: 0'] == '1-class'].copy()
    dataset1['patient'] = 1

    data_01 = pd.concat([dataset0, dataset1], ignore_index=True)

    tmp_columns = ['patient', 'Unnamed: 2'] + columns
    data = data_01[tmp_columns].copy()
    
    X = data.drop(['patient', 'Unnamed: 2'],axis=1)
    y = data.patient
    
    for ii in columns:
        X[ii] = pd.to_numeric(X[ii], errors = "coerce")

	########################################################################################################
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Impute missing value in the training set with the mean of each column
    X_train = X_train.fillna(X_train.mean()).astype(float).copy()   
    
    # Scaler the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
      
    ######################################################################################################
    # 1. FEATURE SELECTION
    
    selected_features = columns
    
    X_train_sel = X_train_scaled[selected_features].copy()
    X_test_sel = X_test_scaled[selected_features].copy()
    
    ######################################################################################################
    
    # 2. TRAIN THE MODEL
    
    # oversampling to balanced the classes    
    oversampler = sv.MWMOTE(proportion=1., random_state=42)
    X_train_over, y_train_over = oversampler.sample(np.array(X_train_sel), np.array(y_train))
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)    

    t = time.time()
    param_grid = {'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 5, 6],
            'n_estimators': [ 50, 60, 100],
            'subsample': [0.7, 0.8, 0.9],
			'scale_pos_weight' : [ ratio, 1] }

    cv_inner = StratifiedKFold( n_splits=3, shuffle=True, random_state=42)
    classifier = XGBClassifier(eval_metric='auc', random_state=42)

    # fit the model
    model = GridSearchCV( estimator = classifier, param_grid = param_grid, cv=cv_inner, scoring='roc_auc' )
    model_fit = model.fit(X_train_over, y_train_over)
    best_params = model_fit.best_params_
    print ('Best parameters =', best_params)
	
    final_model = XGBClassifier(**best_params, eval_metric='auc', random_state=42)
    final_model.fit(X_train_over, y_train_over)
        
    ######################################################################################################
    
    # 3. EVALUATE THE MODEL
    
    y_test_pred = final_model.predict(X_test_sel)
    y_train_pred = final_model.predict(X_train_over)

    y_test_pred_proba = final_model.predict_proba(X_test_sel)
    y_train_pred_proba = final_model.predict_proba(X_train_over)

    b_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)

    elapsed = time.time() - t

    print(f"Balanced accuracy: {b_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Elapsed time : {elapsed:.4f}")
        
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print('TRAIN AUC :',round((roc_auc_score(y_train_over, y_train_pred_proba[:,1]))*100,2), '%')
    print('TEST AUC:',round((roc_auc_score(y_test, y_test_pred_proba[:,1]))*100,2), '%')
    
    ######################################################################################################
    
    # 4. INTERPRETABILITY OF THE MODEL
    
    features_importances = final_model.feature_importances_

    feat_importances = pd.Series(features_importances, index=selected_features)

    result_XG = permutation_importance(final_model, X_test_sel, y_test, n_repeats=10, random_state=42, n_jobs=2)
    permutation_importances = pd.Series(result_XG.importances_mean, index=selected_features)
    
    print('\nFeature importance:')
    print(feat_importances)
    
    print('\nPermutation importance:')
    print(permutation_importances)
    
    fig5 = plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    ax = feat_importances.plot.bar()
    plt.ylabel('feature importance')
    plt.title("Feature importance in XGBoost")

    plt.subplot(1, 2, 2)
    ax = permutation_importances.plot.bar()
    plt.ylabel('permutation importance')
    plt.title("Permutation importance in Xgboost")
    fig5.savefig(dirpath_img +"model_importances.png")
    plt.show()   
    

    explainer_shap = shap.TreeExplainer(final_model)
    shap_values = explainer_shap.shap_values(X_test_sel) 

    fig6 = shap.summary_plot(shap_values, X_test_sel, feature_names=selected_features, max_display=10,show=False)
    plt.savefig(dirpath_img +"shap.png")
    plt.show()
    
    print('y_pred = ', y_test_pred)
    print('y_true = ', np.array(y_test))
    
    begin_loop = True
    
    while begin_loop:
        index = input('Insert the index of the instance you want to analyze with LIME (press Enter to exit) = ')
        
        if index == '':
            break
        else:
            index = int(index)

        aa = columns[0];   bb = columns[1]
        index_ii = data.index[(data[aa]==np.array(X_test[aa])[index]) & (data[bb]==np.array(X_test[bb])[index])].tolist()
        print('pz = ', data['Unnamed: 2'][index_ii])
            
        print('true class = ', np.array(y_test)[index])
        print('predict class = ', y_test_pred[index])
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train_over, feature_names=selected_features,discretize_continuous=True)            
        exp = explainer_lime.explain_instance(np.array(X_test_sel)[index], final_model.predict_proba, num_features=10)
        exp.show_in_notebook(show_table=True, show_all=False)

        fig = exp.as_pyplot_figure()
        plt.show()
            
        
        