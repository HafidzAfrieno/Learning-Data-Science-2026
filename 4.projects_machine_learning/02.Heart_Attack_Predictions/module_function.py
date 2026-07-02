import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

# metrix Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_curve,auc

# metrix Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold,RandomizedSearchCV,GridSearchCV

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def evalute_models_classification(model_dict,y_test,averages=None):
    score_list = []
    for model_name,y_pred in model_dict.items():
        print(f"Menguji model: {model_name}")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=averages)
        recall = recall_score(y_test, y_pred, average = averages)
        f1 = f1_score(y_test, y_pred, average=averages)

        model_score = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }
        print(f"=== CLASSIFICATION REPORT: {model_name} ===")
        print(classification_report(y_test, y_pred))
        print("\n" + "="*40 + "\n") # Pembatas antar model

        score_list.append(model_score)
    df_final = pd.DataFrame(score_list)
    print("\n" + "="*40 + "\nProses Training Selesai!")
    return df_final

def evaluate_models_Regression(model_dict, y_test):
    score_list = []
    for model_name, y_pred in model_dict.items():
        print(f"Menguji model: {model_name}")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # RMSE adalah akar dari MSE

        model_score = {
            'Model': model_name,
            'R2-Score': round(r2, 4),
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4)
        }
        score_list.append(model_score)
    df_final = pd.DataFrame(score_list)
    print("\n" + "="*40 + "\nProses Training Selesai!")
    return df_final

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def plot_confusion_matrix(model_dict,y_test,labels=[...]):
    num_models = len(model_dict)
    ncols = 3
    nrows = int(np.ceil(num_models / ncols))

    fig, axes = plt.subplots( nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()  
    for i, (model_name, y_pred) in enumerate(model_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[i])
        axes[i].set_title(f"Confusion Matrix — {model_name}")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
    plt.tight_layout()
    plt.show()

def plot_roc_curves(model_dict,X_test,y_test):
    plt.figure(figsize=(8,6))
    for name,model in model_dict.items():
        if hasattr(model,'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr,tpr,_=roc_curve(y_test,y_prob)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def cross_validate_model(models_dict, X, y, cv_folds=5, mode='classification'):
    cv_result = []
    if mode == 'classification':
        cv_strategy = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
        scoring_metrics = {
            'Accuracy': 'accuracy',
            'Precision': 'precision_macro',
            'Recall': 'recall_macro',
            'F1-Score': 'f1_macro'
        }
        print(f"=== MENJALANKAN STRATIFIED {cv_folds}-FOLD CROSS VALIDATION (KLASIFIKASI) ===\n")
    elif mode == 'regression':
        cv_strategy = KFold(n_splits=cv_folds,shuffle=True,random_state=42)
        scoring_metrics = {
            'R2-Score': 'r2',
            'MAE': 'neg_mean_absolute_error',
            'MSE': 'neg_mean_squared_error'
        }
        print(f"=== MENJALANKAN {cv_folds}-FOLD CROSS VALIDATION (REGRESI) ===\n")
    else:
        raise ValueError("Mode harus diisi 'classification' atau 'regression'!")
    
    for model_name,model in models_dict.items():
        print(f"Menguji model: {model_name}")
        scores = cross_validate(model,X,y,cv=cv_strategy,scoring=scoring_metrics,return_train_score=False,n_jobs=-1)
        model_summary = {'Model': model_name}

        for metric_name in scoring_metrics.keys():
            # Mengambil hasil score kunci 'test_NamaMetrik'
            raw_scores = scores[f'test_{metric_name}']
            # Khusus MAE dan MSE di regresi, scikit-learn mengembalikan nilai negatif. Kita positifkan lagi.
            if mode == 'regression' and metric_name in ['MAE', 'MSE']:
                mean_score = -np.mean(raw_scores)
            else:
                mean_score = np.mean(raw_scores)
            model_summary[f'{metric_name}'] = mean_score
            
        # Tambahan metrik RMSE (akar dari MSE) jika dalam mode regresi
        if mode == 'regression':
            model_summary['RMSE'] = np.sqrt(model_summary['MSE'])
            
        cv_result.append(model_summary)
        
    print("\n" + "="*40 + "\nProses CV Selesai!")
    return pd.DataFrame(cv_result)

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#  

def Hyperparameter_Tuning(method,pipeline_model,cv,type_model,param,x_train,x_test,y_train,y_test):
    if type_model == 'classification':
        scoring_metric = 'f1'
    elif type_model == 'regression':
        scoring_metric = 'r2'
    else:
        raise ValueError("type_model harus berupa 'classification' atau 'regression'")
    
    if method == 'randomcv':
        tuned_model = RandomizedSearchCV(estimator=pipeline_model,param_distributions=param,cv=cv,scoring=scoring_metric,
                                         n_iter=10,random_state=42,n_jobs=-1)
    elif method == 'gridcv':
        tuned_model = GridSearchCV(estimator=pipeline_model,param_grid=param,cv=cv,scoring=scoring_metric,n_jobs=-1)
    else:
        raise ValueError("method harus berupa 'randomcv' atau 'gridcv'")

    tuned_model.fit(x_train,y_train)
    best = tuned_model.best_estimator_

    y_pred = best.predict(x_test)
    best_cv_score = tuned_model.best_score_
    model_name = type(best.steps[-1][1]).__name__

    if type_model == 'regression':
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        residuals = (np.array(y_test) - np.array(y_pred)).flatten()

        axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")
        
        axes[1].hist(residuals, bins=45, color="seagreen", edgecolor="black")
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution")
        
        plt.suptitle(f'{model_name}\n(Best CV R²: {best_cv_score:.4f})', fontsize=14, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.show()
        
    elif type_model == 'classification':
        fig, ax = plt.subplots(figsize=(6, 5))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        
        plt.title(f'{model_name}\n(Best CV F1-Score: {best_cv_score:.4f})', fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
    return best

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def plot_tuned_feature_importance(tuned_model):
    model_step = tuned_model.steps[-1][1]
    model_name = type(model_step).__name__

    try:
        feature_names = tuned_model[:-1].get_feature_names_out()
    except Exception:
        raise AttributeError("Gagal mengambil nama fitur. Pastikan pipeline Anda mendukung get_feature_names_out().")

    if hasattr(model_step,'feature_importances_'):
        nilai_kontribusi  = model_step.feature_importances_
    elif hasattr(model_step,'coef_'):
        nilai_kontribusi = model_step.coef_

    plt.figure(figsize=(15, 6))
    importances = pd.Series(nilai_kontribusi,index=feature_names).sort_values(ascending=True).tail(15)
    importances.plot(kind='barh',color='steelblue')
    plt.suptitle(f"Top 15 Feature Importance in {model_name}-Tuned",fontsize=16,fontweight="bold",y=1.02,)
    plt.tight_layout()
    plt.show()

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================# 

def feature_importance(result_crossValidate,model_dict):
    kolom_skor = [col for col in result_crossValidate.columns if 'r2-score' in col.lower() or 'accuracy' in col.lower()]
    metrix_score_colom = kolom_skor[0]
    df_top2 = result_crossValidate.sort_values(by=metrix_score_colom,ascending=False).head(2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (_, df_score) in zip(axes, df_top2.iterrows()):
        name = df_score['Model']
        skor = df_score[metrix_score_colom]
        
        pipe = model_dict[name]
        model = pipe.named_steps['model']
        feature_names = pipe[:-1].get_feature_names_out()
        
        if hasattr(model,'feature_importances_'):
            nilai_kontribusi  = model.feature_importances_
        elif hasattr(model,'coef_'):
            nilai_kontribusi = model.coef_
        else:
            continue
            
        importances = pd.Series(nilai_kontribusi, index=feature_names).sort_values(ascending=True).tail(15)
        importances.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'{name}\n({metrix_score_colom}: {skor:.4f})')
    plt.suptitle("Importance Feature In Models ",fontsize=16,fontweight="bold",y=1.02,)
    plt.tight_layout()
    plt.show()

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def plot_all_modelsRegression_predictions(models_dict, X_train, y_train, X_test, y_test):
    num_models = len(models_dict)
    ncols = 3
    nrows = int(np.ceil(num_models / ncols))

    fig, axes = plt.subplots( nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()  

    for idx, (model_name, pipeline) in enumerate(models_dict.items()):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        axes[idx].scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")
        
        # Hitung batas minimum dan maksimum untuk garis diagonal
        lo = min(np.min(y_test), np.min(y_pred))
        hi = max(np.max(y_test), np.max(y_pred))
        
        axes[idx].plot([lo, hi], [lo, hi], "r--", label="Perfect")
        axes[idx].set_xlabel("Actual")
        axes[idx].set_ylabel("Predicted")
        axes[idx].set_title(f"{model_name}")
        axes[idx].legend()
        axes[idx].grid(True, linestyle="--", alpha=0.6)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Perbandingan Actual vs Predicted — Semua Model",fontsize=16,fontweight="bold",y=1.02,)
    plt.tight_layout()
    plt.show()

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================#    

def plot_residuals(result_crossValidate,model_dict,x_test,y_test):
    kolom_skor = [col for col in result_crossValidate.columns if 'r2-score' in col.lower() or 'accuracy' in col.lower()]
    metrix_score_colom = kolom_skor[0]
    df_top = result_crossValidate.sort_values(by=metrix_score_colom,ascending=False).iloc[0]

    name = df_top['Model']
    skor = df_top[metrix_score_colom]
    pipe = model_dict[name]

    y_pred = pipe.predict(x_test)
    residuals = (np.array(y_test) - np.array(y_pred)).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted — {name}")
    axes[1].hist(residuals, bins=45, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual distribution")
    plt.suptitle(f'{name}\n({metrix_score_colom}: {skor:.4f})',fontsize=16,fontweight="bold",y=1.02,)
    plt.tight_layout()
    plt.show()

#===============================================================================================================================================================================#    
#===============================================================================================================================================================================# 

