import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score)

class ModelKmeansPlus:
    def __init__(self,rangeMax:int):
        self.sil_scores = []
        self.ch_scores = []
        self.db_scores = []
        self.inertias = []
        self.k_range = range(2, rangeMax)

    def fit_model(self,X_cleaned):
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42).fit(X_cleaned)
            model_kmeans = kmeans.labels_
            self.sil_scores.append(silhouette_score(X_cleaned, model_kmeans))
            self.ch_scores.append(calinski_harabasz_score(X_cleaned, model_kmeans))
            self.db_scores.append(davies_bouldin_score(X_cleaned, model_kmeans))
            self.inertias.append(kmeans.inertia_)

    def score_df(self):
        df_scores = pd.DataFrame({
        'n_clusters(k)': list(self.k_range),
        'Inertia (WCSS)': self.inertias,
        'Silhouette Score':self.sil_scores,
        'Calinski-Harabasz Score': self.ch_scores,
        'Davies-Bouldin Score':self.db_scores
        })
        return df_scores

    def select_bestScore(self):
        print('halo')


class ModelVizulazations(ModelKmeansPlus):
    def __init__(self):
        super().__init__()

    def Analysis_AllMetriks(self):
        df_scores = self.score_df()
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
       
        sns.lineplot(ax=axes[0, 0], x='n_clusters(k)', y='Inertia (WCSS)', data=df_scores, marker='o', markersize=8, linewidth=2, color='#7b1fa2')
        axes[0, 0].set_title('Inertia / WCSS (Elbow Method)\n(Mencari Titik Belokan Siku)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Jumlah Cluster (k)')
        axes[0, 0].set_xticks(list(self.k_range))
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
        best_k_sil = int(df_scores.loc[df_scores['Silhouette Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[0, 1], x='n_clusters(k)', y='Silhouette Score', data=df_scores, marker='o', markersize=8, linewidth=2, color='#2b5c8f')
        axes[0, 1].axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best k = {best_k_sil}')
        axes[0, 1].set_title('Silhouette Score\n(Mendekati +1 = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Jumlah Cluster (k)')
        axes[0, 1].set_xticks(list(self.k_range))
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend()
        
        best_k_ch = int(df_scores.loc[df_scores['Calinski-Harabasz Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[1, 0], x='n_clusters(k)', y='Calinski-Harabasz Score', data=df_scores, marker='s', markersize=8, linewidth=2, color='#2e7d32')
        axes[1, 0].axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'Puncak k = {best_k_ch}')
        axes[1, 0].set_title('Calinski-Harabasz Score\n(Semakin Tinggi = Terbaik)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Jumlah Cluster (k)')
        axes[1, 0].set_xticks(list(self.k_range))
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend()
        
        best_k_db = int(df_scores.loc[df_scores['Davies-Bouldin Score'].idxmin(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[1, 1], x='n_clusters(k)', y='Davies-Bouldin Score', data=df_scores, marker='^', markersize=8, linewidth=2, color='#c62828')
        axes[1, 1].axvline(x=best_k_db, color='green', linestyle='--', alpha=0.7, label=f'Terendah k = {best_k_db}')
        axes[1, 1].set_title('Davies-Bouldin Score\n(Semakin Rendah = Terbaik)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Jumlah Cluster (k)')
        axes[1, 1].set_xticks(list(self.k_range))
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)
        axes[1, 1].legend()
        plt.suptitle('Evaluasi Lengkap Metrik Klastering K-Means Berdasarkan Jumlah k', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()



    

