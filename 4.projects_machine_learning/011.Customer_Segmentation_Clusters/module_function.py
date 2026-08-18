import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score)

def preprocessing_score(func):
    @wraps(func)
    def excuted(self,*args, **kwargs):
        df = func(self, *args, **kwargs)
        scaler = MinMaxScaler()
        norm_sil = scaler.fit_transform(df[['Silhouette Score']])
        norm_ch = scaler.fit_transform(df[['Calinski-Harabasz Score']])
        norm_db = 1 - scaler.fit_transform(df[['Davies-Bouldin Score']])
        df['Composite_Score'] = (norm_sil + norm_ch + norm_db) / 3
        return df
    return excuted

class ModelKmeansPlus:
    def __init__(self,rangeMax:int,df_raw:pd.DataFrame):
        self.sil_scores = []
        self.ch_scores = []
        self.db_scores = []
        self.inertias = []
        self.df = df_raw
        start_range = 2 if rangeMax > 2 else 2
        self.k_range = range(start_range, rangeMax)

    def fit_model(self,X_cleaned):
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42).fit(X_cleaned)
            model_kmeans = kmeans.labels_
            self.sil_scores.append(silhouette_score(X_cleaned, model_kmeans))
            self.ch_scores.append(calinski_harabasz_score(X_cleaned, model_kmeans))
            self.db_scores.append(davies_bouldin_score(X_cleaned, model_kmeans))
            self.inertias.append(kmeans.inertia_)
        print("Model Berhasil Di Training")

    @preprocessing_score
    def score_df(self):
        df_scores = pd.DataFrame({
        'n_clusters(k)': list(self.k_range),
        'Inertia (WCSS)': self.inertias,
        'Silhouette Score':self.sil_scores,
        'Calinski-Harabasz Score': self.ch_scores,
        'Davies-Bouldin Score':self.db_scores})
        return df_scores

    def fit_bestModel(self,X_cleaned):
        df_scored = self.score_df()
        best_k_idx = df_scored['Composite_Score'].idxmax()
        best_k = int(df_scored.loc[best_k_idx, 'n_clusters(k)'])
        best_kmeans = KMeans(n_clusters=best_k,init='k-means++', random_state=42)
        self.df['Cluster'] = best_kmeans.fit_predict(X_cleaned)
        return best_kmeans,self.df

    def convert_toPCA(self, X_cleaned, best_kmeans):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_cleaned)
        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_pca['Cluster'] = self.df['Cluster'].values
        centroids_pca = pca.transform(best_kmeans.cluster_centers_)
        return df_pca, centroids_pca, pca
    
class ModelVizulazations(ModelKmeansPlus):
    def __init__(self,rangeMax=10, df_raw=None):
        super().__init__(rangeMax=rangeMax, df_raw=df_raw)

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

    def Analysis_Clusstering(self, X_cleaned, best_kmeans):
        df_pca, centroids_pca, pca = self.convert_toPCA(X_cleaned, best_kmeans)

        sns.set_theme(style="whitegrid")
        palette = 'Set2'
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(ax=axes[0], x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette=palette, s=90, alpha=0.8, style='Cluster')
        axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, c='red', marker='X', edgecolor='black', linewidth=1.5, label='Centroids') # Plot Centroid
        axes[0].set_title('Visualisasi Klaster (PCA 2D Projection)', fontsize=13, fontweight='bold', pad=12)
        axes[0].set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
        axes[0].set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
        axes[0].legend(title='Cluster', loc='upper right')

        sns.scatterplot(ax=axes[1], x='Annual Income ($)', y='Spending Score (1-100)', hue='Cluster', data=self.df, palette=palette, s=90, style='Cluster')
        axes[1].set_title('Segmentasi Pelanggan: Income vs Spending', fontsize=13, fontweight='bold', pad=12)
        axes[1].set_xlabel('Annual Income ($)')
        axes[1].set_ylabel('Spending Score (1-100)')
        axes[1].legend(title='Cluster', loc='upper right')
        plt.suptitle('Analisis Visualisasi Klastering Pelanggan', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()