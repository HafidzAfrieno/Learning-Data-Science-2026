import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class GaussianMixtureClustering:
    """
    Kelas inti untuk menangani logika pemodelan, training, perhitungan metrik evaluasi,
    serta transformasi data untuk segmentasi menggunakan Gaussian Mixture.
    """
    def __init__(self, range_max: int, df_raw: pd.DataFrame):
        if range_max <= 2:
            raise ValueError("range_max harus lebih besar dari 2.")
        
        self.df = df_raw.copy()
        self.k_range = range(2, range_max)
        self.sil_scores = []
        self.ch_scores = []
        self.db_scores = []
        self.bic_scores = []
        self.df_scores = None

    def fit_model(self, X_cleaned: np.ndarray) -> None:
        """Melakukan training GMM untuk setiap nilai k dalam k_range."""
        for k in self.k_range:
            gmm = GaussianMixture(n_components=k,covariance_type='full',random_state=42).fit(X_cleaned)
            model_gmm = gmm.predict(X_cleaned)
            
            self.sil_scores.append(silhouette_score(X_cleaned, model_gmm))
            self.ch_scores.append(calinski_harabasz_score(X_cleaned, model_gmm))
            self.db_scores.append(davies_bouldin_score(X_cleaned, model_gmm))
            self.bic_scores.append(gmm.bic(X_cleaned))
        print("[INFO] Model berhasil dilatih untuk semua rentang k.")

    def compute_scores_dataframe(self) -> pd.DataFrame:
        """Membuat DataFrame metrik evaluasi dan menghitung Composite Score secara normalisasi."""
        df_scores = pd.DataFrame({
            'n_clusters(k)': list(self.k_range),
            'BIC Score': self.bic_scores,
            'Silhouette Score': self.sil_scores,
            'Calinski-Harabasz Score': self.ch_scores,
            'Davies-Bouldin Score': self.db_scores
        })

        scaler = MinMaxScaler()
        norm_sil = scaler.fit_transform(df_scores[['Silhouette Score']])
        norm_ch = scaler.fit_transform(df_scores[['Calinski-Harabasz Score']])
        norm_db = 1 - scaler.fit_transform(df_scores[['Davies-Bouldin Score']]) 
        
        df_scores['Composite_Score'] = (norm_sil + norm_ch + norm_db) / 3
        self.df_scores = df_scores
        return df_scores

    def fit_best_model(self, X_cleaned: np.ndarray):
        """Memilih k terbaik berdasarkan Composite Score tertinggi dan melatih ulang model."""
        if self.df_scores is None:
            self.compute_scores_dataframe()

        best_idx = self.df_scores['Composite_Score'].idxmax()
        best_k = int(self.df_scores.loc[best_idx, 'n_clusters(k)'])
        
        best_gmm = GaussianMixture(n_components=best_k,covariance_type='full',random_state=42)
        self.df['Cluster'] = best_gmm.fit_predict(X_cleaned)
        
        print(f"[INFO] Model optimal dipilih dengan jumlah cluster (k) = {best_k}")
        return best_gmm, self.df

    def convert_to_pca(self, X_cleaned: np.ndarray, best_gmm: GaussianMixture):
            """Mereduksi dimensi data menggunakan PCA untuk keperluan visualisasi 2D."""
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_cleaned)
            
            df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
            df_pca['Cluster'] = self.df['Cluster'].values
            centroids_pca = pca.transform(best_gmm.means_)
            
            return df_pca, centroids_pca, pca

class ClusteringVisualizer:
    """
    Kelas terpisah khusus untuk menangani seluruh visualisasi/plotting data.
    Menerapkan prinsip Single Responsibility Principle (SRP).
    """
    def __init__(self, model_instance: GaussianMixtureClustering):
        self.model = model_instance

    def plot_evaluation_metrics(self) -> None:
        """Membuat plot evaluasi lengkap (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin)."""
        df_scores = self.model.df_scores
        if df_scores is None:
            df_scores = self.model.compute_scores_dataframe()

        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        k_list = list(self.model.k_range)

        # --- Plot 1: BIC Score (Semakin Rendah = Terbaik) ---
        best_k_bic = int(df_scores.loc[df_scores['BIC Score'].idxmin(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[0, 0], x='n_clusters(k)', y='BIC Score', data=df_scores, marker='o', markersize=8, linewidth=2, color='#7b1fa2')
        axes[0, 0].axvline(x=best_k_bic, color='green', linestyle='--', alpha=0.7, label=f'Terendah k = {best_k_bic}')
        axes[0, 0].set_title('BIC Score (Bayesian Information Criterion)\n(Semakin Rendah = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Jumlah Cluster (k)')
        axes[0, 0].set_xticks(list(k_list))
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        axes[0, 0].legend()

        # 2. Silhouette Score
        best_k_sil = int(df_scores.loc[df_scores['Silhouette Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[0, 1], x='n_clusters(k)', y='Silhouette Score', data=df_scores, marker='o', color='#2b5c8f', linewidth=2)
        axes[0, 1].axvline(x=best_k_sil, color='red', linestyle='--', label=f'Best k = {best_k_sil}')
        axes[0, 1].set_title('Silhouette Score (Max = Best)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xticks(k_list)
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend()

        # 3. Calinski-Harabasz Score
        best_k_ch = int(df_scores.loc[df_scores['Calinski-Harabasz Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[1, 0], x='n_clusters(k)', y='Calinski-Harabasz Score', data=df_scores, marker='s', color='#2e7d32', linewidth=2)
        axes[1, 0].axvline(x=best_k_ch, color='red', linestyle='--', label=f'Best k = {best_k_ch}')
        axes[1, 0].set_title('Calinski-Harabasz Score (Max = Best)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xticks(k_list)
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend()

        # 4. Davies-Bouldin Score
        best_k_db = int(df_scores.loc[df_scores['Davies-Bouldin Score'].idxmin(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[1, 1], x='n_clusters(k)', y='Davies-Bouldin Score', data=df_scores, marker='^', color='#c62828', linewidth=2)
        axes[1, 1].axvline(x=best_k_db, color='green', linestyle='--', label=f'Best k = {best_k_db}')
        axes[1, 1].set_title('Davies-Bouldin Score (Min = Best)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xticks(k_list)
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)
        axes[1, 1].legend()

        plt.suptitle('Evaluasi Metrik Klastering K-Means', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_cluster_results(self, X_cleaned: np.ndarray, best_gmm: GaussianMixture, feature_x: str, feature_y: str) -> None:
        """Visualisasi hasil klaster dalam bentuk PCA 2D dan scatter plot fitur asli."""
        df_pca, centroids_pca, pca = self.model.convert_to_pca(X_cleaned, best_gmm)

        sns.set_theme(style="whitegrid")
        palette = 'Set2'
        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot PCA 2D
        sns.scatterplot(ax=axes[0], x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette=palette, s=90, alpha=0.8, style='Cluster')
        axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, c='red', marker='X', edgecolor='black', linewidth=1.5, label='Centroids')
        axes[0].set_title('Visualisasi Klaster (PCA 2D)', fontsize=13, fontweight='bold', pad=12)
        axes[0].set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)')
        axes[0].set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)')
        axes[0].legend(title='Cluster', loc='upper right')

        # Plot Fitur Asli (Dinamis)
        sns.scatterplot(ax=axes[1], x=feature_x, y=feature_y, hue='Cluster', data=self.model.df, palette=palette, s=90, style='Cluster')
        axes[1].set_title(f'Segmentasi: {feature_x} vs {feature_y}', fontsize=13, fontweight='bold', pad=12)
        axes[1].set_xlabel(feature_x)
        axes[1].set_ylabel(feature_y)
        axes[1].legend(title='Cluster', loc='upper right')

        plt.suptitle('Analisis Visualisasi Klaster Pelanggan', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_cluster_characteristic(self,num_cols:list):
        """Visualisasi hasil karakteristik dan keseimbangan data klaster dalam bentuk barplot 2D"""
        if self.model.df_scores is None:
            self.model.compute_scores_dataframe()

        best_idx = self.model.df_scores['Composite_Score'].idxmax()
        best_k = int(self.model.df_scores.loc[best_idx, 'n_clusters(k)'])

        features = [col for col in num_cols if col != 'Cluster']
        n_features = len(features)
        if n_features == 0:
            print("[WARNING] Tidak ada fitur numerik untuk divisualisasikan.")
            return
        df_profile = self.model.df.groupby('Cluster')[features].mean().reset_index()
        palette = 'Set2'

        n_show = len(features)
        n_cols = 3
        nrows = ceil(n_show/n_cols)
        _,axes = plt.subplots(nrows,n_cols,figsize=(25,6*nrows))
        axes = axes.flatten()

        for i, col in enumerate(features):
            counts = df_profile[col]
            sns.barplot(ax=axes[i], x='Cluster', y=col, data=df_profile,hue='Cluster',legend=False,palette=palette, edgecolor='black', alpha=0.85)
            axes[i].set_title(f'Rata-Rata {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Rata-Rata')
            for j, v in enumerate(counts.values):
                    axes[i].text(j, v + (max(counts.values) * 0.02),f'{v:.1f}', ha="center", fontweight="bold")
        for j in range(n_show, len(axes)):
            axes[j].axis('off')
        plt.suptitle(f'Analisis Karakteristik Pelanggan Per Klaster (k={best_k})', fontsize=15, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()
