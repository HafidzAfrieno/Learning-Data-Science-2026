import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class AggloMerativeClustering:
    """
    Kelas inti untuk menangani logika pemodelan, training, perhitungan metrik evaluasi,
    serta transformasi data untuk segmentasi menggunakan Agglo Merative.
    """

    def __init__(self, range_max: int, df_raw: pd.DataFrame):
            if range_max <= 2:
                raise ValueError("range_max harus lebih besar dari 2.")
            
            self.df = df_raw.copy()
            self.k_range = range(2, range_max)
            self.sil_scores = []
            self.ch_scores = []
            self.db_scores = []
            self.df_scores = None

    def fit_model(self, X_cleaned: np.ndarray) -> None:
            """Melakukan training AGL untuk setiap nilai k dalam k_range."""
            for k in self.k_range:
                agl = AgglomerativeClustering(n_clusters=k,linkage='ward')
                model_agl = agl.fit_predict(X_cleaned)
                self.sil_scores.append(silhouette_score(X_cleaned, model_agl))
                self.ch_scores.append(calinski_harabasz_score(X_cleaned, model_agl))
                self.db_scores.append(davies_bouldin_score(X_cleaned, model_agl))
            print("[INFO] Model berhasil dilatih untuk semua rentang k.")

    def compute_scores_dataframe(self) -> pd.DataFrame:
        """Membuat DataFrame metrik evaluasi dan menghitung Composite Score secara normalisasi."""
        df_scores = pd.DataFrame({
            'n_clusters(k)': list(self.k_range),
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
        # best_k = int(self.df_scores.loc[best_idx, 'n_clusters(k)'])
        best_k = 6
        
        best_agl= AgglomerativeClustering(n_clusters=best_k,linkage='ward')
        self.df['Cluster'] = best_agl.fit_predict(X_cleaned) 
        
        print(f"[INFO] Model optimal dipilih dengan jumlah cluster (k) = {best_k}")
        return best_agl, self.df

    def convert_to_pca(self, X_cleaned: np.ndarray):
        """Mereduksi dimensi data menggunakan PCA untuk keperluan visualisasi 2D."""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_cleaned)

        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_pca['Cluster'] = self.df['Cluster']
        centroids_pca = df_pca.groupby('Cluster')[['PCA1', 'PCA2']].mean().values
        
        return df_pca, centroids_pca, pca

class ClusteringVisualizer:
    """
    Kelas terpisah khusus untuk menangani seluruh visualisasi/plotting data.
    Menerapkan prinsip Single Responsibility Principle (SRP).
    """
    def __init__(self, model_instance: AggloMerativeClustering):
        self.model = model_instance

    def plot_evaluation_metrics(self,X_cleaned: np.ndarray) -> None:
        """Membuat plot evaluasi lengkap (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin)."""
        df_scores = self.model.df_scores
        if df_scores is None:
            df_scores = self.model.compute_scores_dataframe()

        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        k_list = list(self.model.k_range)

        # --- Plot 1: Silhouette Score (Mendekati +1 Lebih Baik) ---
        best_k_sil = int(df_scores.loc[df_scores['Silhouette Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[0, 0], x='n_clusters(k)', y='Silhouette Score', data=df_scores, marker='o', markersize=8, linewidth=2, color='#2b5c8f')
        axes[0, 0].axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best k = {best_k_sil}')
        axes[0, 0].set_title('Silhouette Score\n(Mendekati +1 = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Jumlah Cluster (k)')
        axes[0, 0].set_xticks(list(k_list))
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        axes[0, 0].legend()

        # --- Plot 2: Calinski-Harabasz Score (Makin Tinggi Lebih Baik) ---
        best_k_ch = int(df_scores.loc[df_scores['Calinski-Harabasz Score'].idxmax(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[0, 1], x='n_clusters(k)', y='Calinski-Harabasz Score', data=df_scores, marker='s', markersize=8, linewidth=2, color='#2e7d32')
        axes[0, 1].axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'Puncak k = {best_k_ch}')
        axes[0, 1].set_title('Calinski-Harabasz Score\n(Semakin Tinggi = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Jumlah Cluster (k)')
        axes[0, 1].set_xticks(list(k_list))
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend()

        # --- Plot 3: Davies-Bouldin Score (Makin Rendah Lebih Baik) ---
        best_k_db = int(df_scores.loc[df_scores['Davies-Bouldin Score'].idxmin(), 'n_clusters(k)'])
        sns.lineplot(ax=axes[1, 0], x='n_clusters(k)', y='Davies-Bouldin Score', data=df_scores, marker='^', markersize=8, linewidth=2, color='#c62828')
        axes[1, 0].axvline(x=best_k_db, color='green', linestyle='--', alpha=0.7, label=f'Terendah k = {best_k_db}')
        axes[1, 0].set_title('Davies-Bouldin Score\n(Semakin Rendah = Terbaik)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Jumlah Cluster (k)')
        axes[1, 0].set_xticks(list(k_list))
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend()

        # --- Plot 4: Dendrogram Visualisasi Hirarki ---
        Z = linkage(X_cleaned, method='ward')
        dendrogram(Z, ax=axes[1, 1], truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
        axes[1, 1].set_title('Dendrogram (Pohon Hirarki)\n(Struktur Penggabungan Cluster)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Sampel / Sub-cluster')
        axes[1, 1].set_ylabel('Jarak Jangkauan (Euclidean)')
        axes[1, 1].grid(False)

        plt.suptitle('Evaluasi Metrik Klastering Agglomerative Berdasarkan (K) Clusster', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_cluster_results(self, X_cleaned: np.ndarray, feature_x: str, feature_y: str) -> None:
        """Visualisasi hasil klaster dalam bentuk PCA 2D dan scatter plot fitur asli."""
        df_pca, centroids_pca, pca = self.model.convert_to_pca(X_cleaned)

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