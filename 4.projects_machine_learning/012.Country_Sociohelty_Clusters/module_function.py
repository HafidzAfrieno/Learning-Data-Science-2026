import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def get_optimal_eps_range_quantile(X_cleaned: np.ndarray, min_samples: int, num_steps: int = 30):
    k = max(2, min(min_samples, X_cleaned.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(X_cleaned)

    distances, _ = nn.kneighbors(X_cleaned)
    d_k = distances[:, -1]
    eps_start, eps_stop = np.quantile(d_k, [0.25, 0.75]) #quantile high
    if eps_start >= eps_stop:
        eps_start, eps_stop = d_k.min(), d_k.max()

    eps_range = np.linspace(eps_start, eps_stop, num=num_steps)
    return eps_range

class DbscanClustering:
    """
    Kelas inti untuk menangani logika pemodelan, training, perhitungan metrik evaluasi,
    serta transformasi data untuk segmentasi menggunakan DBSCAN.
    """
    def __init__(self,df_raw:pd.DataFrame):
        self.df_scores = pd.DataFrame
        self.df = df_raw
        self.eps_range = []
        self.sil_scores = []
        self.ch_scores = []
        self.db_scores = []
        self.eps_values = []
        self.n_clusters_found = []

    def fit_model(self,X_cleaned:np.ndarray)->None:
        """Melakukan training DBSCAN untuk setiap nilai eps dalam EPSILON"""
        min_samples_val = 3 #menggunakan standart sendiri
        self.eps_range = get_optimal_eps_range_quantile(X_cleaned=X_cleaned, min_samples=min_samples_val)

        for eps in self.eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples_val)
            labels = dbscan.fit_predict(X_cleaned)

            core_mask = labels != -1
            unique_labels = set(labels[core_mask])
        
            if len(unique_labels) >= 2:
                X_valid = X_cleaned[core_mask]
                labels_valid = labels[core_mask]
                
                self.eps_values.append(round(eps, 2))
                self.sil_scores.append(silhouette_score(X_valid, labels_valid))
                self.ch_scores.append(calinski_harabasz_score(X_valid, labels_valid))
                self.db_scores.append(davies_bouldin_score(X_valid, labels_valid))
                self.n_clusters_found.append(len(unique_labels))

        if not self.eps_values:
            print("[WARNING] Tidak ada konfigurasi eps yang menghasilkan minimal 2 cluster valid.")
        else:
            print(f"[INFO] Model berhasil dilatih. Ditemukan {len(self.eps_values)} konfigurasi eps valid.")

    def compute_scores_dataframe(self) -> pd.DataFrame:
        """Membuat DataFrame metrik evaluasi dan menghitung Composite Score secara normalisasi."""

        df_scores = pd.DataFrame({
            'Clusters Found': self.n_clusters_found,
            'eps': self.eps_values,
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

    def best_composcore(self):
        if self.df_scores is None:
            self.compute_scores_dataframe()
        best_idx = self.df_scores['Composite_Score'].idxmax()
        best_eps = self.df_scores.loc[best_idx,'eps']
        return best_eps

    def fit_best_model(self, X_cleaned: np.ndarray):
        """Memilih epsilon terbaik berdasarkan Composite Score tertinggi dan melatih ulang model."""
        min_samples_val = 3
        best_eps = self.best_composcore()
        best_dbscan = DBSCAN(eps=best_eps, min_samples=min_samples_val)
        labels = best_dbscan.fit_predict(X_cleaned)
        self.df['Cluster'] = labels
    
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        print(f'[INFO] Model optimal dipilih dengan epsilon (eps) = {best_eps:.2f} dan Jumlah Cluster (k={self.n_clusters})')
        print(f'[INFO] Jumlah data noise (outlier): {n_noise}')
        return best_dbscan, self.df

    def convert_to_pca(self, X_cleaned: np.ndarray):
        """Mereduksi dimensi data menggunakan PCA untuk keperluan visualisasi 2D."""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_cleaned)
        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

        df_pca['Cluster'] = self.df['Cluster'].values
        valid_clusters = df_pca[df_pca['Cluster'] != -1]
        centroids_pca = valid_clusters.groupby('Cluster')[['PCA1', 'PCA2']].mean().values
        return df_pca, centroids_pca, pca

class ClusteringVisualizer:
    """
    Kelas terpisah khusus untuk menangani seluruh visualisasi/plotting data.
    Menerapkan prinsip Single Responsibility Principle (SRP).
    """
    def __init__(self, model_instance: DbscanClustering):
            self.model = model_instance

    def plot_evaluation_metrics(self) -> None:
        """Membuat plot evaluasi lengkap (Silhouette, Calinski-Harabasz, Davies-Bouldin, Composite Score)."""
        df_scores = self.model.df_scores
        if df_scores is None:
            df_scores = self.model.compute_scores_dataframe()
            
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        unique_eps = sorted(df_scores['eps'].unique())

        # --- Plot 1: Silhouette Score (Mendekati +1 Lebih Baik) ---
        best_k_sil = df_scores.loc[df_scores['Silhouette Score'].idxmax(), 'eps']
        sns.lineplot(ax=axes[0, 0], x='eps', y='Silhouette Score', data=df_scores, marker='o', markersize=8, linewidth=2, color='#2b5c8f')
        axes[0, 0].axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best k = {best_k_sil}')
        axes[0, 0].set_title('Silhouette Score\n(Mendekati +1 = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Jumlah Epsilon (eps)')
        axes[0, 0].set_xticks(unique_eps)
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        axes[0, 0].legend()

        # --- Plot 2: Calinski-Harabasz Score (Makin Tinggi Lebih Baik) ---
        best_k_ch = df_scores.loc[df_scores['Calinski-Harabasz Score'].idxmax(), 'eps']
        sns.lineplot(ax=axes[0, 1], x='eps', y='Calinski-Harabasz Score', data=df_scores, marker='s', markersize=8, linewidth=2, color='#2e7d32')
        axes[0, 1].axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'Puncak k = {best_k_ch}')
        axes[0, 1].set_title('Calinski-Harabasz Score\n(Semakin Tinggi = Terbaik)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Jumlah Epsilon (eps)')
        axes[0, 1].set_xticks(unique_eps)
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend()

        # --- Plot 3: Davies-Bouldin Score (Makin Rendah Lebih Baik) ---
        best_k_db = df_scores.loc[df_scores['Davies-Bouldin Score'].idxmin(), 'eps']
        sns.lineplot(ax=axes[1, 0], x='eps', y='Davies-Bouldin Score', data=df_scores, marker='^', markersize=8, linewidth=2, color='#c62828')
        axes[1, 0].axvline(x=best_k_db, color='green', linestyle='--', alpha=0.7, label=f'Terendah k = {best_k_db}')
        axes[1, 0].set_title('Davies-Bouldin Score\n(Semakin Rendah = Terbaik)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Jumlah Epsilon (eps)')
        axes[1, 0].set_xticks(unique_eps)
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend()

        # --- Plot 4 (BARU): Normalized Composite Score (Gabungan 3 Metrik) ---
        best_k_comp = df_scores.loc[df_scores['Composite_Score'].idxmax(), 'eps']
        sns.lineplot(ax=axes[1, 1], x='eps', y='Composite_Score', data=df_scores, marker='D', markersize=8, linewidth=2, color='#7b1fa2')
        axes[1, 1].axvline(x=best_k_comp, color='purple', linestyle='--', alpha=0.7, label=f'Rekomendasi k = {best_k_comp}')
        axes[1, 1].set_title('Normalized Composite Score\n(Konsensus / Gabungan 3 Metrik)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Jumlah Epsilon (eps)')
        axes[1, 1].set_xticks(unique_eps)
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)
        axes[1, 1].legend()

        plt.suptitle('Evaluasi Metrik Klastering Spectral Berdasarkan (K) Clusster', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_cluster_results(self, X_cleaned: np.ndarray, feature_x: str, feature_y: str) -> None:
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
            best_k = self.model.n_clusters
            features = [col for col in num_cols if col != 'Cluster']
            n_features = len(features)

            if n_features == 0:
                print("[WARNING] Tidak ada fitur numerik untuk divisualisasikan.")
                return
            
            df_valid = self.model.df[self.model.df['Cluster'] != -1]
            if df_valid.empty:
                print('[WARNING] Semua data teridentifikasi sebagai noise (-1).')
                return
            df_profile = df_valid.groupby('Cluster')[features].mean().reset_index()

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
            plt.suptitle(f'Analisis Karakteristik Pelanggan Per Klaster (k={best_k})', fontsize=20, fontweight='bold', y=1.05)
            plt.tight_layout()
            plt.show()