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