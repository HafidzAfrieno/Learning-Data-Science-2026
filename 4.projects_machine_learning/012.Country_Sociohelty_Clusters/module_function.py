import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class DbscanClustering:
    """
    Kelas inti untuk menangani logika pemodelan, training, perhitungan metrik evaluasi,
    serta transformasi data untuk segmentasi menggunakan DBSCAN.
    """
    def __init__(self,df_raw:pd.DataFrame):
        self.eps_range = np.arange(0.1, 1.0, 0.1)
        self.df = df_raw.copy()
        self.df_scores = None
        self.sil_scores = []
        self.ch_scores = []
        self.db_scores = []
        self.eps_values = []
        self.n_clusters_found = []

    def fit_model(self,X_cleaned:np.ndarray)->None:
        """Melakukan training DBSCAN untuk setiap nilai eps dalam EPSILON"""
        dim = X_cleaned.shape[1]
        min_samples_val = 2 * dim 
        for eps in self.eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples_val)
            labels = dbscan.fit_predict(X_cleaned)
            core_mask = labels != -1 # Filter titik noise (-1) untuk perhitungan metrik evaluasi
            unique_labels = set(labels[core_mask])

            if len(unique_labels) >= 2:
                X_valid = X_cleaned[core_mask]
                labels_valid = labels[core_mask]
                self.eps_values.append(round(eps, 2))
                self.sil_scores.append(silhouette_score(X_valid, labels_valid))
                self.ch_scores.append(calinski_harabasz_score(X_valid, labels_valid))
                self.db_scores.append(davies_bouldin_score(X_valid, labels_valid))
                self.n_clusters_found.append(len(unique_labels))

    def compute_scores_dataframe(self) -> pd.DataFrame:
        """Membuat DataFrame metrik evaluasi dan menghitung Composite Score secara normalisasi."""
        scaler = MinMaxScaler()
        df_scores = pd.DataFrame({
            'Clusters Found': self.n_clusters_found,
            'eps': self.eps_values,
            'Silhouette Score': self.sil_scores,
            'Calinski-Harabasz Score': self.ch_scores,
            'Davies-Bouldin Score': self.db_scores
        })
        norm_sil = scaler.fit_transform(df_scores[['Silhouette Score']])
        norm_ch = scaler.fit_transform(df_scores[['Calinski-Harabasz Score']])
        norm_db = 1 - scaler.fit_transform(df_scores[['Davies-Bouldin Score']])
        df_scores['Composite_Score'] = (norm_sil + norm_ch + norm_db) / 3
        self.df_scores = df_scores
        return df_scores

    def fit_best_model(self, X_cleaned: np.ndarray):
        print()
    
    