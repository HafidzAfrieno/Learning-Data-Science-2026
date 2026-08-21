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
    def __init__(self):
        pass
    