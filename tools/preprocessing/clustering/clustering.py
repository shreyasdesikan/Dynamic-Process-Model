import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

class BatchClustering:
    def __init__(self, data_dir, method="kmeans", n_clusters=3, pca_components=2):
        self.data_dir = data_dir
        self.method = method
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        
