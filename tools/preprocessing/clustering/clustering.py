import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import logging
import chardet
from scipy.signal import butter, filtfilt

# Set up logging
logging.basicConfig(level=logging.INFO, filename='clustering_log.txt', format='%(asctime)s - %(levelname)s - %(message)s')

class BatchClustering:
    def __init__(self, data_dir, method="kmeans", n_clusters=3, var_threshold=0.90, min_components=2):
        self.data_dir = data_dir
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.var_threshold = var_threshold
        self.min_components = min_components
        self.batches = {}  # raw DataFrames by batch
        self.features = None  # summary feature DataFrame
        self.clean_features = None
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=var_threshold, svd_solver="auto", random_state=0)
        self.labels_ = None
        self.silhouette_scores_ = None
        logging.info(f"Initialized BatchClustering with parameters: {vars(self)}")

    def detect_encoding(self, file_path):
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
            encoding = result['encoding'] or 'utf-8'
            logging.info(f"Detected encoding for {file_path}: {encoding}")
            return encoding
        except Exception as e:
            logging.error(f"Error detecting encoding for {file_path}: {e}")
            return 'utf-8'

    def load_batches(self):
        """Load all .txt files from the specified directory"""
        if not os.path.exists(self.data_dir):
            logging.error(f"Directory {self.data_dir} does not exist.")
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist.")

        required_cols = ['c', 'T_PM', 'd10', 'd50', 'd90', 'T_TM', 'mf_PM', 'mf_TM', 
                        'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
        separators = ['\t', ',', ';', ' ']

        for file in os.listdir(self.data_dir):
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(self.data_dir, file)
            try:
                encoding = self.detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                logging.info(f"First lines of {file}: {first_lines}")

                df = None
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding, skipinitialspace=True)
                        if len(df.columns) > 1:
                            logging.info(f"Successfully loaded with separator '{sep}' for {file}")
                            break
                    except Exception as e:
                        logging.warning(f"Error with separator '{sep}' for {file}: {e}")
                
                if df is None or df.empty:
                    logging.warning(f"File {file} is empty or has invalid format.")
                    continue

                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logging.warning(f"File {file} is missing columns: {missing_cols}")
                    continue

                if df.select_dtypes(include=np.number).empty:
                    logging.warning(f"File {file} contains no numeric data.")
                    continue

                self.batches[file] = df
                logging.info(f"Loaded {file} with shape {df.shape}, columns: {list(df.columns)}")
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")

        if not self.batches:
            logging.error("No valid data loaded.")
            raise ValueError("No valid data loaded.")


    def apply_moving_average(self, df, window=5):
        """Apply moving average to smooth the data"""
        for col in ['d10', 'd50', 'd90']:
            if col in df.columns:
                df[col] = df[col].rolling(window=window, min_periods=1).mean()
        return df

    def apply_low_pass_filter(self, df, cutoff=0.1):
        """Apply low-pass filter to reduce noise"""
        for col in ['d10', 'd50', 'd90']:
            if col in df.columns:
                b, a = butter(2, cutoff, btype='low', fs=1.0)
                df[col] = filtfilt(b, a, df[col])
        return df
         
    def exclude_noisy_files(self, max_std_threshold=0.001):
        """Exclude files with excessive noise"""
        noisy_files = []
        for file_name, df in self.batches.items():
            if df[['d10', 'd50', 'd90']].std().max() > max_std_threshold:
                noisy_files.append(file_name)
        for file in noisy_files:
            del self.batches[file]
        logging.info(f"Excluded noisy files: {noisy_files}")
         
    def remove_peaks(self, threshold=0.0003):
        """Remove unrealistic peaks from d10, d50, d90"""
        try:
            for file_name, df in self.batches.items():
                for col in ['d10', 'd50', 'd90']:
                    if col in df.columns:
                        mask = (df[col] <= threshold)
                        df[col] = df[col][mask]
                df.dropna(inplace=True)
                if df.empty:
                    del self.batches[file_name]
                    logging.warning(f"File {file_name} became empty after peak removal and was deleted")
            logging.info("Peak removal completed successfully")
        except Exception as e:
            logging.error(f"Error during peak removal: {e}")
            
    def extract_features(self, steady_frac: float = 1.0):
        """Extract features from each batch"""
        rows = []
        for name, df in self.batches.items():
            try:
                df = df.fillna(df.mean(numeric_only=True))
                n_ss = max(1, int(len(df) * steady_frac))
                ss = df.tail(n_ss)
                
                if ss['d50'].std() / ss['d50'].mean() > 0.1:
                    logging.warning(f"Batch {name} may not be in steady state (high variance in d50)")

                feature_dict = {'batch': name}
                feature_dict['avg_d10'] = ss['d10'].mean()
                feature_dict['avg_d50'] = ss['d50'].mean()
                feature_dict['avg_d90'] = ss['d90'].mean()
                feature_dict['avg_c'] = ss['c'].mean()
                feature_dict['avg_T_PM'] = ss['T_PM'].mean()
                feature_dict['avg_T_TM'] = ss['T_TM'].mean()
                feature_dict['avg_mf_PM'] = df['mf_PM'].mean()
                feature_dict['avg_mf_TM'] = df['mf_TM'].mean()
                feature_dict['avg_Q_g'] = df['Q_g'].mean()
                feature_dict['avg_w_crystal'] = df['w_crystal'].mean()
                feature_dict['avg_c_in'] = df['c_in'].mean()
                feature_dict['avg_T_PM_in'] = df['T_PM_in'].mean()
                feature_dict['avg_T_TM_in'] = df['T_TM_in'].mean()
                rows.append(feature_dict)
            except Exception as e:
                logging.error(f"Error extracting features from {name}: {e}")

        self.features = pd.DataFrame(rows)
        self.clean_features = self.features.copy()
        logging.info(f"Features extracted for {len(self.features)} batches")

    def remove_outliers(self, z_thresh=3.0, iso_frac=0.05):
        """Remove outliers using Z-Score and Isolation Forest"""
        df = self.clean_features.copy()
        try:
            df_z = df.drop(columns=['batch']).apply(zscore)
            mask_z = (np.abs(df_z) < z_thresh).all(axis=1)
            df = df[mask_z]
            logging.info(f"{len(self.clean_features) - len(df)} outliers removed with Z-Score")

            iso = IsolationForest(contamination=iso_frac, random_state=0)
            iso_mask = iso.fit_predict(df.drop(columns=['batch'])) == 1
            df = df[iso_mask]
            logging.info(f"{len(self.clean_features) - len(df)} outliers removed with Isolation Forest")

            removed = self.features[~self.features['batch'].isin(df['batch'])]['batch']
            logging.info(f"Removed batches as outliers: {removed.tolist()}")
            self.clean_features = df
            return df
        except Exception as e:
            logging.error(f"Error removing outliers: {e}")
            return self.clean_features
    
    def prepare(self):
        """Prepare data for clustering"""
        try:
            X = self.clean_features.drop(columns=['batch']).values
            self.X_scaled = self.scaler.fit_transform(X)
            logging.info(f"Scaled data with shape {self.X_scaled.shape}")
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")

    def run_pca(self, show=True):
        """Apply PCA and visualize results"""
        try:
            n_components = max(self.min_components, int(self.var_threshold * self.X_scaled.shape[1]))
            self.pca.n_components = n_components
            self.X_use = self.pca.fit_transform(self.X_scaled)
            var_sum = self.pca.explained_variance_ratio_.sum() * 100
            logging.info(f"PCA retained {self.X_use.shape[1]} components (cumulative variance {var_sum:.1f}%)")

            if show and self.X_use.shape[1] >= 2:
                fig = px.scatter(
                    x=self.X_use[:, 0], y=self.X_use[:, 1],
                    labels={'x': f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)',
                            'y': f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)'},
                    title="PCA (Raw)"
                )
                fig.show()
            elif show and self.X_use.shape[1] >= 3:
                fig = px.scatter_3d(
                    x=self.X_use[:, 0], y=self.X_use[:, 1], z=self.X_use[:, 2],
                    labels={'x': f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)',
                            'y': f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)',
                            'z': f'PC3 ({self.pca.explained_variance_ratio_[2]*100:.1f}%)'},
                    title="PCA (3D Raw)"
                )
                fig.show()

            return self.X_use
        except Exception as e:
            logging.error(f"Error running PCA: {e}")
            return None

    def find_k(self, max_k=10, show_elbow=True):
        """Find optimal number of clusters"""
        try:
            Ks = range(2, max_k + 1)
            inertias, silhs = [], []
            for k in Ks:
                km = KMeans(n_clusters=k, random_state=0, n_init=10)
                lbl = km.fit_predict(self.X_use)
                inertias.append(km.inertia_)
                silhs.append(silhouette_score(self.X_use, lbl))
            
            best_k = Ks[np.argmax(silhs)]
            logging.info(f"Best k = {best_k}, Silhouette = {max(silhs):.3f}")

            if show_elbow:
                plt.figure(figsize=(6, 4))
                plt.plot(Ks, inertias, 'o-k')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Inertia')
                plt.title('Elbow method for optimal k')
                plt.grid(True)
                plt.show()

            return best_k
        except Exception as e:
            logging.error(f"Error finding k: {e}")
            return None

    def compute_silhouette_per_cluster(self):
        """Calculate Silhouette score per cluster"""
        try:
            if len(set(self.labels_)) <= 1 or (-1 in self.labels_ and len(set(self.labels_)) <= 2):
                logging.warning("Too few clusters for Silhouette score calculation")
                return None

            self.silhouette_scores_ = silhouette_samples(self.X_use, self.labels_)
            cluster_silhouettes = {}
            for cluster_id in np.unique(self.labels_):
                if cluster_id == -1:
                    continue
                mask = self.labels_ == cluster_id
                cluster_silhouettes[cluster_id] = np.mean(self.silhouette_scores_[mask])
                logging.info(f"Silhouette score for cluster {cluster_id}: {cluster_silhouettes[cluster_id]:.3f}")

            sil_avg = silhouette_score(self.X_use, self.labels_)
            logging.info(f"Average Silhouette score: {sil_avg:.3f}")
            return cluster_silhouettes, sil_avg
        except Exception as e:
            logging.error(f"Error calculating Silhouette score: {e}")
            return None

    def cluster(self):
        """Perform clustering with the specified method"""
        try:
            if self.method == "kmeans":
                model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
                self.labels_ = model.fit_predict(self.X_use)
            elif self.method == "dbscan":
                model = DBSCAN(eps=0.5, min_samples=5)
                self.labels_ = model.fit_predict(self.X_use)
            elif self.method == "hdbscan":
                model = HDBSCAN(min_cluster_size=5)
                self.labels_ = model.fit_predict(self.X_use)
            else:
                logging.error(f"Unknown method: {self.method}")
                raise ValueError(f"Unknown method: {self.method}")

            self.clean_features['cluster'] = self.labels_
            ncl = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            if ncl > 1:
                sil_avg = silhouette_score(self.X_use, self.labels_)
                logging.info(f"{self.method.upper()}: {ncl} clusters, average Silhouette score={sil_avg:.3f}")
                self.compute_silhouette_per_cluster()
                self.plot_silhouette()
            else:
                logging.warning("Too few clusters for Silhouette score calculation")

            return self.labels_
        except Exception as e:
            logging.error(f"Clustering error: {e}")
            return None

    def plot_silhouette(self):
        """Create silhouette plot for clusters"""
        try:
            if self.silhouette_scores_ is None or len(set(self.labels_)) <= 1:
                logging.warning("No silhouette scores available or too few clusters")
                return

            fig = go.Figure()
            y_lower = 10
            unique_labels = sorted([lbl for lbl in np.unique(self.labels_) if lbl != -1])
            for cluster_id in unique_labels:
                mask = self.labels_ == cluster_id
                cluster_scores = np.sort(self.silhouette_scores_[mask])[::-1]
                y_upper = y_lower + len(cluster_scores)
                fig.add_trace(go.Scatter(
                    x=cluster_scores,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    fill='tozerox',
                    name=f'Cluster {cluster_id}',
                    text=[f'Cluster {cluster_id}' for _ in cluster_scores]
                ))
                y_lower = y_upper + 10

            avg_score = silhouette_score(self.X_use, self.labels_)
            fig.add_vline(x=avg_score, line_dash="dash", line_color="red", 
                          annotation_text=f"Average: {avg_score:.3f}", annotation_position="top")
            fig.update_layout(
                title="Silhouette plot for clusters",
                xaxis_title="Silhouette coefficient",
                yaxis_title="Cluster",
                showlegend=True
            )
            fig.show()
        except Exception as e:
            logging.error(f"Error creating silhouette plot: {e}")

    def plot_clusters(self):
        """Visualize clusters in PCA space"""
        try:
            pcs = self.X_use
            labels = self.labels_
            fig = px.scatter(
                x=pcs[:, 0], y=pcs[:, 1], color=labels.astype(str),
                labels={'x': f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)',
                        'y': f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)'},
                title=f"{self.method.upper()} Clusters"
            )
            fig.update_layout(legend_title="Cluster")
            fig.show()
        except Exception as e:
            logging.error(f"Error plotting clusters: {e}")

    def get_cluster_assignments(self):
        """Output cluster assignments with Silhouette scores"""
        try:
            cluster_dict = {}
            for cluster_id in np.unique(self.labels_):
                mask = self.clean_features['cluster'] == cluster_id
                batch_names = self.clean_features.loc[mask, 'batch'].tolist()
                cluster_dict[cluster_id] = batch_names

            logging.info("=== Cluster Assignments ===")
            for cluster_id, batches in sorted(cluster_dict.items()):
                logging.info(f"Cluster {cluster_id}: {len(batches)} batches")
                examples = batches[:3] + (["..."] if len(batches) > 3 else [])
                logging.info(f"  Examples: {examples}")
                if self.silhouette_scores_ is not None and cluster_id != -1:
                    mask = self.labels_ == cluster_id
                    sil_score = np.mean(self.silhouette_scores_[mask])
                    logging.info(f"  Silhouette score: {sil_score:.3f}")

            return cluster_dict
        except Exception as e:
            logging.error(f"Error getting cluster assignments: {e}")
            return None

if __name__ == "__main__":
    try:
        bc = BatchClustering(
            data_dir="Data",
            method="kmeans",
            n_clusters=2,
            var_threshold=0.50,
            min_components=2
        )
        bc.load_batches()
        bc.exclude_noisy_files(max_std_threshold=0.004)
        bc.remove_peaks(threshold=0.0005)
        bc.extract_features(steady_frac=0.3)
        bc.remove_outliers(z_thresh=3.0, iso_frac=0.02)
        bc.prepare()
        bc.run_pca(show=True)
        if bc.method == "kmeans":
            bc.n_clusters = bc.find_k(max_k=8)
        bc.cluster()
        bc.plot_clusters()
        bc.plot_silhouette()
        bc.get_cluster_assignments()
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
