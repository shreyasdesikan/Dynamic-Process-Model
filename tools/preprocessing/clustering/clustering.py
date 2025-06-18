import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

class BatchClustering:
    def __init__(self, data_dir, method="kmeans", n_clusters=3, var_threshold: float = 0.90):
        self.data_dir = data_dir
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.batches = {}        # raw DataFrames by batch
        self.features = None     # summary feature DataFrame
        self.clean_features = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=var_threshold, svd_solver="full", random_state=0)
        self.labels_ = None

    def load_batches(self):
        for fname in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, fname)
            df = pd.read_csv(path, sep='\t')
            self.batches[fname] = df
        return self.batches

    def extract_features(self, steady_frac: float = 0.2):
        """Compute summary features for each batch"""
        rows = []
        for name, df in self.batches.items():
            n_ss = max(1, int(len(df)*steady_frac))
            ss = df.tail(n_ss)
            feature_dict = {'batch': name}
            
            # Particle sizes
            feature_dict['avg_d10'] = ss['d10'].mean()
            feature_dict['avg_d50'] = ss['d50'].mean()
            feature_dict['avg_d90'] = ss['d90'].mean()
            
            # Process states
            feature_dict['avg_c'] = ss['c'].mean()
            feature_dict['avg_T_PM'] = ss['T_PM'].mean()
            feature_dict['avg_T_TM'] = ss['T_TM'].mean()
            
            # Input parameters (controls)
            feature_dict['avg_mf_PM'] = df['mf_PM'].mean()  # Process flow rate
            feature_dict['avg_mf_TM'] = df['mf_TM'].mean()  # Tempering flow rate
            feature_dict['avg_Q_g'] = df['Q_g'].mean()      # Gas flow rate
            feature_dict['avg_w_crystal'] = df['w_crystal'].mean()  # Crystal loading
            
            # Input conditions
            feature_dict['avg_c_in'] = df['c_in'].mean()
            feature_dict['avg_T_PM_in'] = df['T_PM_in'].mean()
            feature_dict['avg_T_TM_in'] = df['T_TM_in'].mean()
            
            rows.append(feature_dict)
            
        self.features = pd.DataFrame(rows)
        self.clean_features = self.features.copy()
        print(f"Extracted features for {len(self.features)} batches")
        return self.features

    def remove_outliers(self, z_thresh=3.0, iso_frac=0.05):
        """
        1) Z-score > 3: Removes extreme values (like those 200m particles). Drop rows whose ANY feature has |z|>z_thresh
        2) IsolationForest: Catches additional anomalies (5% contamination rate)
        """
        df = self.clean_features.copy()
        # 1) Z-score filter
        z_mask = (np.abs(zscore(df.select_dtypes(float))) > z_thresh).any(axis=1)
        print(f"Z-score >{z_thresh}: removing {z_mask.sum()} batches")
        df = df[~z_mask]

        # 2) IsolationForest
        iso = IsolationForest(contamination=iso_frac, random_state=0)
        iso_lbl = iso.fit_predict(df.select_dtypes(float))
        iso_mask = iso_lbl == -1
        print(f"IsolationForest flagged {iso_mask.sum()} batches")
        df = df.loc[~iso_mask].reset_index(drop=True)

        self.clean_features = df
        return df

    def run_pca(self):
        """Scale clean features and apply PCA"""
        X = self.clean_features.drop(columns=['batch']).values
        X_scaled = self.scaler.fit_transform(X)
        print(f"Prepared matrix: {X.shape}")
        
        # Apply PCA
        self.X_use = self.pca.fit_transform(X_scaled)
        kept = self.X_use.shape[1]
        sum = self.pca.explained_variance_ratio_.sum() * 100
        print(f"PCA kept {kept} components (cumulative {sum:.1f}% variance)")
        
        # Print variance explained by each component
        evr = self.pca.explained_variance_ratio_
        for i, p in enumerate(evr, 1):
            print(f"  PC{i}: {p*100:.1f}%")
        print(f"  Total: {evr.sum()*100:.1f}%\n")
        
        return self.X_use

    def find_k(self, max_k=10):
        """Elbow & silhouette to suggest k (only for kmeans)"""
        Ks = range(2, max_k+1)
        inertias, silhs = [], []
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            lbl = km.fit_predict(self.X_use)
            inertias.append(km.inertia_)
            silhs.append(silhouette_score(self.X_use, lbl))

        fig,(a1,a2) = plt.subplots(1,2,figsize=(12,4))
        a1.plot(Ks, inertias,'-o'); a1.set(title='Elbow',xlabel='k')
        a2.plot(Ks, silhs,'-o');  a2.set(title='Silhouette',xlabel='k')
        plt.tight_layout(); plt.show()

        best = Ks[np.argmax(silhs)]
        print(f"Suggested k = {best} (silh={max(silhs):.3f})\n")
        return best

    def cluster(self):
        """Cluster via the selected method, store labels in self.labels_"""
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
            raise ValueError(f"Unknown method: {self.method}")

        self.clean_features['cluster'] = self.labels_
        ncl = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        sil = (silhouette_score(self.X_use, self.labels_) if ncl>1 else np.nan)
        print(f"{self.method.upper()}: {ncl} clusters, silhouette={sil:.3f}")
        return self.labels_

    def plot_clusters(self):
        """Plot colored PCA according to self.labels_"""
        pcs = self.X_use
        labels = self.labels_
        unique = np.unique(labels)
        cmap = plt.get_cmap('tab10', len(unique))

        if pcs.shape[1] >= 3:
            fig = plt.figure(figsize=(7,6))
            ax  = fig.add_subplot(111, projection='3d')
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                ax.scatter(pcs[mask,0], pcs[mask,1], pcs[mask,2],
                           color=cmap(i), label=f"{lbl}", s=50, alpha=0.7)
            ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({self.pca.explained_variance_ratio_[2]*100:.1f}%)")
            ax.set_title(f"{self.method.upper()} Clusters")
            ax.legend(title="Cluster")
            plt.show()
        else:
            plt.figure(figsize=(6,5))
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                plt.scatter(pcs[mask,0], pcs[mask,1],
                            color=cmap(i), label=f"{lbl}", s=60, alpha=0.7)
            plt.xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.title(f"{self.method.upper()} Clusters")
            plt.legend(title="Cluster")
            plt.grid(alpha=0.3)
            plt.show()
            
    def get_cluster_assignments(self):
        """Return dictionary mapping cluster labels to batch filenames"""
        cluster_dict = {}
        for cluster_id in np.unique(self.labels_):
            mask = self.clean_features['cluster'] == cluster_id
            batch_names = self.clean_features.loc[mask, 'batch'].tolist()
            cluster_dict[cluster_id] = batch_names
        
        # Print summary
        print("\n=== Cluster Assignments ===")
        for cluster_id, batches in sorted(cluster_dict.items()):
            print(f"Cluster {cluster_id}: {len(batches)} batches")
            # Show first few examples
            examples = batches[:3]
            if len(batches) > 3:
                examples.append('...')
            print(f"  Examples: {examples}")
        
        return cluster_dict
                

if __name__ == "__main__":
    bc = BatchClustering(
        data_dir="Data",
        method="kmeans", # "kmeans", "dbscan" or "hdbscan"
        n_clusters=2,
        var_threshold=0.90,
    )

    # 1) Load & feature‐extract
    bc.load_batches()
    bc.extract_features(steady_frac=1)

    # 2) Remove outliers
    bc.remove_outliers(z_thresh=3.0, iso_frac=0.05)

    # 3) Scale & run PCA
    bc.run_pca()

    # 4) If using kmeans, auto‐tune k
    if bc.method == "kmeans":
        suggested_k = bc.find_k(max_k=8)
        bc.n_clusters = suggested_k

    # 5) Cluster & visualize
    bc.cluster()
    bc.plot_clusters()
    
    # 6) Print clusters
    bc.get_cluster_assignments()
