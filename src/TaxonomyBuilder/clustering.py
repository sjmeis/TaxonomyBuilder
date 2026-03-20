import logging
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

logger = logging.getLogger(__name__)

try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    HAS_GPU_CLUSTERING = True
except ImportError:
    HAS_GPU_CLUSTERING = False
    from sklearn.cluster import HDBSCAN as skHDBSCAN
    from umap import UMAP as cpuUMAP

class ClusterEngine:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        if self.use_gpu:
            logger.info("ClusterEngine: Initialized with GPU (RAPIDS) support.")
        else:
            logger.info("ClusterEngine: Initialized on CPU.")

    def reduce_dimensions(self, embeddings, n_components=10, n_neighbors=15, random_state=42):
        """Reduces high-dim embeddings to low-dim space for HDBSCAN."""
        logger.info(f"Reducing dimensions to {n_components} components...")

        n_samples = embeddings.shape[0]
        safe_neighbors = min(n_neighbors, n_samples - 1)
        init_method = 'spectral'
        if n_samples < n_components:
            init_method = 'random'
            logger.warning(f"Small dataset detected ({n_samples} samples). Switching UMAP to random init.")
        
        if self.use_gpu:
            reducer = cuUMAP(n_components=n_components, n_neighbors=safe_neighbors, random_state=random_state, init=init_method)
        else:
            reducer = cpuUMAP(n_components=n_components, n_neighbors=safe_neighbors, random_state=random_state, init=init_method)
            
        return reducer.fit_transform(embeddings)

    def cluster(self, data, min_cluster_size=5, max_cluster_size=None):
        """Performs HDBSCAN clustering."""
        logger.info(f"Clustering with min_size={min_cluster_size}...")
        
        # HDBSCAN parameters vary slightly between implementations
        params = {
            "min_cluster_size": min_cluster_size,
            "allow_single_cluster": False,
            "metric": "l2"
        }
        
        if self.use_gpu:
            model = cuHDBSCAN(**params)
        else:
            model = skHDBSCAN(**params)
            
        labels = model.fit_predict(data)
        return labels, model
    
    def soft_cluster(self, reduced_embeddings, labels, model):
        """
        Assigns noise points (-1) to the nearest cluster centroid.
        """
        new_labels = labels.copy()
        noise_mask = (labels == -1)
        
        if not np.any(noise_mask):
            return new_labels

        # calculate Centroids for all valid clusters
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        if not unique_clusters:
            return new_labels
    
        centroids = np.array([
            reduced_embeddings[labels == c].mean(axis=0) 
            for c in unique_clusters
        ])
        noise_embeddings = reduced_embeddings[noise_mask]
        if noise_embeddings.ndim == 1:
            noise_embeddings = noise_embeddings.reshape(1, -1)

        # calculate similarity (n_noise x n_clusters) to find the index of the closest centroid for every noise point
        closest_indices = pairwise_distances_argmin(noise_embeddings, centroids)
        
        # map indices back to actual cluster IDs
        assigned_clusters = np.array([unique_clusters[i] for i in closest_indices])
        
        new_labels[noise_mask] = assigned_clusters
        return new_labels