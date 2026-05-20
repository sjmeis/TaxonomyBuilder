import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from tqdm.auto import tqdm
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import string

from .data import TextDataset
from torch.utils.data import DataLoader
from .clustering import ClusterEngine
from .prompt_utils import PromptBuilder

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxonomyBuilder:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True, working_dir="taxonomy_output", llm_provider=None):
        """
        Initializes the TaxonomyBuilder with hardware detection and model loading.
        
        Args:
            model_name (str): The HuggingFace/SentenceTransformer model to use.
            use_gpu (bool): If True, attempts to use CUDA for torch and cuml.
        """
        self.set_verbosity()

        self.embedding_model_name = embedding_model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.use_cuml = False
        self.cluster_engine = ClusterEngine(use_gpu=use_gpu)
        
        # check for RAPIDS (cuml) availability
        if use_gpu:
            try:
                import cuml
                self.use_cuml = True
                logger.info("GPU detected: Using cuML for accelerated clustering.")
            except ImportError:
                logger.warning("cuml not found. Falling back to CPU-based scikit-learn.")

        self.llm = llm_provider

        # placeholders
        self.data = None
        self.domain_keywords = None
        self.keyword_embeddings = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.clustering_model = None
        self.taxonomy_labels = None

        self.levels = {}  # i.e., {level_index: {cluster_id: label_text}}
        self.hierarchy = {} #i.e.,  {level_index: {child_id: parent_id}}

        # setup working paths
        self.work_dir = Path.cwd() / working_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # load embedding model
        logger.info(f"Loading model '{embedding_model_name}' on {self.device}...")
        self._load_models()

    def set_verbosity(self, level=logging.INFO):
        """Sets the logging level for the builder and muffles external API logs."""
        logging.getLogger("TaxonomyBuilder").setLevel(level)
        
        # Always keep these at Warning or higher to protect tqdm
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        return self

    def _load_models(self):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

    def set_llm(self, provider_name, api_key, model_endpoint):
        """
        Helper method to switch LLM providers easily.
        """
        from .llm import OpenAIProvider, AnthropicProvider, GoogleProvider
        
        if provider_name.lower() == "openai":
            self.llm = OpenAIProvider(api_key=api_key, model=model_endpoint)
        elif provider_name.lower() == "anthropic":
            self.llm = AnthropicProvider(api_key=api_key, model=model_endpoint)
        elif provider_name.lower() == "google":
            self.llm = GoogleProvider(api_key=api_key, model=model_endpoint)
        else:
            raise ValueError(f"Provider {provider_name} is not supported yet.")
        
        logger.info(f"LLM Provider set to {provider_name} using model {model_endpoint}")

    def ingest_data(self, texts, keywords=None):
        """
        Validates and stores the input text and domain context.
        
        Args:
            texts (list[str]): The raw documents/phrases to taxonomize.
            keywords (list[str], optional): Keywords describing the domain.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All inputs in 'texts' must be strings.")

        self.data = texts
        self.embeddings = None
        self.encode()
        logger.info(f"Ingested and embedded {len(self.data)} text documents.")

        self.domain_keywords = keywords if keywords else []
        if self.domain_keywords:
            self.keyword_embeddings = self.embedding_model.encode(self.domain_keywords, convert_to_tensor=True)
            logger.info(f"Domain context set with {len(self.domain_keywords)} keywords.")
            
        return self

    def _get_data_loader(self, batch_size):
        dataset = TextDataset(self.data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def encode(self, batch_size=32, show_progress=True):
        """Embed all of the ingested data to have it ready asap."""
        if self.data is None:
            raise ValueError("You must call ingest_data first!")

        loader = self._get_data_loader(batch_size)
        all_embeddings = []

        pbar = tqdm(
            loader, 
            desc="Encoding Batches", 
            disable=not show_progress,
            unit="batch"
        )

        self.embedding_model.eval()

        with torch.no_grad():
            for batch in pbar:
                batch_emb = self.embedding_model.encode(
                    batch, 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                )
                all_embeddings.append(batch_emb.cpu())
        
        self.embeddings = torch.cat(all_embeddings, dim=0)
        return self
    
    def filter_by_domain(self, percentile=25):
        """
        Removes texts that are least similar to the domain keywords.
        
        Args:
            percentile (int): The bottom % of texts to drop (0-100).
        """
        if not self.domain_keywords:
            logger.warning("No domain keywords found. Skipping filtering.")
            return self

        if self.embeddings is None:
            raise ValueError("Please run encode() before filtering.")

        logger.info(f"Filtering out the bottom {percentile}th percentile of irrelevant texts...")

        embeddings_np = self.embeddings
        if torch.is_tensor(embeddings_np):
            embeddings_np = embeddings_np.detach().cpu().numpy()
        kw_embeddings_np = self.keyword_embeddings
        if torch.is_tensor(kw_embeddings_np):
            kw_embeddings_np = kw_embeddings_np.detach().cpu().numpy()

        ## score = (average of keyword comparisons + max of keyword comparisons) / 2
        keyword_sim_matrix = cosine_similarity(embeddings_np, kw_embeddings_np)

        # average keyword similarity
        keyword_scores = keyword_sim_matrix.mean(axis=1)

        # max keyword score
        max_scores = keyword_sim_matrix.max(axis=1)

        # combine
        scores = (keyword_scores + max_scores) / 2

        # set threshold value based on input percentile
        threshold = np.percentile(scores, percentile)

        # update data via mask to reflect percentile choice
        keep_mask = scores >= threshold
        original_count = len(self.data)
        self.data = [d for i, d in enumerate(self.data) if keep_mask[i]]
        self.embeddings = embeddings_np[keep_mask]
        self.relevance_scores = scores[keep_mask]

        logger.info(f"Filtered {original_count - len(self.data)} texts. {len(self.data)} remain.")
        
        return self
    
    def fit_clusters(self, n_components=10, min_cluster_size=5, soft_cluster=True):
        """
        Important pipeline step before taxonomy building: Reduce -> Cluster.
        
        Args:
            n_components (int): Target dimensions for UMAP (Default: 5)
            min_cluster_size (int): Smallest grouping to be a cluster (Default: 15)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run encode() first.")
        
        embeddings_to_reduce = self.embeddings
        if torch.is_tensor(embeddings_to_reduce):
            embeddings_to_reduce = embeddings_to_reduce.detach().cpu().numpy()

        self.reduced_embeddings = self.cluster_engine.reduce_dimensions(
            embeddings_to_reduce, 
            n_components=n_components
        )

        self.cluster_labels, self.clustering_model = self.cluster_engine.cluster(
            self.reduced_embeddings, 
            min_cluster_size=min_cluster_size
        )

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        logger.info(f"Found {n_clusters} clusters. Noise points: {list(self.cluster_labels).count(-1)}")

        if soft_cluster:
            logger.info("Soft clustering noise points...")
            self.cluster_labels = self.cluster_engine.soft_cluster(
                self.reduced_embeddings, 
                self.cluster_labels,
                self.clustering_model
            )
            logger.info("Soft clustering complete.")
        
        return self.cluster_labels
    
    def configure_labeling(self, name, definition):
        """Initializes the prompt builder with domain context."""
        self.prompt_builder = PromptBuilder(name, definition)
        return self

    def add_label_example(self, statements, description):
        """Helper to add few-shot examples to the prompt builder."""
        if not hasattr(self, 'prompt_builder'):
            raise ValueError("Run configure_labeling() first to initiztalize the PromptBuilder.")
        self.prompt_builder.add_example(statements, description)
        return self
    
    def label_clusters(self, max_samples=100, top_k_exemplars=True):
        """
        Generates a natural language label for every cluster.
        
        Args:
            max_samples (int): Max number of statements to send to the LLM per cluster.
            top_k_exemplars (bool): If True, picks texts closest to the centroid. 
                                If False, picks random samples.
        """
        if self.cluster_labels is None or self.llm is None:
            raise ValueError("Ensure clusters are fit (fit_clusters) and LLM provider is set (set_llm).")

        unique_clusters = [c for c in np.unique(self.cluster_labels) if c != -1]
        self.taxonomy_labels = {}

        logger.info(f"Labeling {len(unique_clusters)} clusters...")

        for cluster_id in tqdm(unique_clusters, desc="Labeling Clusters"):
            # get cluster members
            indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # select representative statements, either by distance to centroid or by random sampling
            if top_k_exemplars:
                # claculate centroid and get max_samples
                cluster_points = self.reduced_embeddings[indices]
                centroid = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                
                best_indices = indices[np.argsort(distances)[:max_samples]]
                statements = [self.data[i] for i in best_indices]
            else:
                # random sampling with max_samples
                sample_size = min(len(indices), max_samples)
                sample_indices = np.random.choice(indices, sample_size, replace=False)
                statements = [self.data[i] for i in sample_indices]

            prompt = self.prompt_builder.build(statements)
            
            try:
                raw_response = self.llm.generate(prompt)
                clean_label = raw_response.split("Description:")[-1].strip()
                self.taxonomy_labels[cluster_id] = clean_label
            except Exception as e:
                logger.error(f"Failed to label cluster {cluster_id}: {e}")
                self.taxonomy_labels[cluster_id] = f"Cluster_{cluster_id}_(Error)"

        return self.taxonomy_labels
    
    def get_report(self):
        """Returns a list of dictionaries containing the taxonomy results."""        
        report = []
        for cluster_id, label in self.taxonomy_labels.items():
            count = list(self.cluster_labels).count(cluster_id)
            report.append({
                "cluster_id": cluster_id,
                "label": label,
                "size": count
            })
        return pd.DataFrame(report).sort_values("size", ascending=False)
    
    def to_dataframe(self):
        """Returns a DataFrame with the original text and its final taxonomy label."""
        df = pd.DataFrame({
            "text": self.data,
            "cluster_id": self.cluster_labels,
            "relevance_score": getattr(self, 'relevance_scores', None)
        })
        # map IDs to their generated labels
        df["taxonomy_label"] = df["cluster_id"].map(self.taxonomy_labels).fillna("NOISE")
        return df
    
    def consolidate_labels(self, similarity_threshold=0.95):
        """
        Finds semantically identical labels and merges their clusters using an LLM summary.
        """
        if not self.taxonomy_labels:
            raise ValueError("No labels found. Run label_clusters() first.")

        cluster_ids = list(self.taxonomy_labels.keys())
        label_texts = [self.taxonomy_labels[cid] for cid in cluster_ids]
        
        label_embeds = self.embedding_model.encode(label_texts, convert_to_tensor=True)
        
        # compute cross-similarity
        from sentence_transformers import util
        cosine_scores = util.cos_sim(label_embeds, label_embeds).cpu().numpy()

        # find groups to merge
        processed = set()
        merge_groups = []

        for i in range(len(cluster_ids)):
            if i in processed:
                continue
            
            # find all indices j that are highly similar to i
            similar_indices = np.where(cosine_scores[i] >= similarity_threshold)[0]
            similar_indices = [j for j in similar_indices if j != i]

            if similar_indices:
                group = [i] + similar_indices
                merge_groups.append(tuple(cluster_ids[idx] for idx in group))
                processed.update(group)
            else:
                processed.add(i)

        if not merge_groups:
            logger.info("No labels were similar enough to consolidate.")
            return self

        logger.info(f"Consolidating {len(merge_groups)} groups of similar labels...")

        # use LLM to generate a single label for each merged group
        agg_template = """You will be given a list of statements. They express similar meaning. 
    Combine them into one coherent description that captures the essence of all.
    Statements: {}
    Output:::
    Description: """

        processed = set()
        for group in tqdm(merge_groups, desc="Aggregating Labels"):
            valid_group = [cid for cid in group if cid in self.taxonomy_labels and cid not in processed]
            if len(valid_group) < 2:
                continue

            statements = [self.taxonomy_labels[cid] for cid in valid_group]
            prompt = agg_template.format(str(statements))
            
            new_label = self.llm.generate(prompt).split("Description:")[-1].strip()
            
            # update the id
            master_id = valid_group[0]
            self.taxonomy_labels[master_id] = new_label
            
            # update original data
            for other_id in valid_group[1:]:
                self.cluster_labels[self.cluster_labels == other_id] = master_id
                if other_id in self.taxonomy_labels:
                    del self.taxonomy_labels[other_id]
            processed.update(valid_group)

        return self
    
    def build_hierarchy(self, stop_at=10, max_levels=5, soft_cluster=True):
        """
        This is where we build up from the first level!
        I.e., we recursively cluster labels into higher-level categories.
        
        Args:
            stop_at (int): Stop recursing when we have fewer than this many labels.
            max_levels (int): Safety break to prevent infinite loops.
        """
        current_level = 0
        # store the base level (should be built)
        if not self.taxonomy_labels:
            print("Base level does not exist! Please run label_clusters() first.")
        self.levels[0] = self.taxonomy_labels.copy()
        
        while len(self.levels[current_level]) > stop_at and current_level < max_levels:
            logger.info(f"--- Building Taxonomy Level {current_level + 1} ---")
            
            # get previous level
            previous_labels = list(self.levels[current_level].values())
            previous_ids = list(self.levels[current_level].keys())

            if len(previous_labels) < stop_at*2 or current_level == max_levels - 1:
                is_top = True
            else:
                is_top = False

            # embed the labels
            nodes_embeddings = self.embedding_model.encode(previous_labels)
            nodes_embeddings = np.array(nodes_embeddings)

            if nodes_embeddings.ndim == 1:
                nodes_embeddings = nodes_embeddings.reshape(1, -1)
            if len(previous_labels) < 2:
                logger.info("Only one label remains. Stopping hierarchy build.")
                break

            num_nodes = len(previous_labels)
            dynamic_neighbors = min(15, max(2, int(np.sqrt(num_nodes))))

            reduced_nodes = self.cluster_engine.reduce_dimensions(
                nodes_embeddings, 
                n_components=min(5, num_nodes - 1),
                n_neighbors=dynamic_neighbors
            )

            # cluster labels
            new_cluster_ids, level_cluster_model = self.cluster_engine.cluster(
                reduced_nodes, 
                min_cluster_size=3
            )
            
            # soft cluster if desired
            if soft_cluster:
                new_cluster_ids = self.cluster_engine.soft_cluster(
                    nodes_embeddings, new_cluster_ids, level_cluster_model
                )

            # run labeling process
            current_level += 1
            self.levels[current_level] = {}
            self.hierarchy[current_level] = {}

            unique_parents = np.unique(new_cluster_ids)
            if len(unique_parents) == 0:
                logger.warning("No clusters found at this level. Stopping hierarchy build.")
                break

            for p_id in tqdm(unique_parents, desc="Labeling Taxonomy Level {}".format(current_level)):
                child_indices = np.where(new_cluster_ids == p_id)[0]
                child_labels = [previous_labels[i] for i in child_indices]
                
                if is_top:
                    prompt = self.prompt_builder.build_top(child_labels)
                    parent_label = self.llm.generate(prompt).split("Title:")[-1].strip()
                else:
                    prompt = self.prompt_builder.build_mid(child_labels)
                    parent_label = self.llm.generate(prompt).split("Description:")[-1].strip()
                
                self.levels[current_level][p_id] = parent_label
                
                for i in child_indices:
                    child_id = previous_ids[i]
                    self.hierarchy[current_level][child_id] = p_id

        logger.info(f"Hierarchy complete. Reached {current_level} levels.")
        return self
    
    def _get_alphabet_code(self, n, uppercase=True):
        """Converts an integer to an Excel-style alphabet string (A, B, ..., AA, AB...)."""
        chars = string.ascii_uppercase if uppercase else string.ascii_lowercase
        result = ""
        while n >= 0:
            result = chars[n % 26] + result
            n = (n // 26) - 1
        return result
    
    def to_hierarchy_dataframe(self):    
        """
        Export the complete built taxonomy to a DataFrame with infinite-depth coding.
        Code format example: TB.A.0.a.1.b.10042
        """
        if hasattr(self, 'relevance_scores'):
            global_ranks = ((-self.relevance_scores).argsort().argsort())
            rank_lookup = {i: rank for i, rank in enumerate(global_ranks)}
        else:
            # Fallback if filtering wasn't run
            rank_lookup = {i: i for i in range(len(self.data))}

        level_indices = sorted(self.levels.keys(), reverse=True)
        root_level = level_indices[0]
        results = []

        def get_children(parent_id, current_lvl):
            if current_lvl <= 0: return []
            return [child for child, p in self.hierarchy[current_lvl].items() if p == parent_id]

        def walk(parent_id, current_lvl, current_code, current_path_labels):
            label = self.levels[current_lvl][parent_id]
            new_path_labels = current_path_labels + [label]
            depth = root_level - current_lvl
            
            # Leaf Level (Statements)
            if current_lvl == 0:
                indices = np.where(self.cluster_labels == parent_id)[0]
                for i in indices:
                    stmt = self.data[i]
                    # calculate relevance score suffix
                    global_ordinal = rank_lookup[i]
                    final_code = f"{current_code}{10000+global_ordinal}"
                    
                    row = {"Code": final_code, "Statement": stmt}
                    
                    for d, path_label in enumerate(new_path_labels):
                        row[f"Level_{d}_Label"] = path_label
                        
                    results.append(row)
                return

            # Internal Nodes (Recursion)
            children = get_children(parent_id, current_lvl)
            for idx, child_id in enumerate(children):
                if depth % 3 == 0:
                    segment = self._get_alphabet_code(idx, uppercase=True)
                elif depth % 3 == 1:
                    segment = str(idx)
                else:
                    segment = self._get_alphabet_code(idx, uppercase=False)
                
                next_code = f"{current_code}{segment}."
                walk(child_id, current_lvl - 1, next_code, new_path_labels)

        # start Recursion
        for root_idx, root_id in enumerate(self.levels[root_level]):
            root_seg = self._get_alphabet_code(root_idx, uppercase=True)
            initial_code = f"TB.{root_seg}."
            walk(root_id, root_level, initial_code, [])

        return pd.DataFrame(results)