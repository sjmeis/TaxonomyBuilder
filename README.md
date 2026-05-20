<div align="center">

  # TaxonomyBuilder: Building Domain-specific Taxonomies from the Ground Up

  [![PyPI version](https://img.shields.io/pypi/v/taxonomybuilder.svg)](https://pypi.org/project/taxonomybuilder/)
  [![License](https://img.shields.io/github/license/sjmeis/TaxonomyBuilder.svg)](https://github.com/sjmeis/TaxonomyBuilder/blob/main/LICENSE)

</div>

A robust, high-performance Python framework for transforming massive, unstructured text datasets into structured, hierarchical taxonomies. Originally published as part of the CustomNLP4U 2026 paper: *Building a Custom Taxonomy of AI Skills and Tasks from the Ground Up with Job Postings*

## 🚀 Overview
**TaxonomyBuilder** bridges the gap between raw data and structured knowledge. It leverages **Sentence Transformers** for semantic representation, **RAPIDS (cuML)** for GPU-accelerated clustering (if applicable), and **LLMs (user-defined: OpenAI, Anthropic, etc.)** for natural language categorization and recursive hierarchy building. The end product is a semantically meaningful taxonomy built from the ground up!


## 🛠 Installation

```bash
# Core installation
pip install taxonomybuilder

# For GPU acceleration (Requires CUDA)
pip install taxonomybuilder[gpu]
```


## 📖 Quick Start: The Full Pipeline
Using **TaxonomyBuilder** is straightforward and simple, and highly automated! Nevertheless, you are given the opportunity to inject your *domain expertise*.

Here is how to go from a list of raw strings to a multi-leveled hierarchy in minutes.

```python
from TaxonomyBuilder import TaxonomyBuilder

# Step 1: Initialize (with GPU support) - make sure to specify your preferred sentence embedding model!
tb = TaxonomyBuilder(embedding_model_name="all-MiniLM-L6-v2", use_gpu=True)

# Step 2: Setup your preferred LLM Provider (currently supported: OpenAI, Google, Anthropic)
tb.set_llm(provider_name="openai", api_key="your-api-key", model_endpoint="gpt-4o-mini")

# Step 3: Ingest and Filter your Data. You are also encouraged to provide keywords to "anchor" the domain and filter out noise.
texts = ["Automate cloud backups to...", "Debug python scripts for...", "Develop machine learning solutions...", "Fix the broken coffee machine...", ...]
keywords = ["Software Engineering", "DevOps", "Programming"]

(tb.ingest_data(texts, keywords=keywords)
   .encode(batch_size=16)
   .filter_by_domain(percentile=25)) # Drop 25% least relevant texts, according to your defined domain

# Step 4: Build the Bottom Level of your Taxonomy via Clustering (soft cluster = include "noise" points)
tb.fit_clusters(n_components=10, min_cluster_size=5, soft_cluster=True)

# Step 5: Configure Labeling and Add Examples
(tb.configure_labeling(name="Technical Task", definition="A specific action performed by an engineer.")
   .add_label_example(["Fixing a syntax error", "Refactoring a loop", "Making sense of spaghetti code"], "Code Debugging"))

# Step 6: Generate Labels & Consolidate
tb.label_clusters()
tb.consolidate_labels(similarity_threshold=0.95) # this removes redundant cluster labels - optional!

# Step 7: Build the Hierarchical Taxonomy
tb.build_hierarchy(stop_at=10, max_levels=5) # Stops when the top level has 10 or fewer categories, OR when five levels have been built

# Step 8: Export Results (also check get_report and to_dataframe for exporting base level results)
df = tb.to_hierarchy_dataframe()
df.to_csv("taxonomy_results.csv")
```

## 🧠 Key Features

### ⚡ GPU-Accelerated Clustering
If a compatible GPU is detected, **TaxonomyBuilder** automatically uses cuML for UMAP and HDBSCAN, allowing you to cluster millions of documents in seconds rather than hours.

### 🎯 Two-Fold Domain Filtering
Our relevance scoring ensures your taxonomy isn't polluted by "off-topic" data. We score every text based on:

 1. Mean Similarity: Average distance to all keywords.
 2. Max Similarity: Highest match to any single keyword.

### 🌲 Recursive Hierarchical Logic
Unlike flat clustering, **TaxonomyBuilder** re-clusters the labels of the previous level to create a parent-child tree. It automatically switches prompts at the "Top Level" to ensure broad categories (e.g., "Operations") aren't labeled as granular tasks (e.g., "Password Reset").


## 📁 Project Structure

```plaintext
TaxonomyBuilder/
├── src/TaxonomyBuilder/
   ├── core.py           # Main Logic
   ├── clustering.py     # GPU/CPU Dispatcher (UMAP/HDBSCAN)
   ├── data.py           # PyTorch Dataset & Dataloaders
   ├── llm.py            # LLM Provider Interface
   └── prompt_utils.py   # Dynamic Template & Few-shot Logic
```

## 💡 Tips and Hints
 - Domain Context: We highly recommend seeding the process with domain-specific keywords. Additionally, make sure to add example for the labeling process (max. 3)!
 - Memory Management: If you have a massive dataset, set `batch_size` lower in `.encode()` to avoid out-of-memory issues.
 - Consolidation: If your taxonomy has too many "similar" sounding categories, lower the `similarity_threshold` in `consolidate_labels` to group more labels together.
 - The "Noise": Any text marked as -1 by HDBSCAN will be labeled as noise (-1) unless `soft_cluster=True` is used. It is up to you whether you want to include these points or not!

## 
If you use or build upon `TaxonomyBuilder`, we would appreciate it if you cited the original work:

```
Bib entry coming soon!
```