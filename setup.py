from setuptools import setup, find_packages

setup(
    name="TaxonomyBuilder",
    version="0.1.0",
    author="Stephen Meisenbacher",
    description="A robust tool for the automated building of hierarchical taxonomies using LLMs and GPU-accelerated clustering.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "transformers",
        "torch",
        "sentence-transformers",
        "openai",
        "tqdm",
        "scikit-learn",
        "hdbscan",
        "tensorflow",
        "umap-learn",
        "openai",
        "google-genai",
        "anthropic"
    ],
    extras_require={
        "gpu": [
            "cuml",
        ],
        "dev": [
            "pytest",
            "black",
            "ruff",
            "ipykernel"
        ]
    },
    python_requires=">=3.8",
)