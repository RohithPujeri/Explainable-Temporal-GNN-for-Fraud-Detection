# An Explainable Temporal Graph-Based Framework for Robust Fraud Detection in Financial Transaction Networks

This repository contains a Jupyter notebook (`rohit-final-updated.ipynb`) implementing and evaluating fraud-detection models on two large-scale financial transaction graph datasets: **Elliptic++** (Bitcoin transactions) and **DGraphFin** (financial social network). The notebook combines traditional ML baselines, static and temporal Graph Neural Networks (GNNs), and a suite of explainability techniques, with statistical robustness checks across multiple runs.

## Datasets

| Dataset | Type | Role | Source |
|---|---|---|---|
| **Elliptic++** | Bitcoin transaction + wallet (actor) graph | Primary dataset | [git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) (auto-downloaded in Cell 3) |
| **DGraphFin** | Financial social network graph | Secondary / cross-dataset generalisation | Expects `dgraphfin.npz`, e.g. via a Kaggle dataset mount at `/kaggle/input/.../dgraphfin.npz` |

The notebook was authored for the **Kaggle** environment (paths reference `/kaggle/working` and `/kaggle/input`). To run elsewhere, update `TX_DIR`, `AC_DIR`, and `DGRAPHFIN_PATH` in Cell 3 accordingly. Elliptic++ transaction/actor CSVs are downloaded automatically from GitHub; DGraphFin's `dgraphfin.npz` must be supplied manually (it is not auto-downloaded).

## Notebook Structure

- **Part A** — Environment setup and data download
- **Part B** — Data loading (Elliptic++ transactions, actors/wallets, DGraphFin)
- **Part C** — EDA: Elliptic++ transactions (class balance, temporal patterns, feature distributions, graph topology, subgraph viz)
- **Part D** — EDA: DGraphFin (label distribution, edge types, feature comparison)
- **Part E** — EDA: Wallet-level (dual-level) analysis
- **Part F** — Preprocessing (temporal train/val/test split, scaling)
- **Part G** — Traditional ML baselines with hyperparameter tuning (Random Forest, XGBoost) on both datasets
- **Part H** — Graph Neural Networks: static GCN, static GAT, and a novel `TemporalSnapshotGCN` (GCN + learnable time embedding + GRU) on Elliptic++, plus GAT on DGraphFin for cross-dataset generalisation
- **Part I** — GNN embedding visualisation (t-SNE) comparing static vs. temporal embeddings
- **Part J** — 4-way ablation study (tabular vs. graph, random vs. temporal split, static vs. temporal model)
- **Part K** — Per-timestep temporal validation across all models, plus comparison against published Elliptic/Elliptic++ benchmarks
- **Part L** — Explainability: SHAP (XGBoost), GNNExplainer (GCN), GAT attention-weight analysis, degree/PageRank centrality and neighbourhood homophily, and quantitative faithfulness/sparsity/agreement metrics
- **Part M** — Statistical robustness across 5 random seeds with paired Wilcoxon significance tests
- **Part N** — Comprehensive results tables, comparison charts, confusion matrices, ROC/PR curves, and a final summary answering the research questions

## Key Modelling Contributions

- **`TemporalSnapshotGCN`**: a genuinely temporal GNN combining a learnable per-timestep embedding, a spatial GCN pathway, and a GRU over timestep summaries, designed to capture the evolution of fraud patterns over time (rather than treating the graph as static).
- **Concept-drift quantification**: explicit ablation isolating the cost of temporal (vs. random) train/test splitting and the extent to which the temporal model recovers that cost.
- **Quantitative explainability evaluation**: permutation-based faithfulness testing, sparsity analysis (F1 retained by top-K features), and cross-method agreement between SHAP and GNNExplainer feature rankings.

## How to Run

1. Open the notebook in a Kaggle notebook environment (or adapt the paths in Cell 3 for local/Colab use).
2. Ensure a GPU runtime is available (CUDA is used automatically if present; the notebook falls back to CPU otherwise — training will be considerably slower on CPU, especially the temporal GNN and the 5-seed robustness study).
3. Run all cells top to bottom. Package installation (`torch-geometric`, `shap`) happens in Cell 1.
4. Provide the DGraphFin `.npz` file at the path expected in Cell 3, or adjust `DGRAPHFIN_PATH`.

See `requirements.txt` for the Python package dependencies (install with `pip install -r requirements.txt`). Note that `torch` and `torch-geometric` may require environment-specific installation (e.g. matching CUDA version) — see the [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) install instructions if the plain pip install fails.

## Outputs

Running the notebook produces (inline, not saved to disk by default):
- EDA plots for both datasets
- Trained RF/XGBoost/GCN/GAT/TemporalSnapshotGCN models and their metrics (Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR)
- Ablation and temporal-validation charts
- SHAP, GNNExplainer, and attention-based explainability visualisations
- A comprehensive results table and final summary printed at the end of the notebook
