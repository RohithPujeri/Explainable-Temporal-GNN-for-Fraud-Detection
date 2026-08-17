# An Explainable Temporal Graph-Based Framework for Robust Fraud Detection in Financial Transaction Networks

An end-to-end, reproducible study comparing traditional machine learning baselines against
static and temporal Graph Neural Networks (GNNs) for fraud detection on two large-scale
financial transaction graphs, with a quantitative explainability evaluation.

The complete pipeline — data acquisition, exploratory analysis, preprocessing, modelling,
ablation, temporal validation, explainability and statistical robustness — lives in a single
notebook: **`Rohith-code.ipynb`**.

---

## 1. Repository Structure

```
.
├── Rohith-code.ipynb     # Complete pipeline (Parts A–N), with saved outputs
└── README.md             # This file
```

The notebook is committed **with its executed outputs**, so all figures, metrics and tables
can be inspected on GitHub without re-running the pipeline.

---

## 2. Datasets

| Dataset | Role | Nodes | Edges | Features | Labels |
|---|---|---|---|---|---|
| **Elliptic++** (Bitcoin transactions) | Primary | 203,769 | 234,355 | 182 (93 local + 89 aggregated) | Illicit / Licit / Unknown, 49 timesteps |
| **Elliptic++** (Actors / wallets) | Dual-level EDA | 822,942 wallet labels | 477,117 Addr→Tx, 837,124 Tx→Addr | 56 | Wallet-level classes |
| **DGraphFin** (financial social network) | Secondary / cross-dataset | 3,700,550 | 4,300,999 | 17 | Fraud / Normal + 2 background classes |

Class imbalance on labelled Elliptic++ transactions is **9.2 : 1** (Licit : Illicit).

### Obtaining the data

**Elliptic++** is downloaded automatically by the notebook (Part A, Cell 3) directly from the
official repository, which avoids the Git-LFS pointer problem that occurs when the CSVs are
re-uploaded to other hosts:

```
https://github.com/git-disl/EllipticPlusPlus
```

**DGraphFin** (`dgraphfin.npz`) must be provided manually. Place it in the `./data/` directory
created by Part A; the notebook checks that path first and then falls back to a recursive search
under `./data/`, so any subdirectory works. To use a different location, edit `DGRAPHFIN_PATH`
in Cell 3. DGraphFin is available from the DGraph benchmark:

```
https://dgraph.xinye.com/dataset
```

---

## 3. Environment

The notebook was developed and executed on **Python 3.12 with a CUDA-enabled NVIDIA GPU
(16 GB VRAM)**. A GPU is strongly recommended: Part M alone trains 15 GNNs.
The code falls back to CPU automatically, but the full run becomes impractically slow.

```bash
pip install torch-geometric shap
```

Core dependencies (most are pre-installed in standard deep-learning images):

```
torch, torch-geometric, xgboost, scikit-learn, shap,
numpy, pandas, matplotlib, seaborn, networkx, scipy
```

Full disk requirement is roughly **1.6 GB** for the downloaded Elliptic++ CSVs
(`txs_features.csv` alone is 663 MB).

---

## 4. Notebook Structure

| Part | Contents |
|---|---|
| **A** | Environment setup, package install, Git-LFS-safe Elliptic++ download |
| **B** | Data loading (transactions, actors/wallets, DGraphFin) with header auto-detection |
| **C** | EDA — Elliptic++ transactions: class distribution, temporal fraud patterns, feature distributions, correlation, graph topology, subgraph visualisation |
| **D** | EDA — DGraphFin: label distribution, edge types, edge timestamps, feature separability |
| **E** | Wallet-level (dual-level) analysis and cross-level edge statistics |
| **F** | Preprocessing: dynamic temporal split (60/20/20 by timestep) and standardisation |
| **G** | ML baselines with grid-search hyperparameter tuning (Random Forest, XGBoost) |
| **H** | GNNs: static GCN, static GAT, and the proposed TemporalSnapshotGCN |
| **H-2** | DGraphFin cross-dataset generalisation (explicitly *not* framed as temporal validation) |
| **I** | t-SNE visualisation of static vs temporal GCN embeddings |
| **J** | Four-way ablation study |
| **K** | Per-timestep temporal validation of all models + comparison against published baselines |
| **L** | Explainability: SHAP, GNNExplainer, GAT attention, centrality, homophily, and a quantitative faithfulness/sparsity/agreement evaluation |
| **M** | Statistical robustness: 5 seeds per GNN + paired Wilcoxon signed-rank tests |
| **N** | Comprehensive results table, confusion matrices, ROC/PR curves, research-question summary |

---

## 5. Models

**Traditional baselines**
- **Random Forest** — grid over `n_estimators ∈ {100, 200, 300}` × `max_depth ∈ {10, 15, 20}`, `class_weight='balanced'`.
- **XGBoost** — grid over `n_estimators` × `max_depth ∈ {6, 8, 10}` × `learning_rate ∈ {0.05, 0.1}`, `scale_pos_weight` set from the training imbalance, `aucpr` eval metric.

Hyperparameters are selected on the validation split; the model is then refit on train + validation and evaluated once on the held-out test period.

**Graph neural networks** (input features augmented with in-degree, out-degree and total degree)
- **FraudGCN** — 3-layer GCN with batch normalisation and dropout.
- **FraudGAT** — 2-layer multi-head GAT (4 heads) exposing attention weights for explainability.
- **TemporalSnapshotGCN** *(the proposed contribution)* — a learnable temporal positional
  embedding concatenated to node features, a two-layer GCN spatial pathway, a 2-layer **GRU**
  over per-timestep graph summaries, and a **learned sigmoid gate** that fuses the spatial and
  temporal representations per node before classification.

**Training protocol**
- Class weighting uses `min(√(n₀/n₁), 5.0)` — the geometric mean between full inverse-frequency
  weighting (which over-predicts the minority class) and no weighting, capped to keep training
  stable on the extremely imbalanced DGraphFin graph.
- Adam (`lr=0.01`, `weight_decay=5e-4`) with `ReduceLROnPlateau`.
- Model selection on the **validation mask**; the test mask is evaluated **once**.
- Decision threshold chosen by **Youden's J** on validation (clipped to `[0.2, 0.7]`), with both
  default-0.5 and tuned-threshold F1 reported for transparency.

---

## 6. Results

### Elliptic++ — strict temporal split (train ≤ ts29, val ≤ ts39, test > ts39)

Train 26,381 · Validation 8,999 · Test 11,184. Fraud rate: train 10.88 %, val 11.53 %, **test 5.69 %**.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.9744 | 0.9654 | 0.5708 | 0.7174 | 0.9425 | 0.7513 |
| **XGBoost** | **0.9753** | 0.9245 | 0.6164 | **0.7396** | 0.9374 | **0.7713** |
| GCN | 0.8627 | 0.2012 | 0.4764 | 0.2829 | 0.7406 | 0.2678 |
| GAT | 0.6785 | 0.1107 | 0.6619 | 0.1897 | 0.7418 | 0.1255 |
| TemporalSnapshotGCN | 0.7791 | 0.1502 | 0.6195 | 0.2418 | **0.7762** | 0.3410 |

### DGraphFin — predefined (non-temporal) masks

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.6604 | 0.0235 | 0.6367 | 0.0453 | 0.7167 | 0.0264 |
| XGBoost | 0.6026 | 0.0226 | 0.7188 | 0.0438 | 0.7163 | 0.0270 |
| GAT | 0.7484 | 0.0158 | 0.3074 | 0.0300 | 0.5590 | 0.0151 |

### Ablation study

| Configuration | F1 | Interpretation |
|---|---|---|
| A1 — XGBoost, tabular features | 0.7396 | No graph structure |
| A2 — GCN, **random** split | 0.8141 | Graph structure adds **+0.0745** F1 under i.i.d. conditions |
| A3 — GCN, **temporal** split | 0.2829 | Concept drift costs **−0.5312** F1 |
| A4 — TemporalSnapshotGCN, temporal split | 0.2418 | Temporal modelling does not recover the gap (−0.0411 vs A3) |

### Statistical robustness (5 seeds: 42, 123, 456, 789, 1024)

| Model | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|
| GCN | 0.3010 ± 0.0278 | 0.7473 ± 0.0137 | 0.3062 ± 0.0436 |
| GAT | 0.1938 ± 0.0168 | 0.7638 ± 0.0206 | 0.1482 ± 0.0234 |
| TemporalSnapshotGCN | 0.2223 ± 0.0332 | 0.7595 ± 0.0118 | **0.3365 ± 0.0361** |

Paired Wilcoxon signed-rank tests: TemporalGCN vs GCN `p = 0.0625`; TemporalGCN vs GAT
`p = 0.1875` — neither significant at α = 0.05. With n = 5 the minimum achievable p-value is
≈ 0.06, so these are reported as **indicative trends, not formal significance claims**.

### Explainability

- **Faithfulness** — randomly permuting the top-10 SHAP features drops XGBoost F1 by **0.4595**, confirming the attributions identify genuinely predictive features (permutation is used rather than zeroing, so the marginal distribution is preserved).
- **Sparsity** — a small subset of features retains most of the model's F1; the full incremental curve (top-5 → top-100) is printed in Part L.
- **Cross-explainer agreement** — overlap between SHAP top-20 and GNNExplainer top-20 is reported; partial overlap is expected because XGBoost and GCN learn different representations.
- **Attention** — GAT attention is analysed at the **edge** level and deliberately not compared feature-to-feature with SHAP/GNNExplainer, since the granularities differ.
- **Fraud homophily** — mean illicit-neighbour ratio is **0.5341** for illicit nodes vs **0.0058** for licit nodes, a strong structural signal that motivates graph-based methods.

---

## 7. Honest Discussion of the Findings

The headline result is **negative for GNNs under realistic evaluation, and this is reported deliberately rather than hidden**:

1. **Graph structure genuinely helps — but only under i.i.d. evaluation.** GCN beats XGBoost by +0.075 F1 on a random split (A2 vs A1). Much of the GNN literature on Elliptic reports exactly this setting.
2. **A strict temporal split reverses the conclusion.** The same GCN loses 0.53 F1 when the split respects time (A3), while the tabular models remain strong. The test period has a fraud rate roughly half that of the training period (5.69 % vs 10.88 %), so message passing over an evolving graph degrades far more sharply than per-node feature models.
3. **The proposed temporal model helps on ranking, not on thresholded F1.** TemporalSnapshotGCN achieves the best GNN AUC-ROC (0.7762) and the best GNN AUC-PR (0.3410, vs 0.2678 for static GCN — a relative improvement of ~27 %), but its F1 remains below static GCN and the difference across seeds is not statistically significant. The reasonable reading is that temporal context improves the *ordering* of suspicious nodes while calibration under drift remains unresolved.
4. **DGraphFin is hard for every method here.** All F1 scores are below 0.05 at a ~1.3 % fraud rate. AUC-ROC around 0.72 for the tree models shows some signal exists, but no configuration in this study is deployment-viable on that graph.

Read together, these results argue that **evaluation protocol matters more than architecture** in this problem: papers reporting strong GNN gains on Elliptic under random splits may not transfer to production, where models are always applied to future transactions.

---

## 8. Comparison Against Published Baselines

Part K compares against published numbers for context:

| Method | F1 | AUC-ROC | Source |
|---|---|---|---|
| GCN | 0.55 | 0.85 | Weber et al., KDD 2019 Workshop |
| Random Forest | 0.60 | 0.87 | Weber et al., KDD 2019 Workshop |
| EvolveGCN | 0.58 | 0.87 | Pareja et al., AAAI 2020 |
| Skip-GCN | 0.68 | 0.92 | Elmougy & Liu, KDD 2023 |
| XGBoost (this work) | 0.7396 | 0.9374 | — |

**This comparison is approximate, not like-for-like.** Weber et al. and EvolveGCN use the
original Elliptic dataset (166 features); this work uses Elliptic++ (182 features plus a wallet
layer), and split boundaries differ across papers. The table is included for orientation only
and no claim of state-of-the-art is made.

---

## 9. Reproducibility and Known Limitations

**Reproducibility**
- Global seed `SEED = 42` for NumPy, PyTorch and CUDA; the robustness study sweeps five seeds explicitly.
- Data acquisition is scripted, so no manual file placement is required for Elliptic++.
- Splits are derived dynamically from the observed timestep range rather than hard-coded, so the notebook does not silently break if the dataset version changes.

**Limitations, stated openly**
- **Structural feature leak (mild).** Degree features are computed on the *full* graph, including test-period edges. Degree is not a label signal, so the leak is mild, but it is a leak and is flagged in Part F of the notebook.
- **Wallet features are not fully loaded.** `wallets_features.csv` (578 MB) is skipped to preserve memory, so the dual-level analysis is descriptive rather than a trained wallet-level model.
- **n = 5 seeds** limits statistical power, as noted above.
- **DGraphFin masks are not temporal**, so Part H-2 tests cross-dataset generalisation only — no temporal robustness claim is made from it.
- GNNExplainer is wrapped in a `try/except`; on some `torch-geometric` versions it may be skipped, and the notebook continues without it.

---

## 10. Research Questions Addressed

- **RQ1 — Graph construction.** Both datasets are converted to PyTorch Geometric graphs (203,769 nodes / 234,355 edges and 3.7 M nodes / 4.3 M edges respectively), augmented with three structural degree features and per-timestep snapshot processing.
- **RQ2 — Do graph methods beat traditional ML?** Yes under a random split (+0.075 F1), no under a temporal split (−0.46 F1 vs XGBoost). Concept drift, quantified at 0.53 F1, is the dominant factor.
- **RQ3 — Are the explanations trustworthy?** Yes on faithfulness (0.46 F1 drop under permutation of top-10 features) and sparsity; cross-explainer agreement is partial and expected to be, and strong fraud homophily (0.534 vs 0.006) independently corroborates the structural signal.

---

## 11. References

1. Weber, M. et al. *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD 2019 Workshop on Anomaly Detection in Finance.
2. Pareja, A. et al. *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs.* AAAI 2020.
3. Elmougy, Y. and Liu, L. *Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics.* KDD 2023.
4. Kipf, T. N. and Welling, M. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
5. Veličković, P. et al. *Graph Attention Networks.* ICLR 2018.
6. Lundberg, S. M. and Lee, S.-I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
7. Ying, Z. et al. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
8. Huang, X. et al. *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection.* NeurIPS 2022 Datasets and Benchmarks.

---

## 12. How to Run

1. Open `Rohith-code.ipynb` in any Jupyter environment with a CUDA-enabled GPU.
2. Place `dgraphfin.npz` in `./data/`, or edit `DGRAPHFIN_PATH` in Part A to point at your copy.
3. Run all cells in order. Parts A–B download and load ~1.6 GB of data; Part M is the longest stage (15 GNN training runs).
4. All data paths are relative to `DATA_ROOT = './data'`; change `DATA_ROOT`, `TX_DIR` or `AC_DIR` in Part A to relocate the downloads.

---

## 13. Licence and Data Usage

Code in this repository may be released under the licence of your choice (MIT is a common
default — add a `LICENSE` file to make the choice explicit). The datasets are **not**
redistributed here; both remain subject to the licences and terms of their original providers
(the Elliptic++ repository and the DGraph benchmark). Please cite the original dataset papers
in any derived work.
