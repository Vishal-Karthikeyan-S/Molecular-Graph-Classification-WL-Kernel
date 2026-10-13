# Molecular Graph Classification: Weisfeiler–Lehman Subtree Kernel vs. GNNs on NCI1

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

A complete, reproducible, research-grade machine learning project for **Machine Learning with graphs (Course Code: 23CSE479, 30-mark academic assessment)** implementing the **1-dimensional Weisfeiler–Lehman (WL) Subtree Kernel from scratch** and evaluating it against conventional graph-level feature baselines and Graph Convolutional Networks (GCN) on the **NCI1 molecular graph classification benchmark**.

---

## 1. Project Overview & Abstract

In chemoinformatics and computational drug discovery, chemical compounds are modeled as graphs with atoms as nodes and chemical bonds as edges. Graph classification requires learning representations that are invariant to graph isomorphisms while capturing both discrete atomic properties and complex bonding topologies.

This project implements:
1. **1-WL Subtree Kernel from Scratch**: Deterministic neighborhood color refinement, multiset compression, graph-level subtree histograms, and exact normalized Gram matrix generation without using `GraKeL` or other prebuilt kernel libraries.
2. **Precomputed Kernel SVM**: Scikit-Learn `SVC` supporting decoupled train-train and test-train Gram matrices to eliminate data leakage.
3. **Conventional Graph Feature Baseline**: Non-structural graph descriptors (density, node/edge counts, degree statistics, atom histograms) with an RBF SVM.
4. **Graph Convolutional Network (GCN)**: 3-layer neural message passing baseline with global mean pooling, dropout, and early stopping.
5. **Interactive Streamlit Web Application**: An exploratory dashboard allowing graph visualization, interactive subtree exploration, empirical metrics comparison, and single-compound inference.

---

## 2. Experimental Benchmark Results on NCI1

All models were evaluated under identical, reproducible stratified splits (Train: 70%, Val: 10%, Test: 20%) on the NCI1 dataset:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Macro-F1 | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph Feature Baseline + RBF SVM** | 0.6625 | 0.6757 | 0.6250 | 0.6494 | 0.6620 | 0.7125 | 0.042s |
| **WL Subtree Kernel ($h=0$) + SVM** | 0.5250 | 0.5188 | 0.6917 | 0.5929 | 0.5114 | 0.5823 | 0.021s |
| **WL Subtree Kernel ($h=1$) + SVM** | 0.6583 | 0.6900 | 0.5750 | 0.6273 | 0.6559 | 0.7078 | 0.042s |
| **WL Subtree Kernel ($h=2$) + SVM** | 0.7125 | 0.7576 | 0.6250 | 0.6849 | 0.7103 | 0.7685 | 0.062s |
| **WL Subtree Kernel ($h=3$) + SVM** | **0.7500** | **0.7885** | **0.6833** | **0.7321** | **0.7489** | **0.7881** | **0.083s** |
| **3-Layer GCN (PyG Neural Baseline)** | 0.6708 | 0.6783 | 0.6500 | 0.6638 | 0.6707 | 0.7594 | 41.307s |

### Key Theoretical & Empirical Insights:
1. **Importance of Neighborhood Depth**: At $h=0$ (bag-of-atoms), accuracy is only $52.5\%$. At $h=1$ (1-hop bond environments), accuracy surges to $65.8\%$. At $h=3$ (3-hop functional groups), accuracy reaches **$75.0\%$**.
2. **Kernel vs. GNN Efficiency**: The WL subtree kernel achieves +7.92% higher test accuracy than the 3-layer GCN while training in **0.08 seconds vs. 41.31 seconds** (~500x faster).
3. **Theoretical Alignment**: In alignment with Morris et al. (2019) and Xu et al. (2019), standard GCNs with mean aggregation are bounded in expressive power by the 1-WL isomorphism test. On discrete molecular benchmarks, exact subtree enumeration with convex SVM optimization provides a superior inductive bias compared to gradient descent on parameterized neural layers.

---

## 3. Mathematical Formulation of the WL Subtree Kernel

The 1-WL subtree kernel (Shervashidze et al., 2011) proceeds as follows:

1. **Iteration 0 (Initialization)**: Assign initial categorical label $\ell_0(v)$ based on the atom type:
   ```math
   \ell_0(v) = \text{atom\_type}(v), \quad \forall v \in V
   ```
   Count label frequencies across graph $G$:
   ```math
   \phi_0(G) = \begin{bmatrix} c(G, \ell_1) & c(G, \ell_2) & \dots & c(G, \ell_{|\Sigma_0|}) \end{bmatrix}^\top
   ```

2. **Iteration $t \in \{1, \dots, h\}$ (Neighborhood Relabeling)**:
   - For every node $v$, collect the multiset of neighbor labels:
     ```math
     \mathcal{M}_t(v) = \{\!\{ \ell_{t-1}(u) : u \in \mathcal{N}(v) \}\!\}
     ```
   - Sort $\mathcal{M}_t(v)$ deterministically in ascending order: $(s_1, s_2, \dots, s_k)$.
   - Compress the tuple signature into a unique integer identifier:
     ```math
     \ell_t(v) = \text{hash}\Big( t, \ell_{t-1}(v), (s_1, s_2, \dots, s_k) \Big)
     ```
   - Extract histogram of newly discovered labels: $\phi_t(G)$.

3. **Concatenated Feature Vector**:
   ```math
   \phi(G) = \begin{bmatrix} \phi_0(G)^\top & \phi_1(G)^\top & \dots & \phi_h(G)^\top \end{bmatrix}^\top \in \mathbb{R}^D
   ```

4. **Normalized Kernel Matrix**:
   ```math
   K_{\text{norm}}(G_i, G_j) = \frac{\langle \phi(G_i), \phi(G_j) \rangle}{\|\phi(G_i)\|_2 \cdot \|\phi(G_j)\|_2}
   ```

---

## 4. Repository Structure

```text
MLG ass3/
├── app.py                      # Interactive Streamlit dashboard
├── README.md                   # Full documentation & empirical summary
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration parameters
├── pytest.ini                  # Pytest configuration
├── data/
│   ├── README.md               # Dataset documentation
│   └── NCI1/                   # Downloaded NCI1 raw graphs
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data downloading, validation & stats
│   ├── preprocessing.py        # MolecularGraph structures & validation
│   ├── wl_kernel.py            # 1-WL Subtree Kernel from scratch
│   ├── wl_svm.py               # Precomputed Kernel SVM classifier
│   ├── baseline.py             # Conventional graph feature baseline
│   ├── gcn_model.py            # 3-layer GCN model & training loop
│   ├── experiments.py          # End-to-end benchmark suite
│   ├── analysis.py             # Research evaluation & reporting
│   └── visualization.py        # Visualizations & graph plots
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing
│   └── test_wl_kernel.py       # Unit tests for WL kernel properties
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive Jupyter analysis
├── scripts/
│   ├── download_data.py        # Dataset downloader CLI
│   ├── run_experiments.py      # Benchmark runner CLI
│   └── generate_report.py      # Report generator CLI
├── models/
│   └── best_gcn.pt             # Saved GCN checkpoint
├── results/
│   ├── metrics/                # Benchmark JSON files & stats
│   └── plots/                  # Generated publication-quality figures
└── reports/
    └── project_report.md       # Comprehensive 30-mark research report
```

---

## 5. Quickstart & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests
```bash
pytest
```
Expected output: **7 passed** verifying label updates, C3 vs P3 isomorphism, kernel symmetry, Mercer positive semi-definiteness, and test-train dimensions.

### Step 3: Run the Benchmark Pipeline
```bash
python scripts/run_experiments.py --subset 1200 --seed 42
```
Or to run on all 4,110 graphs:
```bash
python scripts/run_experiments.py --subset 0
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore interactive molecular graphs, hand-crafted WL subtree relabeling, and model comparisons.

---

## 6. Generated Visualizations

All figures are automatically saved in `results/plots/`:
- `dataset_distribution.png`: Class balance and atom count statistics.
- `model_comparison.png`: Accuracy, F1, and ROC-AUC comparisons across all models.
- `wl_depth_effect.png`: Scaling behavior of accuracy and feature dimensions as $h$ increases from 0 to 3.
- `runtime_comparison.png`: Wall-clock execution time comparison.
- `confusion_matrices.png`: Confusion matrices side-by-side.
- `gcn_training_curves.png`: Training and validation loss curves for GCN.

---

## 7. Citation & References

```bibtex
@article{shervashidze2011weisfeiler,
  title={Weisfeiler-lehman graph kernels},
  author={Shervashidze, Nino and Schweitzer, Pascal and Van Leeuwen, Erik Jan and Mehlhorn, Kurt and Borgwardt, Karsten M},
  journal={Journal of Machine Learning Research},
  volume={12},
  number={9},
  pages={2539--2561},
  year={2011}
}
```
