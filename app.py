"""
Streamlit Web Dashboard for Molecular Graph Classification.
Provides interactive exploration for:
1. Overview: Problem formulation & theoretical architecture
2. Dataset Explorer: NCI1 metrics, graph stats, and interactive molecular visualizer
3. WL Kernel Explorer: Subtree relabeling step-by-step on hand-crafted & real graphs
4. Model Comparison: Empirically measured metrics, plots, confusion matrices, and runtimes
5. Prediction Demo: Live molecule selection, decision score breakdown, and classification
"""

import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.data_loader import load_nci1_dataset
from src.preprocessing import pyg_to_molecular_graph, MolecularGraph
from src.wl_kernel import WeisfeilerLehmanKernel, hand_constructed_example
from src.visualization import plot_molecular_graph


st.set_page_config(
    page_title="Molecular Graph Classification: WL Kernel vs GNN",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_dataset():
    """Loads NCI1 dataset."""
    return load_nci1_dataset("data")


@st.cache_data
def load_cached_results():
    """Loads saved benchmark results and dataset statistics."""
    results_file = "results/metrics/benchmark_results.json"
    stats_file = "results/metrics/dataset_stats.json"
    
    res = []
    stats = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            res = json.load(f)
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
    return res, stats


dataset = load_cached_dataset()
results, dataset_stats = load_cached_results()

# Sidebar Navigation
st.sidebar.title("🧪 Navigation")
menu = st.sidebar.radio(
    "Select Section:",
    [
        "📌 Project Overview",
        "📊 Dataset Explorer",
        "🔬 WL Kernel Explorer",
        "📈 Empirical Model Comparison",
        "🎯 Interactive Prediction Demo"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Course**: Graph Representation Learning\n\n"
    "**Dataset**: NCI1 (National Cancer Institute)\n\n"
    "**Core Method**: Weisfeiler–Lehman Subtree Kernel from scratch"
)

# -----------------------------------------------------------------------------
# Section 1: Project Overview
# -----------------------------------------------------------------------------
if menu == "📌 Project Overview":
    st.markdown("<div class='main-header'>Molecular Graph Classification Using the Weisfeiler–Lehman Subtree Kernel</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Comparative Evaluation with Graph Neural Networks on the NCI1 Molecular Dataset</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🎯 Problem Statement")
        st.write(
            """
            In chemoinformatics and drug discovery, molecules are naturally modeled as attributed graphs where:
            - **Nodes** correspond to atoms with categorical types (Carbon, Oxygen, Nitrogen, etc.).
            - **Edges** correspond to chemical bonds.
            - **Graph-level Target**: Predict whether a compound inhibits the growth of human non-small cell lung cancer cell lines (NCI1 benchmark).
            """
        )

        st.subheader("🧬 Weisfeiler–Lehman (WL) Subtree Kernel Algorithm")
        st.markdown(r"""
        The Weisfeiler–Lehman subtree kernel (Shervashidze et al., 2011) maps graphs into a reproducing kernel Hilbert space (RKHS)
        by counting common subtree patterns:

        1. **Initial Categorical Labels**: Each atom receives an initial label based on its atomic species:
           $$\ell_0(v) = \text{atom}(v), \quad \forall v \in V$$

        2. **Neighborhood Multiset Aggregation**: For each atom, collect the multiset of neighbor labels:
           $$\mathcal{M}_t(v) = \{\ell_{t-1}(u) : u \in \mathcal{N}(v)\}$$

        3. **Deterministic Relabeling**: Sort neighbor labels deterministically and map the signature to a unique integer ID:
           $$\ell_t(v) = \text{hash}\Big(t, \ell_{t-1}(v), \text{sort}(\mathcal{M}_t(v))\Big)$$

        4. **Feature Vector**: Graph feature $\phi(G)$ is the concatenated histogram of subtree labels from $t=0$ to $h$:
           $$\phi(G) = [\phi_0(G)^\top \;\; \phi_1(G)^\top \;\; \dots \;\; \phi_h(G)^\top]^\top \in \mathbb{R}^D$$

        5. **Kernel Computation**: Normalized inner product with cosine normalization:
           $$K_{\text{norm}}(G_i, G_j) = \frac{\langle \phi(G_i), \phi(G_j) \rangle}{\|\phi(G_i)\|_2 \cdot \|\phi(G_j)\|_2}$$
        """)
    with col2:
        st.subheader("📊 Key Benchmark Highlights")
        if results:
            best_model = max(results, key=lambda x: x["accuracy"])
            st.success(f"🏆 **Top Performer**: {best_model['model']}\n\n"
                       f"- **Accuracy**: {best_model['accuracy']*100:.2f}%\n\n"
                       f"- **F1-Score**: {best_model['f1']:.4f}\n\n"
                       f"- **ROC-AUC**: {best_model['roc_auc']:.4f}\n\n"
                       f"- **Training Time**: {best_model['train_time_sec']:.2f} seconds")
        st.info("""
        **Pipeline Architecture**:
        1. `src/data_loader.py`: Download & validation
        2. `src/preprocessing.py`: Lightweight MolecularGraph data structures
        3. `src/wl_kernel.py`: 1-WL subtree kernel from scratch
        4. `src/wl_svm.py`: Exact precomputed Gram SVM
        5. `src/baseline.py`: Conventional graph features + SVM
        6. `src/gcn_model.py`: 3-layer GCN baseline
        7. `src/experiments.py`: Unified comparative test suite
        """)

# -----------------------------------------------------------------------------
# Section 2: Dataset Explorer
# -----------------------------------------------------------------------------
elif menu == "📊 Dataset Explorer":
    st.markdown("<div class='main-header'>NCI1 Molecular Dataset Explorer</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Molecular Graphs", f"{dataset_stats.get('num_graphs', len(dataset)):,}")
    col2.metric("Target Classes", f"{dataset_stats.get('num_classes', 2)} (Binary)")
    col3.metric("Atom Categories", f"{dataset_stats.get('num_atom_categories', 37)}")
    col4.metric("Average Atoms / Molecule", f"{dataset_stats.get('node_statistics', {}).get('mean', 29.8):.1f}")

    tab1, tab2 = st.tabs(["🔬 Interactive Molecule Viewer", "📈 Distribution Statistics"])
    
    with tab1:
        st.subheader("Select a Molecule to Visualize")
        graph_idx = st.slider("Select Molecule Index", min_value=0, max_value=min(len(dataset)-1, 500), value=15)
        pyg_data = dataset[graph_idx]
        mg = pyg_to_molecular_graph(pyg_data, graph_id=graph_idx)
        
        col_view, col_info = st.columns([3, 2])
        with col_view:
            fig = plot_molecular_graph(mg)
            st.pyplot(fig)
        with col_info:
            status_text = "Active (Cancer Inhibitor)" if mg.graph_label == 1 else "Inactive"
            st.write(f"### Properties of Molecule #{graph_idx}")
            st.write(f"- **Cancer Inhibition Activity**: `{status_text}`")
            st.write(f"- **Atom Count (Nodes)**: {mg.num_nodes}")
            st.write(f"- **Bond Count (Edges)**: {len(mg.edge_list)}")
            st.write(f"- **Atom Labels Present**: {sorted(list(set(mg.node_labels)))}")
            st.write(f"- **Adjacency Connectivity**: {len(mg.adjacency_list)} nodes connected")

    with tab2:
        st.subheader("Class & Graph Size Distributions")
        plot_path = "results/plots/dataset_distribution.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="NCI1 Class Balance & Graph Size Statistics", use_container_width=True)
        else:
            st.warning("Distribution plot not found. Run benchmark experiments first.")

# -----------------------------------------------------------------------------
# Section 3: WL Kernel Explorer
# -----------------------------------------------------------------------------
elif menu == "🔬 WL Kernel Explorer":
    st.markdown("<div class='main-header'>Weisfeiler–Lehman Subtree Kernel Explorer</div>", unsafe_allow_html=True)
    st.write("Demonstration of the neighborhood aggregation and multiset relabeling algorithm implemented strictly from scratch.")

    tab1, tab2 = st.tabs(["🧪 Hand-Crafted C3 vs P3 Example", "🧬 Real Molecular Subtree Relabeling"])
    
    with tab1:
        st.subheader("Distinguishing Cycle (C3) vs Path (P3)")
        st.markdown("""
        Consider two graphs having identical node counts (3 atoms) and identical atom labels `[1, 1, 1]`:
        - **Graph A ($C_3$)**: Triangle cycle ($0-1-2-0$), each atom has degree 2.
        - **Graph B ($P_3$)**: Linear chain ($0-1-2$), atoms have degrees 1, 2, 1.
        """)
        graphs, wl_example = hand_constructed_example()
        phi = wl_example.transform(graphs).toarray()
        K = wl_example.compute_kernel_matrix(wl_example.transform(graphs))
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("#### Graph A: Triangle Cycle (C3)")
            figA = plot_molecular_graph(graphs[0], title="Graph A (C3 Cycle)")
            st.pyplot(figA)
            st.write(f"WL Feature Vector $\phi(G_A)$: `{phi[0].tolist()}`")
        with col_g2:
            st.write("#### Graph B: Linear Chain (P3)")
            figB = plot_molecular_graph(graphs[1], title="Graph B (P3 Path)")
            st.pyplot(figB)
            st.write(f"WL Feature Vector $\phi(G_B)$: `{phi[1].tolist()}`")

        st.write("#### Resulting Gram Matrix $K$:")
        st.dataframe(pd.DataFrame(K, index=["Graph A (C3)", "Graph B (P3)"], columns=["Graph A (C3)", "Graph B (P3)"]))
        st.success("✅ At iteration $h=0$, both graphs had identical feature vector $[3]$. At iteration $h=1$, 1-WL partitioned the nodes based on subtree topology!")

    with tab2:
        st.subheader("Subtree Depth Relabeling on NCI1 Molecules")
        selected_h = st.selectbox("Select Subtree Depth $h$:", [0, 1, 2, 3], index=2)
        sample_mol = pyg_to_molecular_graph(dataset[0], graph_id=0)
        wl_sample = WeisfeilerLehmanKernel(h=selected_h, normalize=True)
        phi_sample = wl_sample.fit_transform([sample_mol])
        
        st.write(f"- **Vocabulary size (unique subtree patterns)**: {wl_sample.feature_dim}")
        st.write(f"- **Non-zero features in Molecule #0**: {phi_sample.nnz}")
        st.write("First 10 learned subtree signature mappings:")
        sample_signatures = list(wl_sample.label_dict.items())[:10]
        st.json({f"Subtree ID {v}": str(k) for k, v in sample_signatures})

# -----------------------------------------------------------------------------
# Section 4: Model Comparison
# -----------------------------------------------------------------------------
elif menu == "📈 Empirical Model Comparison":
    st.markdown("<div class='main-header'>Empirical Model Comparison on NCI1</div>", unsafe_allow_html=True)
    st.write("Rigorous comparative evaluation across all implemented architectures.")

    if results:
        df_res = pd.DataFrame(results)
        st.subheader("📊 Performance Summary Table")
        display_cols = ["model", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "train_time_sec", "inference_time_sec"]
        st.dataframe(
            df_res[display_cols].style.format({
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1": "{:.4f}",
                "macro_f1": "{:.4f}",
                "roc_auc": "{:.4f}",
                "train_time_sec": "{:.3f}s",
                "inference_time_sec": "{:.3f}s"
            }).highlight_max(subset=["accuracy", "f1", "roc_auc"], color="#D1FAE5"),
            use_container_width=True
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("🎯 Metrics Comparison")
            comp_plot = "results/plots/model_comparison.png"
            if os.path.exists(comp_plot):
                st.image(comp_plot, use_container_width=True)
        with col_p2:
            st.subheader("📈 WL Subtree Depth Scaling")
            depth_plot = "results/plots/wl_depth_effect.png"
            if os.path.exists(depth_plot):
                st.image(depth_plot, use_container_width=True)

        col_p3, col_p4 = st.columns(2)
        with col_p3:
            st.subheader("⏱️ Runtime & Efficiency")
            runtime_plot = "results/plots/runtime_comparison.png"
            if os.path.exists(runtime_plot):
                st.image(runtime_plot, use_container_width=True)
        with col_p4:
            st.subheader("🧩 Confusion Matrices")
            cm_plot = "results/plots/confusion_matrices.png"
            if os.path.exists(cm_plot):
                st.image(cm_plot, use_container_width=True)
    else:
        st.warning("No benchmark results found. Run `python3 -m src.experiments` first.")

# -----------------------------------------------------------------------------
# Section 5: Prediction Demo
# -----------------------------------------------------------------------------
elif menu == "🎯 Interactive Prediction Demo":
    st.markdown("<div class='main-header'>Interactive Prediction & Decision Score Demo</div>", unsafe_allow_html=True)
    st.write("Inspect how the trained Weisfeiler–Lehman Subtree Kernel model makes decisions on arbitrary molecular graphs.")

    # Pick molecules from diverse classes
    col_ctrl1, col_ctrl2 = st.columns([2, 2])
    with col_ctrl1:
        class_filter = st.radio("Filter Molecules by Ground Truth Activity:", ["All", "Active Only (Class 1)", "Inactive Only (Class 0)"], horizontal=True)
    
    # Pre-select representative molecules
    if class_filter == "Active Only (Class 1)":
        sample_indices = [1650, 1700, 1800, 2000, 2200, 2500, 3000, 3500, 4000]
    elif class_filter == "Inactive Only (Class 0)":
        sample_indices = [10, 25, 50, 100, 250, 500, 750, 1000, 1200, 1500]
    else:
        sample_indices = [10, 50, 100, 500, 1650, 1800, 2200, 2500, 3000, 4000]

    with col_ctrl2:
        pred_idx = st.selectbox("Choose a Benchmark Molecule Index:", sample_indices, index=0)

    pyg_m = dataset[pred_idx]
    mol = pyg_to_molecular_graph(pyg_m, graph_id=pred_idx)
    
    col_vis, col_pred = st.columns([3, 2])
    with col_vis:
        fig_demo = plot_molecular_graph(mol, title=f"Molecule #{pred_idx} (Ground Truth: {'Active' if mol.graph_label==1 else 'Inactive'})")
        st.pyplot(fig_demo)

    with col_pred:
        st.subheader("Model Inference & Decision Boundary")
        true_label = mol.graph_label
        true_desc = "Active (Cancer Inhibitor)" if true_label == 1 else "Inactive"

        # Check if saved WL model is available for live inference
        model_path = "models/wl_clf_h2.pkl"
        if os.path.exists(model_path):
            import pickle
            with open(model_path, "rb") as f:
                saved_wl_clf = pickle.load(f)
            preds, scores = saved_wl_clf.predict([mol])
            pred_class = int(preds[0])
            score_val = float(scores[0])
            pred_desc = "Active (Cancer Inhibitor)" if pred_class == 1 else "Inactive"
            is_correct = (pred_class == true_label)

            if is_correct:
                st.success(f"🎯 **Predicted**: `{pred_desc}` (Class {pred_class}) - **MATCH**")
            else:
                st.error(f"⚠️ **Predicted**: `{pred_desc}` (Class {pred_class}) - Misclassified")

            st.metric("SVM Margin / Decision Score", f"{score_val:+.4f}")
            st.caption("Score > 0 predicts Active; Score < 0 predicts Inactive. Magnitude reflects margin distance.")
        else:
            st.info("Interactive model checkpoint not found. Displaying graph metadata.")

        st.write("---")
        st.write(f"- **Ground Truth**: `{true_desc}` (Class {true_label})")
        st.write(f"- **Atom Count (Vertices)**: {mol.num_nodes}")
        st.write(f"- **Chemical Bonds (Edges)**: {len(mol.edge_list)}")
        st.write(f"- **Distinct Atom Types**: {len(set(mol.node_labels))}")

