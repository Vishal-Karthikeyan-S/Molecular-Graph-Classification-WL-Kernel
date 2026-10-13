"""
Visualization utilities for graph representation, molecular structures,
evaluation curves, confusion matrices, and model comparison bar charts.
"""

import os
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from src.preprocessing import MolecularGraph


# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def plot_dataset_distributions(stats: Dict[str, Any], save_path: str = "results/plots/dataset_distribution.png") -> None:
    """Plots class balance and node count distribution."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Class balance bar plot
    classes = list(stats["class_distribution"].keys())
    counts = list(stats["class_distribution"].values())
    axes[0].bar(classes, counts, color=["#3498db", "#e74c3c"], width=0.5, edgecolor="black")
    axes[0].set_title("NCI1 Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Molecules", fontsize=11)
    for i, v in enumerate(counts):
        axes[0].text(i, v + 25, str(v), ha="center", fontweight="bold")

    # Node stats summary bar
    n_stats = stats["node_statistics"]
    stat_names = ["Min Nodes", "Median Nodes", "Mean Nodes", "Max Nodes"]
    stat_vals = [n_stats["min"], n_stats["median"], n_stats["mean"], n_stats["max"]]
    axes[1].bar(stat_names, stat_vals, color="#2ecc71", width=0.5, edgecolor="black")
    axes[1].set_title("Molecular Graph Size (Nodes)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count / Atoms", fontsize=11)
    for i, v in enumerate(stat_vals):
        axes[1].text(i, v + 1.5, f"{v:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved dataset distribution plot to {save_path}")


def plot_molecular_graph(mg: MolecularGraph, title: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """Visualizes an individual MolecularGraph with atom labels and bond connectivity."""
    G = mg.to_networkx()
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by atom label
    node_labels = nx.get_node_attributes(G, 'label')
    colors = [node_labels.get(n, 0) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, cmap=plt.cm.tab20, node_size=350, edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: f"A{node_labels[n]}" for n in G.nodes()}, font_size=8, ax=ax)
    
    status = "Active (1)" if mg.graph_label == 1 else "Inactive (0)"
    heading = title or f"Molecule #{mg.graph_id} [{status}] ({mg.num_nodes} atoms)"
    ax.set_title(heading, fontsize=11, fontweight="bold")
    ax.axis("off")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def plot_model_comparison(results: List[Dict[str, Any]], save_path: str = "results/plots/model_comparison.png") -> None:
    """Plots comparative bar charts across models for Accuracy, F1, and AUC."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    aucs = [r.get("roc_auc", 0.5) for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    rects1 = ax.bar(x - width, accs, width, label='Accuracy', color="#2980b9", edgecolor="black")
    rects2 = ax.bar(x, f1s, width, label='F1-Score', color="#27ae60", edgecolor="black")
    rects3 = ax.bar(x + width, aucs, width, label='ROC-AUC', color="#8e44ad", edgecolor="black")

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: WL Kernel Variants vs. Baselines on NCI1', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {save_path}")


def plot_wl_depth_analysis(wl_results: List[Dict[str, Any]], save_path: str = "results/plots/wl_depth_effect.png") -> None:
    """Plots accuracy and feature dimension vs WL iteration depth h."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hs = [r["h"] for r in wl_results]
    accs = [r["accuracy"] for r in wl_results]
    feats = [r["num_features"] for r in wl_results]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    color = '#1f77b4'
    ax1.set_xlabel('WL Subtree Iteration Depth (h)', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color, fontsize=12)
    line1 = ax1.plot(hs, accs, marker='o', linewidth=2.5, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(hs)
    for i, txt in enumerate(accs):
        ax1.annotate(f"{txt:.3f}", (hs[i], accs[i] + 0.005), fontweight="bold")

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Total Subtree Features (Vocabulary Size)', color=color, fontsize=12)
    line2 = ax2.plot(hs, feats, marker='s', linestyle='--', linewidth=2, color=color, label='Feature Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Effect of Subtree Depth (h) on Accuracy & Feature Dimensionality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved WL depth plot to {save_path}")


def plot_runtime_comparison(results: List[Dict[str, Any]], save_path: str = "results/plots/runtime_comparison.png") -> None:
    """Plots training time across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = [r["model"] for r in results]
    train_times = [r["train_time_sec"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(models, train_times, color="#34495e", edgecolor="black")
    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Training Execution Time Comparison", fontsize=13, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(train_times)*0.01, bar.get_y() + bar.get_height()/2, f"{w:.2f}s", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved runtime comparison plot to {save_path}")


def plot_confusion_matrices(results: List[Dict[str, Any]], save_path: str = "results/plots/confusion_matrices.png") -> None:
    """Plots confusion matrices side-by-side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, r in enumerate(results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Inactive (0)", "Active (1)"],
                    yticklabels=["Inactive (0)", "Active (1)"])
        axes[idx].set_title(r["model"], fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices plot to {save_path}")


def plot_gcn_loss_curves(history: Dict[str, List[float]], save_path: str = "results/plots/gcn_training_curves.png") -> None:
    """Plots GCN training and validation loss & accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.set_title("GCN Training and Validation Dynamics", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved GCN loss curves to {save_path}")
