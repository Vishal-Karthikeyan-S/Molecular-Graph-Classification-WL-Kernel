"""
Module 7: Experimental Pipeline.

Coordinates the rigorous, end-to-end benchmark on NCI1:
1. Conventional Graph-Level Features Baseline + RBF SVM
2. Weisfeiler-Lehman Subtree Kernel + SVM across depths h = 0, 1, 2, 3
3. 3-layer Graph Convolutional Network (GCN) with early stopping
Standardizes train/test splits, computes all evaluation metrics, saves results to JSON,
and produces comparative analysis artifacts.
"""

import os
import json
import yaml
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import load_nci1_dataset, inspect_dataset, save_dataset_metadata
from src.preprocessing import preprocess_dataset, MolecularGraph
from src.wl_svm import WLSVMClassifier
from src.baseline import GraphFeatureBaseline
from src.gcn_model import GCNTrainer
from src.visualization import (
    plot_dataset_distributions, plot_model_comparison,
    plot_wl_depth_analysis, plot_runtime_comparison,
    plot_confusion_matrices, plot_gcn_loss_curves
)


def run_benchmark(
    config_path: str = "config.yaml",
    subset_size: int = 1200,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the comparative evaluation across all models on NCI1.
    
    Args:
        config_path: Path to YAML configuration.
        subset_size: Number of graphs to sample for reproducible benchmark.
                     If None or <= 0, uses entire dataset (4110 graphs).
        random_seed: Random seed for reproducible splitting.
    """
    print(f"=== Starting Molecular Graph Classification Benchmark ===")
    start_total_time = time.perf_counter()

    # Load config if exists
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # 1. Load dataset
    pyg_dataset = load_nci1_dataset("data")
    stats = inspect_dataset(pyg_dataset)
    save_dataset_metadata(stats, "results/metrics/dataset_stats.json")
    plot_dataset_distributions(stats, "results/plots/dataset_distribution.png")

    total_graphs = len(pyg_dataset)
    print(f"Total dataset: {total_graphs} graphs.")
    
    if subset_size and 0 < subset_size < total_graphs:
        print(f"Sampling stratified subset of {subset_size} graphs for efficient benchmark...")
        all_ys = [int(d.y.item()) for d in pyg_dataset]
        # Stratified sampling
        indices, _ = train_test_split(
            np.arange(total_graphs),
            train_size=subset_size,
            stratify=all_ys,
            random_state=random_seed
        )
        indices = sorted(indices.tolist())
        selected_pyg = [pyg_dataset[i] for i in indices]
    else:
        selected_pyg = list(pyg_dataset)

    # Preprocess into MolecularGraph objects
    print(f"Preprocessing {len(selected_pyg)} graphs...")
    molecular_graphs = preprocess_dataset(selected_pyg)
    y_all = np.array([g.graph_label for g in molecular_graphs])

    # Stratified Train/Val/Test Split (70% train, 10% val, 20% test)
    # Stratified 80/20 train_val / test
    idx_train_val, idx_test = train_test_split(
        np.arange(len(molecular_graphs)),
        test_size=0.20,
        stratify=y_all,
        random_state=random_seed
    )
    # Stratified 70/10 split from train_val for GCN validation
    y_train_val = y_all[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=0.125,  # 0.125 * 0.8 = 0.10 of total
        stratify=y_train_val,
        random_state=random_seed
    )

    train_graphs = [molecular_graphs[i] for i in idx_train]
    val_graphs = [molecular_graphs[i] for i in idx_val]
    test_graphs = [molecular_graphs[i] for i in idx_test]

    train_pyg = [selected_pyg[i] for i in idx_train]
    val_pyg = [selected_pyg[i] for i in idx_val]
    test_pyg = [selected_pyg[i] for i in idx_test]

    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    print(f"Split sizes: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    all_results: List[Dict[str, Any]] = []
    wl_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 2. Graph Feature Baseline + SVM
    # -------------------------------------------------------------
    print("\n--- Training Baseline: Classical Graph Features + RBF SVM ---")
    baseline = GraphFeatureBaseline(C=1.0, kernel="rbf")
    baseline.fit(train_graphs, y_train)
    res_baseline = baseline.evaluate(test_graphs, y_test)
    all_results.append(res_baseline)
    print(f"Baseline Results: Acc={res_baseline['accuracy']:.4f}, F1={res_baseline['f1']:.4f}, AUC={res_baseline['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 3. Weisfeiler-Lehman Subtree Kernel + SVM (h = 0, 1, 2, 3)
    # -------------------------------------------------------------
    for h in [0, 1, 2, 3]:
        print(f"\n--- Training Weisfeiler-Lehman Subtree Kernel (h={h}) + SVM ---")
        wl_clf = WLSVMClassifier(h=h, C=1.0, normalize=True)
        wl_clf.fit(train_graphs, y_train)
        res_wl = wl_clf.evaluate(test_graphs, y_test)
        all_results.append(res_wl)
        wl_results.append(res_wl)
        print(f"WL (h={h}) Results: Acc={res_wl['accuracy']:.4f}, F1={res_wl['f1']:.4f}, AUC={res_wl['roc_auc']:.4f}, Features={res_wl['num_features']}")

    # -------------------------------------------------------------
    # 4. Graph Convolutional Network (GCN) Baseline
    # -------------------------------------------------------------
    print("\n--- Training Graph Convolutional Network (GCN) Baseline ---")
    gcn_trainer = GCNTrainer(
        in_channels=37,
        hidden_dim=64,
        num_classes=2,
        dropout=0.3,
        lr=0.005,
        weight_decay=1e-4
    )
    gcn_trainer.fit(
        train_pyg,
        val_pyg,
        batch_size=64,
        epochs=60,
        patience=12,
        checkpoint_path="models/best_gcn.pt"
    )
    res_gcn = gcn_trainer.evaluate(test_pyg, batch_size=64)
    all_results.append(res_gcn)
    print(f"GCN Results: Acc={res_gcn['accuracy']:.4f}, F1={res_gcn['f1']:.4f}, AUC={res_gcn['roc_auc']:.4f}")

    # Plot GCN loss curves
    if "training_history" in res_gcn:
        plot_gcn_loss_curves(res_gcn["training_history"], "results/plots/gcn_training_curves.png")

    # -------------------------------------------------------------
    # 5. Save Results & Generate Comparative Plots
    # -------------------------------------------------------------
    # Exclude training_history from final clean metrics table
    clean_results = []
    for r in all_results:
        item = {k: v for k, v in r.items() if k != "training_history"}
        clean_results.append(item)

    metrics_path = "results/metrics/benchmark_results.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved benchmark metrics to {metrics_path}")

    # Comparative plots
    plot_model_comparison(clean_results, "results/plots/model_comparison.png")
    plot_wl_depth_analysis(wl_results, "results/plots/wl_depth_effect.png")
    plot_runtime_comparison(clean_results, "results/plots/runtime_comparison.png")
    plot_confusion_matrices(clean_results, "results/plots/confusion_matrices.png")

    total_time = time.perf_counter() - start_total_time
    print(f"\n=== Benchmark Complete in {total_time:.2f} seconds ===")

    return {
        "results": clean_results,
        "wl_results": wl_results,
        "stats": stats
    }


if __name__ == "__main__":
    run_benchmark(subset_size=1200)
