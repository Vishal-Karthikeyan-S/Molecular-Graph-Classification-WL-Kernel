"""
Module 5: Graph-Level Feature Baseline.

Extracts classical non-structural graph features:
- Number of nodes
- Number of edges / edge density
- Mean, standard deviation, and maximum node degree
- Degree histogram (binned)
- Atom label category frequencies (bag of atoms)

Trains a Standard Scaler + Support Vector Classifier (RBF kernel) as a competitive
conventional baseline to assess the specific contribution of subtree neighborhood context.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.preprocessing import MolecularGraph


class GraphFeatureExtractor:
    """
    Extracts summary graph descriptors and atom composition vectors.
    """
    def __init__(self, max_atom_types: int = 37, max_degree_bins: int = 6):
        self.max_atom_types = max_atom_types
        self.max_degree_bins = max_degree_bins

    def extract_single_graph(self, g: MolecularGraph) -> np.ndarray:
        num_nodes = g.num_nodes
        num_edges = len(g.edge_list)
        density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        
        # Degrees
        degrees = [len(g.adjacency_list.get(u, [])) for u in range(num_nodes)]
        if degrees:
            mean_deg = float(np.mean(degrees))
            std_deg = float(np.std(degrees))
            max_deg = float(np.max(degrees))
        else:
            mean_deg, std_deg, max_deg = 0.0, 0.0, 0.0

        # Degree histogram up to max_degree_bins
        deg_hist = np.zeros(self.max_degree_bins, dtype=np.float64)
        for d in degrees:
            bin_idx = min(d, self.max_degree_bins - 1)
            deg_hist[bin_idx] += 1.0
        if num_nodes > 0:
            deg_hist /= num_nodes

        # Atom type counts
        atom_counts = np.zeros(self.max_atom_types, dtype=np.float64)
        for lbl in g.node_labels:
            if 0 <= lbl < self.max_atom_types:
                atom_counts[lbl] += 1.0
        if num_nodes > 0:
            atom_counts /= num_nodes  # normalized atom proportions

        features = np.concatenate([
            [float(num_nodes), float(num_edges), float(density), mean_deg, std_deg, max_deg],
            deg_hist,
            atom_counts
        ])
        return features

    def extract_dataset(self, graphs: List[MolecularGraph]) -> np.ndarray:
        feats = [self.extract_single_graph(g) for g in graphs]
        return np.array(feats, dtype=np.float64)


class GraphFeatureBaseline:
    """
    Conventional Graph Feature + Scaled SVM Baseline.
    """
    def __init__(self, C: float = 1.0, kernel: str = "rbf"):
        self.extractor = GraphFeatureExtractor()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel=kernel, probability=False, random_state=42))
        ])
        self.fit_time: float = 0.0
        self.pred_time: float = 0.0

    def fit(self, train_graphs: List[MolecularGraph], y_train: np.ndarray) -> "GraphFeatureBaseline":
        start_t = time.perf_counter()
        X_train = self.extractor.extract_dataset(train_graphs)
        self.pipeline.fit(X_train, y_train)
        self.fit_time = time.perf_counter() - start_t
        return self

    def predict(self, test_graphs: List[MolecularGraph]) -> Tuple[np.ndarray, np.ndarray]:
        start_t = time.perf_counter()
        X_test = self.extractor.extract_dataset(test_graphs)
        preds = self.pipeline.predict(X_test)
        scores = self.pipeline.decision_function(X_test)
        self.pred_time = time.perf_counter() - start_t
        return preds, scores

    def evaluate(self, test_graphs: List[MolecularGraph], y_test: np.ndarray) -> Dict[str, Any]:
        y_test = np.array(y_test)
        preds, scores = self.predict(test_graphs)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, preds).tolist()
        
        try:
            auc = float(roc_auc_score(y_test, scores))
        except Exception:
            auc = 0.5

        return {
            "model": "Graph Features Baseline (SVM)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "train_time_sec": float(self.fit_time),
            "inference_time_sec": float(self.pred_time)
        }
