"""
Module 4: SVM Classification with Precomputed WL Kernel.

Implements train/test evaluation using scikit-learn's SVC(kernel='precomputed').
Strictly prevents data leakage by computing K_train (N_train x N_train) from training graphs,
and K_test_train (N_test x N_train) from test graphs using the fitted vocabulary.
Provides decision scores, cross-validation, and metrics logging.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.preprocessing import MolecularGraph
from src.wl_kernel import WeisfeilerLehmanKernel


class WLSVMClassifier:
    """
    Classifier wrapping WeisfeilerLehmanKernel and scikit-learn SVC(kernel='precomputed').
    """
    def __init__(self, h: int = 3, C: float = 1.0, normalize: bool = True, probability: bool = False):
        self.h = h
        self.C = C
        self.normalize = normalize
        self.probability = probability
        self.kernel_extractor = WeisfeilerLehmanKernel(h=h, normalize=normalize)
        self.clf = SVC(C=C, kernel="precomputed", probability=probability, random_state=42)
        
        self.phi_train = None
        self.y_train = None
        self.fit_time: float = 0.0
        self.pred_time: float = 0.0

    def fit(self, train_graphs: List[MolecularGraph], y_train: np.ndarray) -> "WLSVMClassifier":
        """
        Fits WL subtree vocabulary on train_graphs, builds train Gram matrix,
        and trains SVC.
        """
        start_t = time.perf_counter()
        self.y_train = np.array(y_train)
        
        # 1. Fit WL extractor on training graphs
        self.phi_train = self.kernel_extractor.fit_transform(train_graphs)
        
        # 2. Compute symmetric Gram matrix K_train (shape: N_train x N_train)
        K_train = self.kernel_extractor.compute_kernel_matrix(self.phi_train)
        
        # 3. Fit SVC
        self.clf.fit(K_train, self.y_train)
        self.fit_time = time.perf_counter() - start_t
        return self

    def predict(self, test_graphs: List[MolecularGraph]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms test_graphs using learned vocabulary, computes K_test_train,
        and returns predictions and decision scores.
        
        Returns:
            (predictions, decision_function_scores)
        """
        start_t = time.perf_counter()
        if self.phi_train is None:
            raise RuntimeError("Model must be fitted before predict()")

        # 1. Transform test graphs
        phi_test = self.kernel_extractor.transform(test_graphs)
        
        # 2. Compute test-train Gram matrix (shape: N_test x N_train)
        K_test_train = self.kernel_extractor.compute_kernel_matrix(self.phi_train, phi_test)
        
        # 3. Predict via SVC
        preds = self.clf.predict(K_test_train)
        scores = self.clf.decision_function(K_test_train)
        self.pred_time = time.perf_counter() - start_t
        return preds, scores

    def evaluate(self, test_graphs: List[MolecularGraph], y_test: np.ndarray) -> Dict[str, Any]:
        """
        Computes evaluation metrics on test set.
        """
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
            "model": f"WL-Kernel (h={self.h}) + SVM",
            "h": self.h,
            "C": self.C,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "train_time_sec": float(self.fit_time),
            "inference_time_sec": float(self.pred_time),
            "num_features": int(self.kernel_extractor.feature_dim)
        }
