"""
Module 6: Graph Convolutional Network (GCN) Baseline.

Implements a deep graph neural network architecture using PyTorch and PyTorch Geometric:
- 3 Graph Convolutional Layers (GCNConv) with ReLU and Dropout
- Global Mean Pooling across nodes
- Multi-layer perceptron (MLP) classification head
- Early stopping based on validation loss / accuracy
- Full training loop tracking loss curves and inference latency
"""

import time
import os
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import numpy as np


class MolecularGCN(nn.Module):
    """
    3-layer Graph Convolutional Network with Global Mean Pooling.
    """
    def __init__(self, in_channels: int = 37, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout_rate = dropout
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. First Graph Conv + ReLU + Dropout
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        # 2. Second Graph Conv + ReLU + Dropout
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        # 3. Third Graph Conv + ReLU
        h = self.conv3(h, edge_index)
        h = F.relu(h)
        
        # 4. Global Graph Pooling
        h_graph = global_mean_pool(h, batch)
        
        # 5. MLP Classifier
        out = F.relu(self.fc1(h_graph))
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        logits = self.fc2(out)
        return logits


class GCNTrainer:
    """
    Trainer for GCN on PyG Data batches.
    """
    def __init__(
        self,
        in_channels: int = 37,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = MolecularGCN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        self.best_model_state = None
        self.train_time: float = 0.0

    def fit(
        self,
        train_dataset,
        val_dataset,
        batch_size: int = 64,
        epochs: int = 60,
        patience: int = 15,
        checkpoint_path: str = "models/best_gcn.pt"
    ) -> "GCNTrainer":
        """
        Trains GCN with early stopping on validation loss.
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        start_t = time.perf_counter()

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y.squeeze())
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch.num_graphs

            train_loss = total_train_loss / len(train_dataset)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    loss = self.criterion(out, batch.y.squeeze())
                    total_val_loss += loss.item() * batch.num_graphs
                    preds = out.argmax(dim=-1)
                    correct += int((preds == batch.y.squeeze()).sum())

            val_loss = total_val_loss / len(val_dataset)
            val_acc = correct / len(val_dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.best_model_state, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # Early stop triggered
                    break

        self.train_time = time.perf_counter() - start_t
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self

    def predict(self, test_dataset, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs inference and returns predicted classes, probabilities of class 1, and inference time.
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        
        all_preds = []
        all_probs = []
        start_t = time.perf_counter()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=-1)
                preds = out.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())
                
        inf_time = time.perf_counter() - start_t
        return np.array(all_preds), np.array(all_probs), inf_time

    def evaluate(self, test_dataset, batch_size: int = 64) -> Dict[str, Any]:
        """
        Evaluates on test dataset.
        """
        y_true = np.array([int(d.y.item()) for d in test_dataset])
        preds, probs, inf_time = self.predict(test_dataset, batch_size=batch_size)

        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, preds).tolist()
        
        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = 0.5

        return {
            "model": "Graph Convolutional Network (GCN)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "train_time_sec": float(self.train_time),
            "inference_time_sec": float(inf_time),
            "training_history": self.history
        }
