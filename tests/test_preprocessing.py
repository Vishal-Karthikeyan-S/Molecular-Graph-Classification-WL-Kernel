"""
Tests for Module 2: Graph Preprocessing.
"""

import pytest
import torch
from torch_geometric.data import Data
from src.preprocessing import pyg_to_molecular_graph, preprocess_dataset, MolecularGraph


def test_pyg_to_molecular_graph_conversion():
    # Construct a dummy PyG Data object
    # 3 nodes, 2 edges: 0-1, 1-2
    # Categorical labels: one-hot for class 0, 1, 2 (dim 5)
    x = torch.zeros((3, 5))
    x[0, 2] = 1.0  # node 0 has label 2
    x[1, 4] = 1.0  # node 1 has label 4
    x[2, 2] = 1.0  # node 2 has label 2
    
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    y = torch.tensor([1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    mg = pyg_to_molecular_graph(data, graph_id=42)
    assert mg.graph_id == 42
    assert mg.num_nodes == 3
    assert mg.node_labels == [2, 4, 2]
    assert mg.graph_label == 1
    assert mg.adjacency_list[1] == [0, 2]
    assert (0, 1) in mg.edge_list and (1, 2) in mg.edge_list
    assert mg.validate() is True


def test_degree_as_label_option():
    # 3 nodes: star around node 0 (0-1, 0-2)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)
    data = Data(edge_index=edge_index, y=torch.tensor([0]))
    data.num_nodes = 3
    
    mg = pyg_to_molecular_graph(data, graph_id=1, use_degree_as_label=True)
    assert mg.node_labels == [2, 1, 1]
