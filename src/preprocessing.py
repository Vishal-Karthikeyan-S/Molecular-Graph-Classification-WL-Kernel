"""
Module 2: Graph Preprocessing.
Converts PyTorch Geometric Data objects into clean, lightweight molecular graph structures
preserving atom categorical labels, adjacency structures, and edge information.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx


@dataclass
class MolecularGraph:
    """
    Lightweight, validated in-memory molecular graph representation.
    """
    graph_id: int
    num_nodes: int
    node_labels: List[int]
    adjacency_list: Dict[int, List[int]]
    edge_list: List[Tuple[int, int]]
    graph_label: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validates graph structure consistency."""
        assert len(self.node_labels) == self.num_nodes, (
            f"Node labels count ({len(self.node_labels)}) != num_nodes ({self.num_nodes})"
        )
        assert self.graph_label in (0, 1), f"Unexpected graph label {self.graph_label}"
        for u, neighbors in self.adjacency_list.items():
            assert 0 <= u < self.num_nodes, f"Node index {u} out of bounds"
            for v in neighbors:
                assert 0 <= v < self.num_nodes, f"Neighbor {v} of {u} out of bounds"
                # Undirected consistency check
                assert u in self.adjacency_list.get(v, []), (
                    f"Edge ({u}, {v}) is not symmetric in adjacency list"
                )
        return True

    def to_networkx(self) -> nx.Graph:
        """Converts to a NetworkX Graph for visualization and inspection."""
        G = nx.Graph()
        for i, label in enumerate(self.node_labels):
            G.add_node(i, label=label)
        for u, v in self.edge_list:
            G.add_edge(u, v)
        G.graph["label"] = self.graph_label
        G.graph["id"] = self.graph_id
        return G


def pyg_to_molecular_graph(data: Data, graph_id: int = 0, use_degree_as_label: bool = False) -> MolecularGraph:
    """
    Converts a PyTorch Geometric Data object into an independent MolecularGraph.
    
    Args:
        data: PyG Data object with edge_index, x (one-hot atom labels), y (graph label).
        graph_id: Unique identifier for the graph.
        use_degree_as_label: If True, uses node degree instead of atom label. (Default: False).
            In NCI1, chemistry is determined by atom types (C, N, O, S, halogens), so categorical
            atom label preservation is paramount.
    """
    num_nodes = data.num_nodes
    
    # Extract node labels
    if use_degree_as_label:
        # Compute degrees from edge_index
        degrees = [0] * num_nodes
        if data.edge_index is not None and data.edge_index.numel() > 0:
            src = data.edge_index[0].tolist()
            for u in src:
                degrees[u] += 1
        node_labels = degrees
    else:
        if hasattr(data, 'x') and data.x is not None:
            # x is one-hot vector representing categorical atom label
            node_labels = data.x.argmax(dim=-1).tolist()
        else:
            # Fallback to degree if no attributes exist
            degrees = [0] * num_nodes
            if data.edge_index is not None and data.edge_index.numel() > 0:
                src = data.edge_index[0].tolist()
                for u in src:
                    degrees[u] += 1
            node_labels = degrees

    # Build adjacency list and unique undirected edge list
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    edge_set = set()
    
    if data.edge_index is not None and data.edge_index.numel() > 0:
        edges = data.edge_index.t().tolist()
        for u, v in edges:
            if u != v: # eliminate self-loops if any
                adj[u].append(v)
                edge_set.add((min(u, v), max(u, v)))
                
    # Deduplicate and sort neighbors for deterministic operations
    for u in adj:
        adj[u] = sorted(list(set(adj[u])))
        
    edge_list = sorted(list(edge_set))
    graph_label = int(data.y.item()) if hasattr(data, 'y') and data.y is not None else 0

    mg = MolecularGraph(
        graph_id=graph_id,
        num_nodes=num_nodes,
        node_labels=node_labels,
        adjacency_list=adj,
        edge_list=edge_list,
        graph_label=graph_label,
        metadata={"num_edges": len(edge_list)}
    )
    mg.validate()
    return mg


def preprocess_dataset(pyg_dataset, use_degree_as_label: bool = False) -> List[MolecularGraph]:
    """
    Preprocesses the complete or sliced PyG dataset into a list of MolecularGraphs.
    """
    graphs = []
    for idx, data in enumerate(pyg_dataset):
        mg = pyg_to_molecular_graph(data, graph_id=idx, use_degree_as_label=use_degree_as_label)
        graphs.append(mg)
    return graphs


if __name__ == "__main__":
    from src.data_loader import load_nci1_dataset
    ds = load_nci1_dataset("data")
    print(f"Pre-processing first 10 graphs...")
    mgs = preprocess_dataset(ds[:10])
    print(f"Sample graph 0: Nodes={mgs[0].num_nodes}, Edges={len(mgs[0].edge_list)}, Label={mgs[0].graph_label}")
    print(f"Graph 0 node labels: {mgs[0].node_labels[:8]}...")
    print(f"Graph 0 adj for node 0: {mgs[0].adjacency_list[0]}")
