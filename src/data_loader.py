"""
Module 1: Dataset Loading and Exploration.
Loads the NCI1 molecular graph dataset, verifies schema, computes graph statistics,
and handles automated downloading with fallbacks.
"""

import os
import sys
import json
import zipfile
import urllib.request
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data


def ensure_nci1_raw_data(data_root: str = "data") -> None:
    """
    Ensures that raw NCI1 dataset files exist in data_root/NCI1/raw.
    If not, downloads and unzips from authoritative mirrors.
    """
    raw_dir = os.path.join(data_root, "NCI1", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    required_files = [
        "NCI1_A.txt",
        "NCI1_graph_indicator.txt",
        "NCI1_graph_labels.txt",
        "NCI1_node_labels.txt"
    ]
    
    if all(os.path.exists(os.path.join(raw_dir, f)) for f in required_files):
        return

    # Direct mirror URLs
    urls = [
        "https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip",
        "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip"
    ]
    
    zip_path = os.path.join(raw_dir, "NCI1.zip")
    downloaded = False
    for url in urls:
        try:
            print(f"Downloading NCI1 dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    if not downloaded:
        raise RuntimeError(
            "Could not download NCI1 dataset automatically. "
            "Please manually download NCI1.zip and extract into data/NCI1/raw/"
        )
        
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    # Check if nested NCI1 folder was created during extraction
    nested_dir = os.path.join(raw_dir, "NCI1")
    if os.path.isdir(nested_dir):
        for item in os.listdir(nested_dir):
            src_file = os.path.join(nested_dir, item)
            dst_file = os.path.join(raw_dir, item)
            if not os.path.exists(dst_file):
                os.rename(src_file, dst_file)
        os.rmdir(nested_dir)
        
    print("Dataset raw files verified successfully.")


def load_nci1_dataset(data_root: str = "data") -> TUDataset:
    """
    Loads the NCI1 dataset via PyTorch Geometric TUDataset.
    Includes automated fallback downloading if required.
    """
    ensure_nci1_raw_data(data_root)
    dataset = TUDataset(root=data_root, name="NCI1")
    return dataset


def inspect_dataset(dataset: TUDataset) -> Dict[str, Any]:
    """
    Inspects and computes comprehensive graph statistics for NCI1:
    - total graphs, total nodes, total edges
    - node count distributions (min, max, mean, median, std)
    - edge count distributions
    - class distribution
    - atom label category distributions
    """
    num_graphs = len(dataset)
    num_nodes_list = []
    num_edges_list = []
    labels_list = []
    unique_atom_types = set()
    
    for i, data in enumerate(dataset):
        num_nodes_list.append(data.num_nodes)
        # Note: edge_index has shape [2, 2 * num_undirected_edges]
        num_edges_list.append(data.num_edges // 2)
        labels_list.append(int(data.y.item()))
        
        if hasattr(data, 'x') and data.x is not None:
            atom_cats = data.x.argmax(dim=-1).tolist()
            unique_atom_types.update(atom_cats)

    num_nodes_arr = np.array(num_nodes_list)
    num_edges_arr = np.array(num_edges_list)
    labels_arr = np.array(labels_list)

    class_0_count = int(np.sum(labels_arr == 0))
    class_1_count = int(np.sum(labels_arr == 1))

    stats = {
        "num_graphs": num_graphs,
        "num_classes": int(dataset.num_classes),
        "num_node_features": int(dataset.num_node_features),
        "num_atom_categories": len(unique_atom_types),
        "atom_category_ids": sorted(list(unique_atom_types)),
        "class_distribution": {
            "0 (Inactive)": class_0_count,
            "1 (Active)": class_1_count
        },
        "class_balance_ratio": float(class_1_count / num_graphs),
        "node_statistics": {
            "min": int(np.min(num_nodes_arr)),
            "max": int(np.max(num_nodes_arr)),
            "mean": float(np.mean(num_nodes_arr)),
            "median": float(np.median(num_nodes_arr)),
            "std": float(np.std(num_nodes_arr))
        },
        "edge_statistics": {
            "min": int(np.min(num_edges_arr)),
            "max": int(np.max(num_edges_arr)),
            "mean": float(np.mean(num_edges_arr)),
            "median": float(np.median(num_edges_arr)),
            "std": float(np.std(num_edges_arr))
        }
    }
    return stats


def save_dataset_metadata(stats: Dict[str, Any], filepath: str = "results/metrics/dataset_stats.json") -> None:
    """Saves dataset statistics to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset metadata to {filepath}")


if __name__ == "__main__":
    ds = load_nci1_dataset("data")
    print(f"Loaded NCI1: {len(ds)} graphs.")
    stats = inspect_dataset(ds)
    save_dataset_metadata(stats)
    print("Class distribution:", stats["class_distribution"])
    print("Node statistics:", stats["node_statistics"])
