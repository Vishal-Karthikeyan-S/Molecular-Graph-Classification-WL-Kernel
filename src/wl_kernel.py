"""
Module 3: Weisfeiler-Lehman (WL) Subtree Kernel From Scratch.

The 1-dimensional Weisfeiler-Lehman (1-WL) graph isomorphism test and its
corresponding subtree kernel (Shervashidze et al., 2011, JMLR) iteratively computes
compressed multiset signatures representing subtree patterns rooted at each node.

ALGORITHM OVERVIEW:
1. Iteration 0: Assign initial node labels l_0(v) from atom categorical features.
   Construct feature histogram phi_0(G) counting frequencies of each initial label.
2. Iterations 1..h:
   - For each node v in G, gather multiset of neighbor labels: {l_{t-1}(u) : u in N(v)}.
   - Sort the neighbor multiset deterministically: (s_1, s_2, ..., s_k).
   - Form tuple signature: (l_{t-1}(v), (s_1, s_2, ..., s_k)).
   - Map tuple signature to a new integer label using a global, deterministic dictionary
     maintained across all graphs in the corpus.
   - Construct histogram phi_t(G) counting occurrences of new labels in G.
3. Feature Representation:
   phi(G) = [phi_0(G) || phi_1(G) || ... || phi_h(G)]
4. Kernel Matrix:
   K(G_i, G_j) = phi(G_i)^T phi(G_j)
   Or normalized kernel: K_norm(G_i, G_j) = K(G_i, G_j) / sqrt(K(G_i, G_i) * K(G_j, G_j))
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy import sparse
from src.preprocessing import MolecularGraph


class WeisfeilerLehmanKernel:
    """
    Weisfeiler-Lehman Subtree Kernel implemented strictly from scratch.
    
    Supports:
    - Multiple iteration depths h (e.g. h=0, 1, 2, 3)
    - Deterministic, collision-free label compression via Python tuple hashing
    - Train/test feature extraction without test data leakage
    - Exact Gram matrix calculation (train-train and test-train)
    - Kernel normalization option
    """
    def __init__(self, h: int = 3, normalize: bool = True):
        """
        Args:
            h: Maximum subtree height / iteration depth (h >= 0).
            normalize: If True, normalizes Gram matrix: K(G_i, G_j) / sqrt(K_ii * K_jj).
        """
        assert h >= 0, f"Iteration depth h must be non-negative, got {h}"
        self.h = h
        self.normalize = normalize
        
        # State learned from training data (or fit collection)
        # label_dict: maps signature tuple -> integer label id
        self.label_dict: Dict[Tuple, int] = {}
        self.next_label_id: int = 0
        self.fitted: bool = False
        self.feature_dim: int = 0
        self.train_diag: Optional[np.ndarray] = None

    def _get_label_id(self, signature: Tuple, allow_new: bool = True) -> Optional[int]:
        """
        Returns the unique integer ID for a signature tuple.
        If allow_new is True and signature is not seen, assigns the next ID.
        If allow_new is False and signature not seen, returns None (unseen subtree).
        """
        if signature in self.label_dict:
            return self.label_dict[signature]
        if allow_new:
            new_id = self.next_label_id
            self.label_dict[signature] = new_id
            self.next_label_id += 1
            return new_id
        return None

    def fit_transform(self, graphs: List[MolecularGraph]) -> sparse.csr_matrix:
        """
        Fits the vocabulary of subtree patterns on training graphs and extracts feature vectors.
        
        Args:
            graphs: List of training MolecularGraph objects.
            
        Returns:
            scipy.sparse.csr_matrix of shape (N_train, total_features)
        """
        self.label_dict = {}
        self.next_label_id = 0
        self.fitted = True
        
        feature_matrix = self._extract_features(graphs, allow_new_labels=True)
        self.feature_dim = self.next_label_id
        # Resize to total vocabulary size
        feature_matrix = feature_matrix.tocsr()
        return feature_matrix

    def transform(self, graphs: List[MolecularGraph]) -> sparse.csr_matrix:
        """
        Extracts feature vectors for unseen test graphs using the fitted vocabulary.
        
        Args:
            graphs: List of test MolecularGraph objects.
            
        Returns:
            scipy.sparse.csr_matrix of shape (N_test, total_features)
        """
        if not self.fitted:
            raise RuntimeError("WeisfeilerLehmanKernel must be fitted before calling transform().")
            
        return self._extract_features(graphs, allow_new_labels=False)

    def _extract_features(self, graphs: List[MolecularGraph], allow_new_labels: bool) -> sparse.csr_matrix:
        """
        Core WL iterative feature extraction across graphs.
        
        For each graph:
        - Maintains current node labels at iteration t.
        - Accumulates counts in a sparse dictionary.
        """
        num_graphs = len(graphs)
        # We will build a COO matrix: rows, cols, data
        rows: List[int] = []
        cols: List[int] = []
        data_vals: List[float] = []

        # Current labels for each graph: list of lists
        # graphs_node_labels[i][u] = label ID of node u in graph i
        graphs_node_labels: List[List[int]] = []

        # Iteration 0: Initial categorical labels
        for g_idx, g in enumerate(graphs):
            current_g_labels = []
            for u in range(g.num_nodes):
                # Signature for initial label is just a tuple ("init", original_label)
                sig = ("init", g.node_labels[u])
                lbl_id = self._get_label_id(sig, allow_new=allow_new_labels)
                current_g_labels.append(lbl_id)
                if lbl_id is not None:
                    rows.append(g_idx)
                    cols.append(lbl_id)
                    data_vals.append(1.0)
            graphs_node_labels.append(current_g_labels)

        # Iterations 1 through h
        for it in range(1, self.h + 1):
            next_graphs_node_labels: List[List[int]] = []
            for g_idx, g in enumerate(graphs):
                next_g_labels = []
                cur_labels = graphs_node_labels[g_idx]

                for u in range(g.num_nodes):
                    cur_u_label = cur_labels[u]
                    # Gather neighbor labels
                    nbrs = g.adjacency_list.get(u, [])
                    nbr_labels = [cur_labels[v] for v in nbrs]
                    
                    # Sort neighbor labels deterministically (handle None for unseen test labels)
                    # For sorting, treat None as a sentinel
                    sorted_nbr_labels = tuple(sorted(nbr_labels, key=lambda x: (x is None, x)))
                    
                    # Relabeling signature: (iteration, node_current_label, sorted_neighbor_labels)
                    sig = (it, cur_u_label, sorted_nbr_labels)
                    new_lbl_id = self._get_label_id(sig, allow_new=allow_new_labels)
                    next_g_labels.append(new_lbl_id)

                    if new_lbl_id is not None:
                        rows.append(g_idx)
                        cols.append(new_lbl_id)
                        data_vals.append(1.0)

                next_graphs_node_labels.append(next_g_labels)
            graphs_node_labels = next_graphs_node_labels

        total_cols = self.next_label_id if allow_new_labels else self.feature_dim
        coo = sparse.coo_matrix((data_vals, (rows, cols)), shape=(num_graphs, total_cols), dtype=np.float64)
        # Sum duplicates (since multiple nodes in the same graph have the same label)
        csr = coo.tocsr()
        return csr

    def compute_kernel_matrix(self, phi_train: sparse.csr_matrix, phi_test: Optional[sparse.csr_matrix] = None) -> np.ndarray:
        """
        Computes the Gram kernel matrix via dot product:
        If phi_test is None:
            K_train = phi_train @ phi_train.T  (shape: N_train x N_train)
        If phi_test is provided:
            K_test_train = phi_test @ phi_train.T  (shape: N_test x N_train)
            
        Applies cosine normalization if self.normalize is True.
        """
        if phi_test is None:
            # Train-train symmetric Gram matrix
            K = (phi_train @ phi_train.T).toarray()
            diag = np.diag(K).copy()
            # Numerical safety: diagonal should be > 0 for non-empty graphs
            diag[diag < 1e-12] = 1e-12
            self.train_diag = diag

            if self.normalize:
                d_sqrt = np.sqrt(diag)
                K = K / np.outer(d_sqrt, d_sqrt)
                np.fill_diagonal(K, 1.0)
            return K
        else:
            # Test-train Gram matrix (shape: N_test x N_train)
            K_test_train = (phi_test @ phi_train.T).toarray()
            if self.normalize:
                # Test diagonals
                test_sq_norm = phi_test.multiply(phi_test).sum(axis=1).A1
                test_sq_norm[test_sq_norm < 1e-12] = 1e-12
                test_d_sqrt = np.sqrt(test_sq_norm)

                train_d_sqrt = np.sqrt(self.train_diag)
                K_test_train = K_test_train / np.outer(test_d_sqrt, train_d_sqrt)
            return K_test_train


def hand_constructed_example() -> Tuple[List[MolecularGraph], WeisfeilerLehmanKernel]:
    """
    Constructs a minimal hand-crafted pair of graphs demonstrating WL subtree relabeling.
    Graph A: Triangle (cycle C3) with identical atom labels [1, 1, 1]
    Graph B: 3-node path (P3) with identical atom labels [1, 1, 1]
    
    Demonstrates that at h=0, both have 3 nodes with label 1 (identical phi_0 = [3]),
    but at h=1, 1-WL distinguishes the cycle (nodes have degree 2) from the path
    (nodes have degrees 1 and 2).
    """
    # Graph A: C3 cycle
    gA = MolecularGraph(
        graph_id=101,
        num_nodes=3,
        node_labels=[1, 1, 1],
        adjacency_list={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        edge_list=[(0, 1), (0, 2), (1, 2)],
        graph_label=1
    )
    # Graph B: P3 path (0 - 1 - 2)
    gB = MolecularGraph(
        graph_id=102,
        num_nodes=3,
        node_labels=[1, 1, 1],
        adjacency_list={0: [1], 1: [0, 2], 2: [1]},
        edge_list=[(0, 1), (1, 2)],
        graph_label=0
    )
    
    wl = WeisfeilerLehmanKernel(h=1, normalize=False)
    phi = wl.fit_transform([gA, gB])
    return [gA, gB], wl


if __name__ == "__main__":
    graphs, wl = hand_constructed_example()
    phi = wl.transform(graphs).toarray()
    print("Hand-crafted Example (Cycle C3 vs Path P3):")
    print(f"Features shape: {phi.shape}")
    print(f"Graph A (C3) features: {phi[0]}")
    print(f"Graph B (P3) features: {phi[1]}")
    K = wl.compute_kernel_matrix(wl.transform(graphs))
    print("Kernel Matrix K:\n", K)
    print("Label Dictionary size:", len(wl.label_dict))
    for sig, lbl_id in wl.label_dict.items():
        print(f"  ID {lbl_id}: {sig}")
