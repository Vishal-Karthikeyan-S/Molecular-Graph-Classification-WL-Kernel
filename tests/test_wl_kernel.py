"""
Tests for Module 3: Weisfeiler-Lehman Subtree Kernel.
Verifies label updates, feature dimensions, kernel symmetry, non-negative diagonals,
deterministic behavior, and proper handling of train-test matrices.
"""

import pytest
import numpy as np
from scipy import sparse
from src.preprocessing import MolecularGraph
from src.wl_kernel import WeisfeilerLehmanKernel, hand_constructed_example


@pytest.fixture
def sample_graphs():
    # Graph 1: Simple 3-node path: 0 - 1 - 2, labels [0, 1, 0]
    g1 = MolecularGraph(
        graph_id=1,
        num_nodes=3,
        node_labels=[0, 1, 0],
        adjacency_list={0: [1], 1: [0, 2], 2: [1]},
        edge_list=[(0, 1), (1, 2)],
        graph_label=1
    )
    # Graph 2: 3-node triangle: 0-1, 1-2, 2-0, labels [0, 1, 0]
    g2 = MolecularGraph(
        graph_id=2,
        num_nodes=3,
        node_labels=[0, 1, 0],
        adjacency_list={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        edge_list=[(0, 1), (1, 2), (0, 2)],
        graph_label=0
    )
    # Graph 3: 4-node star: center 0 (label 1), leaves 1,2,3 (labels 0, 0, 0)
    g3 = MolecularGraph(
        graph_id=3,
        num_nodes=4,
        node_labels=[1, 0, 0, 0],
        adjacency_list={0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},
        edge_list=[(0, 1), (0, 2), (0, 3)],
        graph_label=1
    )
    return [g1, g2, g3]


def test_hand_constructed_example():
    """Verify that 1-WL distinguishes the C3 cycle from the P3 path."""
    graphs, wl = hand_constructed_example()
    phi = wl.transform(graphs).toarray()
    # Initial label counts must be identical (both have three '1' atoms)
    assert phi[0, 0] == phi[1, 0] == 3.0
    # At h=1, C3 has 3 nodes with signature (0, (0, 0)), P3 has only 1 node with (0, (0, 0)) and 2 nodes with (0, (0,))
    assert phi[0, 1] == 3.0
    assert phi[0, 2] == 0.0
    assert phi[1, 1] == 1.0
    assert phi[1, 2] == 2.0


def test_kernel_symmetry_and_diagonal(sample_graphs):
    """Verify that K_train is symmetric, finite, and positive along the diagonal."""
    for h in [0, 1, 2]:
        wl = WeisfeilerLehmanKernel(h=h, normalize=True)
        phi_train = wl.fit_transform(sample_graphs)
        K_train = wl.compute_kernel_matrix(phi_train)
        
        # Check shape
        assert K_train.shape == (3, 3)
        # Check symmetry
        assert np.allclose(K_train, K_train.T, atol=1e-7)
        # Check normalized diagonal entries are 1.0
        assert np.allclose(np.diag(K_train), np.ones(3), atol=1e-7)
        # Check values are finite
        assert np.all(np.isfinite(K_train))
        # Off-diagonals are in [-1, 1] (or [0, 1] for counts)
        assert np.all(K_train >= 0.0) and np.all(K_train <= 1.000001)


def test_unnormalized_kernel_properties(sample_graphs):
    """Verify unnormalized kernel satisfies Mercer properties."""
    wl = WeisfeilerLehmanKernel(h=2, normalize=False)
    phi_train = wl.fit_transform(sample_graphs)
    K_train = wl.compute_kernel_matrix(phi_train)
    
    # Check positive semi-definiteness: eigenvalues >= -1e-7
    eigenvals = np.linalg.eigvalsh(K_train)
    assert np.all(eigenvals >= -1e-7)
    # Check diagonal elements are positive
    assert np.all(np.diag(K_train) > 0)


def test_train_test_split_matrix_shapes(sample_graphs):
    """Verify that test-train kernel matrix matches (N_test, N_train) dimensions."""
    train_graphs = sample_graphs[:2]
    test_graphs = sample_graphs[2:]
    
    wl = WeisfeilerLehmanKernel(h=1, normalize=True)
    phi_train = wl.fit_transform(train_graphs)
    K_train = wl.compute_kernel_matrix(phi_train)
    
    phi_test = wl.transform(test_graphs)
    K_test_train = wl.compute_kernel_matrix(phi_train, phi_test)
    
    assert K_train.shape == (2, 2)
    assert K_test_train.shape == (1, 2)
    assert np.all(np.isfinite(K_test_train))


def test_deterministic_feature_extraction(sample_graphs):
    """Verify that repeated calls yield identical features."""
    wl1 = WeisfeilerLehmanKernel(h=2, normalize=False)
    phi1 = wl1.fit_transform(sample_graphs).toarray()
    
    wl2 = WeisfeilerLehmanKernel(h=2, normalize=False)
    phi2 = wl2.fit_transform(sample_graphs).toarray()
    
    assert np.array_equal(phi1, phi2)
