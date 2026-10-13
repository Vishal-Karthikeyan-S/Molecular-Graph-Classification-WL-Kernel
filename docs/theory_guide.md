# Theoretical Foundations: Weisfeiler–Lehman Subtree Kernel & Expressivity

This document outlines the rigorous mathematical properties, complexity bounds, and theoretical connections between the 1-dimensional Weisfeiler–Lehman (1-WL) graph kernel and Graph Neural Networks (GNNs).

---

## 1. Mercer Kernel Positive Semi-Definiteness

### Definition
A symmetric function $K: \mathcal{G} \times \mathcal{G} \to \mathbb{R}$ is a positive semi-definite (PSD) Mercer kernel if, for any finite collection of graphs $\{G_1, \dots, G_N\}$ and any vector $c = [c_1, \dots, c_N]^\top \in \mathbb{R}^N$:
$$\sum_{i=1}^N \sum_{j=1}^N c_i c_j K(G_i, G_j) \ge 0$$

### Proof Sketch
Let $\phi(G) \in \mathbb{R}^D$ denote the concatenated subtree histogram feature vector:
$$\phi(G) = \left[ \phi_0(G)^\top \,\|\, \phi_1(G)^\top \,\|\, \dots \,\|\, \phi_h(G)^\top \right]^\top$$
The unnormalized kernel is the Euclidean inner product in $\mathbb{R}^D$:
$$K(G_i, G_j) = \langle \phi(G_i), \phi(G_j) \rangle = \phi(G_i)^\top \phi(G_j)$$
Thus:
$$c^\top K c = \sum_{i=1}^N \sum_{j=1}^N c_i c_j \phi(G_i)^\top \phi(G_j) = \left\| \sum_{i=1}^N c_i \phi(G_i) \right\|_2^2 \ge 0$$
Since Euclidean norms are non-negative, $K$ is positive semi-definite ($K \succeq 0$).

The cosine-normalized kernel:
$$K_{\text{norm}}(G_i, G_j) = \frac{\langle \phi(G_i), \phi(G_j) \rangle}{\|\phi(G_i)\|_2 \|\phi(G_j)\|_2} = \langle \tilde{\phi}(G_i), \tilde{\phi}(G_j) \rangle$$
represents the inner product between normalized vectors on the unit hypersphere $\mathbb{S}^{D-1}$, preserving the Mercer condition.

---

## 2. Computational Complexity Bounds

For a collection of $N$ graphs with maximum vertices $n$ and maximum edges $m$:

| Stage | Operation | Time Complexity | Space Complexity |
| :--- | :--- | :---: | :---: |
| **Color Refinement** | Multiset collection & string sorting ($h$ iterations) | $\mathcal{O}(h \cdot N \cdot m \log \text{deg}_{\max})$ | $\mathcal{O}(N \cdot n)$ |
| **Dictionary Indexing** | Subtree signature prefix tree / hash table lookup | $\mathcal{O}(h \cdot N \cdot n)$ | $\mathcal{O}(|\Sigma_h|)$ |
| **Feature Extraction** | Sparse histogram construction (`scipy.sparse.csr_matrix`) | $\mathcal{O}(N \cdot n \cdot h)$ | $\mathcal{O}(N \cdot n \cdot h)$ |
| **Gram Matrix** | Pairwise sparse matrix dot product ($K = \tilde{\Phi} \tilde{\Phi}^\top$) | $\mathcal{O}(N^2 \cdot \text{nnz}(\tilde{\Phi}) / N)$ | $\mathcal{O}(N^2)$ |
| **SVM Dual Solver** | Convex quadratic programming with precomputed $K$ | $\mathcal{O}(N^2 \text{ to } N^3)$ | $\mathcal{O}(N^2)$ |

---

## 3. Expressive Capacity: 1-WL vs. Graph Neural Networks

1. **Morris et al. (2019)**: Proved that standard message passing neural networks (MPNNs) are at most as powerful as the 1-WL graph isomorphism test.
2. **Xu et al. (2019)**: Showed that standard GCNs with mean aggregation fail to distinguish multisets of differing cardinalities, making them strictly less powerful than 1-WL.
3. **Finite Sample Regularization**: On datasets with small molecular graphs ($N=4,110$), non-parametric subtree histograms with convex maximum-margin classifiers exhibit lower sample variance and avoid optimization plateaus.
