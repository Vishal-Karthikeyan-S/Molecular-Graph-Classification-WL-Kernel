# NCI1 Dataset Information

The NCI1 (National Cancer Institute) molecular graph dataset is a benchmark for graph classification.
Each graph corresponds to a chemical compound screened for activity against non-small cell lung cancer cell lines.

- **Source**: [TU Dortmund Graph Kernel Benchmark Library](https://chrsmrrs.github.io/graphkerneldatasets/)
- **Total Graphs**: 4,110
- **Task**: Binary graph classification (0: inactive, 1: active against cancer cell lines)
- **Node Attributes**: Categorical atom types represented as 37-dimensional one-hot vectors.
- **Edges**: Chemical bonds connecting atoms.

The automated data loader downloads and prepares these files in this directory.
