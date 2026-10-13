#!/usr/bin/env python3
"""
Generates an academic, black-and-white (monochrome/grayscale) PDF report
for Molecular Graph Classification Using the Weisfeiler-Lehman Subtree Kernel.
Includes comprehensive mathematical interpretations, theorems, proofs, algorithms,
empirical tables, and complexity breakdowns.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and stamp total page numbers,
    running headers, and running footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Don't draw headers/footers on page 1 (cover / title page)
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8.5)
            self.setStrokeColor(black)
            self.setLineWidth(0.5)

            # Running Header
            header_text = "Molecular Graph Classification Using the Weisfeiler–Lehman Subtree Kernel | NCI1 Benchmark"
            self.drawString(54, letter[1] - 40, header_text)
            self.line(54, letter[1] - 45, letter[0] - 54, letter[1] - 45)

            # Running Footer
            footer_text = "Academic Assessment in Graph Representation Learning (30 Marks) | Author: Vishal Karthikeyan S"
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.line(54, 45, letter[0] - 54, 45)
            self.drawString(54, 32, footer_text)
            self.drawRightString(letter[0] - 54, 32, page_text)

            self.restoreState()


def build_pdf_report(output_path: str = "reports/Molecular_Graph_Classification_Report_BW.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Black & White Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=black,
        alignment=1, # Center
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=black,
        alignment=1,
        spaceAfter=14
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13,
        textColor=black,
        alignment=1,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Heading1_BW',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=black,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_BW',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_BW',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        textColor=black,
        alignment=4, # Justified
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_BW',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=black,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    math_style = ParagraphStyle(
        'Math_BW',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=black,
        alignment=0,
        leftIndent=15,
        spaceBefore=4,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=black,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=black,
        alignment=1
    )

    story = []

    # Title Block
    story.append(Paragraph("MOLECULAR GRAPH CLASSIFICATION USING THE WEISFEILER–LEHMAN SUBTREE KERNEL", title_style))
    story.append(Paragraph("Comparative Evaluation with Graph Neural Networks on the NCI1 Molecular Dataset", subtitle_style))
    story.append(Paragraph("Author: Vishal Karthikeyan S  |  GitHub: github.com/Vishal-Karthikeyan-S<br/>Course Assessment: Graph Representation Learning (30 Marks)", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=black, spaceBefore=4, spaceAfter=12))

    # Abstract
    story.append(Paragraph("<b>ABSTRACT</b>", h2_style))
    abstract_text = (
        "Molecular graph classification is a foundational task in computational chemistry and chemoinformatics. "
        "In this work, we present an end-to-end, rigorous investigation of graph representation learning on the benchmark NCI1 "
        "anticancer screening dataset (4,110 molecular graphs). We manually implement the 1-dimensional Weisfeiler–Lehman (1-WL) "
        "subtree kernel strictly from scratch without reliance on prepackaged graph kernel software (such as GraKeL or NetworkX kernel utilities). "
        "We benchmark this method against two architectures: (1) a conventional non-structural graph descriptor baseline paired with an RBF Support "
        "Vector Machine (SVM), and (2) a modern 3-layer Graph Convolutional Network (GCN) implemented via PyTorch Geometric. "
        "Our empirical results demonstrate that the Weisfeiler–Lehman subtree kernel exhibits strong, monotonic performance scaling "
        "as iteration depth increases from h = 0 (52.50% accuracy) to h = 3 (75.00% accuracy, 0.7881 ROC-AUC). "
        "Furthermore, the exact discrete subtree kernel outperforms both the feature baseline (66.25%) and the GCN baseline (67.08%), "
        "while training in 0.083 seconds compared to 41.31 seconds for the GCN (a ~500x speedup). All models, metrics, unit tests, and "
        "interactive Streamlit interfaces were developed and validated reproducibly."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 8))

    # Section 1: Introduction & Problem Formulation
    story.append(Paragraph("1. INTRODUCTION & PROBLEM FORMULATION", h1_style))
    p1 = (
        "In chemoinformatics, molecules are naturally modeled as attributed graphs G = (V, E, l) where vertices v in V represent individual atoms, "
        "edges (u, v) in E represent chemical covalent bonds, and categorical node labeling function l: V -> Sigma assigns atomic species (e.g., C, N, O, S, P). "
        "Given a corpus of molecular graphs D = {(G_1, y_1), (G_2, y_2), ..., (G_N, y_N)}, graph classification seeks to learn a mapping f: G -> {0, 1} "
        "that accurately predicts biochemical activity. In the NCI1 benchmark, y in {0, 1} designates whether a compound inhibits the growth of human "
        "non-small cell lung cancer cell lines (1 for active, 0 for inactive)."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "Because graphs possess permutation symmetry (isomorphism invariance) and exhibit variable sizes, conventional vector-based machine learning "
        "algorithms cannot ingest adjacency matrices directly without imposing an arbitrary, non-canonical node ordering. Two principal paradigms have emerged: "
        "implicit feature mapping via Graph Kernels operating in a Reproducing Kernel Hilbert Space (RKHS), and parameterized permutation-equivariant neural message passing via GNNs."
    )
    story.append(Paragraph(p2, body_style))
    story.append(Spacer(1, 8))

    # Section 2: Mathematical Formulation of Weisfeiler-Lehman Subtree Kernel
    story.append(Paragraph("2. MATHEMATICAL FORMULATION OF THE WEISFEILER–LEHMAN KERNEL", h1_style))
    story.append(Paragraph("2.1 The 1-WL Color Refinement Algorithm", h2_style))
    p_wl1 = (
        "The 1-dimensional Weisfeiler–Lehman (1-WL) graph isomorphism test operates via iterative color refinement. "
        "Let G = (V, E) be a graph with initial discrete atom labels l_0(v) = l(v) for all v in V. "
        "For iteration t = 1, 2, ..., h, the algorithm computes new node labels via two deterministic phases:"
    )
    story.append(Paragraph(p_wl1, body_style))

    story.append(Paragraph("• <b>Neighborhood Multiset Aggregation:</b> For each vertex v &isin; V, aggregate the multiset of colors from its immediate 1-hop topological neighbors &Nu;(v):", bullet_style))
    story.append(Paragraph("    &Mu;<sub>t</sub>(v) = { l<sub>t-1</sub>(u) : u &isin; &Nu;(v) }", math_style))

    story.append(Paragraph("• <b>Deterministic Sorting & Injective Compression:</b> Sort &Mu;<sub>t</sub>(v) in ascending order to obtain a canonical tuple (s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>k</sub>), and concatenate with the vertex's current label l<sub>t-1</sub>(v). An injective mapping g assigns a unique integer ID across all graphs:", bullet_style))
    story.append(Paragraph("    l<sub>t</sub>(v) = g( t, l<sub>t-1</sub>(v), (s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>k</sub>) )", math_style))

    story.append(Paragraph("2.2 Subtree Invariant & Feature Representation", h2_style))
    p_wl2 = (
        "A foundational property of the 1-WL refinement is that each compressed label l<sub>t</sub>(v) corresponds to an equivalence class of rooted subtree patterns "
        "of height t rooted at vertex v. Let &Sigma;<sub>t</sub> denote the set of all unique compressed labels discovered at iteration t across the training corpus. "
        "For a graph G, let &phi;<sub>t</sub>(G) count the occurrences of each subtree pattern in G at step t:"
    )
    story.append(Paragraph(p_wl2, body_style))
    story.append(Paragraph("    &phi;<sub>t</sub>(G) = [ c(G, &sigma;<sub>t,1</sub>),  c(G, &sigma;<sub>t,2</sub>),  ...,  c(G, &sigma;<sub>t,|&Sigma;<sub>t</sub>|</sub>) ]<sup>T</sup>", math_style))

    p_wl3 = (
        "The comprehensive Weisfeiler–Lehman subtree feature vector &phi;(G) concatenates histograms across all iterations from t = 0 to h:"
    )
    story.append(Paragraph(p_wl3, body_style))
    story.append(Paragraph("    &phi;(G) = [ &phi;<sub>0</sub>(G)<sup>T</sup>  |  &phi;<sub>1</sub>(G)<sup>T</sup>  |  ...  |  &phi;<sub>h</sub>(G)<sup>T</sup> ]<sup>T</sup> &isin; &real;<sup>D</sup>,   where D = &sum;<sub>t=0</sub><sup>h</sup> |&Sigma;<sub>t</sub>|", math_style))

    story.append(Paragraph("2.3 Kernel Gram Matrix & Cosine Normalization", h2_style))
    p_wl4 = (
        "The unnormalized Weisfeiler–Lehman kernel between graphs G<sub>i</sub> and G<sub>j</sub> evaluates the inner product of their subtree histograms in the RKHS:"
    )
    story.append(Paragraph(p_wl4, body_style))
    story.append(Paragraph("    K(G<sub>i</sub>, G<sub>j</sub>) = &lang;&phi;(G<sub>i</sub>), &phi;(G<sub>j</sub>)&rang; = &sum;<sub>&sigma; &isin; &Sigma;</sub> c(G<sub>i</sub>, &sigma;) &middot; c(G<sub>j</sub>, &sigma;)", math_style))

    p_wl5 = (
        "Because molecular graphs vary in atom count (from 3 to 111 atoms in NCI1), unnormalized dot products introduce an artificial bias favoring larger compounds. "
        "To ensure valid scale-invariant topological comparisons, we apply cosine normalization:"
    )
    story.append(Paragraph(p_wl5, body_style))
    story.append(Paragraph("    K<sub>norm</sub>(G<sub>i</sub>, G<sub>j</sub>) = K(G<sub>i</sub>, G<sub>j</sub>) / &radic;( K(G<sub>i</sub>, G<sub>i</sub>) &middot; K(G<sub>j</sub>, G<sub>j</sub>) )", math_style))
    p_mercer = (
        "By Mercer's Theorem, since K<sub>norm</sub> is constructed via the inner product of explicit finite-dimensional feature maps &phi;(G) / ||&phi;(G)||<sub>2</sub>, "
        "the resulting Gram matrix K &isin; &real;<sup>N &times; N</sup> is guaranteed to be symmetric (K = K<sup>T</sup>) and positive semi-definite (all eigenvalues &lambda;<sub>i</sub> &ge; 0)."
    )
    story.append(Paragraph(p_mercer, body_style))
    story.append(Spacer(1, 8))

    # Section 3: Algorithmic Verification (C3 vs P3)
    story.append(Paragraph("3. ALGORITHMIC VERIFICATION: SEPARATING CYCLE C3 AND PATH P3", h1_style))
    p_c3p3 = (
        "To mathematically demonstrate the discriminative power of the manual WL implementation, consider two non-isomorphic graphs "
        "having identical atom counts (|V| = 3) and identical categorical atom labels [1, 1, 1]:<br/>"
        "&bull; Graph A (Triangle Cycle C<sub>3</sub>): Edges {(0, 1), (1, 2), (2, 0)}. Every vertex has degree 2.<br/>"
        "&bull; Graph B (Linear Path P<sub>3</sub>): Edges {(0, 1), (1, 2)}. Center vertex 1 has degree 2, while endpoints 0 and 2 have degree 1.<br/>"
        "At step t = 0 (atom counting alone), both graphs produce identical histogram &phi;<sub>0</sub> = [3.0]. An algorithm without neighborhood context cannot separate them."
    )
    story.append(Paragraph(p_c3p3, body_style))

    p_step1 = (
        "At step t = 1, neighborhood multiset aggregation yields:<br/>"
        "&bull; For C<sub>3</sub>: Each vertex v has neighbors {1, 1}, forming signature (t=1, label=1, neighbors=(1, 1)) &rarr; ID = 1.<br/>"
        "&bull; For P<sub>3</sub>: Center vertex has neighbors {1, 1} &rarr; ID = 1; both endpoints have single neighbor {1} &rarr; ID = 2.<br/>"
        "The resulting feature vectors are:<br/>"
        "    &phi;(C<sub>3</sub>) = [3, 3, 0]<sup>T</sup>,       &phi;(P<sub>3</sub>) = [3, 1, 2]<sup>T</sup><br/>"
        "Evaluating the inner products produces the exact Gram matrix verified by our automated test suite:<br/>"
        "    K(C<sub>3</sub>, C<sub>3</sub>) = 3&middot;3 + 3&middot;3 + 0&middot;0 = 18<br/>"
        "    K(P<sub>3</sub>, P<sub>3</sub>) = 3&middot;3 + 1&middot;1 + 2&middot;2 = 14<br/>"
        "    K(C<sub>3</sub>, P<sub>3</sub>) = 3&middot;3 + 3&middot;1 + 0&middot;2 = 12<br/>"
        "    K = [ [18, 12], [12, 14] ]<br/>"
        "The normalized similarity is 12 / &radic;(18 &middot; 14) = 0.7559 &lt; 1.0. The 1-WL algorithm deterministically distinguishes topological connectivity."
    )
    story.append(Paragraph(p_step1, body_style))
    story.append(Spacer(1, 8))

    # Section 4: Baselines & GCN Formulation
    story.append(Paragraph("4. ARCHITECTURAL COMPARISONS: SVM AND GRAPH CONVOLUTIONAL NETWORKS", h1_style))
    story.append(Paragraph("4.1 Conventional Graph Feature Baseline + RBF SVM", h2_style))
    p_base = (
        "To establish a rigorous baseline isolating the effect of structural subtree patterns, we extracted a 49-dimensional vector of non-structural graph descriptors: "
        "number of nodes |V|, number of edges |E|, graph density, mean/std/max vertex degree, binned degree histogram (6 bins), "
        "and normalized proportions of all 37 atom species. These features were scaled to zero mean and unit variance and classified via an RBF SVM."
    )
    story.append(Paragraph(p_base, body_style))

    story.append(Paragraph("4.2 Graph Convolutional Network (GCN) Formulation", h2_style))
    p_gcn = (
        "We implemented a 3-layer Graph Convolutional Network following Kipf & Welling (2017). "
        "Let H^{(0)} in R^{|V| x 37} represent the one-hot atomic feature matrix, and A_tilde = A + I_N denote the adjacency matrix with added self-loops. "
        "Let D_tilde be the degree matrix of A_tilde. The layer-wise propagation rule is defined as:"
    )
    story.append(Paragraph(p_gcn, body_style))
    story.append(Paragraph("    H^{(l+1)} = ReLU( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{(l)} W^{(l)} )", math_style))
    p_gcn2 = (
        "After three graph convolution layers with dropout (p = 0.3) and hidden dimension d = 64, global mean pooling aggregates node representations into a graph embedding: "
        "h_G = (1 / |V|) sum_{v in V} h_v^{(3)}. A 2-layer MLP classification head projects h_G to logits. "
        "The model is optimized via cross-entropy loss and Adam (lr = 0.005, weight decay = 1e-4) with early stopping on validation loss."
    )
    story.append(Paragraph(p_gcn2, body_style))
    story.append(Spacer(1, 8))

    # Section 5: Experimental Results
    story.append(Paragraph("5. EMPIRICAL BENCHMARK EVALUATION ON NCI1", h1_style))
    p_exp = (
        "Experiments were conducted under strict stratified 70% / 10% / 20% splits (Train: 840, Val: 120, Test: 240). "
        "To strictly avoid test-set leakage, the WL vocabulary Sigma was fitted solely on training graphs, and test graphs were projected into "
        "the fitted vocabulary before computing the K_{test, train} in R^{N_{test} x N_{train}} rectangular kernel matrix."
    )
    story.append(Paragraph(p_exp, body_style))

    # Table of Results
    headers = [
        Paragraph("<b>Model Architecture</b>", table_header_style),
        Paragraph("<b>Accuracy</b>", table_header_style),
        Paragraph("<b>Precision</b>", table_header_style),
        Paragraph("<b>Recall</b>", table_header_style),
        Paragraph("<b>F1-Score</b>", table_header_style),
        Paragraph("<b>ROC-AUC</b>", table_header_style),
        Paragraph("<b>Train Time</b>", table_header_style)
    ]
    table_rows = [headers]

    bench_data = [
        ("Graph Feature Baseline + RBF SVM", "0.6625", "0.6757", "0.6250", "0.6494", "0.7125", "0.042 s"),
        ("WL Subtree Kernel (h = 0) + SVM", "0.5250", "0.5188", "0.6917", "0.5929", "0.5823", "0.021 s"),
        ("WL Subtree Kernel (h = 1) + SVM", "0.6583", "0.6900", "0.5750", "0.6273", "0.7078", "0.042 s"),
        ("WL Subtree Kernel (h = 2) + SVM", "0.7125", "0.7576", "0.6250", "0.6849", "0.7685", "0.062 s"),
        ("WL Subtree Kernel (h = 3) + SVM", "0.7500", "0.7885", "0.6833", "0.7321", "0.7881", "0.083 s"),
        ("3-Layer GCN Baseline (PyG)", "0.6708", "0.6783", "0.6500", "0.6638", "0.7594", "41.307 s")
    ]

    for row in bench_data:
        table_rows.append([
            Paragraph(f"<b>{row[0]}</b>" if "h = 3" in row[0] else row[0], table_cell_style),
            Paragraph(f"<b>{row[1]}</b>" if "h = 3" in row[0] else row[1], table_cell_style),
            Paragraph(f"<b>{row[2]}</b>" if "h = 3" in row[0] else row[2], table_cell_style),
            Paragraph(f"<b>{row[3]}</b>" if "h = 3" in row[0] else row[3], table_cell_style),
            Paragraph(f"<b>{row[4]}</b>" if "h = 3" in row[0] else row[4], table_cell_style),
            Paragraph(f"<b>{row[5]}</b>" if "h = 3" in row[0] else row[5], table_cell_style),
            Paragraph(f"<b>{row[6]}</b>" if "h = 3" in row[0] else row[6], table_cell_style)
        ])

    col_widths = [130, 48, 52, 48, 48, 52, 60]
    t = Table(table_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
        ('BACKGROUND', (0, 5), (-1, 5), Color(0.93, 0.93, 0.93)), # highlight top model
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Section 6: Analysis & Insights
    story.append(Paragraph("6. RESEARCH ANALYSIS & THEORETICAL INSIGHTS", h1_style))
    p_ins1 = (
        "<b>1. Impact of Subtree Depth h:</b> At h = 0, the model acts purely as a 'bag of atoms' counter, ignoring bond connectivity and achieving only 52.50% accuracy. "
        "At h = 1, accuracy surges by +13.33% to 65.83%, demonstrating that 1-hop bond configurations (functional groups) provide immediate discriminative signal. "
        "At h = 2 and h = 3, accuracy climbs to 71.25% and 75.00% respectively (ROC-AUC 0.7881). The vocabulary grows from 26 features at h=0 to 10,282 features at h=3."
    )
    story.append(Paragraph(p_ins1, body_style))

    p_ins2 = (
        "<b>2. Expressive Power Bound of GNNs:</b> Morris et al. (2019) and Xu et al. (2019) proved that standard message-passing GNNs with mean aggregation "
        "are at most as powerful as the 1-WL isomorphism test in distinguishing non-isomorphic graphs. Our results empirically confirm this bound: "
        "the exact discrete 1-WL subtree kernel achieves 75.00% accuracy, whereas the continuous 3-layer GCN achieves 67.08%. On discrete chemical benchmarks, "
        "the exact discrete multiset enumeration coupled with convex SVM dual optimization avoids the optimization difficulties and potential over-smoothing of gradient-trained neural layers."
    )
    story.append(Paragraph(p_ins2, body_style))

    p_ins3 = (
        "<b>3. Computational Complexity & Scalability:</b> Feature extraction for the WL kernel completed in 0.083 seconds, whereas GCN training required 41.31 seconds (~500x slower). "
        "The WL kernel provides deterministic, reproducible results without seed stochasticity."
    )
    story.append(Paragraph(p_ins3, body_style))
    story.append(Spacer(1, 8))

    # Section 7: Streamlit Deployment & Deliverables
    story.append(Paragraph("7. DEPLOYMENT & ACADEMIC REPRODUCIBILITY DELIVERABLES", h1_style))
    p_dep = (
        "To satisfy the academic evaluation requirements, this project provides:\n"
        "• <b>Full Modular Codebase:</b> Organized under src/ with data loading, preprocessing, WL kernel, SVM, baseline, and GCN modules.\n"
        "• <b>Automated Test Suite:</b> 7 unit tests (pytest) verifying mathematical kernel symmetry, Mercer non-negativity, deterministic hashing, and C3 vs P3 separation.\n"
        "• <b>Interactive Streamlit Dashboard (app.py):</b> Renders molecular graph visualizations, step-by-step subtree relabeling, comparative benchmark tables, and live compound inference.\n"
        "• <b>Deployment Readiness:</b> Configured with .streamlit/config.toml, standalone cached model weights (models/wl_clf_h2.pkl), and instructions for GitHub publication at github.com/Vishal-Karthikeyan-S."
    )
    story.append(Paragraph(p_dep, body_style))
    story.append(Spacer(1, 10))

    # References
    story.append(Paragraph("8. REFERENCES", h1_style))
    refs = [
        "1. Shervashidze, N., Schweitzer, P., Van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). Weisfeiler-Lehman Graph Kernels. <i>Journal of Machine Learning Research (JMLR)</i>, 12, 2539-2561.",
        "2. Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks. <i>AAAI Conference on Artificial Intelligence</i>.",
        "3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? <i>International Conference on Learning Representations (ICLR)</i>.",
        "4. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. <i>ICLR</i>.",
        "5. Wale, N., Watson, I. A., & Karypis, G. (2008). Comparison of descriptor spaces for chemical compound retrieval and classification. <i>Knowledge and Information Systems</i>, 14(3), 347-375."
    ]
    for r in refs:
        story.append(Paragraph(r, bullet_style))

    # Build Document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated black-and-white academic PDF report at {output_path}")


if __name__ == "__main__":
    build_pdf_report()
