from graphviz import Digraph

# Define the graph
dot = Digraph(format='pdf')
dot.attr(rankdir="TB", size='6')

# Input
dot.node("Input", "Input\n(Flattened window)")

# LSTM
dot.node("BiLSTM", "Bidirectional LSTM\n(num_layers=2, dropout=0.2)")

# Attention
dot.node("Attention", "Temporal Attention\n(Linear → Softmax → Weighted Sum)")

# FC Block
dot.node("FC1", "Linear(256 → 64)\nReLU")
dot.node("FC2", "Linear(64 → 64)\nReLU + Residual")

# Output Heads
dot.node("Head1", "Output Head 1\n(c, T_PM)")
dot.node("Head2", "Output Head 2\n(d50, d90, d10)")
dot.node("Head3", "Output Head 3\n(T_TM)")

# Output Representation
dot.node("SmoothHead", "Smooth Head")
dot.node("NoisyHead", "Noisy Head")

# Edges
dot.edges([
    ("Input", "BiLSTM"),
    ("BiLSTM", "Attention"),
    ("Attention", "FC1"),
    ("FC1", "FC2"),
    ("FC2", "Head1"),
    ("FC2", "Head2"),
    ("FC2", "Head3"),
    ("Head1", "SmoothHead"),
    ("Head3", "SmoothHead"),
    ("Head2", "NoisyHead")
])

# Render to SVG
pdf_path = "C:/Users/shrey/OneDrive/Documents/TUD/SEM 4 DOCS/MLME/Project descriptions-20250530/project_release/release/results/ann/modelgraph/bilstm_multihead_simple"
dot.render(pdf_path, cleanup=True)

pdf_path
