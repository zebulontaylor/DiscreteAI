#!/usr/bin/env python3
"""
a2a_visualizer.py

Visualize any2any gate model as a directed graph, pruning gates not upstream of outputs.
"""

import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Add interactive support using pyvis if available
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


def load_layers(checkpoint_path):
    """
    Load model_state_dict from a checkpoint file and return the list of layer tensors.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        layers = ckpt['model_state_dict']
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    return layers


def build_graph(layers):
    """
    Build a directed graph of inputs and gates based on hard-selected connections.
    Returns the graph and the inferred input length.
    """
    num_gates = len(layers)
    valid_inputs = layers[0].shape[1]
    input_len = valid_inputs - num_gates

    # Stack logits and compute selection via argmax
    stacked = torch.cat(layers, dim=0)  # [num_gates, valid_inputs, 2]
    probs = torch.softmax(stacked, dim=1)
    a_probs = probs[..., 0]
    b_probs = probs[..., 1]
    a_idx = a_probs.argmax(dim=1).numpy()
    b_idx = b_probs.argmax(dim=1).numpy()

    G = nx.DiGraph()
    # Add input nodes
    for i in range(input_len):
        G.add_node(f"I{i}", type='input')
    # Add gate nodes
    for g in range(num_gates):
        G.add_node(f"G{g}", type='gate')
    # Add edges from selected inputs/gates to each gate
    for g in range(num_gates):
        for idx in (a_idx[g], b_idx[g]):
            if idx < input_len:
                src = f"I{idx}"
            else:
                src = f"G{idx - input_len}"
            G.add_edge(src, f"G{g}")
    return G, input_len


def prune_graph(G, input_len):
    """
    Prune out any gates not upstream of the output gates.
    """
    output_len = input_len // 2
    outputs = [f"G{g}" for g in range(output_len)]
    keep = set()
    for out in outputs:
        if out in G:
            keep |= nx.ancestors(G, out)
            keep.add(out)
    subG = G.subgraph(keep).copy()
    # Mark output gates for coloring
    for out in outputs:
        if out in subG.nodes:
            subG.nodes[out]['type'] = 'output'
    return subG


def visualize_graph(G, output_path=None):
    """
    Draw the graph using an interactive layout if possible; else use layered static layout.
    """
    # Interactive visualization if pyvis is installed and HTML output desired
    if PYVIS_AVAILABLE and (output_path is None or output_path.lower().endswith('.html')):
        try:
            net = Network(directed=True, height="750px", width="100%")
            for n, data in G.nodes(data=True):
                t = data.get('type')
                if t == 'input':
                    color = 'red'
                elif t == 'output':
                    color = 'green'
                else:
                    color = 'lightblue'
                net.add_node(n, label=n, color=color)
            for src, dst in G.edges():
                net.add_edge(src, dst)
            html_path = output_path if output_path and output_path.lower().endswith('.html') else 'graph.html'
            net.show(html_path)
            print(f"Interactive graph saved to {html_path}")
            return
        except Exception as e:
            print(f"Interactive visualization failed ({e}), falling back to static layout.")

    # Static visualization: simple force-directed layout
    pos = nx.spring_layout(G, seed=42, k=1)
    node_colors = []
    for n in G.nodes():
        t = G.nodes[n].get('type')
        if t == 'input':
            node_colors.append('red')
        elif t == 'output':
            node_colors.append('green')
        else:
            node_colors.append('lightblue')
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        arrows=True, node_size=600, font_size=8
    )
    plt.title('any2any NAND Gate Graph (Pruned)')
    # Save static graph, handling unsupported HTML extension
    if output_path:
        if output_path.lower().endswith(('.html', '.htm')):
            # Matplotlib cannot save HTML; switch to PNG
            png_path = output_path.rsplit('.', 1)[0] + '.png'
            plt.savefig(png_path, bbox_inches='tight')
            print(f"Static graph saved to {png_path} (PNG format)")
        else:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Static graph saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize any2any model gate connections"
    )
    parser.add_argument(
        "checkpoint", help="Path to any2any checkpoint (.pth)"
    )
    parser.add_argument(
        "-o", "--output", help="Output image file (e.g. graph.png)"
    )
    args = parser.parse_args()

    layers = load_layers(args.checkpoint)
    G, input_len = build_graph(layers)
    pruned = prune_graph(G, input_len)
    visualize_graph(pruned, args.output)


if __name__ == "__main__":
    main() 