# parser.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
import json

torch.set_printoptions(sci_mode=False, linewidth=500)

def parse_gate_configuration(base_layers):
    gate_probs = base_layers[0]
    input1_layers = base_layers[::2]
    input2_layers = base_layers[1::2]
    num_gates = gate_probs.shape[0]

    gates = []

    for i in range(num_gates):
        gate_type = torch.argmax(gate_probs[i]).item()
        input1_index = torch.argmax(input1_layers[i][gate_type]).item()
        input2_index = torch.argmax(input2_layers[i][gate_type]).item()

        gates.append({
            'gate_type': gate_type,
            'input1': input1_index,
            'input2': input2_index
        })

    return gates

def assign_layers(G):
    layers = {node: 0 for node in G.nodes if "Input" in node}
    
    def dfs(node):
        if node in layers:
            return layers[node]
        predecessors = list(G.predecessors(node))
        if not predecessors:
            layers[node] = 0
        else:
            layers[node] = max(dfs(pred) for pred in predecessors) + 1
        return layers[node]
    
    for node in G.nodes:
        dfs(node)
    
    for node, layer in layers.items():
        G.nodes[node]['layer'] = layer

def create_graph(gates, input_len=8, output_count=4, prune_leaves=True):
    G = nx.DiGraph(directed=True)
    gate_types = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR"]
    node_colors = []
    for i in range(input_len):
        G.add_node(f"Input {i+1}")
        node_colors.append("#a7fc9c")
    for i in range(len(gates)):
        if i >= len(gates) - output_count:
            node_colors.append("#9c9ffc")
        else:
            node_colors.append("#c7c9c7")
        G.add_node(f"{gate_types[gates[i]['gate_type']]} ({i})")
    for idx, gate in enumerate(gates):
        G.add_edge(list(G.nodes)[gate["input1"]], list(G.nodes)[idx+input_len])
        if not list(G.nodes)[idx+input_len].startswith("NOT"):
            G.add_edge(list(G.nodes)[gate["input2"]], list(G.nodes)[idx+input_len])

    if prune_leaves:
        total_pruned = 0
        for i in range(16):
            to_remove = []
            for idx, n in enumerate(G.nodes):
                if idx >= len(G.nodes) - output_count:
                    continue
                if idx < input_len:
                    continue
                edges = G.out_edges(n)
                if len(edges) == 0:
                    to_remove.append((idx, n))
            for idx, n in to_remove[::-1]:
                total_pruned += 1
                node_colors.pop(idx)
                G.remove_node(n)
        print(f"Pruned {total_pruned} nodes.")

    assign_layers(G)
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(3, figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, font_size=12, node_color=node_colors, node_size=2000, width=2.0, font_weight='bold')
    plt.show()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    base_layers = checkpoint['model_state_dict']
    return base_layers

def print_layer_probabilities(base_layers):
    gate_probs = base_layers[0]
    print("Gate Probabilities:")
    print(gate_probs)

    for idx, layer in enumerate(base_layers[1:], start=1):
        input("")
        print(f"Layer {idx}:")
        print(layer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse gate configuration from a checkpoint.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint to load")
    parser.add_argument('--prune_leaves', action='store_true', help="Prune leaf nodes with no connections")
    parser.add_argument('--show_layers', action='store_true', help="Print layers of the model")
    args = parser.parse_args()

    base_layers = load_checkpoint(args.checkpoint)
    gates = parse_gate_configuration(base_layers)

    print("Gate Configuration:")
    for gate in gates:
        print(f"Gate Type: {gate['gate_type']}, Input1: {gate['input1']}, Input2: {gate['input2']}")

    if args.show_layers:
        print_layer_probabilities(base_layers)
    create_graph(gates, prune_leaves=args.prune_leaves)
