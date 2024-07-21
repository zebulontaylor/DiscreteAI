# parser.py
import torch
import networkx as nx
import matplotlib.pyplot as plt

def parse_gate_configuration(gate_probs, input1, input2):
    num_gates, num_gate_types, num_possible_inputs = input1.shape[0], input1.shape[1], input1.shape[2]
    
    gates = []

    for i in range(num_gates):
        gate_type = torch.argmax(gate_probs[i]).item()
        input1_index = torch.argmax(input1[i, gate_type]).item()
        input2_index = torch.argmax(input2[i, gate_type]).item()

        gates.append({
            'gate_type': gate_type,
            'input1': input1_index,
            'input2': input2_index
        })
    
    return gates

def create_graph(gates: list[dict], input_len: int = 16, output_count: int = 8, prune_leaves: bool = True):
    G = nx.DiGraph()
    gate_types = [
        "AND",
        "OR",
        "NOT",
        "NAND",
        "NOR",
        "XOR",
        "XNOR"
    ]
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
        G.add_edge(list(G.nodes)[gate["input2"]], list(G.nodes)[idx+input_len])

    if prune_leaves:
        total_pruned = 0
        for i in range(10):
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

    pos = nx.spring_layout(G, k=.8, iterations=30, scale=2.)
    plt.figure(3, figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, font_size=12, node_color=node_colors, node_size=2000, width=2.0, font_weight='bold')
    plt.show()

if __name__ == "__main__":
    import json
    with open("output.json", "r") as f:
        gates = json.loads(f.read())

    for gate in gates:
        print(f"Gate Type: {gate['gate_type']}, Input1: {gate['input1']}, Input2: {gate['input2']}")

    create_graph(gates)
