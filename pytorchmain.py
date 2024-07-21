# main.py
import torch
from tqdm import trange
from logic import evaluate_instance, correct_behavior, generate_inputs
from parser import parse_gate_configuration, create_graph
from pprint import pprint
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running pytorch on {'cuda' if torch.cuda.is_available() else 'cpu'}.")

num_gates = 64
num_gate_types = 7
num_possible_inputs = num_gates + 16  # 2 inputs, 8 bits each

with torch.no_grad():
    sample_probs = torch.rand((num_gates, num_gate_types), dtype=torch.float32, requires_grad=True, device=device)
    sample_input1 = torch.zeros((num_gates, num_gate_types, num_possible_inputs), dtype=torch.float32, requires_grad=True, device=device)
    sample_input2 = torch.zeros((num_gates, num_gate_types, num_possible_inputs), dtype=torch.float32, requires_grad=True, device=device)

    for i in range(num_gates):
        valid_inputs = i + 16
        sample_input1[i, :, :valid_inputs] = torch.rand((num_gate_types, valid_inputs), dtype=torch.float32, device=device)
        sample_input2[i, :, :valid_inputs] = torch.rand((num_gate_types, valid_inputs), dtype=torch.float32, device=device)

    sample_probs = torch.nn.functional.normalize(sample_probs, p=1, dim=1)
    sample_input1 = torch.nn.functional.normalize(sample_input1, p=1, dim=0)
    sample_input2 = torch.nn.functional.normalize(sample_input2, p=1, dim=0)

print("Parameters:", sample_input1.numel() * 2 + sample_input2.numel() * 2 + sample_probs.numel())

learning_rate = 0.01
num_epochs = 40000

model_state = (sample_probs, sample_input1, sample_input2)

import signal
import sys

losses = []

def signal_handler(signal, frame):
    model = parse_gate_configuration(*model_state)
    pprint(model)
    pprint(losses)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10_000)

for i in range(3):
    model_state[i].requires_grad_()

optimizer = torch.optim.Adam(model_state, lr=learning_rate)

for epoch in (pbar := trange(num_epochs, desc="Training Epochs")):
    optimizer.zero_grad(set_to_none=True)

    loss = evaluate_instance(*model_state, generate_inputs, correct_behavior)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model_state, max_norm=1.0)
    with torch.no_grad():
        torch.nn.functional.normalize(model_state[0], p=1, dim=1, out=model_state[0])
        torch.nn.functional.normalize(model_state[1], p=1, dim=0, out=model_state[1])
        torch.nn.functional.normalize(model_state[2], p=1, dim=0, out=model_state[2])

    optimizer.step()
    for i in range(3):
        model_state[i].data.clamp_(0, 1)

    pbar.set_description(f"Epoch {epoch}: loss = {loss.item():.4f}; grad magn.: {torch.sum(torch.abs(model_state[0].grad))+torch.sum(torch.abs(model_state[1].grad))+torch.sum(torch.abs(model_state[2].grad)):.4f}")

    if epoch % 50 == 0 and epoch != 0:
        losses.append(loss.item())
        with open("output.json", "w") as f:
            f.write(json.dumps(parse_gate_configuration(*model_state)))

print("Final Instance (input1):\n", model_state[1])
print("Final Instance (input2):\n", model_state[2])
print("Final Instance (gate_probs):\n", model_state[0])
pprint(losses)
