# main.py
import torch
from tqdm import tqdm, trange
from logic import evaluate_instance, correct_behavior, generate_inputs
from parser import parse_gate_configuration, create_graph
from pprint import pprint
import json
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running pytorch on {'cuda' if torch.cuda.is_available() else 'cpu'}.")

num_gates = 12
num_gate_types = 4
num_possible_inputs = num_gates + 8  # 2 inputs, 4 bits each

def create_model():
    gate_probs = torch.rand((num_gates, num_gate_types-1), dtype=torch.float32, requires_grad=True, device=device)
    base_layers = [gate_probs]
    for i in range(num_gates):
        valid_inputs = i + 8
        base_layers.append(torch.rand((num_gate_types, valid_inputs-1), dtype=torch.float32, device=device, requires_grad=True))
        base_layers.append(torch.rand((num_gate_types, valid_inputs-1), dtype=torch.float32, device=device, requires_grad=True))
    return base_layers

losses = []

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10_000, sci_mode=False)

def save_checkpoint(epoch, model, optimizer, losses, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': [layer.detach().cpu() for layer in model],
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, checkpoint_path)
    tqdm.write(f"Checkpoint saved at epoch {epoch}: {losses[-1]}")

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model_state_dicts = checkpoint['model_state_dict']
    if len(model) != len(model_state_dicts):
        raise ValueError("Checkpoint model size does not match the current model size.")
    for layer, state_dict in zip(model, model_state_dicts):
        if layer.size() != state_dict.size():
            raise ValueError("Checkpoint layer size does not match the current model layer size.")
        layer.copy_(state_dict.to(device))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint['losses']
    print(f"Checkpoint loaded from epoch {start_epoch}")
    return start_epoch, losses

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with optional checkpoint loading.")
parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to load")
parser.add_argument('--new', action='store_true', help="Start a new model from scratch")
args = parser.parse_args()

learning_rate = .01
num_epochs = 10000

optim = torch.optim.AdamW
optim_args = {
    "lr": learning_rate,
    #"fused": True,
    #"rho": 0.95,
}

if args.new or not args.checkpoint:
    base_layers = create_model()
    losses = []
    start_epoch = 0
    optimizer = optim(base_layers, **optim_args)
else:
    base_layers = create_model()
    optimizer = optim(base_layers, **optim_args)
    if args.checkpoint and os.path.exists(args.checkpoint):
        with torch.no_grad():
            start_epoch, losses = load_checkpoint(args.checkpoint, base_layers, optimizer)
    else:
        start_epoch = 0

#optimizer = optim(base_layers, **optim_args)
param_count = sum(param.numel() for param in base_layers)
print("Parameters:", param_count)

for epoch in (pbar := trange(start_epoch, num_epochs, desc="Training Epochs")):
    optimizer.zero_grad(set_to_none=True)

    loss = evaluate_instance(base_layers, generate_inputs, correct_behavior, dropout=.0)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(base_layers, max_norm=1.0)

    optimizer.step()

    grad_magn = 0
    for param in base_layers:
        grad_magn += torch.sum(torch.abs(param.grad))

    pbar.set_description(f"Epoch {epoch}: loss = {loss.item():.4f}; grad magn.: {grad_magn:.4f}")

    if epoch % 50 == 0 and epoch != start_epoch:
        losses.append(loss.item())
        with open("output.json", "w") as f:
            f.write(json.dumps(parse_gate_configuration(base_layers)))
        save_checkpoint(epoch, base_layers, optimizer, losses)

print("Final Instance (gate_probs):\n", base_layers[0])
pprint(losses)
