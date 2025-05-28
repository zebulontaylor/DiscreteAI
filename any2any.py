#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from unifiedbackprop import (
    DEVICE, NUM_GATES, generate_inputs, correct_behavior,
    compute_nand, hinge_overlap_loss, LEARNING_RATE,
    NUM_EPOCHS, CHECKPOINT_DIR, save_checkpoint
)
import wandb
import dotenv

dotenv.load_dotenv()

# Configuration
INPUT_LEN = 8 # Number of original input bits
OUTPUT_LEN = INPUT_LEN // 2  # Number of output bits
NUM_ITER = NUM_GATES+1 # Number of iterations for signal propagation
CYCLE_WEIGHT = 1e1  # Weight for cycle-penalty regularization
# Curriculum learning: threshold to advance output bits
HARD_LOSS_THRESHOLD = 1
TAU = 1.0
USE_GUMBEL = True
USE_FREEZE = False


def create_graph_model(input_len=INPUT_LEN, num_gates=NUM_GATES, num_gate_types=1):
    """
    Initialize a fully-connected gate graph where each gate can connect
    to any original input or any other gate's output.
    """
    valid_inputs = input_len + num_gates
    layers = []
    for _ in range(num_gates):
        layer = torch.rand(
            (num_gate_types, valid_inputs, 2),
            dtype=torch.float32, device=DEVICE
        ) * 0.05
        layer.requires_grad = True
        layers.append(layer)
    return layers


def cycle_penalty(layers_logits, input_len=INPUT_LEN, weight=CYCLE_WEIGHT):
    """
    Penalize self-loops and mutual loops based on the learned adjacency.
    """
    num_gates = len(layers_logits)
    device = layers_logits[0].device
    adj = torch.zeros((num_gates, num_gates), device=device)
    for i, layer in enumerate(layers_logits):
        probs = torch.softmax(layer, dim=1)  # (1, valid_inputs, 2)
        # focus on connections from gates (skip original inputs)
        a_probs = probs[0, input_len:, 0]
        b_probs = probs[0, input_len:, 1]
        adj[i] = a_probs + b_probs
    # self loops
    diag = torch.diagonal(adj)
    self_pen = (diag**2).sum()
    # mutual loops
    mutual = (adj * adj.T * (1 - torch.eye(num_gates, device=device))).sum()
    return weight * (self_pen + mutual)


def evaluate_graph(
    layers_logits, num_iterations, generate_inputs_fn, correct_behavior_fn,
    input_len=INPUT_LEN, num_tests=256, target_bits=None
):
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    # Sample binary inputs
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)

    # Initialize state: clamp inputs and set initial gate activations
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    # Precompute adjacency matrix for soft connections
    stacked_logits = torch.cat(layers_logits, dim=0)
    probs = torch.softmax(stacked_logits, dim=1)
    a_probs = probs[..., 0]
    b_probs = probs[..., 1]
    M = a_probs.unsqueeze(2) * b_probs.unsqueeze(1)
    # Disallow outgoing connections from output gates: zero contributions where features correspond to output gates
    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+OUTPUT_LEN, :] = 0
    mask[:, input_len:input_len+OUTPUT_LEN] = 0
    M = M * mask.unsqueeze(0)

    # Propagate signals (vectorized)
    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M)
        state = torch.cat([sample_inputs, gate_out], dim=1)

    # Compute loss against correct behavior using the current number of bits
    expected = correct_behavior_fn(sample_inputs, (num_tests, OUTPUT_LEN))
    actual = state[:, input_len:input_len+OUTPUT_LEN]
    # Apply curriculum mask if specified
    if target_bits is not None and target_bits < OUTPUT_LEN:
        expected = expected[:, :target_bits]
        actual = actual[:, :target_bits]

    # binary cross-entropy loss (clamped to avoid log(0))
    bce = F.binary_cross_entropy(actual.clamp(min=1e-7, max=1-1e-7), expected, reduction='sum')
    return bce


def loss_hard_graph(
    layers_logits, num_iterations, generate_inputs_fn, correct_behavior_fn,
    input_len=INPUT_LEN, num_tests=256, target_bits=None
):
    """
    Hard-selection loss for the graph: discrete connections via argmax and iterative propagation.
    """
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    # Build hard adjacency matrices
    stacked = torch.cat(layers_logits, dim=0)  # [num_gates, total_dim, 2]
    probs = torch.softmax(stacked, dim=1)
    a_probs = probs[..., 0]
    b_probs = probs[..., 1]
    a_idx = a_probs.argmax(dim=1)
    b_idx = b_probs.argmax(dim=1)
    a_hard = torch.zeros_like(a_probs)
    b_hard = torch.zeros_like(b_probs)
    idx = torch.arange(num_gates, device=device)
    a_hard[idx, a_idx] = 1.0
    b_hard[idx, b_idx] = 1.0
    M_hard = a_hard.unsqueeze(2) * b_hard.unsqueeze(1)  # [num_gates, total_dim, total_dim]
    # Disallow outgoing connections from output gates: zero contributions where features correspond to output gates
    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+OUTPUT_LEN, :] = 0
    mask[:, input_len:input_len+OUTPUT_LEN] = 0
    M_hard = M_hard * mask.unsqueeze(0)

    # Sample inputs and initialize state
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)

    # Initialize state: clamp inputs and set initial gate activations
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    # Iterative propagation with hard weights
    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M_hard)
        # rebuild state for next iteration without in-place ops
        state = torch.cat([sample_inputs, gate_out], dim=1)

    # Compute hard MSE against correct behavior
    expected = correct_behavior_fn(sample_inputs, (num_tests, OUTPUT_LEN))
    actual = state[:, input_len:input_len+OUTPUT_LEN]

    if target_bits is not None and target_bits < OUTPUT_LEN:
        expected = expected[:, :target_bits]
        actual = actual[:, :target_bits]
    
    return (actual - expected).pow(2).sum()


# Add Gumbel-Softmax selection loss function
def loss_gumbel_graph(
    layers_logits, num_iterations, generate_inputs_fn, correct_behavior_fn,
    input_len=INPUT_LEN, num_tests=256, target_bits=None, tau=TAU, hard=False
):
    """
    Gumbel-Softmax selection loss for the graph: discrete connections via Gumbel-Softmax and iterative propagation.
    """
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    # Build Gumbel-Softmax adjacency matrices
    stacked = torch.cat(layers_logits, dim=0)  # [num_gates, total_dim, 2]
    # Permute to apply gumbel_softmax on last dimension
    logits = stacked.permute(0, 2, 1)  # [num_gates, 2, total_dim]
    samples = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [num_gates, 2, total_dim]
    a_hard = samples[:, 0, :]  # [num_gates, total_dim]
    b_hard = samples[:, 1, :]  # [num_gates, total_dim]
    M_gumbel = a_hard.unsqueeze(2) * b_hard.unsqueeze(1)  # [num_gates, total_dim, total_dim]
    # Disallow outgoing connections from output gates: zero contributions where features correspond to output gates
    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+OUTPUT_LEN, :] = 0
    mask[:, input_len:input_len+OUTPUT_LEN] = 0
    M_gumbel = M_gumbel * mask.unsqueeze(0)

    # Sample inputs and initialize state
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    # Iterative propagation with Gumbel-Softmax weights
    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M_gumbel)
        state = torch.cat([sample_inputs, gate_out], dim=1)

    # Compute loss against correct behavior
    expected = correct_behavior_fn(sample_inputs, (num_tests, OUTPUT_LEN))
    actual = state[:, input_len:input_len+OUTPUT_LEN]
    if target_bits is not None and target_bits < OUTPUT_LEN:
        expected = expected[:, :target_bits]
        actual = actual[:, :target_bits]

    bce = F.binary_cross_entropy(actual.clamp(min=1e-7, max=1-1e-7), expected, reduction='sum')
    return bce

# Helper function to freeze gates upstream of a resolved output bit
def freeze_upstream_gates(layers_logits, input_len, target_gate_idx):
    """
    Freeze parameters (requires_grad=False) for all gates upstream of the specified gate index,
    including the gate itself.
    """
    stacked = torch.cat(layers_logits, dim=0)
    probs = torch.softmax(stacked, dim=1)
    a_probs = probs[..., 0]
    b_probs = probs[..., 1]
    a_idx = a_probs.argmax(dim=1)
    b_idx = b_probs.argmax(dim=1)

    upstream = set()
    stack = [target_gate_idx]
    while stack:
        idx = stack.pop()
        if idx in upstream:
            continue
        upstream.add(idx)
        for parent in (a_idx[idx].item(), b_idx[idx].item()):
            if parent >= input_len:
                stack.append(parent - input_len)

    # Exclude output gates (first OUTPUT_LEN gates) from freezing
    upstream = {idx for idx in upstream if idx >= OUTPUT_LEN}

    for idx in upstream:
        layer = layers_logits[idx]
        with torch.no_grad():
            layer.fill_(-1e9)
            layer[..., a_idx[idx], 0] = +1e9
            layer[..., b_idx[idx], 1] = +1e9
        layer.requires_grad_(False)

    tqdm.write(f"Froze {len(upstream)} gates upstream of output bit {target_gate_idx}")

def train():
    run = wandb.init(project="nandai", config={
        "learning_rate": LEARNING_RATE,
        "num_gates": NUM_GATES,
        "num_iterations": NUM_ITER,
        "cycle_weight": CYCLE_WEIGHT,
        "hard_loss_threshold": HARD_LOSS_THRESHOLD,
        "tau": TAU,
        "use_gumbel": USE_GUMBEL,
        "any2any": True,
        "use_freeze": USE_FREEZE
    })

    tqdm.write(f"Running any2any on {'cuda' if DEVICE.type == 'cuda' else 'cpu'}")

    # Initialize curriculum to train on LSB first
    current_bits = 1
    tqdm.write(f"Curriculum: training on first {current_bits} output bit(s)")
    model = create_graph_model()
    optimizer = torch.optim.RMSprop(model, lr=LEARNING_RATE)

    pbar = trange(0, NUM_EPOCHS, desc=f'Training(bits={current_bits})')
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        # Evaluate loss using Gumbel-Softmax selection on current curriculum bits
        if USE_GUMBEL:
            loss = loss_gumbel_graph(
                model, NUM_ITER, generate_inputs, correct_behavior,
                target_bits=current_bits
            )
        else:
            loss = evaluate_graph(
                model, NUM_ITER, generate_inputs, correct_behavior,
                target_bits=current_bits
            )

        with torch.no_grad():
            hard_mse = loss_hard_graph(model, NUM_ITER, generate_inputs, correct_behavior, target_bits=current_bits)
        
        # Advance curriculum if threshold reached
        if hard_mse.item() < HARD_LOSS_THRESHOLD and current_bits < OUTPUT_LEN:
            # Freeze gates upstream of the just-solved output bit
            if USE_FREEZE:
                freeze_upstream_gates(model, INPUT_LEN, current_bits - 1)
            current_bits += 1
            tqdm.write(f"Threshold reached: advancing to first {current_bits} output bit(s)")

            with torch.no_grad():
                hard_mse = loss_hard_graph(model, NUM_ITER, generate_inputs, correct_behavior, target_bits=current_bits)

        div_loss = hinge_overlap_loss(model)
        cyc = cycle_penalty(model)
        total_loss = loss + cyc + div_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model, max_norm=0.5)
        optimizer.step()

        run.log({
            "loss": loss.item(),
            "hard_loss": hard_mse.item(),
            "current_bits": current_bits
        })

        pbar.set_description(f'Training(bits={current_bits}) - BCE: {loss.item():7.3f}, Hard Loss: {hard_mse.item():5.1f}, Div Loss: {div_loss.item():4.1f}, Cycle Loss: {cyc.item():4.1f}')

        if epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, [total_loss.item()])
        
        if hard_mse.item() < HARD_LOSS_THRESHOLD and current_bits == OUTPUT_LEN:
            run.finish()
            break

    print("Training complete")

if __name__ == '__main__':
    train()