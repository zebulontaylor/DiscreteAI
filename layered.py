#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from unifiedbackprop import (
    DEVICE,
    generate_inputs,
    correct_behavior,
    compute_nand,
    hinge_overlap_loss,
    LEARNING_RATE,
    NUM_EPOCHS,
    save_checkpoint,
    load_checkpoint
)
from torch.optim.lr_scheduler import CyclicLR

# Configuration variables
LAYER_SIZES = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 12]
INPUT_LEN = 8
CHECKPOINT_PATH = None


def create_layered_model(layer_sizes, input_len=8):
    """
    Initialize a layered feed-forward gate model.
    Each layer has `num_gates` gates, and each gate can connect to any original input
    or any gate output from all previous layers.
    """
    layers = []
    for layer_idx, num_gates in enumerate(layer_sizes):
        valid_inputs = input_len + sum(layer_sizes[:layer_idx])
        # logits shape: (num_gates, valid_inputs, 2) for two input selections per gate
        layer = torch.rand((num_gates, valid_inputs, 2), dtype=torch.float32, device=DEVICE) * 0.05
        layer.requires_grad_(True)
        layers.append(layer)
    return layers


def evaluate_layered_graph(
    layers_logits,
    generate_inputs_fn,
    correct_behavior_fn,
    input_len=8,
    num_tests=256
):
    """
    Forward pass: compute outputs of each gate layer-by-layer in a feed-forward manner.
    Returns binary cross-entropy loss and mean-squared-error.
    """
    device = layers_logits[0].device
    # Sample inputs
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    # Accumulate all outputs (start with original inputs)
    outputs_all = sample_inputs

    # Layer-by-layer propagation with limited access
    layer_sizes = LAYER_SIZES
    for layer_idx, layer_logits in enumerate(layers_logits):
        # Determine which previous gate layers to include (last two)
        prev_start = max(0, layer_idx - 2)
        # Build allowed indices: original inputs + outputs of prev two gate layers
        allowed_idx = list(range(input_len))
        offset = input_len
        for j, size in enumerate(layer_sizes):
            if j >= layer_idx:
                break
            if j >= prev_start:
                allowed_idx.extend(range(offset, offset + size))
            offset += size
        # Gather features from allowed inputs
        feat = outputs_all[:, allowed_idx]

        # Compute soft connection weights
        probs = torch.softmax(layer_logits, dim=1)
        a_probs = probs[..., 0]
        b_probs = probs[..., 1]
        # Full weight tensor
        M = a_probs.unsqueeze(2) * b_probs.unsqueeze(1)
        # Restrict weights to allowed inputs indices so dims match feat
        M = M[:, allowed_idx, :][:, :, allowed_idx]
        # Compute NAND on selected features
        N = compute_nand(feat.unsqueeze(2), feat.unsqueeze(1))
        # Gate outputs
        gate_out = torch.einsum('bij,gij->bg', N, M)
        # Append new gate outputs
        outputs_all = torch.cat([outputs_all, gate_out], dim=1)

    # Compute expected behavior and build full target [a, b, a+b]
    output_len = input_len // 2
    expected_sum = correct_behavior_fn(sample_inputs, (num_tests, output_len))
    # Full expected: original inputs + sum bits
    expected_full = torch.cat([sample_inputs, expected_sum], dim=1)
    # Predicted sum bits from final layer
    pred_sum = outputs_all[:, -output_len:]
    # Full prediction: original inputs + predicted sum
    pred_full = torch.cat([sample_inputs, pred_sum], dim=1)
    # Compute losses over full output vector
    diff = pred_full - expected_full
    mse = diff.pow(2).sum()
    bce = F.binary_cross_entropy(
        pred_full.clamp(min=1e-7, max=1 - 1e-7), expected_full, reduction='sum'
    )
    return bce, mse


def evaluate_layered_graph_hard(
    layers_logits,
    generate_inputs_fn,
    correct_behavior_fn,
    input_len=8,
    num_tests=256
):
    """
    Hard propagation: discrete gate connections (argmax) and compute MSE loss.
    """
    device = layers_logits[0].device
    # Sample inputs
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    # Accumulate all outputs (start with original inputs)
    outputs_all = sample_inputs

    # Layer-by-layer hard propagation with limited access
    layer_sizes = LAYER_SIZES
    for layer_idx, layer_logits in enumerate(layers_logits):
        # Determine which previous gate layers to include (last two)
        prev_start = max(0, layer_idx - 2)
        # Build allowed indices
        allowed_idx = list(range(input_len))
        offset = input_len
        for j, size in enumerate(layer_sizes):
            if j >= layer_idx:
                break
            if j >= prev_start:
                allowed_idx.extend(range(offset, offset + size))
            offset += size
        feat = outputs_all[:, allowed_idx]

        # Compute hard connection masks
        probs = torch.softmax(layer_logits, dim=1)
        a_probs = probs[..., 0]
        b_probs = probs[..., 1]
        num_gates = a_probs.shape[0]
        a_idx = a_probs.argmax(dim=1)
        b_idx = b_probs.argmax(dim=1)
        a_hard = torch.zeros_like(a_probs)
        b_hard = torch.zeros_like(b_probs)
        idxs = torch.arange(num_gates, device=device)
        a_hard[idxs, a_idx] = 1.0
        b_hard[idxs, b_idx] = 1.0
        M_hard = a_hard.unsqueeze(2) * b_hard.unsqueeze(1)
        # Restrict hard weights to allowed inputs indices
        M_hard = M_hard[:, allowed_idx, :][:, :, allowed_idx]
        # Compute NAND on selected features
        N = compute_nand(feat.unsqueeze(2), feat.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M_hard)
        outputs_all = torch.cat([outputs_all, gate_out], dim=1)

    # Compute hard MSE against full target [a, b, a+b]
    output_len = input_len // 2
    expected_sum = correct_behavior_fn(sample_inputs, (num_tests, output_len))
    expected_full = torch.cat([sample_inputs, expected_sum], dim=1)
    pred_sum = outputs_all[:, -output_len:]
    pred_full = torch.cat([sample_inputs, pred_sum], dim=1)
    mse_hard = (pred_full - expected_full).pow(2).sum()
    return mse_hard


def train():
    print(f"Training layered model on {'cuda' if DEVICE.type == 'cuda' else 'cpu'}")
    # Initialize model and optimizer
    layers = create_layered_model(LAYER_SIZES, INPUT_LEN)
    optimizer = torch.optim.RMSprop(layers, lr=LEARNING_RATE)
    scheduler = CyclicLR(
        optimizer,
        base_lr=LEARNING_RATE / 10,
        max_lr=LEARNING_RATE,
        step_size_up=800,
        mode='triangular',
        cycle_momentum=False
    )
    start_epoch = 0
    losses = []

    # Optionally load checkpoint
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        start_epoch, losses = load_checkpoint(CHECKPOINT_PATH, layers, optimizer)

    pbar = trange(start_epoch, NUM_EPOCHS, desc='Training')
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, mse = evaluate_layered_graph(
            layers,
            generate_inputs,
            correct_behavior,
            INPUT_LEN
        )
        # Diversity regularization
        div_loss = hinge_overlap_loss(layers)
        total_loss = loss + div_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(layers, max_norm=0.5)
        optimizer.step()
        scheduler.step()

        # Compute hard MSE for progress reporting
        with torch.no_grad():
            hard_mse = evaluate_layered_graph_hard(
                layers, generate_inputs, correct_behavior, INPUT_LEN
            )
        pbar.set_description(
            f"MSE: {mse.item():7.3f}, BCE: {loss.item():7.3f}, Hard MSE: {hard_mse.item():7.3f}"
        )

        # Save checkpoint periodically
        if epoch % 50 == 0 and epoch != start_epoch:
            losses.append(total_loss.item())
            save_checkpoint(epoch, layers, optimizer, losses)

    print("Training complete")


# Entry point
if __name__ == '__main__':
    train() 