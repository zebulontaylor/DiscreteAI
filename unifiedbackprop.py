#!/usr/bin/env python3

import os
import json
import argparse
from pprint import pprint

import torch
import numpy as np
from tqdm import tqdm, trange
from typing import Callable, List, Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GATES = 48
NUM_GATE_TYPES = 1
LEARNING_RATE = 1e-1
NUM_EPOCHS = 30000
DIVERSITY_WEIGHT = 1e-1
CHECKPOINT_DIR = 'checkpoints'

# --- Logic Functions ---
def normalize_layers(layers: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply softmax to each layer tensor along the feature dimension.
    """
    return [torch.softmax(layer, dim=1) for layer in layers]


def compute_nand(inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise NAND: 1 - inputs1 * inputs2.
    """
    return 1.0 - inputs1 * inputs2


def evaluate_instance(
    layers_logits: List[torch.Tensor],
    num_gates: int,
    generate_inputs_fn: Callable[[int], np.ndarray],
    correct_behavior_fn: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int = 8,
    num_tests: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Generate and move inputs once
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=layers_logits[0].device)
    # Normalize layer logits to probabilities
    layers = normalize_layers(layers_logits)
    # Initialize outputs with input bits
    outputs = sample_inputs
    for i in range(num_gates):
        probs = layers[i]
        a_probs = probs[:, :, 0]
        b_probs = probs[:, :, 1]
        M = a_probs.transpose(0, 1) @ b_probs  # valid_inputs x valid_inputs
        S = M.sum()
        O = outputs[:, :input_len + i]
        OM = O @ M
        quad = (O * OM).sum(dim=1)
        gate_output = (S - quad).unsqueeze(1)
        outputs = torch.cat([outputs, gate_output], dim=1)

    # Compute expected outputs and losses
    expected = correct_behavior_fn(sample_inputs, (num_tests, input_len // 2))
    actual = outputs[:, -expected.shape[1]:]
    diff = actual - expected
    mse = diff.pow(2).sum()
    # clamp probabilities to [eps, 1-eps] to avoid log(0) in BCE
    bce = F.binary_cross_entropy(actual.clamp(min=1e-7, max=1-1e-7), expected, reduction='sum')
    loss = bce
    return loss, mse


def correct_behavior(inputs: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Compute the correct sum-of-bits behavior modulo 2**(num_bits), producing individual output bits.
    """
    device = inputs.device
    num_bits = inputs.shape[1] // 2
    weights = torch.tensor([2**i for i in range(num_bits)], device=device, dtype=torch.float32)

    A = (inputs[:, :num_bits] * weights).sum(dim=1, keepdim=True)
    B = (inputs[:, num_bits:] * weights).sum(dim=1, keepdim=True)
    sum_mod = (A + B) % (2**num_bits)
    outputs = torch.zeros(output_shape, device=device, dtype=torch.float32)
    for bit in range(output_shape[1]):
        outputs[:, bit] = ((sum_mod // (2**bit)) % 2).squeeze()
    return outputs


def generate_inputs(num_samples: int, num_bits: int = 4) -> np.ndarray:
    """
    Generate binary test inputs. If num_samples == 2**(2*num_bits), returns all combinations;
    otherwise returns random binary patterns.
    """
    total_bits = 2 * num_bits
    if num_samples == 2**total_bits:
        inputs = np.zeros((num_samples, total_bits), dtype=np.float32)
        for i in range(2**num_bits):
            for j in range(2**num_bits):
                idx = i * (2**num_bits) + j
                bits_i = [(i >> bit) & 1 for bit in range(num_bits)]
                bits_j = [(j >> bit) & 1 for bit in range(num_bits)]
                inputs[idx, :num_bits] = bits_i
                inputs[idx, num_bits:] = bits_j
    else:
        inputs = np.random.randint(0, 2, size=(num_samples, total_bits)).astype(np.float32)
    return inputs


def loss_hard(
    layers_logits: List[torch.Tensor],
    num_gates: int,
    generate_inputs_fn: Callable[[int], np.ndarray],
    correct_behavior_fn: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int = 8,
    num_tests: int = 256
) -> torch.Tensor:
    # Generate and move inputs once
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=layers_logits[0].device)
    # Normalize layer logits to probabilities
    layers = normalize_layers(layers_logits)
    # Initialize outputs with input bits
    outputs = sample_inputs
    # Compute gate outputs via direct indexing
    for i in range(num_gates):
        probs = layers[i]
        # select highest-prob inputs
        a_idx = probs[:, :, 0].argmax(dim=1).item()
        b_idx = probs[:, :, 1].argmax(dim=1).item()
        O = outputs[:, :input_len + i]
        gate_output = (1.0 - O[:, a_idx] * O[:, b_idx]).unsqueeze(1)
        outputs = torch.cat([outputs, gate_output], dim=1)

    # Compute expected outputs and losses
    expected = correct_behavior_fn(sample_inputs, (num_tests, input_len // 2))
    actual = outputs[:, -expected.shape[1]:]
    diff = actual - expected
    mse_hard = diff.pow(2).sum()
    return mse_hard

def hinge_overlap_loss(layers, max_shared=1, weight=DIVERSITY_WEIGHT):
    """
    Soft-penalty: for each pair of gates i≠j, compute their dot-product
    and only penalize sims > max_shared.
    """
    total = 0.0
    for layer in layers:               # layer.shape = (T, V, C)
        # normalize and flatten each gate's probability map
        probs = torch.softmax(layer, dim=1).view(layer.size(0), -1)  # (T, V*C)
        # Gram matrix of dot products
        G = probs @ probs.T            # (T, T)
        # mask out self-similarities
        mask = 1 - torch.eye(G.size(0), device=G.device)
        # subtract allowed overlap and clamp
        sims = (G - max_shared) * mask
        total += (sims.clamp(min=0)**2).sum()
    return weight * total

# --- Model & Training ---
def create_model():
    """Initialize model layers with random weights."""
    layers = []
    for i in range(NUM_GATES):
        valid_inputs = i * NUM_GATE_TYPES + 8
        layer = torch.rand(
            (NUM_GATE_TYPES, valid_inputs, 2),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True
        )
        layers.append(layer)
    return layers


def save_checkpoint(epoch, model, optimizer, losses):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': [layer.detach().cpu() for layer in model],
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, path)
    tqdm.write(f"Checkpoint saved: epoch {epoch:5d}, loss {losses[-1]:.4f}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE)
    start_epoch = ckpt['epoch']
    state_dicts = ckpt['model_state_dict']
    if len(state_dicts) != len(model):
        raise ValueError("Model size mismatch.")
    for layer, state in zip(model, state_dicts):
        if layer.size() != state.size():
            raise ValueError("Layer size mismatch.")
        # perform in-place update without gradient tracking
        with torch.no_grad():
            layer.copy_(state.to(DEVICE))
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    tqdm.write(f"Checkpoint loaded: resume from epoch {start_epoch}")
    return start_epoch, ckpt['losses']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch model for gate configuration inference."
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        '--new', action='store_true',
        help="Ignore checkpoint and start a new model"
    )
    return parser.parse_args()


def train(args):
    print(f"Running on {'cuda' if DEVICE.type == 'cuda' else 'cpu'}")
    model = create_model()
    optimizer = torch.optim.RMSprop(model, lr=LEARNING_RATE)
    # set up cyclical learning rate scheduler
    scheduler = CyclicLR(
        optimizer,
        base_lr=LEARNING_RATE/10,
        max_lr=LEARNING_RATE,
        step_size_up=800,
        mode='triangular',
        cycle_momentum=False
    )
    start_epoch = 0
    losses = []

    if args.checkpoint and not args.new and os.path.exists(args.checkpoint):
        start_epoch, losses = load_checkpoint(args.checkpoint, model, optimizer)

    pbar = trange(start_epoch, NUM_EPOCHS, desc='Training')
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, mse = evaluate_instance(
            model, NUM_GATES, generate_inputs, correct_behavior
        )
        # hinge-overlap regularization: allow at most one shared input
        div_loss = hinge_overlap_loss(model)
        loss = loss + div_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model, max_norm=0.5)
        optimizer.step()
        # update learning rate
        scheduler.step()

        hard_mse = loss_hard(
            model, NUM_GATES, generate_inputs, correct_behavior
        )
        pbar.set_description(f"Training - MSE: {mse.item():7.3f}, Hard Loss: {hard_mse.item()}")

        if epoch % 50 == 0 and epoch != start_epoch:
            losses.append(loss.item())
            save_checkpoint(epoch, model, optimizer, losses)

    print("Final instance (gate_probs):")
    pprint(model[0])
    pprint(losses)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main() 