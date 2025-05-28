#!/usr/bin/env python3
import os
from typing import Callable, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import wandb
import dotenv

dotenv.load_dotenv()

# Default hyperparameters
DEFAULT_LEARNING_RATE = 1e-1
DEFAULT_NUM_EPOCHS = 30000
DEFAULT_CYCLE_WEIGHT = 1e1
DEFAULT_DIVERSITY_WEIGHT = 1e-1
DEFAULT_HARD_LOSS_THRESHOLD = 1.0
DEFAULT_TAU = 1.0
DEFAULT_USE_GUMBEL = True
DEFAULT_USE_FREEZE = False
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_NUM_TESTS = 256


def compute_nand(inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise NAND: 1 - inputs1 * inputs2.
    """
    return 1.0 - inputs1 * inputs2


def hinge_overlap_loss(layers,
                       max_shared: int = 1,
                       weight: float = DEFAULT_DIVERSITY_WEIGHT) -> torch.Tensor:
    """
    Soft-penalty: for each pair of gates, penalize dot-products > max_shared.
    """
    total = 0.0
    for layer in layers:
        # normalize and flatten each gate's probability map
        probs = torch.softmax(layer, dim=1).view(layer.size(0), -1)  # (T, V*C)
        G = probs @ probs.T
        mask = 1 - torch.eye(G.size(0), device=G.device)
        sims = (G - max_shared) * mask
        total += (sims.clamp(min=0)**2).sum()
    return weight * total


def create_graph_model(input_len: int,
                       num_gates: int,
                       num_gate_types: int = 1):
    """
    Initialize a fully-connected gate graph where each gate can connect
    to any original input or any other gate's output.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_inputs = input_len + num_gates
    layers = []
    for _ in range(num_gates):
        layer = torch.rand((num_gate_types, valid_inputs, 2),
                          dtype=torch.float32,
                          device=device) * 0.05
        layer.requires_grad_(True)
        layers.append(layer)
    return layers


def cycle_penalty(layers_logits,
                  input_len: int,
                  weight: float = DEFAULT_CYCLE_WEIGHT) -> torch.Tensor:
    """
    Penalize self-loops and mutual loops based on learned adjacency.
    """
    num_gates = len(layers_logits)
    device = layers_logits[0].device
    adj = torch.zeros((num_gates, num_gates), device=device)
    for i, layer in enumerate(layers_logits):
        probs = torch.softmax(layer, dim=1)
        a_probs = probs[0, input_len:, 0]
        b_probs = probs[0, input_len:, 1]
        adj[i] = a_probs + b_probs
    diag = torch.diagonal(adj)
    self_pen = (diag**2).sum()
    mutual = (adj * adj.T * (1 - torch.eye(num_gates, device=device))).sum()
    return weight * (self_pen + mutual)


def evaluate_graph(
    layers_logits,
    num_iterations: int,
    generate_inputs_fn: Callable[[int], np.ndarray],
    correct_behavior_fn: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int,
    num_tests: int,
    target_bits: int
) -> torch.Tensor:
    """
    Soft-selection propagation loss (vectorized) using BCE.
    """
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    stacked_logits = torch.cat(layers_logits, dim=0)
    probs = torch.softmax(stacked_logits, dim=1)
    a_probs = probs[..., 0]
    b_probs = probs[..., 1]
    M = a_probs.unsqueeze(2) * b_probs.unsqueeze(1)
    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+target_bits, :] = 0
    mask[:, input_len:input_len+target_bits] = 0
    M = M * mask.unsqueeze(0)

    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M)
        state = torch.cat([sample_inputs, gate_out], dim=1)

    expected = correct_behavior_fn(sample_inputs, (num_tests, target_bits))
    actual = state[:, input_len:input_len+target_bits]
    return F.binary_cross_entropy(
        actual.clamp(min=1e-7, max=1-1e-7),
        expected,
        reduction='sum'
    )


def loss_hard_graph(
    layers_logits,
    num_iterations: int,
    generate_inputs_fn: Callable[[int], np.ndarray],
    correct_behavior_fn: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int,
    num_tests: int,
    target_bits: int
) -> torch.Tensor:
    """
    Hard-selection propagation loss (discrete argmax connections).
    """
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    stacked = torch.cat(layers_logits, dim=0)
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
    M_hard = a_hard.unsqueeze(2) * b_hard.unsqueeze(1)

    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+target_bits, :] = 0
    mask[:, input_len:input_len+target_bits] = 0
    M_hard = M_hard * mask.unsqueeze(0)

    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M_hard)
        state = torch.cat([sample_inputs, gate_out], dim=1)

    expected = correct_behavior_fn(sample_inputs, (num_tests, target_bits))
    actual = state[:, input_len:input_len+target_bits]
    return (actual - expected).pow(2).sum()


def loss_gumbel_graph(
    layers_logits,
    num_iterations: int,
    generate_inputs_fn: Callable[[int], np.ndarray],
    correct_behavior_fn: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int,
    num_tests: int,
    target_bits: int,
    tau: float = DEFAULT_TAU,
    hard: bool = False
) -> torch.Tensor:
    """
    Gumbel-Softmax propagation loss.
    """
    device = layers_logits[0].device
    num_gates = len(layers_logits)
    total_dim = input_len + num_gates

    stacked = torch.cat(layers_logits, dim=0)
    logits = stacked.permute(0, 2, 1)
    samples = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    a_hard = samples[:, 0, :]
    b_hard = samples[:, 1, :]
    M_gumbel = a_hard.unsqueeze(2) * b_hard.unsqueeze(1)

    mask = torch.ones((total_dim, total_dim), device=device)
    mask[input_len:input_len+target_bits, :] = 0
    mask[:, input_len:input_len+target_bits] = 0
    M_gumbel = M_gumbel * mask.unsqueeze(0)

    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)
    state = torch.cat(
        [sample_inputs, torch.full((num_tests, num_gates), 0.5, device=device)],
        dim=1
    )

    for _ in range(num_iterations):
        N = compute_nand(state.unsqueeze(2), state.unsqueeze(1))
        gate_out = torch.einsum('bij,gij->bg', N, M_gumbel)
        state = torch.cat([sample_inputs, gate_out], dim=1)

    expected = correct_behavior_fn(sample_inputs, (num_tests, target_bits))
    actual = state[:, input_len:input_len+target_bits]
    return F.binary_cross_entropy(
        actual.clamp(min=1e-7, max=1-1e-7),
        expected,
        reduction='sum'
    )


def freeze_upstream_gates(
    layers_logits,
    input_len: int,
    target_gate_idx: int
):
    """
    Freeze parameters for all gates upstream of the specified gate.
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

    # Exclude output gates
    upstream = {idx for idx in upstream if idx >= target_gate_idx+1}
    for idx in upstream:
        layer = layers_logits[idx]
        with torch.no_grad():
            layer.fill_(-1e9)
            layer[..., a_idx[idx], 0] = 1e9
            layer[..., b_idx[idx], 1] = 1e9
        layer.requires_grad_(False)
    tqdm.write(f"Froze {len(upstream)} gates upstream of output bit {target_gate_idx}")


def save_checkpoint(
    epoch: int,
    model,
    optimizer,
    losses: list,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': [layer.detach().cpu() for layer in model],
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, path)


def main(
    generate_inputs: Callable[[int], np.ndarray],
    correct_behavior: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
    input_len: int,
    output_bits: int,
    num_gates: int,
    num_iterations: int = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    cycle_weight: float = DEFAULT_CYCLE_WEIGHT,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    hard_loss_threshold: float = DEFAULT_HARD_LOSS_THRESHOLD,
    tau: float = DEFAULT_TAU,
    use_gumbel: bool = DEFAULT_USE_GUMBEL,
    use_freeze: bool = DEFAULT_USE_FREEZE,
    num_tests: int = DEFAULT_NUM_TESTS,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    checkpoint_interval: int = 50,
    project_name: str = "any2any"
):
    """Run generic any-to-any training with the specified problem functions and hyperparameters."""
    run = wandb.init(project=project_name, config={
        "input_len": input_len,
        "output_bits": output_bits,
        "num_gates": num_gates,
        "learning_rate": learning_rate,
        "cycle_weight": cycle_weight,
        "diversity_weight": diversity_weight,
        "hard_loss_threshold": hard_loss_threshold,
        "tau": tau,
        "use_gumbel": use_gumbel,
        "use_freeze": use_freeze,
        "num_tests": num_tests
    })
    model = create_graph_model(input_len=input_len, num_gates=num_gates)
    optimizer = torch.optim.RMSprop(model, lr=learning_rate)
    num_iters = num_iterations if num_iterations else num_gates + 1

    current_bits = 1
    pbar = trange(0, num_epochs, desc=f'Training(bits={current_bits})')
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        if use_gumbel:
            loss = loss_gumbel_graph(
                model, num_iters, generate_inputs, correct_behavior,
                input_len, num_tests, current_bits, tau=tau, hard=False
            )
        else:
            loss = evaluate_graph(
                model, num_iters, generate_inputs, correct_behavior,
                input_len, num_tests, current_bits
            )

        with torch.no_grad():
            hard_mse = loss_hard_graph(
                model, num_iters, generate_inputs, correct_behavior,
                input_len, num_tests, current_bits
            )

        if hard_mse.item() < hard_loss_threshold and current_bits < output_bits:
            if use_freeze:
                freeze_upstream_gates(model, input_len, current_bits - 1)
            current_bits += 1
            tqdm.write(f"Threshold reached, advancing to first {current_bits} bits")
            with torch.no_grad():
                hard_mse = loss_hard_graph(
                    model, num_iters, generate_inputs, correct_behavior,
                    input_len, num_tests, current_bits
                )

        div_loss = hinge_overlap_loss(model, weight=diversity_weight)
        cyc_loss = cycle_penalty(model, input_len, weight=cycle_weight)
        total_loss = loss + div_loss + cyc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model, max_norm=0.5)
        optimizer.step()

        run.log({"loss": loss.item(), "hard_loss": hard_mse.item(), "current_bits": current_bits})
        pbar.set_description(
            f'Training(bits={current_bits}) - BCE: {loss.item():8.2f}, Hard: {hard_mse.item():3.1f}, Div: {div_loss.item():4.2f}, Cycle: {cyc_loss.item():4.2f}'
        )

        if epoch % checkpoint_interval == 0:
            save_checkpoint(epoch, model, optimizer, [total_loss.item()], checkpoint_dir)

        if hard_mse.item() < hard_loss_threshold and current_bits == output_bits:
            run.finish()
            break

    print("Training complete")

