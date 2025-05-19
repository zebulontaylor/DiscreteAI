# logic.py
import numpy as np
import torch
from typing import Callable, List, Tuple


def normalize_layers(layers: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply softmax to each layer tensor along the feature dimension.

    Args:
        layers: List of tensors containing unnormalized gate logits.

    Returns:
        List of tensors with probabilities after softmax.
    """
    return [torch.softmax(layer, dim=1) for layer in layers]


def compute_nand(inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise NAND: 1 - inputs1 * inputs2.

    Args:
        inputs1: Tensor of any shape.
        inputs2: Tensor broadcastable to inputs1.

    Returns:
        Tensor of the same shape as inputs1, containing NAND results.
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
    """
    Evaluate a network of NAND gates and compute the loss and sum-of-squared-errors (MSE).

    Args:
        layers_logits: List of unnormalized gate weight tensors.
        num_gates: Number of gates in the network to apply sequentially.
        generate_inputs_fn: Function generating numpy inputs of shape (num_tests, input_len).
        correct_behavior_fn: Function computing expected outputs given inputs and desired output shape.
        input_len: Number of input bits total (default: 8).
        num_tests: Number of test samples (default: 256).

    Returns:
        Tuple:
            loss: Tensor scalar for backward (sum of squared errors, with possible penalties).
            mse: Tensor scalar sum of squared errors between actual and expected.
    """
    # Normalize layer logits to probabilities
    layers = normalize_layers(layers_logits)
    device = layers[0].device

    # Generate test inputs
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)

    # Initialize outputs with the input bits
    outputs = sample_inputs  # shape: [num_tests, current_len]

    # Sequentially apply NAND gates
    for i in range(num_gates):
        probs = layers[i]  # shape: [batch_dim, current_len, 2]

        # Compute weighted adjacency matrix M: [current_len, current_len]
        a_probs = probs[:, :, 0]
        b_probs = probs[:, :, 1]
        M = a_probs.transpose(0, 1) @ b_probs

        # Prepare outputs for pairwise NAND for each sample
        O = outputs  # shape: [num_tests, current_len]
        O1 = O.unsqueeze(2)  # [num_tests, current_len, 1]
        O2 = O.unsqueeze(1)  # [num_tests, 1, current_len]

        # Compute NAND and weight by adjacency
        N = compute_nand(O1, O2)  # [num_tests, current_len, current_len]
        weighted = N * M.unsqueeze(0)  # [num_tests, current_len, current_len]
        gate_outputs = weighted.sum(dim=(1, 2))  # [num_tests]

        # Append new gate outputs as a new column
        outputs = torch.cat([outputs, gate_outputs.unsqueeze(1)], dim=1)

    # Compute expected outputs and losses
    expected = correct_behavior_fn(sample_inputs, (num_tests, input_len // 2))
    actual = outputs[:, -expected.shape[1]:]
    diff = actual - expected
    mse = diff.pow(2).sum()
    loss = mse  # add penalties here if desired

    return loss, mse


def correct_behavior(inputs: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Compute the correct sum-of-bits behavior modulo 2**(num_bits), producing individual output bits.

    Args:
        inputs: Tensor of shape [num_tests, 2 * num_bits] containing binary inputs.
        output_shape: Tuple (num_tests, num_output_bits).

    Returns:
        Tensor of shape output_shape with correct output bits.
    """
    device = inputs.device
    num_bits = inputs.shape[1] // 2
    weights = torch.tensor([2**i for i in range(num_bits)],
                           device=device, dtype=torch.float32)

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

    Args:
        num_samples: Number of samples to generate.
        num_bits: Number of bits per operand (default: 4).

    Returns:
        Numpy array of shape (num_samples, 2 * num_bits) with float32 values 0.0 or 1.0.
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
    """
    Compute sum-of-squared errors using hard connections (max-strength) for each NAND gate.

    Args:
        layers_logits: List of unnormalized gate weight tensors.
        num_gates: Number of gates in the network.
        generate_inputs_fn: Function generating numpy inputs.
        correct_behavior_fn: Function computing expected outputs.
        input_len: Total number of input bits.
        num_tests: Number of samples to generate.

    Returns:
        Tensor scalar sum of squared errors using hard connections.
    """
    layers = normalize_layers(layers_logits)
    device = layers[0].device

    # Generate test inputs
    inputs_np = generate_inputs_fn(num_tests)
    sample_inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device)

    # Initialize outputs
    outputs = sample_inputs  # [num_tests, current_len]

    for i in range(num_gates):
        probs = layers[i]  # [batch_dim, current_len, 2]
        a_probs = probs[:, :, 0]
        b_probs = probs[:, :, 1]

        # Hard one-hot connections
        batch_dim, current_len = a_probs.shape
        a_idx = a_probs.argmax(dim=1)
        b_idx = b_probs.argmax(dim=1)

        a_hard = torch.zeros_like(a_probs)
        b_hard = torch.zeros_like(b_probs)
        a_hard.scatter_(1, a_idx.unsqueeze(1), 1.0)
        b_hard.scatter_(1, b_idx.unsqueeze(1), 1.0)

        # Construct hard adjacency matrix
        M = a_hard.transpose(0, 1) @ b_hard  # [current_len, current_len]

        # Apply NAND with hard connections
        O = outputs  # [num_tests, current_len]
        O1 = O.unsqueeze(2)  # [num_tests, current_len, 1]
        O2 = O.unsqueeze(1)  # [num_tests, 1, current_len]
        N = compute_nand(O1, O2)
        weighted = N * M.unsqueeze(0)
        gate_outputs = weighted.sum(dim=(1, 2))  # [num_tests]

        outputs = torch.cat([outputs, gate_outputs.unsqueeze(1)], dim=1)

    # Compute expected and hard loss
    expected = correct_behavior_fn(sample_inputs, (num_tests, input_len // 2))
    actual = outputs[:, -expected.shape[1]:]
    diff = actual - expected
    mse_hard = diff.pow(2).sum()
    return mse_hard
