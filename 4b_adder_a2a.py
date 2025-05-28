#!/usr/bin/env python3
import numpy as np
import torch
from typing import Tuple
from unified_a2a import main

# 4-bit adder problem definition
NUM_BITS = 4
NUM_GATES = 64


def generate_inputs(num_samples: int) -> np.ndarray:
    """
    Generate binary test inputs for two NUM_BITS-wide numbers.
    If num_samples == 2**(2*NUM_BITS), returns all combinations, else random.
    """
    total_bits = 2 * NUM_BITS
    if num_samples == 2**total_bits:
        inputs = np.zeros((num_samples, total_bits), dtype=np.float32)
        for a in range(2**NUM_BITS):
            for b in range(2**NUM_BITS):
                idx = a * (2**NUM_BITS) + b
                # little-endian bit order
                for i in range(NUM_BITS):
                    inputs[idx, i] = (a >> i) & 1
                    inputs[idx, NUM_BITS + i] = (b >> i) & 1
    else:
        inputs = np.random.randint(0, 2, size=(num_samples, total_bits)).astype(np.float32)
    return inputs


def correct_behavior(inputs: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Compute bitwise sum modulo 2**NUM_BITS and return per-bit outputs.
    """
    device = inputs.device
    # weights for binary to integer conversion
    weights = torch.tensor([2**i for i in range(NUM_BITS)], device=device, dtype=torch.float32)
    A = (inputs[:, :NUM_BITS] * weights).sum(dim=1, keepdim=True)
    B = (inputs[:, NUM_BITS:] * weights).sum(dim=1, keepdim=True)
    sum_mod = (A + B) % (2**NUM_BITS)
    outputs = torch.zeros(output_shape, device=device, dtype=torch.float32)
    # extract bits little-endian
    for i in range(output_shape[1]):
        outputs[:, i] = ((sum_mod // (2**i)) % 2).squeeze()
    return outputs


if __name__ == '__main__':
    # invoke the generic any-to-any trainer
    main(
        generate_inputs=generate_inputs,
        correct_behavior=correct_behavior,
        input_len=2 * NUM_BITS,
        output_bits=NUM_BITS,
        num_gates=NUM_GATES
    ) 