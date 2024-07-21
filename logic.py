# logic.py
import numpy as np
import torch

def evaluate_instance(gate_probs, input1, input2, generate_inputs_fn: callable, correct_behavior_fn: callable) -> torch.Tensor:
    num_gates = gate_probs.shape[0]

    gate_probs = torch.clamp(gate_probs, 0, 1)
    input1 = torch.clamp(input1, 0, 1)
    input2 = torch.clamp(input2, 0, 1)

    gate_probs = torch.nn.functional.normalize(gate_probs, p=1, dim=1)
    input1 = torch.nn.functional.normalize(input1, p=1, dim=0)
    input2 = torch.nn.functional.normalize(input2, p=1, dim=0)

    NUM_TESTS = 8192

    sample_inputs = torch.tensor(generate_inputs_fn(NUM_TESTS), dtype=torch.float32, device=gate_probs.device)
    outputs = sample_inputs

    for i in range(num_gates):
        valid_inputs = i + 16
        i1 = torch.matmul(input1[i, :, :valid_inputs], outputs.T)
        i2 = torch.matmul(input2[i, :, :valid_inputs], outputs.T)

        result_matrix = torch.zeros((NUM_TESTS, 7), dtype=torch.float32, device=gate_probs.device)
        for j in range(7):
            result_matrix[:, j] = predict_gate(j, i1[j, :], i2[j, :])

        result = result_matrix @ gate_probs[i]
        result = result.unsqueeze(1)
        result = torch.relu(result)

        outputs = torch.cat((outputs, result), dim=1)

    expected_outputs = correct_behavior_fn(sample_inputs, (NUM_TESTS, 8))
    m = expected_outputs.shape[0]
    n = expected_outputs.shape[1]
    actual_outputs = outputs[-m:, -n:]
    loss = torch.sum(torch.sum(torch.abs(actual_outputs - expected_outputs), dim=1) ** 2)
    #loss = torch.sum((actual_outputs - expected_outputs) ** 2)

    return loss

def predict_gate(gate_type: int, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if gate_type == 0:
        return input1 * input2
    elif gate_type == 1:
        return input1 + input2 - input1 * input2
    elif gate_type == 2:
        return 1 - input1
    elif gate_type == 3:
        return 1 - (input1 * input2)
    elif gate_type == 4:
        return (1 - input1) * (1 - input2)
    elif gate_type == 5:
        return (1 - input1) * input2 + input1 * (1 - input2)
    elif gate_type == 6:
        return 1 - ((1 - input1) * input2 + input1 * (1 - input2))
    else:
        raise ValueError(f"Invalid gate type: {gate_type}")

def correct_behavior(inputs, shape):
    device = inputs.device
    A = torch.sum(inputs[:, :8] * torch.tensor([2**i for i in range(8)], device=device, dtype=torch.float32), dim=1, keepdim=True)
    B = torch.sum(inputs[:, 8:] * torch.tensor([2**i for i in range(8)], device=device, dtype=torch.float32), dim=1, keepdim=True)
    sum_result = (A + B) % 256
    result = torch.zeros(shape, device=device, dtype=torch.float32)
    for i in range(8):
        result[:, i] = ((sum_result // (2**i)) % 2).squeeze()
    return result

def generate_inputs(num_samples, num_bits=8):
    if num_samples != 256:
        inputs = np.random.randint(0, 2, size=(num_samples, num_bits*2), dtype=np.int8)
    else:
        inputs = np.zeros((256, 16))
        for i in range(256):
            for j in range(256):
                inputs[i*256+j, :8] = np.asarray([(i&(1<<k))>>k for k in range(8)])
                inputs[i*256+j, 8:] = np.asarray([(j&(1<<k))>>k for k in range(8)])
    return inputs.astype(np.float32)
