# logic.py
import numpy as np
import torch

def normalize_layers(layers, dropout):
    new_layers = []
    for layer in layers:
        clamped = torch.sigmoid(5 * (layer - 0.5))
        extended = torch.cat([clamped, torch.clamp(1 - torch.sum(clamped, dim=1), 0, 1).unsqueeze(1)], dim=1)
        if dropout:
            extended = torch.nn.functional.dropout(extended, dropout)
        normalized = torch.nn.functional.normalize(extended, p=1, dim=1)
        new_layers.append(normalized)
    return new_layers

def compute_gate_results(i1, i2):
    AND = i1[0] * i2[0]
    OR = i1[1] + i2[1] - i1[1] * i2[1]
    XOR = i2[2] + i1[2] - 2 * i1[2] * i2[2]
    NOT = 1 - i1[3]
    return torch.stack([AND, OR, XOR, NOT], dim=1)

def evaluate_instance(original_layers, num_gates: int, generate_inputs_fn: callable, correct_behavior_fn: callable, dropout: float = .0, input_len: int = 8) -> torch.Tensor:
    layers = normalize_layers(original_layers, dropout)
    input1s, input2s = layers[::2], layers[1::2]

    NUM_TESTS = 128

    sample_inputs = torch.tensor(generate_inputs_fn(NUM_TESTS), dtype=torch.float32, device=input1s[0].device)
    outputs = sample_inputs

    for i in range(num_gates):
        i1, i2 = torch.matmul(input1s[i], outputs.T), torch.matmul(input2s[i], outputs.T)
        if i == num_gates-1:
            gate_results = i1 + i2 - i1 * i2  # Make outputs OR gates
            gate_results = gate_results.swapaxes(0,1)
        else:
            gate_results = compute_gate_results(i1, i2)
        outputs = torch.cat((outputs, gate_results), dim=1)
    

    expected_outputs = correct_behavior_fn(sample_inputs, (NUM_TESTS, input_len // 2))
    m = expected_outputs.shape[0]
    n = expected_outputs.shape[1]
    actual_outputs = outputs[-m:, -n:]

    #loss = torch.sum(torch.sum(torch.abs(actual_outputs - expected_outputs), dim=0) ** 2)
    loss = torch.sum((actual_outputs - expected_outputs) ** 2)
    #loss = torch.sum(torch.sigmoid(8*(torch.abs(actual_outputs-expected_outputs) - 1)))
    #loss = torch.sum(torch.abs(actual_outputs - expected_outputs) ** 3)

    # penalize gates that are between 0 and 1
    #all_layers = torch.cat(layers[1:], dim=1)
    #loss = loss + torch.sum(gate_probs*(1-gate_probs)) + torch.sum(all_layers*(1-all_layers))

    return loss, torch.sum((actual_outputs - expected_outputs) ** 2)

def correct_behavior(inputs, shape):
    device = inputs.device
    A = torch.sum(inputs[:, :4] * torch.tensor([2**i for i in range(4)], device=device, dtype=torch.float32), dim=1, keepdim=True)
    B = torch.sum(inputs[:, 4:] * torch.tensor([2**i for i in range(4)], device=device, dtype=torch.float32), dim=1, keepdim=True)
    sum_result = (A + B) % 16
    result = torch.zeros(shape, device=device, dtype=torch.float32)
    for i in range(shape[1]):
        result[:, i] = ((sum_result // (2**i)) % 2).squeeze()
    return result

def generate_inputs(num_samples, num_bits=4):
    if num_samples != 256:
        inputs = np.random.randint(0, 2, size=(num_samples, num_bits*2), dtype=np.int8)
    else:
        inputs = np.zeros((256, 8))
        for i in range(16):
            for j in range(16):
                inputs[i*16+j, :4] = np.asarray([(i&(1<<k))>>k for k in range(4)])
                inputs[i*16+j, 4:] = np.asarray([(j&(1<<k))>>k for k in range(4)])
    return inputs.astype(np.float32)
