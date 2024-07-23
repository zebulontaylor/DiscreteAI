# logic.py
import numpy as np
import torch

def evaluate_instance(original_layers, generate_inputs_fn: callable, correct_behavior_fn: callable, input_len: int = 8) -> torch.Tensor:
    layers = []
    for layer in original_layers:
        newlayer = torch.nn.functional.normalize(torch.sigmoid(5*(layer-.5)), p=1, dim=1)
        layers.append(newlayer)

    gate_probs = layers[0]
    inputs = layers[1:]
    input1s = inputs[::2]
    input2s = inputs[1::2]
    num_gates = gate_probs.shape[0]

    NUM_TESTS = 256

    sample_inputs = torch.tensor(generate_inputs_fn(NUM_TESTS), dtype=torch.float32, device=gate_probs.device)
    outputs = sample_inputs

    for i in range(num_gates):
        i1 = torch.matmul(input1s[i], outputs.T)
        i2 = torch.matmul(input2s[i], outputs.T)

        # Vectorized gate predictions
        AND = i1[0] * i2[0]
        OR = i1[1] + i2[1] - i1[1] * i2[1]
        XOR = i2[2] + i1[2] - 2 * i1[2] * i2[2]
        NOT = 1 - i1[3]

        gate_results = torch.stack([AND, OR, XOR, NOT], dim=1)

        result = gate_results @ gate_probs[i]
        result = result.unsqueeze(1)

        outputs = torch.cat((outputs, result), dim=1)

    expected_outputs = correct_behavior_fn(sample_inputs, (NUM_TESTS, input_len // 2))
    m = expected_outputs.shape[0]
    n = expected_outputs.shape[1]
    actual_outputs = outputs[-m:, -n:]
    #print("Given:", actual_outputs, "Correct:", expected_outputs, sep="\n")
    #loss = torch.sum(torch.sum(torch.abs(actual_outputs - expected_outputs), dim=1) ** 2)
    #loss = torch.sum((actual_outputs - expected_outputs) ** 2)
    loss = torch.sum(torch.sigmoid(8*(torch.abs(actual_outputs-expected_outputs) - .5)))
    #loss = torch.sum(torch.abs(actual_outputs - expected_outputs) ** 3)
    # penalize gates that are between 0 and 1
    #all_layers = torch.cat(layers[1:], dim=1)
    #loss = loss + torch.sum(gate_probs*(1-gate_probs)) + torch.sum(all_layers*(1-all_layers))

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
