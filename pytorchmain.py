# pytorchmain.py

import os
import json
import argparse
from pprint import pprint

import torch
from tqdm import trange

from logic import evaluate_instance, correct_behavior, generate_inputs, loss_hard

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GATES = 48
NUM_GATE_TYPES = 1
LEARNING_RATE = 1e-1
NUM_EPOCHS = 100000
CHECKPOINT_DIR = 'checkpoints'


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
    """Save training state to a checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': [layer.detach().cpu() for layer in model],
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, path)
    print(f"Checkpoint saved: epoch {epoch}, loss {losses[-1]:.4f}")


def load_checkpoint(path, model, optimizer):
    """Load training state from a checkpoint file."""
    ckpt = torch.load(path, map_location=DEVICE)
    start_epoch = ckpt['epoch']
    state_dicts = ckpt['model_state_dict']
    if len(state_dicts) != len(model):
        raise ValueError("Model size mismatch.")
    for layer, state in zip(model, state_dicts):
        if layer.size() != state.size():
            raise ValueError("Layer size mismatch.")
        layer.copy_(state.to(DEVICE))
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"Checkpoint loaded: resume from epoch {start_epoch}")
    return start_epoch, ckpt['losses']


def parse_args():
    """Parse command-line arguments."""
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
    """Run the training loop."""
    print(f"Running on {'cuda' if DEVICE.type == 'cuda' else 'cpu'}")
    model = create_model()
    optimizer = torch.optim.RMSprop(model, lr=LEARNING_RATE)
    start_epoch = 0
    losses = []

    # Optionally load from checkpoint
    if args.checkpoint and not args.new and os.path.exists(args.checkpoint):
        start_epoch, losses = load_checkpoint(args.checkpoint, model, optimizer)

    # Training loop with progress bar
    pbar = trange(start_epoch, NUM_EPOCHS, desc='Training')
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, mse = evaluate_instance(
            model, NUM_GATES, generate_inputs, correct_behavior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model, max_norm=0.5)
        optimizer.step()

        hard_mse = loss_hard(
            model, NUM_GATES, generate_inputs, correct_behavior
        )
        pbar.set_postfix({'mse': mse.item(), 'hard_mse': hard_mse.item()})

        if epoch % 50 == 0 and epoch != start_epoch:
            losses.append(loss.item())
            save_checkpoint(epoch, model, optimizer, losses)

    # Final outputs
    print("Final instance (gate_probs):")
    pprint(model[0])
    pprint(losses)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
