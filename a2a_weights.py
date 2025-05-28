#!/usr/bin/env python3
import os
import torch
from any2any import create_graph_model
import unifiedbackprop as ub
import argparse
import numpy as np

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:4.2f}'.format})


def main():
    parser = argparse.ArgumentParser(description="Print gate weights with optional checkpoint loading")
    parser.add_argument('-c', '--checkpoint', type=str, help="Path to a specific checkpoint file")
    parser.add_argument('-d', '--ckpt_dir', default=ub.CHECKPOINT_DIR, help="Directory containing checkpoints")
    parser.add_argument('--new', action='store_true', help="Ignore checkpoints and use initial weights")
    args = parser.parse_args()

    # Initialize model and optimizer
    model = create_graph_model()
    optimizer = torch.optim.RMSprop(model, lr=ub.LEARNING_RATE)

    # Determine checkpoint loading strategy
    if args.new:
        print("Ignoring checkpoints, using initial weights.")
    else:
        if args.checkpoint:
            path = args.checkpoint
            if os.path.isfile(path):
                print(f"Loading checkpoint: {path}")
                ub.load_checkpoint(path, model, optimizer)
            else:
                print(f"Checkpoint file '{path}' not found, using initial weights.")
        else:
            ckpt_dir = args.ckpt_dir
            if os.path.isdir(ckpt_dir):
                files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
                if files:
                    # Sort by epoch number and load the latest
                    ckpts = sorted(files, key=lambda f: int(f.split('_')[1].split('.')[0]))
                    latest = ckpts[-1]
                    path = os.path.join(ckpt_dir, latest)
                    print(f"Loading checkpoint: {path}")
                    ub.load_checkpoint(path, model, optimizer)
                else:
                    print(f"No checkpoints found in '{ckpt_dir}', using initial weights.")
            else:
                print(f"Checkpoint directory '{ckpt_dir}' does not exist, using initial weights.")

    # Print weights for each gate
    for idx, layer in enumerate(model):
        weights = layer.softmax(dim=1).detach().cpu().numpy()
        print(f"\nGate {idx} weights:\n", weights)

if __name__ == '__main__':
    main()