#!/usr/bin/env python

import os.path
import argparse

import torch

from model import GPT, ASCII_TRANSCODER


def main():
    parser = argparse.ArgumentParser("generate.py", description="Uses a trained language model to generate text.")
    parser.add_argument("-n", "--num-chars", help="Number of characters to generate.", default=10000, type=int)
    parser.add_argument(
        "-p",
        "--param-path",
        help="File path containing the trained parameters.",
        default="parameters.pth",
        type=str,
    )
    args = parser.parse_args()

    # Instantiate model.
    model = GPT()
    model = model.to(model.hyper_params.device)

    # Load parameters into model
    param_path = os.path.abspath(os.path.normpath(args.param_path))
    print(f"Loading trained model parameters: {param_path}")
    state_dict = torch.load(param_path)
    model.load_state_dict(state_dict)

    # Generate characters
    characters = generate(model, num_chars=args.num_chars)
    print(characters)


def generate(model, num_chars=10000):
    print(f"Generating {num_chars} characters...\n")
    inputs = torch.zeros((1, 1), dtype=torch.long).to(model.hyper_params.device)
    predictions = model.generate(inputs, num_chars)
    int_sequence = predictions[0].tolist()
    return ASCII_TRANSCODER.decode(int_sequence)


if __name__ == "__main__":
    main()
