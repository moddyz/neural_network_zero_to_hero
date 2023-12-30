#!/usr/bin/env python

import sys
import os.path
import argparse

import torch

from model import GPT, ASCII_TRANSCODER


def main():
    parser = argparse.ArgumentParser("generate.py", description="Uses a trained language model to generate text.")
    parser.add_argument("-n", "--num-chars", help="Number of characters to generate.  Pass in -1 to generate indefinitely", default=-1, type=int)
    parser.add_argument(
        "-ip",
        "--input-parameters-path",
        help="File path containing the trained parameters.",
        default="parameters.pth",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed value to produce deterministic results.",
        default=1337,
        type=int,
    )
    args = parser.parse_args()

    # Seed for deterministic results
    torch.manual_seed(args.seed)

    # Instantiate model.
    model = GPT()
    model = model.to(model.hyper_params.device)

    # Load parameters into model
    input_parameters_path = os.path.abspath(os.path.normpath(args.input_parameters_path))
    print(f"Loading trained model parameters: {input_parameters_path}")
    state_dict = torch.load(input_parameters_path)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode.
    model.eval()

    # Generate characters
    generate(model, num_chars=args.num_chars)


def generate(model, num_chars):
    if num_chars == -1:
        print(f"Generating characters indefinitely...\n")
    else:
        print(f"Generating {num_chars} characters...\n")

    # Start with an whitespace to seed the generation.
    encoded_whitespace = ASCII_TRANSCODER.encode(" ")[0]
    inputs = torch.full((1, 1), encoded_whitespace, dtype=torch.long).to(model.hyper_params.device)

    # Print results as it is being generated.
    for predictions in model.generate(inputs, num_chars):
        int_sequence = predictions[0].tolist()
        next_char = ASCII_TRANSCODER.decode(int_sequence)
        sys.stdout.write(next_char)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
