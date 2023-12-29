#!/usr/bin/env python

"""Train a language model with text from Shakespeare's play, such that it can generate
sentences like Shakespeare."""

import os
import argparse
import collections

import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from model import HyperParameters, GPT
from dataset import AsciiTextFileDataset


def main():
    parser = argparse.ArgumentParser(
        "train.py", description="Launches training for a language model"
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed value to produce deterministic results.",
        default=1337,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        help="Number of training iterations.",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--input-data-path",
        help="File path to the input text data to train on",
        default="input.txt",
        type=str,
    )
    parser.add_argument(
        "-io",
        "--input-optimizer-path",
        help="File path to load existing optimizer state for resuming training.",
        default="",
        type=str,
    )
    parser.add_argument(
        "-ip",
        "--input-parameters-path",
        help="File path to load existing model parameters for resuming training.",
        default="",
        type=str,
    )
    parser.add_argument(
        "-oo",
        "--output-optimizer-path",
        help="File path to save the trained optimizer.",
        default="optimizer.pth",
        type=str,
    )
    parser.add_argument(
        "-op",
        "--output-parameters-path",
        help="File path to save the trained parameters.",
        default="parameters.pth",
        type=str,
    )

    args = parser.parse_args()

    # Seed for deterministic results
    torch.manual_seed(args.seed)

    # Set up our model's hyper parameters.
    hyper_params = HyperParameters(
        train_iters=args.num_iterations,
    )

    # Instantiate the model.
    model = GPT(hyper_params=hyper_params).to(hyper_params.device)

    # Should we load existing parameters?
    if args.input_parameters_path:
        input_parameters = torch.load(args.input_parameters_path)
        model.load_state_dict(input_parameters)

    # Instantiate the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)

    # Should we load existing optimizer state?
    if args.input_optimizer_path:
        input_optimizer_state = torch.load(args.input_optimizer_path)
        optimizer.load_state_dict(input_optimizer_state)

    output_parameters_path = os.path.abspath(os.path.normpath(args.output_parameters_path))
    output_optimizer_path = os.path.abspath(os.path.normpath(args.output_optimizer_path))

    # Train the model.
    for _ in train(args.input_data_path, hyper_params, model, optimizer):

        # Train will yield at checkpoints so we can incrementally save state.
        torch.save(model.state_dict(), output_parameters_path)
        torch.save(optimizer.state_dict(), output_optimizer_path)

    # Save out the parameters one last time.
    torch.save(model.state_dict(), output_parameters_path)
    print(f"Saved out trained model parameters at: {args.output_parameters_path}")


def train(input_path, hyper_params, model, optimizer):

    print(f"Reading {input_path}...")
    full_dataset = AsciiTextFileDataset(input_path, hyper_params.block_size)
    print(f"Length of dataset in characters: {len(full_dataset)}")

    # Separate data into training & validation sets.
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=hyper_params.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=hyper_params.batch_size, shuffle=False)

    print(f"Starting training with {hyper_params}")

    train_dataloader_iter = iter(train_dataloader)

    for train_iter in range(hyper_params.train_iters):
        # Extract some training data.
        inputs, targets = next(train_dataloader_iter)

        # Upload to configured device.
        inputs = inputs.to(hyper_params.device)
        targets = targets.to(hyper_params.device)

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Zero out the gradients
        optimizer.zero_grad(set_to_none=True)

        # Propagate gradients across the computation network.
        loss.backward()

        # Update the parameters based on gradients to minimize loss.
        optimizer.step()

        # Estimate and print the optimization progress at every 1 percentage.
        if train_iter % (hyper_params.train_iters // 100) == 0:
            train_loss = estimate_loss(model, train_dataloader, hyper_params)
            val_loss = estimate_loss(model, val_dataloader, hyper_params)
            progress_percentage = (
                float(train_iter + 1) / hyper_params.train_iters * 100.0
            )
            print(
                f"progress: {progress_percentage:.01f}%, training iteration: {train_iter + 1}/{hyper_params.train_iters}, training loss: {train_loss:.05f}, validation loss: {val_loss:.05f}"
            )

            yield


@torch.no_grad()
def estimate_loss(model, dataloader, hyper_params):
    """This method is called during the model training phase to provide a
    more representative estimate of the loss across our data set.

    Args:
        model (torch.nn.Module): a neural network model
        data (torch.tensor): data set to estimate loss for
        hyper_params (HyperParameters): the parameters for training and evaluating the current model
    """

    out = {}

    # Set the model to evaluation mode.
    model.eval()

    # Initialize storage to hold loss values.
    losses = torch.zeros(hyper_params.eval_iters)

    dataloader_iter = iter(dataloader)

    # Compute the loss multiple times.
    for eval_index in range(hyper_params.eval_iters):
        # Evaluate & store the loss for a batch
        inputs, targets = next(dataloader_iter)
        inputs = inputs.to(hyper_params.device)
        targets = targets.to(hyper_params.device)

        _, loss = model(inputs, targets)
        losses[eval_index] = loss.item()

    # Revert the model back to training mode.
    model.train()

    return losses.mean()


if __name__ == "__main__":
    main()
