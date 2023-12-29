#!/usr/bin/env python

"""Train a language model with text from Shakespeare's play, such that it can generate
sentences like Shakespeare."""

import argparse
import collections

import torch

import matplotlib.pyplot as plt

from model import HyperParameters, GPT, ASCII_TRANSCODER


def main():
    parser = argparse.ArgumentParser(
        "train.py", description="Launches training for a language model"
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
        "--input-path",
        help="File path to the input text data to train on",
        default="input.txt",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--param-path",
        help="File path to store the trained parameters.",
        default="parameters.pth",
        type=str,
    )

    args = parser.parse_args()

    # Set up our model's hyper parameters.
    hyper_params = HyperParameters(
        train_iters=args.num_iterations,
    )

    # Create the model
    model = GPT(hyper_params=hyper_params).to(hyper_params.device)

    # Train the model.
    train(args.input_path, hyper_params, model)

    # Save out the parameters.
    param_path = os.path.abspath(os.path.normpath(args.param_path))
    torch.save(model.state_dict(), param_path)
    print(f"Saved out trained model parameters at: {args.output_parameters}")


def train(input_path, hyper_params, model):

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Length of dataset in characters: {len(text)}")

    # Create a tensor and encode the data.
    encoded_text = ASCII_TRANSCODER.encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)

    # Separate data into training & validation sets.
    train_data_size = int(0.9 * len(data))
    train_data = data[:train_data_size]
    val_data = data[train_data_size:]

    # Seed for deterministic results
    torch.manual_seed(1337)

    print(f"Starting training with {hyper_params}")

    # Create an Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)

    for train_iter in range(hyper_params.train_iters):
        # Extract some training data.
        inputs, targets = get_batch(train_data, hyper_params)

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Zero out the gradients
        optimizer.zero_grad(set_to_none=True)

        # Propagate gradients across the computation network.
        loss.backward()

        # Update the parameters based on gradients to minimize loss.
        optimizer.step()

        # Estimate and print the optimization progress at every 1 percentage.
        if train_iter % (hyper_params.train_iters / 100) == 0:
            train_loss = estimate_loss(model, train_data, hyper_params)
            val_loss = estimate_loss(model, val_data, hyper_params)
            progress_percentage = (
                float(train_iter + 1) / hyper_params.train_iters * 100.0
            )
            print(
                f"progress: {progress_percentage:.01f}%, training iteration: {train_iter + 1}/{hyper_params.train_iters}, training loss: {train_loss:.05f}, validation loss: {val_loss:.05f}"
            )


def get_batch(data, hyper_params):
    """Extract a subset of data to be used for training.

    Args:
        data (torch.tensor): a batch size x context size array of character indices
        hyper_params (HyperParameters): the parameters for training and evaluating the current model
    """

    # Generate random indices
    random_indices = torch.randint(
        len(data) - (hyper_params.block_size + 1), (hyper_params.batch_size,)
    )

    # Extract rows of inputs and their corresponding targets.
    inputs = torch.stack(
        [data[i : i + hyper_params.block_size] for i in random_indices]
    )
    targets = torch.stack(
        [data[i + 1 : i + hyper_params.block_size + 1] for i in random_indices]
    )

    # Upload data to specified device.
    inputs = inputs.to(hyper_params.device)
    targets = targets.to(hyper_params.device)

    return inputs, targets


@torch.no_grad()
def estimate_loss(model, data, hyper_params):
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

    # Compute the loss multiple times.
    for eval_index in range(hyper_params.eval_iters):
        # Evaluate & store the loss for a batch
        inputs, targets = get_batch(data, hyper_params)
        _, loss = model(inputs, targets)
        losses[eval_index] = loss.item()

    # Revert the model back to training mode.
    model.train()

    return losses.mean()


def visualize_weights_vs_gradients(parameters):
    # Visualize weight vs gradientlize histograms
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, p in enumerate(parameters):
        t = p.grad
        print(
            "weight %10s | mean %+f | std %e | grad:data ratio %e"
            % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"{i} {tuple(p.shape)}")
        plt.legend(legends)
        plt.title("weights gradient distribution")

    plt.show()


if __name__ == "__main__":
    main()
