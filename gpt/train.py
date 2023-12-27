#!/usr/bin/env python

"""Train a language model with text from Shakespeare's play, such that it can generate
sentences like Shakespeare."""

import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt


def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Length of dataset in characters: {len(text)}")

    # Get unique characters from data set
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary: {chars}")
    print(f"Vocab size: {vocab_size}")

    # Create mapping & functions to convert between characters and integers.
    char_to_int = {char: i for i, char in enumerate(chars)}
    int_to_char = {i: char for i, char in enumerate(chars)}
    encode = lambda s: [char_to_int[c] for c in s]
    decode = lambda l: [int_to_char[i] for i in l]

    # Create a tensor and encode the data.
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)

    # Separate data into training & validation sets.
    train_data_size = int(0.9 * len(data))
    train_data = data[:train_data_size]
    val_data = data[train_data_size:]

    # Seed for deterministic results
    torch.manual_seed(1337)

    # Set up our model's hyper parameters.
    hyper_params = HyperParameters(
        batch_size=32,
        block_size=8,
        train_iters=10000,
        eval_iters=300,
        learning_rate=1e-2,
        device=get_optimal_device(),
    )

    print(
        f"Starting training with batch size: {hyper_params.batch_size}, block size: {hyper_params.block_size}, number of training iterations: {hyper_params.train_iters}"
    )

    # Create the model
    model = BigramLanguageModel(vocab_size, hyper_params.device).to(hyper_params.device)

    # Create an Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)

    for train_iter in range(hyper_params.train_iters):
        # Extract some training data.
        inputs, targets = get_batch(train_data, hyper_params)

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Estimate and print the optimization progress at every 1 percentage.
        if train_iter % (hyper_params.train_iters / 100) == 0:
            train_loss = estimate_loss(model, train_data, hyper_params)
            val_loss = estimate_loss(model, val_data, hyper_params)
            progress_percentage = float(train_iter) / hyper_params.train_iters * 100.0
            print(
                f"progress: {progress_percentage:.01f}%, training iteration: {train_iter}/{hyper_params.train_iters}, training loss: {train_loss:.05f}, validation loss: {val_loss:.05f}"
            )

        # Zero out the gradients
        optimizer.zero_grad(set_to_none=True)

        # Propagate gradients across the computation network.
        loss.backward()

        # Update the parameters based on gradients to minimize loss.
        optimizer.step()

    # Generate some data
    inputs = torch.zeros((1, 1), dtype=torch.long).to(hyper_params.device)
    predictions = model.generate(inputs, 400)
    int_chars = predictions[0].tolist()
    chars = decode(int_chars)
    string = "".join(chars)
    print(string)


HyperParameters = collections.namedtuple(
    "HyperParameters",
    [
        "batch_size",  # Number of character sequences processed in parallel
        "block_size",  # Block size is the maximum context length to make predictions from.
        "train_iters",  # Number of iterations to train the model.
        "eval_iters",  # Number of iterations for estimating loss.
        "learning_rate",  # The learning rate
        "device",  # The device to store and execute our neutral network.
    ],
)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size).to(device)

    def forward(self, inputs, targets=None):
        # This extracts a tensor with shape (Batch Size, Block Size, Vocab Size)
        # Where each integer from inputs corresponds to a vocab-sized embedding vector.
        logits = self.token_embedding_table(inputs)

        if targets is None:
            # During generation we do not need to compute loss.
            loss = None
        else:
            # Need to generate loss for training.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            # Compute predictions
            logits, _ = self(inputs)

            # Take the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next character from distribution
            next_input = torch.multinomial(probs, num_samples=1)

            # Append index to sequence to generate max_new_tokens number of chars
            inputs = torch.cat((inputs, next_input), dim=1)

        return inputs


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


def get_optimal_device():
    """Pick the optimal device based in the current environment.

    Returns:
        torch.device
    """
    return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    main()
