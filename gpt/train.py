#!/usr/bin/env python

"""Train a language model with text from Shakespeare's play, such that it can generate
sentences like Shakespeare."""

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
    train_data_size = int(0.9*len(data))
    train_data = data[:train_data_size]
    val_data = data[train_data_size:]

    # Seed for deterministic results
    torch.manual_seed(1337)

    # Number of character sequences processed in parallel
    batch_size = 32

    # Block size is the # of characters per training sequence.
    block_size = 8

    # Number of training steps.
    num_steps = 10000

    print(f"Starting training with batch size: {batch_size}, block size: {block_size}, number of steps: {num_steps}")

    # Create the model
    model = BigramLanguageModel(vocab_size)

    # Create an Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(num_steps):

        # Extract some training data.
        inputs, targets = get_batch(train_data, batch_size, block_size)

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        if step % (num_steps/100) == 0:
            print(f"Step: {step}, loss: {loss}")

        # Zero out the gradients
        optimizer.zero_grad(set_to_none=True)

        # Propagate gradients across the computation network.
        loss.backward()

        # Update the parameters based on gradients to minimize loss.
        optimizer.step()

    # Generate some data
    idx = torch.zeros((1, 1), dtype=torch.long)
    predictions = model.generate(idx, 400)
    int_chars = predictions[0].tolist()
    chars = decode(int_chars)
    string = "".join(chars)
    print(string)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # This extracts a tensor with shape (Batch Size, Block Size, Vocab Size)
        # Where each integer from idx corresponds to a vocab-sized embedding vector.
        logits = self.token_embedding_table(idx)

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

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # Compute predictions
            logits, _ = self(idx)

            # Take the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distributions
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append index to sequence to generate max_new_tokens number of chars
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - (block_size + 1), (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


def visualize_weights_vs_gradients(parameters):
    # Visualize weight vs gradientlize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, p in enumerate(parameters):
        t = p.grad
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {tuple(p.shape)}')
        plt.legend(legends)
        plt.title('weights gradient distribution');

    plt.show()

if __name__ == "__main__":

    main()
