"""Contains the GPT model and its components."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from transcoder import AsciiTranscoder


def get_optimal_device():
    """Pick the optimal device based in the current environment.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


ASCII_TRANSCODER = AsciiTranscoder()


"""Parameters for training the GPT model"""
@dataclass
class HyperParameters:
    """Number of character sequences processed in parallel"""
    batch_size: int = 64

    """Block size is the maximum context length to make predictions from."""
    block_size: int = 256

    """Size of a embeddding vector representing a single character."""
    embed_size: int = 384

    """Number of self attention heads"""
    num_heads: int = 6

    """Number of transformer block layers"""
    num_block_layers: int = 6

    """Number of iterations to train the model."""
    train_iters : int = 5000

    """Number of iterations for estimating loss."""
    eval_iters: int = 300

    """The learning rate"""
    learning_rate: float = 3e-4

    """The learning rate"""
    dropout_prob: float = 0.2

    """Size of the vocabulary"""
    vocab_size: int = ASCII_TRANSCODER.vocab_size

    """The device to store and execute our neutral network."""
    device: torch.device = get_optimal_device()

    @property
    def head_size(self):
        assert self.embed_size % self.num_heads == 0
        return self.embed_size // self.num_heads


class GPT(nn.Module):

    def __init__(self, hyper_params=None):
        super().__init__()

        hyper_params = hyper_params or HyperParameters()
        self.hyper_params = hyper_params

        self.token_embedding_table = nn.Embedding(
            hyper_params.vocab_size, hyper_params.embed_size
        ).to(hyper_params.device)

        self.position_embedding_table = nn.Embedding(
            hyper_params.block_size, hyper_params.embed_size
        ).to(hyper_params.device)

        self.blocks = nn.Sequential(*[Block(hyper_params) for _ in range(hyper_params.num_block_layers)])
        self.layer_norm = nn.LayerNorm(hyper_params.embed_size)
        self.linear_head = nn.Linear(hyper_params.embed_size, hyper_params.vocab_size)

        self._device = hyper_params.device

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        # This extracts a tensor with shape (Batch Size, Block Size, Vocab Size)
        # Where each integer from inputs corresponds to a embedding vector.
        token_emb = self.token_embedding_table(inputs)  # B, T, embed_size
        position_emb = self.position_embedding_table(torch.arange(T).to(self._device)) # T, C

        combined_emb = token_emb + position_emb

        # Apply self attention head.
        attended_emb = self.blocks(combined_emb)

        norm_emb = self.layer_norm(attended_emb)

        logits = self.linear_head(norm_emb)  # (B, T, vocab_size)

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
            # Crop input to last block_size tokens for making predictions
            inputs_cropped = inputs[:, -self.hyper_params.block_size:]

            # Compute predictions
            logits, _ = self(inputs_cropped)

            # Take the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next character from distribution
            next_input = torch.multinomial(probs, num_samples=1)

            # Append index to sequence to generate max_new_tokens number of chars
            inputs = torch.cat((inputs, next_input), dim=1)

        return inputs


class Block(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()

        self.self_attention_head = MultiHeadAttention(hyper_params)
        self.feed_forward = FeedForward(hyper_params)
        self.layer_norm_1 = nn.LayerNorm(hyper_params.embed_size)
        self.layer_norm_2 = nn.LayerNorm(hyper_params.embed_size)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.self_attention_head(x)
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x


class Head(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()
        self.key = nn.Linear(hyper_params.embed_size, hyper_params.head_size, bias=False).to(hyper_params.device)
        self.query = nn.Linear(hyper_params.embed_size, hyper_params.head_size, bias=False).to(hyper_params.device)
        self.value = nn.Linear(hyper_params.embed_size, hyper_params.head_size, bias=False).to(hyper_params.device)
        self.register_buffer("tril", torch.tril(torch.ones(hyper_params.block_size, hyper_params.block_size)))
        self.dropout = nn.Dropout(hyper_params.dropout_prob)

    def forward(self, inputs):
        batch_size, block_size, embed_size = inputs.shape

        # Keys are the input elements to compare against
        key = self.key(inputs)

        # Query represents the current element
        query = self.query(inputs)

        # Compute attention scores
        head_size = key.shape[2]
        wei = query @ key.transpose(-2, -1) * (head_size**-0.5)
        wei = wei.masked_fill(self.tril[:block_size, :block_size] == 0, float("-inf")) # replace 0's with negative infinities
        wei = F.softmax(wei, dim=-1) # normalize into probabilities with softmax

        wei = self.dropout(wei)

        # Weighted aggregation
        value = self.value(inputs)
        output = wei @ value

        return output


class FeedForward(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hyper_params.embed_size, 4 * hyper_params.embed_size),
            nn.ReLU(),
            nn.Linear(4 * hyper_params.embed_size, hyper_params.embed_size),
            nn.Dropout(hyper_params.dropout_prob),
        )

    def forward(self, inputs):
        return self.net(inputs)


class MultiHeadAttention(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()
        self.heads = nn.ModuleList([Head(hyper_params) for _ in range(hyper_params.num_heads)])
        self.proj = nn.Linear(hyper_params.embed_size, hyper_params.embed_size)
        self.dropout = nn.Dropout(hyper_params.dropout_prob)

    def forward(self, inputs):
        out = torch.cat([head(inputs) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


