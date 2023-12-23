#!/usr/bin/env python

"""makemore: a character-based language model which will generate Name like words."""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Makemore:
    """
    Learns from a text file with a list of words, then generate words that look like the original words.

    Args:
        file_path (str): Path to the file containing the list of new line separated words.
        num_context (int):
    """

    def __init__(self, file_path, num_context=3, num_dim_per_char=10, num_neurons=200):

        self._num_context = num_context
        self._num_dim_per_char = num_dim_per_char
        self._input_dim = num_context * num_dim_per_char
        self._num_neurons = num_neurons

        # Read the raw data, which is a list of names of people.
        words = self._read_raw_data(file_path)

        # Build mapping of characters to indices
        # We need to represent the characters as numerical elements
        self._index_to_char, self._char_to_index = self._build_char_index_mapping(words)
        num_characters = len(self._index_to_char)

        # Split the raw data into training, validation, and testing groups.
        self._partition_network_data(words)

        # Initialize a random generator.
        self._generator = torch.Generator().manual_seed(2147483647)

        # The embedding layer is used for encoding the input data in a way such that
        # it can be fed into a neural network.
        # In this case, we use a vector of 2 dimensional floats to represent each unique character.
        self._C = torch.randn((num_characters, num_dim_per_char), generator=self._generator)

        # Construct hidden layer #1 (aka neurons / weights)
        self._W1 = torch.randn((self._input_dim, self._num_neurons), generator=self._generator) * (5/3) / (self._input_dim)**0.5  # Weights
        self._b1 = torch.randn(self._num_neurons, generator=self._generator) * 0.01 # Biases

        # Construct output layer
        self._W2 = torch.randn((self._num_neurons, num_characters), generator=self._generator) * 0.1 # Weights
        self._b2 = torch.randn(num_characters, generator=self._generator) * 0 # Biases

        # Batch normalization gain & bias
        self._batch_norm_gain = torch.ones((1, self._num_neurons))
        self._batch_norm_bias = torch.zeros((1, self._num_neurons))

        # All parameters
        self.parameters = [self._C, self._W1, self._b1, self._W2, self._b2]
        for p in self.parameters:
            p.requires_grad = True

    def train(self, num_steps=200000, batch_size=32):
        """Train the current model

        Args:
            num_steps (int): Number of steps used to train the model.
            batch_size (int): Number of randomly selected words to train on in a single step
        """

        losses = []

        for step in range(num_steps):

            #
            # Forward pass
            #

            # Create a random list of indices for sampling from our inputs in this training step.
            batch_indices = torch.randint(0, self._inputs_tr.shape[0], (batch_size,), generator=self._generator)

            loss = self._forward_pass(
                self._inputs_tr,
                self._targets_tr,
                batch_indices,
            )

            # Store losses to later draw the loss over epoch.
            losses.append(loss.item())

            #
            # Backward pass
            #

            # Reset gradient first.
            for parameter in self.parameters:
                parameter.grad = None

            # Calculate gradients for each parameter.
            loss.backward()

            # Update parameter data.
            learning_rate = 0.01 if step < 100000 else 0.001
            for parameter in self.parameters:
                parameter.data += -learning_rate * parameter.grad

            if step % 10000 == 0:
                print(f"{step:7d} / {num_steps:7d}: {loss.item()}")

        with torch.no_grad():
            emb = self._C[self._inputs_tr]
            embcat = emb.view(emb.shape[0], -1)
            h_preact = embcat @ self._W1 + self._b1
            self._mean_tr = h_preact.mean(0, keepdim=True)
            self._std_tr = h_preact.std(0, keepdim=True)

        return losses

    @torch.no_grad()
    def generate_word(self, generator):
        # Sample from model
        word = ""
        context = [self._char_to_index["."]] * self._num_context
        while True:
            # Extract embedded
            emb = self._C[torch.tensor([context])]
            embcat = emb.view(emb.shape[0], self._input_dim)

            # Layer 1
            h_preact = embcat @ self._W1 + self._b1

            h_preact = (
                self._batch_norm_gain * (h_preact - self._mean_tr) / self._std_tr
            ) + self._batch_norm_bias

            h = torch.tanh(h_preact)

            # Layer 2
            logits = h @ self._W2 + self._b2

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)

            # Choose an char index based on probability.
            char_index = torch.multinomial(probs, num_samples=1, generator=generator).item()

            # Slide context window forward
            context = context[1:] + [char_index]

            # Append to word.
            word += self._index_to_char[char_index]

            if char_index == self._char_to_index["."]:
                break

        return word

    def _forward_pass(self, inputs, targets, indices):

        # Get the input entries corresponding to the batch indices.
        inputs_batch = inputs[indices]

        # Translate the inputs to their embedding encoding.
        emb = self._C[inputs_batch]

        # Each input entry is now a 3 by 2 matrix.  Concatenate into a 6 dimensional vector.
        embcat = emb.view(emb.shape[0], self._input_dim)

        # Multiply by weights and add bias.
        h_preact = embcat @ self._W1 + self._b1

        # Batch normalization
        h_preact = (
            self._batch_norm_gain * (h_preact - h_preact.mean(0, keepdim=True)) / h_preact.std(0, keepdim=True)
        ) + self._batch_norm_bias

        # Map values to be within -1 and 1
        h = torch.tanh(h_preact)

        #
        logits = h @ self._W2 + self._b2

        # computes log likelihood for classification purposes
        targets_batch = targets[indices]
        loss = F.cross_entropy(logits, targets_batch)

        return loss

    def compute_training_loss(self):
        return self._forward_pass(
            self._inputs_tr,
            self._targets_tr,
            torch.arange(0, self._inputs_tr.shape[0], dtype=torch.int),
        )

    def compute_validation_loss(self):
        return self._forward_pass(
            self._inputs_va,
            self._targets_va,
            torch.arange(0, self._inputs_va.shape[0], dtype=torch.int),
        )

    @staticmethod
    def _read_raw_data(file_path):
        with open(file_path, "r") as f:
            words = f.read().splitlines()

        print(f"Read {len(words)} words: {words[:6]}")
        return words

    @staticmethod
    def _build_char_index_mapping(words):
        """Utility to create mappings from characters to their index representations"""
        index_to_char = ["."]
        index_to_char.extend(sorted(list(set(''.join(words)))))
        char_to_index = {char: index for index, char in enumerate(index_to_char)}
        return index_to_char, char_to_index

    def _build_network_data(self, words):
        inputs = [] # inputs to the neural net.
        targets = [] # target/label for each data point in inputs.

        for w in words[:]:

            # Initialize context to "..." (expressed in indices)
            context = [self._char_to_index["."]] * self._num_context

            # Fill inputs & targets by scanning each character in the word.
            for char in (w + "."):
                char_index = self._char_to_index[char]
                inputs.append(context)
                targets.append(char_index)

                # Slide context cursor "forwards"
                context = context[1:] + [char_index]

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        return inputs, targets

    def _partition_network_data(self, words):
        # Partition the data into 3 groups:
        # - Training
        # - Validation
        # - Testing
        #
        # Each entry in the inputs will look something like "emm" where the corresponding target element
        # is the character after "emm", for example, "a".
        #
        # Since our context is 3 char wide, we use "." to represent an "blank" character to fill in empty space before or
        # after the word.

        random.seed(42)
        random.shuffle(words)
        num_tr = int(0.8*len(words))
        num_va = int(0.1*len(words))

        self._inputs_tr, self._targets_tr = self._build_network_data(words[:num_tr])
        self._inputs_va, self._targets_va = self._build_network_data(words[num_tr:num_tr + num_va])
        self._inputs_te, self._targets_te = self._build_network_data(words[num_tr + num_va:])


def main():
    model = Makemore("names.txt")
    model.train(num_steps=200000)

    loss_tr = model.compute_training_loss()
    print(f"Training loss: {loss_tr}")

    loss_va = model.compute_validation_loss()
    print(f"Validation loss: {loss_va}")

    generator = torch.Generator().manual_seed(2147483647)
    for _ in range(20):
        print(model.generate_word(generator=generator))



if __name__ == "__main__":
    main()

