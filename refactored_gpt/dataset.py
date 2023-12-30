import torch
from torch.utils.data import Dataset

from transcoder import AsciiTranscoder


class AsciiTextFileDataset(Dataset):
    """Reads a text file and provides character data."""

    _ASCII_TRANSCODER = AsciiTranscoder()

    def __init__(self, text_file_path, block_size):
        with open(text_file_path, "r", encoding="ascii") as f:
            text = f.read()

        encoded_text = self._ASCII_TRANSCODER.encode(text)
        self.data = torch.tensor(encoded_text, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, idx):
        input_data = self.data[idx : idx + self.block_size]
        target_data = self.data[idx + 1 : idx + self.block_size + 1]
        assert input_data.shape == target_data.shape, f"{input_data.shape}, {target_data.shape}, {idx}"
        return input_data, target_data

    @property
    def transcoder(self):
        return self._ASCII_TRANSCODER
