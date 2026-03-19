import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    A simple PyTorch Dataset for handling large lists of strings, allowing for batching and such.
    """
    def __init__(self, texts):
        self.texts = list(texts) if not isinstance(texts, list) else texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]