import torch
from torch.utils.data import Dataset

class ArtTitleDataset(Dataset):
    def __init__(self, image_embeddings, input_ids, attention_mask):
        self.image_embeddings = image_embeddings
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "image_emb": self.image_embeddings[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }