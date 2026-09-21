from __future__ import annotations

from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fusion_backbone import FusionBackbone


class CharacterVocabulary:
    PAD, UNK = 0, 1

    def __init__(self, stoi):
        self.stoi = dict(stoi)

    @classmethod
    def fit(cls, texts, max_size=4096):
        counts = Counter(ch for text in texts for ch in str(text))
        chars = sorted(counts, key=lambda ch: (-counts[ch], ch))[: max_size - 2]
        return cls({ch: i + 2 for i, ch in enumerate(chars)})

    def encode(self, text, max_length):
        ids = [self.stoi.get(ch, self.UNK) for ch in str(text)[:max_length]]
        return ids + [self.PAD] * (max_length - len(ids))

    def __len__(self):
        return len(self.stoi) + 2


class TextFrameDataset(Dataset):
    def __init__(self, frame, vocab, max_length=64):
        self.ids = torch.tensor([vocab.encode(x, max_length) for x in frame.text], dtype=torch.long)
        self.labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        ids = self.ids[i]
        return {"input_ids": ids, "attention_mask": ids.ne(0), "labels": self.labels[i]}


def make_loader(frame, vocab, max_length=64, batch_size=64, shuffle=False, seed=42):
    return DataLoader(
        TextFrameDataset(frame, vocab, max_length), batch_size=batch_size,
        shuffle=shuffle, generator=torch.Generator().manual_seed(int(seed)),
    )


class CharacterFusionModel(nn.Module):
    """Compact reproducible host for feasibility checks, never final paper results."""

    def __init__(self, vocab_size, embedding_dim=32, gru_hidden=24, cnn_channels=24, fused_dim=32):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.host = FusionBackbone(
            encoder_dim=embedding_dim, gru_hidden=gru_hidden, gru_layers=1,
            cnn_channels=cnn_channels, fused_dim=fused_dim, dropout=0.1,
            sentiment_classes=2,
        )

    def forward(self, input_ids, attention_mask):
        return self.host(self.encoder(input_ids), attention_mask.bool())
