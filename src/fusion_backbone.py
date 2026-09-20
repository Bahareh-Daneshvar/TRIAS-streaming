from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.context = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.context, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, L, D]
        u = torch.tanh(self.proj(x))
        scores = torch.einsum("bld,d->bl", u, self.context)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        a = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bl,bld->bd", a, x)
        return pooled, a


class BiGRUBranch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0
        )
        self.attn = AttentionPool(hidden * 2)

    def forward(self, x, mask=None):
        if mask is None:
            h, _ = self.gru(x)
        else:
            lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_h, _ = self.gru(packed)
            h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=x.shape[1])
        return self.attn(h, mask)


class TextCNNBranch(nn.Module):
    def __init__(self, in_dim: int, channels: int = 256, kernels=(2, 3, 4), out_dim: int = 128, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_dim, channels, k) for k in kernels])
        self.proj = nn.Sequential(
            nn.Linear(channels * len(kernels), out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # [B,L,D] -> [B,D,L]
        y = x.transpose(1, 2)
        feats = []
        mask_f = None if mask is None else mask.float().unsqueeze(1)
        for conv in self.convs:
            z = F.relu(conv(y))
            if mask_f is not None:
                k = conv.kernel_size[0]
                kernel = torch.ones(1, 1, k, device=x.device, dtype=mask_f.dtype)
                valid = F.conv1d(mask_f, kernel).squeeze(1) >= float(k)
                z = z.masked_fill(~valid.unsqueeze(1), torch.finfo(z.dtype).min)
                pooled = torch.amax(z, dim=-1)
                # If a sequence is shorter than a kernel, avoid -inf propagation.
                no_valid = ~valid.any(dim=-1)
                if no_valid.any():
                    pooled[no_valid] = 0.0
            else:
                pooled = torch.amax(z, dim=-1)
            feats.append(pooled)
        return self.proj(torch.cat(feats, dim=-1))


class FusionBackbone(nn.Module):
    """Transparent fusion backbone used as the TRIAS evaluation host.

    It consumes token-level contextual embeddings rather than owning a specific
    RoBERTa implementation, making the architecture testable offline.
    """
    def __init__(
        self,
        encoder_dim: int,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        cnn_channels: int = 256,
        cnn_kernels=(2, 3, 4),
        fused_dim: int = 128,
        dropout: float = 0.2,
        sentiment_classes: int = 2,
        risk_level_classes: int = 3,
        risk_type_classes: int = 3,
    ):
        super().__init__()
        self.bigru = BiGRUBranch(encoder_dim, gru_hidden, gru_layers, dropout)
        self.gru_proj = nn.Linear(gru_hidden * 2, fused_dim)
        self.textcnn = TextCNNBranch(encoder_dim, cnn_channels, cnn_kernels, fused_dim, dropout)
        self.gate = nn.Linear(fused_dim * 2, fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(fused_dim, sentiment_classes)
        self.risk_level_head = nn.Linear(fused_dim, risk_level_classes)
        self.risk_type_head = nn.Linear(fused_dim, risk_type_classes)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        vg, attn = self.bigru(hidden_states, attention_mask)
        vg = self.gru_proj(vg)
        vc = self.textcnn(hidden_states, attention_mask)
        g = torch.sigmoid(self.gate(torch.cat([vg, vc], dim=-1)))
        fused = g * vg + (1.0 - g) * vc
        fused = self.dropout(fused)
        return {
            "fused": fused,
            "gate": g,
            "gru_attention": attn,
            "sentiment_logits": self.sentiment_head(fused),
            "risk_level_logits": self.risk_level_head(fused),
            "risk_type_logits": self.risk_type_head(fused),
        }
