from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel
from fusion_backbone import FusionBackbone


class HFFusionModel(nn.Module):
    """Reproducible host model: Hugging Face encoder + fusion backbone.

    The source paper does not identify an exact public checkpoint for its
    "RoBERTa-wwm-ext-small" variant, so `encoder_name` must be logged in every
    experiment and the host is not presented as an exact reproduction of an
    unavailable implementation.
    """

    def __init__(
        self,
        encoder_name: str,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        cnn_channels: int = 256,
        cnn_kernels=(2, 3, 4),
        fused_dim: int = 128,
        dropout: float = 0.2,
        sentiment_classes: int = 2,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = int(self.encoder.config.hidden_size)
        self.host = FusionBackbone(
            encoder_dim=hidden,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            fused_dim=fused_dim,
            dropout=dropout,
            sentiment_classes=sentiment_classes,
        )

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        out = self.host(enc.last_hidden_state, attention_mask.bool())
        return out


def transformer_blocks(encoder: nn.Module):
    """Return ordered transformer blocks for common BERT/RoBERTa-like encoders."""
    candidates = [
        ("encoder", "layer"),                  # BertModel.encoder.layer
        ("encoder", "layers"),                 # some variants
        ("transformer", "layer"),              # Distil-like variants
        ("transformer", "h"),                  # GPT-like variants
    ]
    root = encoder
    # AutoModel often exposes BERT directly, but some wrappers nest base_model.
    if hasattr(root, "base_model") and root.base_model is not root:
        root = root.base_model
    for a, b in candidates:
        if hasattr(root, a):
            x = getattr(root, a)
            if hasattr(x, b):
                blocks = getattr(x, b)
                try:
                    return list(blocks)
                except TypeError:
                    pass
    return []


def set_selective_trainability(model: HFFusionModel, components, top_encoder_blocks: int = 1):
    """Freeze all parameters, then enable an explicit TRIAS route."""
    for p in model.parameters():
        p.requires_grad = False

    comps = tuple(components)
    host = model.host
    modules = {
        "gate": [host.gate],
        "heads": [host.sentiment_head],
        "bigru": [host.bigru, host.gru_proj],
        "textcnn": [host.textcnn],
    }
    for comp in comps:
        for m in modules.get(comp, []):
            for p in m.parameters():
                p.requires_grad = True

    if "encoder_top" in comps:
        blocks = transformer_blocks(model.encoder)
        if not blocks:
            raise RuntimeError("Could not identify transformer blocks; refuse ambiguous partial unfreezing")
        n = min(max(1, int(top_encoder_blocks)), len(blocks))
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
    return comps
