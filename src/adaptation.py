from __future__ import annotations

# Baseline only: scalar severity -> fixed depth. This is retained so TRIAS can be
# compared against a severity-only policy rather than conflating the two ideas.
SEVERITY_ONLY_POLICY = {
    "stable": tuple(),
    "mild": ("gate", "heads"),
    "moderate": ("gate", "heads", "bigru", "textcnn"),
    "severe": ("gate", "heads", "bigru", "textcnn", "encoder_top"),
}


def components_for_severity(severity: str):
    if severity not in SEVERITY_ONLY_POLICY:
        raise ValueError(f"Unknown severity: {severity}")
    return SEVERITY_ONLY_POLICY[severity]


def set_trainable_components(model, components, encoder=None):
    """Apply an explicit component list to the fusion host.

    This function is used by the TRIAS signature router. Encoder block-level
    unfreezing is handled more explicitly by `hf_fusion.set_selective_trainability`.
    """
    for p in model.parameters():
        p.requires_grad = False
    comps = tuple(components)
    mapping = {
        "gate": [model.gate],
        "heads": [model.sentiment_head, model.risk_level_head, model.risk_type_head],
        "bigru": [model.bigru, model.gru_proj],
        "textcnn": [model.textcnn],
    }
    for c in comps:
        for module in mapping.get(c, []):
            for p in module.parameters():
                p.requires_grad = True

    if encoder is not None:
        for p in encoder.parameters():
            p.requires_grad = False
        if "encoder_top" in comps:
            # Kept only for offline scaffold tests. End-to-end experiments must
            # use explicit transformer-block selection in hf_fusion.py.
            params = list(encoder.parameters())
            start = int(len(params) * 0.8)
            for p in params[start:]:
                p.requires_grad = True
    return comps


def set_trainable(model, severity: str, encoder=None):
    """Severity-only baseline retained for controlled comparison."""
    return set_trainable_components(model, components_for_severity(severity), encoder=encoder)
