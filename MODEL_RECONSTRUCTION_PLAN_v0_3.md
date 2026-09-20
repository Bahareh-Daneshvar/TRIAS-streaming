# Fusion host implementation plan v0.3

## Purpose
Provide a transparent and reproducible host model for testing TRIAS. It is not presented as an exact reproduction of any unavailable implementation.

## Architecture retained from the paper
1. contextual token encoder,
2. Bi-GRU branch + attention,
3. TextCNN local branch with kernels 2/3/4,
4. dynamic gated fusion,
5. sentiment classification head for public Data 1.

## Important code correction in v0.3
Padding is now handled explicitly:
- Bi-GRU uses packed sequences so the backward state is not contaminated by padded tokens;
- TextCNN masks convolution windows containing padding before max pooling.

This is necessary for a defensible variable-length text implementation and was missing from the v0.2 scaffold.

## Encoder decision
Do not hard-code an invented `RoBERTa-wwm-ext-small` checkpoint.

Candidate reproducible settings:
- Primary reproducible encoder: `hfl/chinese-roberta-wwm-ext` (public, exact checkpoint name; heavier than the paper's claimed small model).
- Lightweight sensitivity proxy: a publicly named small Chinese RoBERTa checkpoint, clearly described as a proxy rather than the original authors' checkpoint.

The exact primary choice is frozen only after compute feasibility is checked.

## Efficiency strategy
Initial reference training can cache/freeze encoder representations for rapid architecture checks. However, the final severe-shift policy that unfreezes top encoder blocks must be evaluated end-to-end on the live encoder; cached-embedding experiments cannot support claims about encoder adaptation cost or effect.
