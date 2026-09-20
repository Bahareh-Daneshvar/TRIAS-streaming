from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score


def classification_metrics(y_true, probs):
    y = np.asarray(y_true)
    p = np.asarray(probs)
    pred = p.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class_recall": recall_score(y, pred, average=None, zero_division=0).astype(float).tolist(),
    }


def train_sentiment_steps(model, batches, optimizer, device, max_steps=None):
    model.train()
    losses = []
    t0 = time.perf_counter()
    steps = 0
    for batch in batches:
        if max_steps is not None and steps >= max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, mask)
        loss = F.cross_entropy(out["sentiment_logits"], labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        steps += 1
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "steps": steps,
        "wall_seconds": float(time.perf_counter() - t0),
    }


@torch.no_grad()
def predict_sentiment(model, batches, device):
    model.eval()
    probs, ys, fused = [], [], []
    t0 = time.perf_counter()
    n = 0
    for batch in batches:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask)
        pr = torch.softmax(out["sentiment_logits"], dim=-1)
        probs.append(pr.cpu().numpy())
        fused.append(out["fused"].cpu().numpy())
        if "labels" in batch:
            ys.append(batch["labels"].cpu().numpy())
        n += len(ids)
    p = np.concatenate(probs) if probs else np.empty((0, 2))
    z = np.concatenate(fused) if fused else np.empty((0, 0))
    y = np.concatenate(ys) if ys else None
    return {
        "probs": p,
        "fused": z,
        "labels": y,
        "latency_ms_per_sample": 1000.0 * (time.perf_counter() - t0) / max(1, n),
    }
