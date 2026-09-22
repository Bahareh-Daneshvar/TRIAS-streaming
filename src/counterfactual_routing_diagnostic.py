from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from char_fusion import CharacterFusionModel, CharacterVocabulary, make_loader
from drift import MultiViewDriftMonitor
from pilot_b0_b7 import infer_frame, load_split, seed_all, train_frame
from routing import route_from_signature
from temporal_manifest import iter_day_grouped_target_chunks
from train_utils import classification_metrics, train_sentiment_steps


ACTIONS = {
    "none": tuple(),
    "heads": ("heads",),
    "gate_heads": ("gate", "heads"),
    "medium": ("gate", "heads", "bigru", "textcnn"),
    "deep": ("gate", "heads", "bigru", "textcnn", "encoder_top"),
}
COMPONENTS_TO_ACTION = {v: k for k, v in ACTIONS.items()}


def severity_action(severity):
    return {"stable": "none", "mild": "gate_heads", "moderate": "medium", "severe": "deep"}[severity]


def routed_action(scores, monitor):
    components = tuple(route_from_signature(scores, monitor).components)
    if components not in COMPONENTS_TO_ACTION:
        raise RuntimeError(f"Unregistered routed component set: {components}")
    return COMPONENTS_TO_ACTION[components]


def summarize_counterfactuals(frame, seed=42):
    """Compare routed choices with per-chunk one-step counterfactual benefits."""
    x = frame.copy()
    action_cols = [f"benefit_{a}" for a in ACTIONS]
    benefit = x[action_cols].to_numpy(float)
    oracle_i = benefit.argmax(axis=1)
    action_names = np.asarray(list(ACTIONS))
    x["oracle_action"] = action_names[oracle_i]
    x["oracle_benefit"] = benefit[np.arange(len(x)), oracle_i]

    routed = x["routed_action"].tolist()
    severity = x["severity_action"].tolist()
    shuffled = list(routed)
    random.Random(seed + 9001).shuffle(shuffled)
    if sorted(shuffled) != sorted(routed):
        raise AssertionError("Route-shuffle must preserve the action multiset")
    x["shuffled_action"] = shuffled

    rows = []
    for policy, choices in (("routing", routed), ("severity_only", severity), ("route_shuffle", shuffled)):
        chosen = np.array([x.iloc[i][f"benefit_{a}"] for i, a in enumerate(choices)], dtype=float)
        oracle = x["oracle_benefit"].to_numpy(float)
        rows.append({
            "policy": policy,
            "chunks": int(len(x)),
            "action_accuracy": float(np.mean(np.asarray(choices) == x["oracle_action"].to_numpy())),
            "mean_benefit": float(chosen.mean()),
            "mean_regret": float((oracle - chosen).mean()),
            "positive_benefit_rate": float(np.mean(chosen > 0)),
            "harm_rate": float(np.mean(chosen < 0)),
        })
    return x, pd.DataFrame(rows)


def build_model(vocab, args, device):
    return CharacterFusionModel(
        len(vocab), args.embedding_dim, args.gru_hidden, args.cnn_channels, args.fused_dim
    ).to(device)


def main():
    ap = argparse.ArgumentParser(description="One-step counterfactual diagnostic; not a paper result")
    ap.add_argument("--train", type=Path, default=Path("data/raw/train_data.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/raw/test_data.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/counterfactual"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--gru-hidden", type=int, default=12)
    ap.add_argument("--cnn-channels", type=int, default=12)
    ap.add_argument("--fused-dim", type=int, default=16)
    ap.add_argument("--initial-epochs", type=int, default=1)
    ap.add_argument("--initial-lr", type=float, default=2e-3)
    ap.add_argument("--update-lr", type=float, default=5e-4)
    ap.add_argument("--update-steps", type=int, default=1)
    ap.add_argument("--replay-rows", type=int, default=256)
    ap.add_argument("--chunk-target", type=int, default=128)
    ap.add_argument("--monitor-window", type=int, default=128)
    ap.add_argument("--calibration-stride", type=int, default=1)
    ap.add_argument("--n-projections", type=int, default=4)
    ap.add_argument("--max-transitions", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    split = load_split(args.train, args.test)
    vocab = CharacterVocabulary.fit(split.reference.text)
    anchor = build_model(vocab, args, device)
    optimizer = torch.optim.AdamW(anchor.parameters(), lr=args.initial_lr)
    for epoch in range(args.initial_epochs):
        loader = make_loader(split.reference, vocab, args.max_length, args.batch_size, True, args.seed + epoch)
        train_sentiment_steps(anchor, loader, optimizer, device)

    sentinel = copy.deepcopy(anchor).eval()
    for parameter in sentinel.parameters():
        parameter.requires_grad = False
    ref = infer_frame(sentinel, split.reference, vocab, args, device)
    cal = infer_frame(sentinel, split.calibration, vocab, args, device)
    monitor = MultiViewDriftMonitor(n_proj=args.n_projections, seed=args.seed)
    monitor.calibrate_fixed_reference(
        ref["fused"], ref["probs"], cal["fused"], cal["probs"],
        args.monitor_window, args.calibration_stride,
    )
    chunks = list(iter_day_grouped_target_chunks(split.stream, target_rows=args.chunk_target))
    transitions = len(chunks) - 1
    if args.max_transitions > 0:
        transitions = min(transitions, args.max_transitions)

    cached_scores, trailing_x, trailing_p = [], [], []
    for chunk in chunks[:transitions]:
        cur = infer_frame(sentinel, chunk, vocab, args, device)
        trailing_x.extend(cur["fused"]); trailing_p.extend(cur["probs"])
        xw, pw = np.asarray(trailing_x[-args.monitor_window:]), np.asarray(trailing_p[-args.monitor_window:])
        if len(xw) < args.monitor_window:
            xw, pw = cur["fused"], cur["probs"]
        cached_scores.append(monitor.score(ref["fused"], xw, ref["probs"], pw))

    rows = []
    for i in range(transitions):
        current, future, scores = chunks[i], chunks[i + 1], cached_scores[i]
        base_state = copy.deepcopy(anchor.state_dict())
        result = {
            "seed": args.seed, "chunk": i, "current_n": len(current), "future_n": len(future),
            "severity": scores.severity, "z_representation": scores.z_representation,
            "z_output": scores.z_output, "z_dependency": scores.z_dependency,
            "routed_action": routed_action(scores, monitor),
            "severity_action": severity_action(scores.severity),
        }
        baseline_f1 = None
        for action, components in ACTIONS.items():
            branch = build_model(vocab, args, device); branch.load_state_dict(base_state)
            if components:
                train_frame(branch, current, vocab, components, args, device)
            pred = infer_frame(branch, future, vocab, args, device)
            f1 = classification_metrics(pred["labels"], pred["probs"])["macro_f1"]
            result[f"future_f1_{action}"] = f1
            if action == "none": baseline_f1 = f1
        for action in ACTIONS:
            result[f"benefit_{action}"] = result[f"future_f1_{action}"] - baseline_f1
        rows.append(result)

        # The common anchor follows a fixed medium update; routing choices never
        # contaminate the state from which later counterfactual branches start.
        anchor.load_state_dict(base_state)
        train_frame(anchor, current, vocab, ACTIONS["medium"], args, device)

    detail, summary = summarize_counterfactuals(pd.DataFrame(rows), args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.outdir / "counterfactual_chunks.csv", index=False)
    summary.to_csv(args.outdir / "counterfactual_summary.csv", index=False)
    metadata = {
        "status": "diagnostic_not_paper_result", "seed": args.seed,
        "actions": {k: list(v) for k, v in ACTIONS.items()},
        "anchor_policy": "fixed_medium_after_each_branch_set",
        "evaluation": "update_on_chunk_t_then_evaluate_chunk_t_plus_1",
        "calibration": monitor.calibration_meta,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.outdir / "counterfactual_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
