from __future__ import annotations

import argparse, copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from adaptation import components_for_severity, set_trainable_components
from char_fusion import CharacterFusionModel, CharacterVocabulary, make_loader
from data1 import normalize_data1
from data_audit import load_any
from drift import MultiViewDriftMonitor
from routing import route_from_signature
from temporal_manifest import chronological_split_by_day, iter_day_grouped_target_chunks
from train_utils import classification_metrics, predict_sentiment, train_sentiment_steps


POLICIES = (
    "B0_static", "B1_periodic", "B2_sliding", "B3_error_triggered",
    "B4_representation_only", "B5_triview_equal", "B6_severity_only",
    "B7_trias", "A9_route_shuffle",
)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_split(train_path, test_path):
    frames = []
    for path in (train_path, test_path):
        x = normalize_data1(load_any(path)); x["source_file"] = path.name; frames.append(x)
    data = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable")
    return chronological_split_by_day(data, reference_fraction=.20, calibration_fraction=.10)


def reliability_metrics(y, p, bins=10):
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    confidence, pred = p.max(1), p.argmax(1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any(): ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - confidence[mask].mean())
    return {
        "brier": float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(p[np.arange(len(y)), y]))), "ece_10": float(ece),
    }


def infer_frame(model, frame, vocab, args, device):
    return predict_sentiment(model, make_loader(frame, vocab, args.max_length, args.batch_size, False, args.seed), device)


def train_frame(model, frame, vocab, components, args, device):
    if not components or frame.empty: return {"steps": 0, "wall_seconds": 0.0, "trainable_parameters": 0}
    set_trainable_components(model.host, components, encoder=model.encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.update_lr)
    outcome = train_sentiment_steps(
        model, make_loader(frame, vocab, args.max_length, args.batch_size, True, args.seed),
        optimizer, device, max_steps=args.update_steps,
    )
    outcome["trainable_parameters"] = int(sum(p.numel() for p in params))
    return outcome


def policy_components(name, chunk_i, scores, monitor, macro_f1, error_threshold, shuffled):
    medium = ("gate", "heads", "bigru", "textcnn")
    if name == "B0_static": return tuple()
    if name == "B1_periodic": return medium if (chunk_i + 1) % 5 == 0 else tuple()
    if name == "B2_sliding": return medium
    if name == "B3_error_triggered": return medium if macro_f1 < error_threshold else tuple()
    if name == "B4_representation_only": return medium if scores.z_representation >= monitor.z_high[0] else tuple()
    if name == "B5_triview_equal": return medium if scores.severity != "stable" else tuple()
    if name == "B6_severity_only": return components_for_severity(scores.severity)
    if name == "B7_trias": return route_from_signature(scores, monitor).components
    if name == "A9_route_shuffle": return shuffled[chunk_i]
    raise ValueError(name)


def run_policy(name, initial_state, chunks, cached_scores, monitor, vocab, args, device, error_threshold, shuffled):
    model = CharacterFusionModel(len(vocab), args.embedding_dim, args.gru_hidden, args.cnn_channels, args.fused_dim).to(device)
    model.load_state_dict(initial_state)
    ys, ps, rows, replay = [], [], [], pd.DataFrame()
    updates = steps = trainable_sum = 0; seconds = 0.0
    for i, chunk in enumerate(chunks):
        pred = infer_frame(model, chunk, vocab, args, device)
        metric = classification_metrics(pred["labels"], pred["probs"])
        score = cached_scores[i]
        comps = policy_components(name, i, score, monitor, metric["macro_f1"], error_threshold, shuffled)
        update_frame = chunk
        if name == "B2_sliding":
            replay = pd.concat([replay, chunk], ignore_index=True).tail(args.replay_rows); update_frame = replay
        outcome = train_frame(model, update_frame, vocab, comps, args, device)
        if outcome["steps"]:
            updates += 1; steps += outcome["steps"]; seconds += outcome["wall_seconds"]
            trainable_sum += outcome["trainable_parameters"]
        ys.append(pred["labels"]); ps.append(pred["probs"])
        rows.append({"policy": name, "chunk": i, "n": len(chunk), "macro_f1": metric["macro_f1"],
                     "severity": score.severity, "drift_score": score.composite,
                     "components": "+".join(comps) if comps else "none",
                     "update_steps": outcome["steps"], "update_seconds": outcome["wall_seconds"]})
    y, p = np.concatenate(ys), np.concatenate(ps)
    summary = {"policy": name, **classification_metrics(y, p), **reliability_metrics(y, p),
               "rows": len(y), "chunks": len(chunks), "updates": updates, "update_steps": steps,
               "adaptation_seconds": seconds, "cumulative_trainable_parameters": trainable_sum}
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description="TRIAS B0-B7 feasibility pilot; outputs are not paper results")
    ap.add_argument("--train", type=Path, default=Path("data/raw/train_data.csv")); ap.add_argument("--test", type=Path, default=Path("data/raw/test_data.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/pilot_b0_b7")); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=64); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=32); ap.add_argument("--gru-hidden", type=int, default=24); ap.add_argument("--cnn-channels", type=int, default=24); ap.add_argument("--fused-dim", type=int, default=32)
    ap.add_argument("--initial-epochs", type=int, default=2); ap.add_argument("--initial-lr", type=float, default=2e-3); ap.add_argument("--update-lr", type=float, default=5e-4)
    ap.add_argument("--update-steps", type=int, default=2); ap.add_argument("--replay-rows", type=int, default=256)
    ap.add_argument("--chunk-target", type=int, default=128); ap.add_argument("--monitor-window", type=int, default=128); ap.add_argument("--calibration-stride", type=int, default=1); ap.add_argument("--n-projections", type=int, default=8); ap.add_argument("--max-stream-chunks", type=int, default=0)
    args = ap.parse_args(); seed_all(args.seed); device = torch.device(args.device)
    split = load_split(args.train, args.test); vocab = CharacterVocabulary.fit(split.reference.text)
    model = CharacterFusionModel(len(vocab), args.embedding_dim, args.gru_hidden, args.cnn_channels, args.fused_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr)
    for epoch in range(args.initial_epochs):
        train_sentiment_steps(model, make_loader(split.reference, vocab, args.max_length, args.batch_size, True, args.seed + epoch), optimizer, device)
    initial_state = copy.deepcopy(model.state_dict()); sentinel = copy.deepcopy(model).eval()
    for p in sentinel.parameters(): p.requires_grad = False
    ref, cal = infer_frame(sentinel, split.reference, vocab, args, device), infer_frame(sentinel, split.calibration, vocab, args, device)
    monitor = MultiViewDriftMonitor(n_proj=args.n_projections, seed=args.seed)
    monitor.calibrate_fixed_reference(ref["fused"], ref["probs"], cal["fused"], cal["probs"], args.monitor_window, args.calibration_stride)
    error_threshold = max(0.0, classification_metrics(cal["labels"], cal["probs"])["macro_f1"] - .05)
    chunks = list(iter_day_grouped_target_chunks(split.stream, target_rows=args.chunk_target))
    if args.max_stream_chunks > 0: chunks = chunks[:args.max_stream_chunks]
    cached_scores, trailing_x, trailing_p = [], [], []
    for chunk in chunks:
        cur = infer_frame(sentinel, chunk, vocab, args, device); trailing_x.extend(cur["fused"]); trailing_p.extend(cur["probs"])
        xw, pw = np.asarray(trailing_x[-args.monitor_window:]), np.asarray(trailing_p[-args.monitor_window:])
        if len(xw) < args.monitor_window: xw, pw = cur["fused"], cur["probs"]
        cached_scores.append(monitor.score(ref["fused"], xw, ref["probs"], pw))
    routes = [route_from_signature(x, monitor).components for x in cached_scores]
    shuffled = list(routes); random.Random(args.seed + 9001).shuffle(shuffled)
    assert sorted(map(str, routes)) == sorted(map(str, shuffled)), "route-shuffle must preserve action multiset"
    summaries, detail = [], []
    for name in POLICIES:
        s, d = run_policy(name, initial_state, chunks, cached_scores, monitor, vocab, args, device, error_threshold, shuffled); summaries.append(s); detail.extend(d)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.outdir / "pilot_summary.csv", index=False); pd.DataFrame(detail).to_csv(args.outdir / "pilot_chunks.csv", index=False)
    meta = {"status": "feasibility_pilot_not_paper_result", "seed": args.seed, "policies": POLICIES,
            "vocabulary_source": "reference_only", "calibration": monitor.calibration_meta,
            "thresholds": monitor.thresholds, "error_trigger_threshold": error_threshold,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.outdir / "pilot_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__": main()
