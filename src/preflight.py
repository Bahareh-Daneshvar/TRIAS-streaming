from __future__ import annotations
from pathlib import Path
import json

REQUIRED = ("train_data.csv", "test_data.csv")
OPTIONAL = ("news_seed.xlsx",)


def inspect_data_dir(root: Path | None = None) -> dict:
    project = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    found = {name: (raw / name).exists() for name in REQUIRED + OPTIONAL}
    return {
        "raw_dir": str(raw),
        "required": {k: found[k] for k in REQUIRED},
        "optional": {k: found[k] for k in OPTIONAL},
        "ready_for_phase1": all(found[k] for k in REQUIRED),
    }


def main():
    report = inspect_data_dir()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ready_for_phase1"]:
        print("\nPhase 1 needs train_data.csv and test_data.csv in data/raw/. news_seed.xlsx is optional.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
