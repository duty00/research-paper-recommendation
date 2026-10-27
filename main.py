"""Local training and prediction only. This program never submits to Kaggle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_priority.data import find_data_dir, load_dataset
from paper_priority.experiment import predict_bundle, run_experiment, save_prepared, validate_submission


def main():
    parser = argparse.ArgumentParser(description="CO3029 paper priority baseline; Python 3.11/3.12, CPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run", "predict", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", help="Folder containing train.csv, Test.csv and Submission.csv")
        if name in ("audit", "run"):
            sub.add_argument("--out", required=True, type=Path)
        if name == "run":
            sub.add_argument("--seed", type=int, default=261)
            sub.add_argument("--folds", type=int, choices=[3, 5], default=5)
        if name == "predict":
            sub.add_argument("--bundle", required=True, type=Path)
            sub.add_argument("--output", required=True, type=Path)
        if name == "check":
            sub.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    try:
        data = load_dataset(find_data_dir(args.data_dir))
        if args.command == "audit":
            save_prepared(data, args.out)
            print(json.dumps(data.audit, indent=2, ensure_ascii=False))
        elif args.command == "run":
            run_experiment(data, args.out, args.seed, args.folds)
        elif args.command == "predict":
            print(json.dumps(predict_bundle(args.bundle, data, args.output), indent=2))
        else:
            print(json.dumps(validate_submission(args.submission, [r["id"] for r in data.sample]), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
