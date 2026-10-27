"""CPU-only v2 experiments and saved-model inference. Never submits to Kaggle."""
import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset, find_data_dir
from paper_priority.improvement import run_improvement, predict_improvement


def main():
    parser = argparse.ArgumentParser(description='Development-only v2 search and reproducible predictions.')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--reference', type=Path, default=Path('outputs/my_run01'))
    predict = sub.add_parser('predict')
    predict.add_argument('--bundle', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    for command in (run, predict):
        command.add_argument('--data-dir')
    args = parser.parse_args()
    try:
        data = load_dataset(find_data_dir(args.data_dir))
        if args.command == 'run':
            run_improvement(data, args.out, args.reference)
        else:
            print(json.dumps(predict_improvement(args.bundle, data, args.output), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
