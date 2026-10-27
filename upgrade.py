import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset, find_data_dir
from paper_priority.large_embeddings import LARGE_ENCODERS, ensure_large, large_cache
from paper_priority.upgrade_experiment import run_upgrade, predict_upgrade


def main():
    parser = argparse.ArgumentParser(description='V6 large encoders and paired seven-fold validation against v5.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'predict'):
        sub = commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--assets', type=Path, default=Path('assets/encoders'))
        sub.add_argument('--cache', type=Path, default=Path('.cache/embeddings'))
        if name == 'run':
            sub.add_argument('--out', type=Path, required=True)
            sub.add_argument('--reference-v5', type=Path, default=Path('outputs/v5_run01'))
        elif name == 'predict':
            sub.add_argument('--bundle', type=Path, required=True)
            sub.add_argument('--output', type=Path, required=True)
            sub.add_argument('--refit', action='store_true', help='Independently refit all heads from raw training data.')
    args = parser.parse_args()
    data = load_dataset(find_data_dir(args.data_dir))
    if args.command == 'prepare':
        for name in LARGE_ENCODERS:
            ensure_large(name, args.assets, download=True)
            for rows in (data.train, data.test):
                values, _ = large_cache(rows, name, args.assets, args.cache)
                print(name, values.shape, flush=True)
    elif args.command == 'run':
        run_upgrade(data, args.out, args.reference_v5, args.assets, args.cache)
    else:
        print(json.dumps(predict_upgrade(args.bundle, data, args.output, args.assets, args.cache, refit=args.refit), indent=2))


if __name__ == '__main__':
    main()
