"""V3 commands: prepare public encoders, train locally, or reproduce a CSV."""
import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset, find_data_dir
from paper_priority.embeddings import ENCODERS, ensure_encoder, embedding_cache
from paper_priority.semantic_experiment import run_semantic, predict_semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'predict'):
        sub = commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--assets', type=Path, default=Path('assets/encoders'))
        sub.add_argument('--cache', type=Path, default=Path('.cache/embeddings'))
        if name == 'run':
            sub.add_argument('--reference', type=Path, default=Path('outputs/my_run01'))
            sub.add_argument('--out', type=Path, required=True)
        elif name == 'predict':
            sub.add_argument('--bundle', type=Path, required=True)
            sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        data = load_dataset(find_data_dir(args.data_dir))
        if args.command == 'prepare':
            for encoder in ENCODERS:
                ensure_encoder(encoder, args.assets, download=True)
                for rows in (data.train, data.test):
                    embedding_cache(rows, encoder, args.assets, args.cache)
        elif args.command == 'run':
            run_semantic(data, args.out, args.reference, args.assets, args.cache)
        else:
            print(json.dumps(predict_semantic(args.bundle, data, args.output, args.assets, args.cache), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
