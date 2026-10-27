"""Reproduce the frozen V6 candidate 2 from raw competition files.

No historical outputs, model bundle or cross-validation cache is required for train.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from paper_priority.data import find_data_dir, load_dataset, sha256, write_csv, write_json
from paper_priority.embeddings import ensure_encoder
from paper_priority.encoders_v4 import ensure_extra
from paper_priority.metadata_experiment import fit, predict, predict_saved
from paper_priority.models import rows_to_x
from paper_priority.refine_experiment import write_submission
from paper_priority.refine_models import aggregate
from paper_priority.semantic_experiment import versions
from paper_priority.upgrade_experiment import load_assets

ENCODERS = ('e5', 'bge', 'e5base')
SPEC = dict(weights={'bge_authors75': .25, 'bge_svc_c4': .28125,
                     'e5base_c4_g2': .1875, 'old_v3': .28125},
            aggregation='probability', family='bge')
CONFIG = dict(candidate='bge_authors75_mix25', fraction=.6, policy='utility_half')
REFERENCE_SHA256 = '31e9d52b555354168da504597eb671f97dee775533f2d09ae76d48c8f6e498aa'
INPUT_SHA256 = dict(train='d591c126e1a7554895a72749f04ea95be199526d1b05ebe42803783d8be4822a',
                    test='735631a93b61a10947716bd1ad9eb61bd4ec25eeb4f946adc6e6de5e80413cb8',
                    sample='88a1d0ff0b3b00b8a380fe87d1b353e6cdf2045ef06a571c1149e2ffb8e38816')


def prepare(assets):
    for name in ENCODERS:
        ensure = ensure_encoder if name == 'e5' else ensure_extra
        folder, _ = ensure(name, assets, download=True)
        print(f'Ready: {name} -> {folder}', flush=True)


def train(data, out, assets, cache, verify_reference):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty output directory; existing runs are preserved.')
    inputs_match = all(data.audit['files'][k]['sha256'] == digest for k, digest in INPUT_SHA256.items())
    if verify_reference and not inputs_match:
        raise ValueError('Reference input hashes differ. Use the original competition files.')
    start = time.perf_counter()
    # This is a fixed-configuration replay, not a new model-selection experiment.
    train_e, test_e, manifests, caches = load_assets(data, ENCODERS, assets, cache)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'reports/config.json', dict(**CONFIG, spec=SPEC, classifier_seed=261,
               input_files=data.audit['files'], versions=versions(),
               mode='Fixed V6 candidate2 replay; no CV or leaderboard-driven retuning.'))
    x, xt = rows_to_x(data.train), rows_to_x(data.test)
    y = np.asarray([r['Label'] for r in data.train], dtype=int)
    models, predictions = {}, {}
    with threadpool_limits(limits=1):
        for name in sorted(SPEC['weights']):
            print(f'Full-data fit: {name}', flush=True)
            models[name] = fit(name, train_e, x, y, np.arange(len(y)))
            predictions[name] = predict(name, models[name], test_e, xt, np.arange(len(xt)))
        components = aggregate(predictions, SPEC)
        check = write_submission(out/'submissions/v6_candidate_2.csv', data, components, CONFIG)
    bundle = dict(format_version=61, **CONFIG, spec=SPEC, models=models, versions=versions(),
                  encoders=manifests, train_sha256=data.audit['files']['train']['sha256'],
                  raw_test_sha256=data.audit['files']['test']['sha256'], test_ids=[r['id'] for r in data.test])
    model_path = out/'models/v6_candidate_2.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    write_csv(out/'predictions/v6_candidate_2_test.csv',
              [dict(id=r['id'], ranking_score=float(c[0]), probability_read=float(c[1]))
               for r, c in zip(data.test, components)], ['id', 'ranking_score', 'probability_read'])
    matches = check['sha256'] == REFERENCE_SHA256
    report = dict(**check, reference_sha256=REFERENCE_SHA256, reference_match=matches,
                  reference_inputs_match=inputs_match, runtime_seconds=round(time.perf_counter()-start, 3),
                  validation='Fixed-model replay only; does not calculate a new CV or Kaggle score.')
    write_json(out/'reports/reproduction.json', report)
    root = Path(__file__).resolve().parent
    sources = [root/'reproduce.py', root/'requirements.txt', root/'requirements-semantic.txt',
               *sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
               versions=versions(), input_files=data.audit['files'], caches=caches,
               code_sha256={str(p.relative_to(root)):sha256(p) for p in sources},
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(report, indent=2), flush=True)
    if verify_reference and not matches:
        raise ValueError('CSV differs from the reference; inspect the saved reproduction report and environment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'predict'):
        sub = commands.add_parser(name)
        sub.add_argument('--assets', type=Path, default=Path('assets/encoders'))
        if name != 'prepare':
            sub.add_argument('--data-dir')
            sub.add_argument('--cache', type=Path, default=Path('.cache/embeddings'))
        if name == 'train':
            sub.add_argument('--out', type=Path, required=True)
            sub.add_argument('--verify-reference', action='store_true', help='Require the original raw-data and CSV hashes.')
        elif name == 'predict':
            sub.add_argument('--bundle', type=Path, required=True)
            sub.add_argument('--output', type=Path, required=True)
            sub.add_argument('--refit', action='store_true', help='Refit all classifier heads independently before prediction.')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.assets)
        else:
            data = load_dataset(find_data_dir(args.data_dir))
            if args.command == 'train':
                train(data, args.out, args.assets, args.cache, args.verify_reference)
            else:
                print(json.dumps(predict_saved(args.bundle, data, args.output, args.assets, args.cache,
                                              refit=args.refit), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
