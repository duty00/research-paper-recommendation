"""Analyze v1 development OOF mistakes; never reads holdout predictions."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from paper_priority.data import find_data_dir, load_dataset, write_json
from paper_priority.metric import make_priorities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--reference', type=Path, default=Path('outputs/my_run01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_dataset(find_data_dir(args.data_dir))
    if args.output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Analysis output must not overwrite original data.')
    by_id = {r['id']: r for r in data.train}
    with (args.reference / 'reports/development_oof.csv').open(encoding='utf-8-sig', newline='') as stream:
        oof = list(csv.DictReader(stream))
    rows = [by_id[r['id']] for r in oof]
    ids = np.asarray([r['id'] for r in oof])
    y = np.asarray([int(r['Label']) for r in oof])
    gain = np.asarray([float(r['logistic_c1']) for r in oof])
    folds = np.asarray([int(r['fold']) for r in oof])
    target = np.asarray([r['in_target_venues'] == 'True' for r in oof])
    chosen = np.zeros(len(oof), dtype=bool)
    for fold in sorted(set(folds)):
        mask = (folds == fold) & target
        chosen[mask] = make_priorities(ids[mask], gain[mask], .65) > 0

    def summarize(mask):
        return dict(n=int(np.sum(mask)), labels=dict(Counter(map(int, y[mask]))),
                    label1_selected=int(np.sum(mask & chosen & (y == 1))),
                    worthwhile_missed=int(np.sum(mask & ~chosen & (y > 1))),
                    rating4or5_missed=int(np.sum(mask & ~chosen & (y >= 4))))

    report = dict(scope='Development OOF only; no holdout error inspection',
                  overall=summarize(target),
                  by_venue={v: summarize(target & np.asarray([r['venue'] == v for r in rows]))
                            for v in sorted({r['venue'] for r in data.test})},
                  authors_available={str(flag): summarize(target & np.asarray([bool(r['authors']) == flag for r in rows]))
                                     for flag in (True, False)})
    for kind, indices in [('false_positives', np.flatnonzero(target & chosen & (y == 1))),
                          ('missed_high_ratings', np.flatnonzero(target & ~chosen & (y >= 4)))]:
        order = sorted(indices, key=lambda i: -gain[i] if kind == 'false_positives' else gain[i])[:12]
        report[kind] = [dict(id=ids[i], title=rows[i]['title'], venue=rows[i]['venue'],
                             Label=int(y[i]), predicted_gain=round(float(gain[i]), 4)) for i in order]
    write_json(args.output, report)
    print(f'Development analysis: {args.output.resolve()}')


if __name__ == '__main__':
    main()
