"""Reproducible v2 search using the original development partition only."""
from __future__ import annotations

import csv
import importlib.metadata
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .advanced import (MODEL_SPECS, CANDIDATE_WEIGHTS, POLICIES, FRACTIONS,
                       AdvancedModel, make_features, combine_components, policy_priorities)
from .data import paper_groups, sha256, write_csv, write_json
from .experiment import PACKAGES, assert_disjoint_groups, summarize, validate_submission
from .metric import score_priorities
from .models import rows_to_x

SEED = 261
SPLIT_SEEDS = (262, 1262)
MIN_IMPROVEMENT = .002


def original_partition(data, reference):
    with (reference / 'manifest.json').open(encoding='utf-8') as stream:
        manifest = __import__('json').load(stream)
    for role in ('train', 'test'):
        if manifest['input_files'][role]['sha256'] != data.audit['files'][role]['sha256']:
            raise ValueError('Reference run and current raw data hashes differ.')
    with (reference / 'reports/splits.csv').open(encoding='utf-8-sig', newline='') as stream:
        split_rows = list(csv.DictReader(stream))
    ids = [r['id'] for r in data.train]
    by_id = {r['id']: r for r in split_rows}
    if len(split_rows) != len(ids) or len(by_id) != len(ids) or set(by_id) != set(ids):
        raise ValueError('Reference split must cover every training ID exactly once.')
    groups = np.asarray(paper_groups(data.train))
    if any(str(groups[i]) != by_id[row_id]['group'] for i, row_id in enumerate(ids)):
        raise ValueError('Duplicate groups changed since v1.')
    if any(r['partition'] not in ('development', 'holdout') for r in split_rows):
        raise ValueError('Unknown partition in reference split.')
    dev = np.asarray([i for i, row_id in enumerate(ids) if by_id[row_id]['partition'] == 'development'])
    hold = np.asarray([i for i, row_id in enumerate(ids) if by_id[row_id]['partition'] == 'holdout'])
    if len(dev) == 0 or len(hold) == 0:
        raise ValueError('Both development and diagnostic holdout are required.')
    assert_disjoint_groups(groups, dev, hold)
    return dev, hold, groups, by_id


def cv_predictions(x, y, groups, names, split_seed, cache_path):
    predictions = {name: np.zeros((len(y), 2)) for name in names}
    fold_ids = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    for fold, (fit, val) in enumerate(splitter.split(x, y, groups)):
        assert_disjoint_groups(groups, fit, val)
        fold_ids[val] = fold
        cached = {}
        for name in names:
            spec = MODEL_SPECS[name]
            profile = spec['profile']
            if profile not in cached:
                features = make_features(profile)
                train_matrix = features.fit_transform(x[fit])
                cached[profile] = (features, train_matrix, features.transform(x[val]))
            features, train_matrix, val_matrix = cached[profile]
            print(f'Development seed {split_seed}, fold {fold+1}/5: {name}', flush=True)
            model = AdvancedModel(spec, SEED).fit_transformed(features, train_matrix, y[fit])
            predictions[name][val] = model.predict_components_transformed(val_matrix)
    if np.any(fold_ids < 0):
        raise AssertionError('Missing OOF predictions.')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, fold=fold_ids, **predictions)
    return predictions, fold_ids


def evaluate_config(config, components, ids, y, fold_ids, target):
    metrics = []
    for fold in range(5):
        mask = (fold_ids == fold) & target
        if not np.any(mask):
            raise ValueError('No target-venue rows in a development fold.')
        priorities = policy_priorities(ids[mask], components[mask], config['fraction'], config['policy'])
        metrics.append(score_priorities(ids[mask], y[mask], priorities))
    return {**config, **summarize(metrics), 'fold_scores': [m['score'] for m in metrics]}


def fixed_baseline():
    return dict(candidate='v1_logistic', policy='gain', fraction=.65)


def write_v2_submission(path, data, components, config):
    ids = np.asarray([r['id'] for r in data.test])
    priorities = policy_priorities(ids, components, config['fraction'], config['policy'])
    by_id = dict(zip(ids, priorities))
    write_csv(path, [{'id': r['id'], 'priority': format(by_id[r['id']], '.12g')} for r in data.sample],
              ['id', 'priority'])
    return validate_submission(path, list(ids))


def run_improvement(data, out, reference):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output folder is not empty: {out}')
    dev, hold, groups, reference_splits = original_partition(data, reference)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    x, xt = rows_to_x(data.train), rows_to_x(data.test)
    y = np.asarray([r['Label'] for r in data.train], dtype=int)
    ids = np.asarray([r['id'] for r in data.train])
    venues = sorted({r['venue'] for r in data.test})
    target = np.asarray([r['venue'] in venues for r in data.train])
    plan = dict(model_specs=MODEL_SPECS, candidate_weights=CANDIDATE_WEIGHTS,
                policies=POLICIES, fractions=FRACTIONS, split_seeds=SPLIT_SEEDS,
                minimum_mean_improvement=MIN_IMPROVEMENT,
                selection_rule='Best configuration per model on seed 262; top 3 distinct models confirmed with seed 1262 at frozen settings. Mean improvement >= .002 and positive confirmation delta. At most 2 models.',
                reference_manifest_sha256=sha256(reference / 'manifest.json'),
                reference_splits_sha256=sha256(reference / 'reports/splits.csv'),
                input_files=data.audit['files'], primary_venues=venues,
                holdout_status='Already inspected in v1; diagnostic only, not a fresh unbiased test.',
                public_scores_used_for_selection=False)
    write_json(out / 'reports/search_plan.json', plan)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        first, folds1 = cv_predictions(x[dev], y[dev], groups[dev], list(MODEL_SPECS),
                                      SPLIT_SEEDS[0], out / 'predictions/development_seed262.npz')
        # Verify this is exactly the previous development CV partition.
        if any(int(reference_splits[ids[i]]['cv_fold']) != int(folds1[j]) for j, i in enumerate(dev)):
            raise ValueError('Seed 262 does not reproduce the reference CV folds.')
        all_results, best = [], []
        for name, weights in CANDIDATE_WEIGHTS.items():
            components = combine_components(first, weights)
            records = [evaluate_config(dict(candidate=name, policy=policy, fraction=fraction),
                       components, ids[dev], y[dev], folds1, target[dev])
                       for policy in POLICIES for fraction in FRACTIONS]
            all_results.extend(records)
            best.append(max(records, key=lambda r: (r['score'], r['fraction'], r['policy'])))
        ranking = sorted(best, key=lambda r: (-r['score'], r['candidate']))
        baseline1 = evaluate_config(fixed_baseline(), first['v1_logistic'], ids[dev], y[dev], folds1, target[dev])
        shortlist = [r for r in ranking if r['candidate'] != 'v1_logistic'][:3]
        write_json(out / 'reports/phase1.json', dict(baseline=baseline1, ranking=ranking, shortlist=shortlist))
        write_csv(out / 'reports/search_results.csv',
                  [{k: v for k, v in r.items() if k != 'fold_scores'} for r in all_results],
                  ['candidate', 'policy', 'fraction', 'score', 'relevance', 'coverage', 'fold_score_std'])
        needed = sorted({'v1_logistic'} | {base for r in shortlist for base in CANDIDATE_WEIGHTS[r['candidate']]})
        print('Phase 1 frozen. Confirmation models: ' + ', '.join(needed), flush=True)
        second, folds2 = cv_predictions(x[dev], y[dev], groups[dev], needed,
                                       SPLIT_SEEDS[1], out / 'predictions/development_seed1262.npz')
        baseline2 = evaluate_config(fixed_baseline(), second['v1_logistic'], ids[dev], y[dev], folds2, target[dev])
        confirmed = []
        for record in shortlist:
            config = {key: record[key] for key in ('candidate', 'policy', 'fraction')}
            confirmation = evaluate_config(config, combine_components(second, CANDIDATE_WEIGHTS[record['candidate']]),
                                          ids[dev], y[dev], folds2, target[dev])
            combined = (record['score'] + confirmation['score']) / 2
            delta = combined - (baseline1['score'] + baseline2['score']) / 2
            confirmed.append(dict(**config, phase1=record, phase2=confirmation,
                                  mean_score=combined, mean_improvement=delta,
                                  eligible=bool(delta >= MIN_IMPROVEMENT and confirmation['score'] > baseline2['score'])))
        confirmed.sort(key=lambda r: (-r['mean_score'], r['candidate']))
        selected = [r for r in confirmed if r['eligible']][:2]
        frozen = dict(baseline_phase1=baseline1, baseline_phase2=baseline2, confirmation=confirmed,
                      selected=selected, frozen_before_holdout=True,
                      frozen_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / 'reports/selection.json', frozen)
        final_needed = sorted({base for r in selected for base in CANDIDATE_WEIGHTS[r['candidate']]})
        hold_predictions, full_models, test_predictions = {}, {}, {}
        for name in final_needed:
            print(f'Diagnostic holdout (selection frozen): {name}', flush=True)
            model = AdvancedModel(MODEL_SPECS[name], SEED).fit(x[dev], y[dev])
            hold_predictions[name] = model.predict_components(x[hold])
            print(f'Full-data refit: {name}', flush=True)
            full_models[name] = AdvancedModel(MODEL_SPECS[name], SEED).fit(x, y)
            test_predictions[name] = full_models[name].predict_components(xt)
        versions = {p: importlib.metadata.version(p) for p in PACKAGES}
        for number, record in enumerate(selected, 1):
            weights = CANDIDATE_WEIGHTS[record['candidate']]
            components = combine_components(hold_predictions, weights)
            mask = target[hold]
            record['diagnostic_holdout'] = score_priorities(ids[hold][mask], y[hold][mask],
                policy_priorities(ids[hold][mask], components[mask], record['fraction'], record['policy']))
            components = combine_components(test_predictions, weights)
            bundle = dict(format_version=2, candidate=record['candidate'], fraction=record['fraction'],
                          policy=record['policy'], weights=weights, models={k: full_models[k] for k in weights},
                          versions=versions, seed=SEED, train_sha256=data.audit['files']['train']['sha256'],
                          raw_test_sha256=data.audit['files']['test']['sha256'], test_ids=[r['id'] for r in data.test])
            bundle_path = out / f'models/v2_candidate_{number}.joblib'
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, bundle_path, compress=3)
            path = out / f'submissions/v2_candidate_{number}.csv'
            record['submission_check'] = write_v2_submission(path, data, components, record)
            record['submission_file'] = str(path.relative_to(out))
            record['model_file'] = str(bundle_path.relative_to(out))
            write_csv(out / f'predictions/v2_candidate_{number}_test.csv',
                      [dict(id=r['id'], expected_gain=float(c[0]), probability_read=float(c[1]))
                       for r, c in zip(data.test, components)], ['id', 'expected_gain', 'probability_read'])
    results = dict(**frozen, runtime_seconds=round(time.perf_counter() - start, 3),
                   limitations=[
                       'Both CV seeds use the same development rows: this is a stability check, not independent test evidence.',
                       'Selection across 576 configurations can overestimate performance; confirmation uses only 3 frozen configurations.',
                       'The old holdout was reported in v1 and is diagnostic only. It never changes selection or parameters.',
                       'Primary evaluation uses visible test venue names; all training venues remain in model fitting.',
                       'Detailed Evaluation metric implemented; competition abstract conflicts and hidden scorer is unverified.',
                       'No Public score enters fitting or selection. No external data, pretrained model or Kaggle API calls.'])
    write_json(out / 'reports/results.json', results)
    summary = ['# V2: development validation', '',
               'Two CV seeds on the same development rows. No new independent test set.', '',
               '| Candidate | Policy | Fraction | CV seed 262 | CV seed 1262 | Mean delta vs v1 | Eligible |',
               '|---|---|---:|---:|---:|---:|---|',
               f"| v1 logistic | gain | .65 | {baseline1['score']:.6f} | {baseline2['score']:.6f} | 0 | baseline |"]
    for r in confirmed:
        summary.append(f"| {r['candidate']} | {r['policy']} | {r['fraction']:.2f} | {r['phase1']['score']:.6f} | {r['phase2']['score']:.6f} | {r['mean_improvement']:+.6f} | {r['eligible']} |")
    summary += ['', '## Frozen candidates', '']
    for n, r in enumerate(selected, 1):
        summary.append(f"- v2_candidate_{n}.csv: {r['candidate']}; diagnostic holdout {r['diagnostic_holdout']['score']:.6f}; {r['submission_check']['recommended']}/596 recommended.")
    if not selected:
        summary.append('No candidate passed the predefined improvement gate. No submission files produced.')
    summary += ['', '## Limits', ''] + ['- ' + s for s in results['limitations']]
    (out / 'reports/summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    source_files = [root / 'improve.py', root / 'requirements.txt', *sorted((root / 'paper_priority').glob('*.py'))]
    write_json(out / 'manifest.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
               python=platform.python_version(), platform=platform.platform(), packages=versions,
               input_files=data.audit['files'], seed=SEED, split_seeds=SPLIT_SEEDS, threads=1,
               code_sha256={str(p.relative_to(root)): sha256(p) for p in source_files},
               outputs_sha256={str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done. {len(selected)} new candidates: {out.resolve()}', flush=True)
    return results


def predict_improvement(bundle_path, data, output):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Output must not overwrite original data.')
    bundle = joblib.load(bundle_path)
    if bundle['format_version'] != 2:
        raise ValueError('Expected a version 2 model bundle.')
    if bundle['versions'] != {p: importlib.metadata.version(p) for p in PACKAGES}:
        raise ValueError('Install the exact requirements.txt versions first.')
    if bundle['train_sha256'] != data.audit['files']['train']['sha256'] or bundle['raw_test_sha256'] != data.audit['files']['test']['sha256']:
        raise ValueError('Raw data hashes differ from training.')
    if bundle['test_ids'] != [r['id'] for r in data.test]:
        raise ValueError('Test IDs/order changed.')
    with threadpool_limits(limits=1):
        x = rows_to_x(data.test)
        components = combine_components({k: m.predict_components(x) for k, m in bundle['models'].items()}, bundle['weights'])
    return write_v2_submission(output, data, components, bundle)
