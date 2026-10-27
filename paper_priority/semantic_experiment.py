"""V3: frozen semantic encoders plus group-safe supervised heads and fixed blends."""
from __future__ import annotations
import importlib.metadata
import json
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
from .advanced import AdvancedModel, combine_components, policy_priorities
from .data import sha256, write_csv, write_json
from .embeddings import ENCODERS, embedding_cache, ensure_encoder
from .experiment import assert_disjoint_groups, PACKAGES
from .improvement import original_partition, evaluate_config, write_v2_submission
from .metric import score_priorities
from .models import rows_to_x
from .semantic_models import SemanticHead, SEMANTIC_SPECS, BASELINES, WEIGHTS

SEEDS = (262, 1262)
MODEL_SEED = 261
FRACTIONS = tuple(round(v / 100, 2) for v in range(50, 101, 5))
POLICIES = ('gain', 'utility_half')
EXTRA_PACKAGES = ('onnxruntime', 'tokenizers')


def versions():
    return {p: importlib.metadata.version(p) for p in (*PACKAGES, *EXTRA_PACKAGES)}


def train_head(name, embeddings, x, y, train):
    if name in BASELINES:
        return AdvancedModel(BASELINES[name], MODEL_SEED).fit(x[train], y[train])
    spec = SEMANTIC_SPECS[name]
    return SemanticHead(spec, MODEL_SEED).fit(embeddings[spec['encoder']][train], x[train], y[train])


def predict_head(name, model, embeddings, x, indices):
    if name in BASELINES:
        return model.predict_components(x[indices])
    return model.predict_components(embeddings[SEMANTIC_SPECS[name]['encoder']][indices], x[indices])


def cross_validate(names, embeddings, x, y, groups, dev, seed, save_path):
    output = {n: np.zeros((len(dev), 2)) for n in names}
    fold_ids = np.full(len(dev), -1, dtype=int)
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit, val) in enumerate(splits.split(x[dev], y[dev], groups[dev])):
        train, validation = dev[fit], dev[val]
        assert_disjoint_groups(groups, train, validation)
        fold_ids[val] = fold
        for name in names:
            print(f'V3 development seed {seed}, fold {fold+1}/5: {name}', flush=True)
            model = train_head(name, embeddings, x, y, train)
            output[name][val] = predict_head(name, model, embeddings, x, validation)
    if np.any(fold_ids < 0):
        raise AssertionError('Missing OOF rows.')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, ids=dev, fold=fold_ids, **output)
    return output, fold_ids


def fixed_baseline():
    return dict(candidate='v2_title', fraction=.65, policy='gain')


def run_semantic(data, out, reference, assets, cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Use a new empty output folder: {out}')
    dev, holdout, groups, old_splits = original_partition(data, reference)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train_e, test_e, cache_reports, asset_reports = {}, {}, {}, {}
    for name in ENCODERS:
        _, asset_reports[name] = ensure_encoder(name, assets)
        train_e[name], report_train = embedding_cache(data.train, name, assets, cache)
        test_e[name], report_test = embedding_cache(data.test, name, assets, cache)
        cache_reports[name] = dict(train=report_train, test=report_test)
    x, xt = rows_to_x(data.train), rows_to_x(data.test)
    y = np.asarray([r['Label'] for r in data.train], dtype=int)
    ids = np.asarray([r['id'] for r in data.train])
    venues = sorted({r['venue'] for r in data.test})
    target = np.asarray([r['venue'] in venues for r in data.train])
    plan = dict(encoders=ENCODERS, semantic_specs=SEMANTIC_SPECS, baselines=BASELINES,
                weights=WEIGHTS, fractions=FRACTIONS, policies=POLICIES, seeds=SEEDS,
                reference_manifest_sha256=sha256(reference / 'manifest.json'),
                reference_splits_sha256=sha256(reference / 'reports/splits.csv'),
                primary_venues=venues, input_files=data.audit['files'],
                rule='Top 3 distinct semantic candidate definitions on seed262, at frozen fraction/policy; confirmation seed1262. Both seeds must beat v2_title top65; mean delta >= .003. At most two, second OOF gain correlation < .995 with first.',
                selection_uses_public_scores=False,
                caveat='Same development rows across seeds; old holdout diagnostic only, no fresh independent test.',
                assets=asset_reports, embedding_caches=cache_reports)
    write_json(out / 'reports/search_plan.json', plan)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        first, folds1 = cross_validate([*BASELINES, *SEMANTIC_SPECS], train_e, x, y, groups,
                                       dev, SEEDS[0], out / 'predictions/development_seed262.npz')
        if any(int(old_splits[ids[i]]['cv_fold']) != int(folds1[j]) for j, i in enumerate(dev)):
            raise ValueError('Original development folds did not reproduce.')
        all_results, ranking = [], []
        for name, weights in WEIGHTS.items():
            components = combine_components(first, weights)
            records = [evaluate_config(dict(candidate=name, fraction=f, policy=p), components,
                                       ids[dev], y[dev], folds1, target[dev])
                       for f in FRACTIONS for p in POLICIES]
            all_results.extend(records)
            ranking.append(max(records, key=lambda r: (r['score'], r['fraction'], r['policy'])))
        ranking.sort(key=lambda r: (-r['score'], r['candidate']))
        baseline1 = evaluate_config(fixed_baseline(), first['v2_title'], ids[dev], y[dev], folds1, target[dev])
        shortlist = [r for r in ranking if any(k in SEMANTIC_SPECS for k in WEIGHTS[r['candidate']])][:3]
        write_json(out / 'reports/phase1.json', dict(baseline=baseline1, ranking=ranking, shortlist=shortlist))
        write_csv(out / 'reports/search_results.csv',
                  [{k:v for k,v in r.items() if k != 'fold_scores'} for r in all_results],
                  ['candidate', 'fraction', 'policy', 'score', 'relevance', 'coverage', 'fold_score_std'])
        names = sorted({'v2_title'} | {k for r in shortlist for k in WEIGHTS[r['candidate']]})
        print('Phase 1 frozen; checking three configurations on seed1262.', flush=True)
        second, folds2 = cross_validate(names, train_e, x, y, groups, dev, SEEDS[1],
                                        out / 'predictions/development_seed1262.npz')
        baseline2 = evaluate_config(fixed_baseline(), second['v2_title'], ids[dev], y[dev], folds2, target[dev])
        confirmed = []
        for record in shortlist:
            config = {k:record[k] for k in ('candidate', 'fraction', 'policy')}
            confirmation = evaluate_config(config, combine_components(second, WEIGHTS[record['candidate']]),
                                          ids[dev], y[dev], folds2, target[dev])
            mean = (record['score'] + confirmation['score']) / 2
            delta = mean - (baseline1['score'] + baseline2['score']) / 2
            wins = [int(np.sum(np.asarray(record['fold_scores']) > baseline1['fold_scores'])),
                    int(np.sum(np.asarray(confirmation['fold_scores']) > baseline2['fold_scores']))]
            eligible = (delta >= .003 and record['score'] > baseline1['score'] and confirmation['score'] > baseline2['score'])
            confirmed.append(dict(**config, phase1=record, phase2=confirmation, mean_score=mean,
                                  mean_improvement=delta, paired_fold_wins=wins, eligible=bool(eligible)))
        confirmed.sort(key=lambda r: (-r['mean_score'], r['candidate']))
        selected = []
        for record in confirmed:
            if not record['eligible']:
                continue
            if selected:
                a = combine_components(first, WEIGHTS[selected[0]['candidate']])[target[dev], 0]
                b = combine_components(first, WEIGHTS[record['candidate']])[target[dev], 0]
                correlation = float(np.corrcoef(a, b)[0, 1])
                record['gain_correlation_with_first'] = correlation
                if correlation >= .995:
                    continue
            selected.append(record)
            if len(selected) == 2:
                break
        frozen = dict(baseline_phase1=baseline1, baseline_phase2=baseline2,
                      confirmation=confirmed, selected=selected,
                      frozen_at_utc=datetime.now(timezone.utc).isoformat(), frozen_before_holdout=True)
        write_json(out / 'reports/selection.json', frozen)
        needed = sorted({k for r in selected for k in WEIGHTS[r['candidate']]})
        models, hold_predictions, test_predictions = {}, {}, {}
        for name in needed:
            print(f'Diagnostic old holdout: {name}', flush=True)
            model = train_head(name, train_e, x, y, dev)
            hold_predictions[name] = predict_head(name, model, train_e, x, holdout)
            print(f'Full-data refit: {name}', flush=True)
            models[name] = train_head(name, train_e, x, y, np.arange(len(y)))
            test_predictions[name] = predict_head(name, models[name], test_e, xt, np.arange(len(xt)))
        for number, record in enumerate(selected, 1):
            weights = WEIGHTS[record['candidate']]
            hold_c = combine_components(hold_predictions, weights)
            mask = target[holdout]
            record['diagnostic_holdout'] = score_priorities(ids[holdout][mask], y[holdout][mask],
                policy_priorities(ids[holdout][mask], hold_c[mask], record['fraction'], record['policy']))
            components = combine_components(test_predictions, weights)
            record['submission_check'] = write_v2_submission(out / f'submissions/v3_candidate_{number}.csv', data, components, record)
            record['submission_file'] = f'submissions/v3_candidate_{number}.csv'
            record['model_file'] = f'models/v3_candidate_{number}.joblib'
            bundle = dict(format_version=3, candidate=record['candidate'], policy=record['policy'],
                          fraction=record['fraction'], weights=weights,
                          models={k:models[k] for k in weights}, versions=versions(),
                          encoders={SEMANTIC_SPECS[k]['encoder']:asset_reports[SEMANTIC_SPECS[k]['encoder']]
                                    for k in weights if k in SEMANTIC_SPECS},
                          train_sha256=data.audit['files']['train']['sha256'],
                          raw_test_sha256=data.audit['files']['test']['sha256'],
                          test_ids=[r['id'] for r in data.test], model_seed=MODEL_SEED)
            path = out / record['model_file']
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path, compress=3)
            write_csv(out / f'predictions/v3_candidate_{number}_test.csv',
                      [dict(id=r['id'], expected_gain=float(c[0]), probability_read=float(c[1]))
                       for r,c in zip(data.test, components)], ['id', 'expected_gain', 'probability_read'])
    limitations = [
        'Search and confirmation reuse the same development rows; selection scores can be optimistic.',
        'Old holdout has already been viewed during earlier versions; diagnostic, not a fresh unbiased test.',
        'Public/Private labels and membership are never inferred; no Kaggle API or automated upload.',
        'Frozen pretrained models are external resources; their upstream corpora may contain public research text, unknown overlap.',
        'MiniLM is FP32; E5 uses the publisher INT8 ONNX export. Quantization and truncation may affect features.',
        'Test title encoding is frozen inference; no vocabulary/centering/classifier is fitted to test.',
        'The detailed Evaluation metric is implemented; the Overview abstract conflict and hidden scorer remain unverified.',
    ]
    results = dict(**frozen, runtime_seconds=round(time.perf_counter()-start, 3), limitations=limitations)
    write_json(out / 'reports/results.json', results)
    lines = ['# V3 semantic experiment', '',
             'Both CV seeds use the same development rows. Holdout is an old diagnostic set.', '',
             '| Candidate | Policy | Fraction | Seed262 | Seed1262 | Mean CV | Mean delta vs v2 | Eligible |',
             '|---|---|---:|---:|---:|---:|---:|---|',
             f"| v2_title | gain | .65 | {baseline1['score']:.6f} | {baseline2['score']:.6f} | {(baseline1['score']+baseline2['score'])/2:.6f} | 0 | baseline |"]
    for r in confirmed:
        lines.append(f"| {r['candidate']} | {r['policy']} | {r['fraction']:.2f} | {r['phase1']['score']:.6f} | {r['phase2']['score']:.6f} | {r['mean_score']:.6f} | {r['mean_improvement']:+.6f} | {r['eligible']} |")
    lines += ['', '## Selected before diagnostic holdout', '']
    for n,r in enumerate(selected,1):
        lines.append(f"- v3_candidate_{n}.csv: {r['candidate']}; old holdout {r['diagnostic_holdout']['score']:.6f}; {r['submission_check']['recommended']}/596 recommended.")
    if not selected:
        lines.append('No candidate passed the predefined gate; no CSV produced.')
    lines += ['', '## Limits', ''] + ['- '+s for s in limitations]
    (out / 'reports/summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    code = [root/'semantic.py', root/'requirements-semantic.txt', *sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
               python=platform.python_version(), platform=platform.platform(), packages=versions(),
               model_seed=MODEL_SEED, cv_seeds=SEEDS, input_files=data.audit['files'],
               code_sha256={str(p.relative_to(root)):sha256(p) for p in code},
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {len(selected)} semantic candidates in {out.resolve()}', flush=True)
    return results


def predict_semantic(bundle_path, data, output, assets, cache):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Output must not overwrite original data.')
    bundle = joblib.load(bundle_path)
    if bundle['format_version'] != 3 or bundle['versions'] != versions():
        raise ValueError('Expected v3 model and the exact semantic dependency versions.')
    if bundle['train_sha256'] != data.audit['files']['train']['sha256'] or bundle['raw_test_sha256'] != data.audit['files']['test']['sha256']:
        raise ValueError('Raw data changed.')
    if bundle['test_ids'] != [r['id'] for r in data.test]:
        raise ValueError('Test IDs/order changed.')
    test_e = {}
    for name, recorded in bundle['encoders'].items():
        _, current = ensure_encoder(name, assets)
        if current != recorded:
            raise ValueError('Encoder assets/manifest changed.')
        test_e[name], _ = embedding_cache(data.test, name, assets, cache)
    with threadpool_limits(limits=1):
        x = rows_to_x(data.test)
        predictions = {name:predict_head(name, model, test_e, x, np.arange(len(x))) for name,model in bundle['models'].items()}
        return write_v2_submission(output, data, combine_components(predictions, bundle['weights']), bundle)
