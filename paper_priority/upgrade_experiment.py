"""V6: large frozen encoders, paired seven-fold evaluation against fixed v5."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .advanced import policy_priorities
from .data import paper_groups, sha256, write_csv, write_json
from .embeddings import ensure_encoder
from .encoders_v4 import ensure_extra, extra_cache
from .experiment import assert_disjoint_groups, summarize
from .large_embeddings import LARGE_ENCODERS, ensure_large, large_cache
from .models import rows_to_x
from .refine_experiment import fit_model, predict_model, write_submission
from .refine_models import RefinedSVC, aggregate
from .semantic_models import SemanticHead
from .semantic_experiment import versions

BASE_WEIGHTS = {'bge_svc_c4': .375, 'e5base_c4_g2': .25, 'old_v3': .375}
BASE_SPEC = dict(weights=BASE_WEIGHTS, aggregation='probability')
HEADS = {f'{encoder}_{suffix}': dict(encoder=encoder, **head)
         for encoder in LARGE_ENCODERS
         for suffix, head in (
             ('svc_c4_g1', dict(kind='svc', C=4., gamma_factor=1., metadata=False)),
             ('svc_c4_g2', dict(kind='svc', C=4., gamma_factor=2., metadata=False)),
             ('ordinal_c4_meta', dict(kind='ordinal', C=4., metadata=True)))}
FRACTIONS = (.55, .575, .60, .625, .65)
POLICIES = ('gain', 'utility_half')
SEEDS = (361, 1361)


def candidates():
    result = {}
    for head, spec in HEADS.items():
        for q in (.25, .5, .75, 1.):
            weights = {head: q}
            weights.update({k: (1-q)*v for k, v in BASE_WEIGHTS.items() if q < 1})
            result[f'{head}_mix{int(q*100)}'] = dict(weights=weights, aggregation='probability', family=spec['encoder'])
    return result


def fit_head(name, embeddings, x, y, indices):
    if name not in HEADS:
        return fit_model(name, embeddings, x, y, indices)
    spec = HEADS[name]
    cls = RefinedSVC if spec['kind'] == 'svc' else SemanticHead
    return cls(spec, 261).fit(embeddings[spec['encoder']][indices], x[indices], y[indices])


def predict_head(name, model, embeddings, x, indices):
    if name not in HEADS:
        return predict_model(name, model, embeddings, x, indices)
    return model.predict_components(embeddings[HEADS[name]['encoder']][indices], x[indices])


def folds_for(x, y, groups, seed):
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    pairs = []
    for fold, (train, val) in enumerate(splitter.split(x, y, groups)):
        assert_disjoint_groups(groups, train, val)
        if np.any(folds[val] != -1):
            raise AssertionError('A row was validated more than once.')
        folds[val] = fold
        pairs.append((train, val))
    if np.any(folds < 0):
        raise AssertionError('Some rows have no validation prediction.')
    return folds, pairs


def cv(names, embeddings, x, y, groups, seed, path):
    folds, pairs = folds_for(x, y, groups, seed)
    result = {name: np.zeros((len(y), 2)) for name in names}
    for fold, (train, val) in enumerate(pairs):
        for name in names:
            print(f'V6 seed {seed}, fold {fold+1}/7: {name}', flush=True)
            model = fit_head(name, embeddings, x, y, train)
            result[name][val] = predict_head(name, model, embeddings, x, val)
            del model
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError('Nonfinite out-of-fold prediction.')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=np.arange(len(y)), fold=folds, **result)
    return result, folds


def evaluate(config, spec, predictions, ids, y, folds, target):
    from .metric import score_priorities
    metrics = []
    counts = []
    for fold in range(7):
        mask = (folds == fold) & target
        if not mask.any():
            raise ValueError('Empty evaluation fold.')
        components = aggregate({k: predictions[k][mask] for k in spec['weights']}, spec)
        priorities = policy_priorities(ids[mask], components, config['fraction'], config['policy'])
        metrics.append(score_priorities(ids[mask], y[mask], priorities))
        counts.append(int(mask.sum()))
    return dict(**config, **summarize(metrics), fold_scores=[m['score'] for m in metrics], fold_sizes=counts)


def paired_check(first, second, base1, base2):
    deltas = [first['score']-base1['score'], second['score']-base2['score']]
    wins = [int(np.sum(np.asarray(first['fold_scores']) > base1['fold_scores'])),
            int(np.sum(np.asarray(second['fold_scores']) > base2['fold_scores']))]
    return dict(deltas=deltas, paired_fold_wins=wins, mean_improvement=float(np.mean(deltas)),
                eligible=bool(min(deltas) > 0 and min(wins) >= 4 and np.mean(deltas) >= .0015))


def encoder_for(name):
    if name in HEADS:
        return HEADS[name]['encoder']
    if name == 'old_v3':
        return 'e5'
    return 'bge' if name.startswith('bge') else 'e5base'


def load_assets(data, names, assets, cache, training=True):
    train_e, test_e, manifests, caches = {}, {}, {}, {}
    for name in names:
        if name in LARGE_ENCODERS:
            _, manifests[name] = ensure_large(name, assets)
            loader = large_cache
        else:
            _, manifests[name] = ensure_encoder(name, assets) if name == 'e5' else ensure_extra(name, assets)
            loader = extra_cache
        if training:
            train_e[name], caches[name+'_train'] = loader(data.train, name, assets, cache)
        test_e[name], caches[name+'_test'] = loader(data.test, name, assets, cache)
    return train_e, test_e, manifests, caches


def run_upgrade(data, out, reference_v5, assets, cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output folder.')
    # The reference checks the baseline identity; its full-data model is never used in CV.
    reference = json.loads((reference_v5/'reports/results.json').read_text(encoding='utf-8'))
    selected = reference['winner']
    if not selected.get('final_eligible') or selected['fraction'] != .6 or selected['policy'] != 'gain':
        raise ValueError('Unexpected v5 reference policy.')
    reference_bundle = joblib.load(reference_v5/selected['model_file'])
    if reference_bundle['spec'] != dict(BASE_SPEC, family='e5base_c4_g2'):
        # Compare the actual predictive specification; the old family is a search label.
        if any(reference_bundle['spec'][k] != BASE_SPEC[k] for k in ('weights', 'aggregation')):
            raise ValueError('Unexpected v5 baseline components.')
    if reference_bundle['train_sha256'] != data.audit['files']['train']['sha256']:
        raise ValueError('V5 training data changed.')
    del reference_bundle
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    x, xt = rows_to_x(data.train), rows_to_x(data.test)
    y = np.asarray([r['Label'] for r in data.train], dtype=int)
    ids = np.asarray([r['id'] for r in data.train])
    groups = np.asarray(paper_groups(data.train))
    target = np.asarray([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    train_e, test_e, manifests, caches = load_assets(data, ['e5', 'bge', 'e5base', *LARGE_ENCODERS], assets, cache)
    specs = candidates()
    plan = dict(protocol='All 2494 training rows; seven-fold StratifiedGroupKFold. Fit all venues, score declared test venues only. Group normalized title or DOI; no group crosses folds.',
                seeds=SEEDS, heads=HEADS, candidates=specs, fractions=FRACTIONS, policies=POLICIES,
                baseline=dict(spec=BASE_SPEC, fraction=.6, policy='gain'), input_files=data.audit['files'],
                reference_manifest_sha256=sha256(reference_v5/'manifest.json'), encoders=manifests,
                rule='Phase1 seed361: select best config per candidate; shortlist one best candidate per large encoder (2 total). Freeze all settings. Phase2 seed1361: require positive paired mean improvement on both seeds, at least4/7 paired fold wins per seed, and two-seed mean delta >=.0015. At most one CSV, highest two-seed mean among passing candidates.',
                public_scores_used_for_selection=False,
                limitation='New CV protocol, so old CV values are not directly comparable. All labels were previously used; this is adaptive validation, not an untouched test. Both seeds reuse the same rows. No guarantee of Public or Private gain.')
    write_json(out/'reports/search_plan.json', plan)
    base_config = dict(candidate='v5', fraction=.6, policy='gain')
    with threadpool_limits(limits=1):
        first, folds1 = cv([*BASE_WEIGHTS, *HEADS], train_e, x, y, groups, SEEDS[0], out/'predictions/full_seed361.npz')
        base1 = evaluate(base_config, BASE_SPEC, first, ids, y, folds1, target)
        ranking, records = [], []
        for name, spec in specs.items():
            values = [evaluate(dict(candidate=name, fraction=f, policy=p), spec, first, ids, y, folds1, target)
                      for f in FRACTIONS for p in POLICIES]
            records.extend(values)
            ranking.append(max(values, key=lambda r: (r['score'], r['fraction'], r['policy'])))
        ranking.sort(key=lambda r: (-r['score'], r['candidate']))
        shortlist = []
        for encoder in LARGE_ENCODERS:
            shortlist.append(next(r for r in ranking if specs[r['candidate']]['family'] == encoder))
        write_json(out/'reports/frozen_shortlist.json', dict(baseline=base1, ranking=ranking, shortlist=shortlist,
                   frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        write_csv(out/'reports/search_results.csv', records,
                  ['candidate', 'fraction', 'policy', 'score', 'relevance', 'coverage', 'fold_score_std'])
        needed = sorted(set(BASE_WEIGHTS) | {k for r in shortlist for k in specs[r['candidate']]['weights']})
        second, folds2 = cv(needed, train_e, x, y, groups, SEEDS[1], out/'predictions/full_seed1361.npz')
        base2 = evaluate(base_config, BASE_SPEC, second, ids, y, folds2, target)
        confirmation = []
        for r in shortlist:
            config = {k:r[k] for k in ('candidate', 'fraction', 'policy')}
            m = evaluate(config, specs[r['candidate']], second, ids, y, folds2, target)
            confirmation.append(dict(**config, phase1=r, phase2=m, mean_score=(r['score']+m['score'])/2,
                                     **paired_check(r, m, base1, base2)))
        confirmation.sort(key=lambda r: (-r['mean_score'], r['candidate']))
        winner = next((r for r in confirmation if r['eligible']), None)
        write_json(out/'reports/final_selection.json', dict(baseline1=base1, baseline2=base2, confirmation=confirmation,
                   winner=winner, frozen_before_full_fit=True))
        if winner:
            spec = specs[winner['candidate']]
            models, predictions = {}, {}
            for name in sorted(spec['weights']):
                print(f'V6 full-data fit: {name}', flush=True)
                models[name] = fit_head(name, train_e, x, y, np.arange(len(y)))
                predictions[name] = predict_head(name, models[name], test_e, xt, np.arange(len(xt)))
            components = aggregate(predictions, spec)
            winner['submission_file'] = 'submissions/v6_candidate_1.csv'
            winner['model_file'] = 'models/v6_candidate_1.joblib'
            winner['submission_check'] = write_submission(out/winner['submission_file'], data, components, winner)
            encoders = sorted({encoder_for(k) for k in models})
            bundle = dict(format_version=6, candidate=winner['candidate'], spec=spec, fraction=winner['fraction'],
                          policy=winner['policy'], models=models, versions=versions(),
                          encoders={k:manifests[k] for k in encoders}, train_sha256=data.audit['files']['train']['sha256'],
                          raw_test_sha256=data.audit['files']['test']['sha256'], test_ids=[r['id'] for r in data.test])
            path = out/winner['model_file']
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path, compress=3)
            write_csv(out/'predictions/v6_candidate_1_test.csv',
                      [dict(id=r['id'], ranking_score=float(c[0]), probability_read=float(c[1])) for r,c in zip(data.test, components)],
                      ['id', 'ranking_score', 'probability_read'])
    result = dict(baseline1=base1, baseline2=base2, confirmation=confirmation, winner=winner,
                  runtime_seconds=round(time.perf_counter()-start, 3), untouched_holdout=False, limitation=plan['limitation'])
    write_json(out/'reports/results.json', result)
    root = Path(__file__).resolve().parents[1]
    files = [root/'upgrade.py', root/'requirements-semantic.txt', *sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(), versions=versions(),
               input_files=data.audit['files'], code_sha256={str(p.relative_to(root)):sha256(p) for p in files},
               caches=caches, output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {out}; new CSV: {winner is not None}', flush=True)
    return result


def predict_upgrade(bundle_path, data, output, assets, cache, refit=False):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle = joblib.load(bundle_path)
    if bundle['format_version'] != 6 or bundle['versions'] != versions():
        raise ValueError('Wrong model or dependency versions.')
    if (bundle['train_sha256'] != data.audit['files']['train']['sha256'] or
        bundle['raw_test_sha256'] != data.audit['files']['test']['sha256'] or bundle['test_ids'] != [r['id'] for r in data.test]):
        raise ValueError('Raw data changed.')
    train_e, test_e, manifests, _ = load_assets(data, sorted(bundle['encoders']), assets, cache, training=refit)
    if manifests != bundle['encoders']:
        raise ValueError('Encoder assets changed.')
    with threadpool_limits(limits=1):
        x, xt = rows_to_x(data.train), rows_to_x(data.test)
        y = np.asarray([r['Label'] for r in data.train], dtype=int)
        predictions = {}
        for name, saved_model in bundle['models'].items():
            model = fit_head(name, train_e, x, y, np.arange(len(y))) if refit else saved_model
            predictions[name] = predict_head(name, model, test_e, xt, np.arange(len(xt)))
        return write_submission(output, data, aggregate(predictions, bundle['spec']), bundle)
