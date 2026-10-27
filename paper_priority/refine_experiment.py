"""V5: fixed model/ensemble grid, frozen confirmation and an extra split check."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .advanced import policy_priorities
from .data import sha256,write_json,write_csv
from .encoders_v4 import extra_cache,ensure_extra
from .embeddings import ensure_encoder
from .experiment import assert_disjoint_groups,summarize,validate_submission
from .fusion_experiment import fit as old_fit,predict as old_predict
from .improvement import original_partition
from .metric import score_priorities
from .models import rows_to_x
from .refine_models import NEW_SPECS,RefinedSVC,candidate_specs,aggregate
from .semantic_experiment import versions

CANDIDATES=candidate_specs()
FRACTIONS=(.50,.525,.55,.575,.60,.625,.65,.675,.70,.725,.75)
POLICIES=('gain','utility_half')
SEEDS=(262,1262,2262)
BASELINE_SPEC=dict(weights={'bge_svc_c4':.5,'old_v3':.5},aggregation='probability',family='v4')


def fit_model(name,embeddings,x,y,indices):
    if name in NEW_SPECS:
        spec=NEW_SPECS[name]
        return RefinedSVC(spec,261).fit(embeddings[spec['encoder']][indices],x[indices],y[indices])
    return old_fit(name,embeddings,x,y,indices)


def predict_model(name,model,embeddings,x,indices):
    if name in NEW_SPECS:
        return model.predict_components(embeddings[NEW_SPECS[name]['encoder']][indices],x[indices])
    return old_predict(name,model,embeddings,x,indices)


def cached_v4(out,seed,dev,data):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if any(manifest['input_files'][k]['sha256']!=data.audit['files'][k]['sha256'] for k in ('train','test')):
        raise ValueError('V4 input hashes differ.')
    hashes={k.replace('\\','/'):v for k,v in manifest['output_sha256'].items()}
    relative=f'predictions/development_seed{seed}.npz'
    if sha256(out/relative)!=hashes[relative]:
        raise ValueError('Cached v4 OOF changed.')
    with np.load(out/relative,allow_pickle=False) as saved:
        if not np.array_equal(saved['indices'],dev):
            raise ValueError('V4 development order changed.')
        return {k:saved[k] for k in ('bge_svc_c4','e5base_svc_c4','old_v3')},saved['fold']


def cross_validate(names,embeddings,x,y,groups,dev,seed,path,initial=None,expected=None):
    predictions={} if initial is None else dict(initial)
    names=[n for n in names if n not in predictions]
    predictions.update({n:np.zeros((len(dev),2)) for n in names})
    folds=np.full(len(dev),-1,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for fold,(train_local,val_local) in enumerate(splitter.split(x[dev],y[dev],groups[dev])):
        train,val=dev[train_local],dev[val_local]
        assert_disjoint_groups(groups,train,val)
        folds[val_local]=fold
        for name in names:
            print(f'V5 seed {seed}, fold {fold+1}/5: {name}',flush=True)
            model=fit_model(name,embeddings,x,y,train)
            predictions[name][val_local]=predict_model(name,model,embeddings,x,val)
    if expected is not None and not np.array_equal(folds,expected):
        raise ValueError('Cached and current folds differ.')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=dev,fold=folds,**predictions)
    return predictions,folds


def evaluate(config,spec,predictions,ids,y,folds,target):
    metrics=[]
    for fold in range(5):
        mask=(folds==fold)&target
        # Ranking aggregation must be computed on the actual evaluated list.
        components=aggregate({k:predictions[k][mask] for k in spec['weights']},spec)
        priorities=policy_priorities(ids[mask],components,config['fraction'],config['policy'])
        metrics.append(score_priorities(ids[mask],y[mask],priorities))
    return dict(**config,**summarize(metrics),fold_scores=[m['score'] for m in metrics])


def write_submission(path,data,components,config):
    ids=[r['id'] for r in data.test]
    p=policy_priorities(ids,components,config['fraction'],config['policy'])
    mapping=dict(zip(ids,p))
    write_csv(path,[dict(id=r['id'],priority=format(mapping[r['id']],'.12g')) for r in data.sample],['id','priority'])
    return validate_submission(path,ids)


def run_refinement(data,out,reference,reference_v4,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output folder.')
    dev,holdout,groups,_=original_partition(data,reference)
    initial1,expected1=cached_v4(reference_v4,262,dev,data)
    initial2,expected2=cached_v4(reference_v4,1262,dev,data)
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    ids=np.asarray([r['id'] for r in data.train])
    target=np.asarray([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    train_e,test_e,asset_reports={},{},{}
    for name in ('e5','bge','e5base'):
        _,asset_reports[name]=ensure_encoder(name,assets) if name=='e5' else ensure_extra(name,assets)
        train_e[name],_=extra_cache(data.train,name,assets,cache)
        test_e[name],_=extra_cache(data.test,name,assets,cache)
    plan=dict(model_specs=NEW_SPECS,candidates=CANDIDATES,fractions=FRACTIONS,policies=POLICIES,seeds=SEEDS,
              assets=asset_reports,input_files=data.audit['files'],reference_v4_manifest_sha256=sha256(reference_v4/'manifest.json'),
              selection_rule='Phase1: best config per candidate, top3 distinct families. Freeze settings; confirm seed1262. Require both positive deltas, >=3/5 paired fold wins each, mean delta>=.001. Choose one winner by two-seed mean. Freeze winner; stress-check seed2262 without retuning; require third delta>=-.0005 and three-seed mean delta>=.0015. At most one CSV. No holdout inspection.',
              public_scores_used_for_selection=False,limitation='All CV seeds reuse the same development rows; adaptive selection can remain optimistic.')
    write_json(out/'reports/search_plan.json',plan)
    baseline_config=dict(candidate='v4',fraction=.6,policy='gain')
    with threadpool_limits(limits=1):
        first,folds1=cross_validate(list(NEW_SPECS),train_e,x,y,groups,dev,262,
                                   out/'predictions/development_seed262.npz',initial1,expected1)
        base1=evaluate(baseline_config,BASELINE_SPEC,first,ids[dev],y[dev],folds1,target[dev])
        all_records,ranking=[],[]
        for name,spec in CANDIDATES.items():
            records=[evaluate(dict(candidate=name,fraction=f,policy=p),spec,first,ids[dev],y[dev],folds1,target[dev])
                     for f in FRACTIONS for p in POLICIES]
            all_records.extend(records)
            ranking.append(max(records,key=lambda r:(r['score'],r['fraction'],r['policy'])))
        ranking.sort(key=lambda r:(-r['score'],r['candidate']))
        shortlist=[]
        families=set()
        for record in ranking:
            family=CANDIDATES[record['candidate']]['family']
            if family not in families:
                families.add(family)
                shortlist.append(record)
            if len(shortlist)==3:
                break
        write_json(out/'reports/phase1.json',dict(baseline=base1,ranking=ranking,shortlist=shortlist))
        write_csv(out/'reports/search_results.csv',[{k:v for k,v in r.items() if k!='fold_scores'} for r in all_records],
                  ['candidate','fraction','policy','score','relevance','coverage','fold_score_std'])
        needed=sorted({k for r in shortlist for k in CANDIDATES[r['candidate']]['weights']})
        second,folds2=cross_validate(needed,train_e,x,y,groups,dev,1262,
                                    out/'predictions/development_seed1262.npz',initial2,expected2)
        base2=evaluate(baseline_config,BASELINE_SPEC,second,ids[dev],y[dev],folds2,target[dev])
        confirmation=[]
        for r in shortlist:
            config={k:r[k] for k in ('candidate','fraction','policy')}
            m=evaluate(config,CANDIDATES[r['candidate']],second,ids[dev],y[dev],folds2,target[dev])
            deltas=[r['score']-base1['score'],m['score']-base2['score']]
            wins=[int(np.sum(np.asarray(r['fold_scores'])>base1['fold_scores'])),int(np.sum(np.asarray(m['fold_scores'])>base2['fold_scores']))]
            eligible=bool(min(deltas)>0 and min(wins)>=3 and np.mean(deltas)>=.001)
            confirmation.append(dict(**config,phase1=r,phase2=m,mean_score=(r['score']+m['score'])/2,
                                     mean_improvement=float(np.mean(deltas)),paired_fold_wins=wins,eligible=eligible))
        confirmation.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        eligible=[r for r in confirmation if r['eligible']]
        winner=eligible[0] if eligible else None
        write_json(out/'reports/selection_before_third_seed.json',dict(baseline1=base1,baseline2=base2,
                   confirmation=confirmation,winner=winner,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        base3=None
        if winner:
            spec=CANDIDATES[winner['candidate']]
            needed=sorted(set(spec['weights'])|set(BASELINE_SPEC['weights']))
            third,folds3=cross_validate(needed,train_e,x,y,groups,dev,2262,out/'predictions/development_seed2262.npz')
            base3=evaluate(baseline_config,BASELINE_SPEC,third,ids[dev],y[dev],folds3,target[dev])
            config={k:winner[k] for k in ('candidate','fraction','policy')}
            m=evaluate(config,spec,third,ids[dev],y[dev],folds3,target[dev])
            winner['phase3']=m
            winner['third_improvement']=m['score']-base3['score']
            winner['three_seed_mean_improvement']=(2*winner['mean_improvement']+winner['third_improvement'])/3
            winner['three_seed_mean_score']=(winner['phase1']['score']+winner['phase2']['score']+m['score'])/3
            winner['final_eligible']=bool(winner['third_improvement']>=-.0005 and winner['three_seed_mean_improvement']>=.0015)
            write_json(out/'reports/final_selection.json',dict(winner=winner,baseline3=base3,
                       frozen_before_full_fit=True,holdout_inspected=False))
            if winner['final_eligible']:
                models,predictions={},{}
                for name in sorted(spec['weights']):
                    print(f'Full-data fit: {name}',flush=True)
                    models[name]=fit_model(name,train_e,x,y,np.arange(len(y)))
                    predictions[name]=predict_model(name,models[name],test_e,xt,np.arange(len(xt)))
                components=aggregate(predictions,spec)
                winner['submission_file']='submissions/v5_candidate_1.csv'
                winner['model_file']='models/v5_candidate_1.joblib'
                winner['submission_check']=write_submission(out/winner['submission_file'],data,components,winner)
                encoders={NEW_SPECS[k]['encoder'] if k in NEW_SPECS else ('e5' if k=='old_v3' else ('bge' if k.startswith('bge') else 'e5base')) for k in models}
                bundle=dict(format_version=5,candidate=winner['candidate'],spec=spec,fraction=winner['fraction'],policy=winner['policy'],
                            models=models,versions=versions(),encoders={k:asset_reports[k] for k in encoders},
                            train_sha256=data.audit['files']['train']['sha256'],raw_test_sha256=data.audit['files']['test']['sha256'],
                            test_ids=[r['id'] for r in data.test])
                path=out/winner['model_file']
                path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(bundle,path,compress=3)
                write_csv(out/'predictions/v5_candidate_1_test.csv',[dict(id=r['id'],ranking_score=float(c[0]),probability_read=float(c[1])) for r,c in zip(data.test,components)],
                          ['id','ranking_score','probability_read'])
    result=dict(baseline1=base1,baseline2=base2,baseline3=base3,confirmation=confirmation,winner=winner,
                runtime_seconds=round(time.perf_counter()-start,3),holdout_inspected=False,
                limitation='All three seeds reuse development rows; repeated selection can overestimate performance. No guarantee of Public/Private gain.')
    write_json(out/'reports/results.json',result)
    lines=['# V5 development result','','| Model | Seed262 | Seed1262 | Seed2262 |', '|---|---:|---:|---:|',
           f"| v4 | {base1['score']:.6f} | {base2['score']:.6f} | {base3['score'] if base3 else 'not run'} |"]
    if winner:
        lines.append(f"| {winner['candidate']} | {winner['phase1']['score']:.6f} | {winner['phase2']['score']:.6f} | {winner.get('phase3',{}).get('score','not run')} |")
        lines+=['',f"Final gate: {winner.get('final_eligible',False)}",f"Fraction: {winner['fraction']}; policy: {winner['policy']}"]
    if not winner or not winner.get('final_eligible',False):
        lines+=['','No new CSV: no candidate passed every predefined gate.']
    lines+=['',result['limitation']]
    (out/'reports/summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    files=[root/'refine.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),versions=versions(),input_files=data.audit['files'],
               code_sha256={str(p.relative_to(root)):sha256(p) for p in files},
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {out}; new CSV: {bool(winner and winner.get("submission_file"))}',flush=True)
    return result


def predict_refinement(bundle_path,data,output,assets,cache):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle=joblib.load(bundle_path)
    if bundle['format_version']!=5 or bundle['versions']!=versions():
        raise ValueError('Wrong model/dependency versions.')
    if bundle['train_sha256']!=data.audit['files']['train']['sha256'] or bundle['raw_test_sha256']!=data.audit['files']['test']['sha256']:
        raise ValueError('Raw data changed.')
    if bundle['test_ids']!=[r['id'] for r in data.test]:
        raise ValueError('Test IDs changed.')
    embeddings={}
    for name,recorded in bundle['encoders'].items():
        _,current=ensure_encoder(name,assets) if name=='e5' else ensure_extra(name,assets)
        if current!=recorded:
            raise ValueError('Encoder assets changed.')
        embeddings[name],_=extra_cache(data.test,name,assets,cache)
    with threadpool_limits(limits=1):
        x=rows_to_x(data.test)
        predictions={k:predict_model(k,m,embeddings,x,np.arange(len(x))) for k,m in bundle['models'].items()}
        return write_submission(output,data,aggregate(predictions,bundle['spec']),bundle)
