"""Frozen shortlisting of local semantic heads against V6 candidate 2."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .data import paper_groups,sha256,write_json,write_csv
from .models import rows_to_x
from .local_models import HEADS,LocalSemanticHead
from .v7_models import PARENT_SPEC
from .v7_experiment import cached_parent,evaluate,write_submission,PARENT_CONFIG
from .metadata_experiment import fit as parent_fit,predict as parent_predict
from .refine_models import aggregate
from .semantic_experiment import versions
from .upgrade_experiment import folds_for,load_assets

SEEDS=(361,1361,2361)
FRACTIONS=(.575,.6,.625,.65)
MIXTURES=(.15,.3,.5)


def fit(name,embeddings,x,y,indices):
    if name not in HEADS:
        return parent_fit(name,embeddings,x,y,indices)
    spec=HEADS[name]
    return LocalSemanticHead(spec).fit(embeddings[spec['encoder']][indices],x[indices],y[indices])


def predict(name,model,embeddings,x,indices):
    if name not in HEADS:
        return parent_predict(name,model,embeddings,x,indices)
    return model.predict_components(embeddings[HEADS[name]['encoder']][indices],x[indices])


def mixture(name,q):
    return dict(weights={name:q,**{k:(1-q)*w for k,w in PARENT_SPEC['weights'].items()}},aggregation='probability')


def cv(names,embeddings,x,y,groups,seed,path,initial=None,expected=None):
    folds,pairs=folds_for(x,y,groups,seed)
    if expected is not None and not np.array_equal(folds,expected):
        raise ValueError('Cached fold assignment changed.')
    predictions={} if initial is None else dict(initial)
    missing=[name for name in names if name not in predictions]
    predictions.update({name:np.zeros((len(y),2)) for name in missing})
    for fold,(train,val) in enumerate(pairs):
        for name in missing:
            print(f'V8 seed{seed} fold{fold+1}/7: {name}',flush=True)
            model=fit(name,embeddings,x,y,train)
            predictions[name][val]=predict(name,model,embeddings,x,val)
            del model
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=np.arange(len(y)),fold=folds,**predictions)
    return predictions,folds


def run(data,out,reference,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty output directory.')
    start=time.perf_counter()
    initial1,expected1=cached_parent(reference,361,data)
    initial2,expected2=cached_parent(reference,1361,data)
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.array([r['Label'] for r in data.train],int)
    ids=np.array([r['id'] for r in data.train])
    groups=np.array(paper_groups(data.train))
    target=np.array([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    out.mkdir(parents=True,exist_ok=True)
    plan=dict(heads=HEADS,fractions=FRACTIONS,mixtures=MIXTURES,seeds=SEEDS,parent=PARENT_SPEC,
              rule='Seed361: best head/mixture/fraction per encoder and model family (four finalists). Freeze before seed1361. Select exactly one finalist by two-seed mean score, breaking ties by candidate name. Freeze before seed2361. Report third-seed comparison without further tuning. Export this experimental candidate regardless of third-seed direction; label evidence honestly.',
              limitation='Repeated training-label validation, including reused seeds. Third seed is a different partition of the same data, not an untouched dataset. Public improvement is unknown.',
              prior_policy_probe='outputs/v8_probe: alternative selection policies did not provide a material gain and are excluded from this model search.',
              input_files=data.audit['files'],versions=versions())
    write_json(out/'reports/search_plan.json',plan)
    train_e,test_e,manifests,caches=load_assets(data,['e5','bge','e5base'],assets,cache)
    with threadpool_limits(limits=1):
        first,folds1=cv(list(HEADS),train_e,x,y,groups,361,out/'predictions/full_seed361.npz',initial1,expected1)
        base1=evaluate(PARENT_CONFIG,PARENT_SPEC,first,ids,y,folds1,target)
        records=[]
        for name,head in HEADS.items():
            for q in MIXTURES:
                for fraction in FRACTIONS:
                    conf=dict(candidate=name,mix=q,fraction=fraction,alpha=.5)
                    score=evaluate(conf,mixture(name,q),first,ids,y,folds1,target)
                    records.append(dict(score,family=head['encoder']+'_'+head['kind']))
        records.sort(key=lambda r:(-r['score'],r['candidate'],r['mix'],r['fraction']))
        shortlist=[]
        for family in sorted({r['family'] for r in records}):
            shortlist.append(next(r for r in records if r['family']==family))
        write_json(out/'reports/frozen_shortlist.json',dict(baseline=base1,ranking=records,shortlist=shortlist,
                   frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        print('V8 frozen shortlist: '+json.dumps(shortlist),flush=True)
        second,folds2=cv([r['candidate'] for r in shortlist],train_e,x,y,groups,1361,
                         out/'predictions/full_seed1361.npz',initial2,expected2)
        base2=evaluate(PARENT_CONFIG,PARENT_SPEC,second,ids,y,folds2,target)
        confirmation=[]
        for r in shortlist:
            conf={k:r[k] for k in ('candidate','mix','fraction','alpha')}
            m=evaluate(conf,mixture(r['candidate'],r['mix']),second,ids,y,folds2,target)
            deltas=[r['score']-base1['score'],m['score']-base2['score']]
            confirmation.append(dict(**conf,phase1=r,phase2=m,mean_score=(r['score']+m['score'])/2,
                                     deltas=deltas,mean_delta=float(np.mean(deltas))))
        confirmation.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        chosen=confirmation[0]
        write_json(out/'reports/frozen_before_third_seed.json',dict(baseline1=base1,baseline2=base2,
                   confirmation=confirmation,chosen=chosen,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        print('V8 frozen final candidate: '+json.dumps(chosen),flush=True)
        needed=[*PARENT_SPEC['weights'],chosen['candidate']]
        third,folds3=cv(needed,train_e,x,y,groups,2361,out/'predictions/full_seed2361.npz')
        base3=evaluate(PARENT_CONFIG,PARENT_SPEC,third,ids,y,folds3,target)
        spec=mixture(chosen['candidate'],chosen['mix'])
        third_score=evaluate(chosen,spec,third,ids,y,folds3,target)
        chosen.update(phase3=third_score,third_delta=third_score['score']-base3['score'])
        chosen['three_seed_mean_delta']=float(np.mean([*chosen['deltas'],chosen['third_delta']]))
        chosen['positive_on_all_seeds']=bool(min(*chosen['deltas'],chosen['third_delta'])>0)
        models,predictions={},{}
        for name in needed:
            print(f'V8 full-data fit: {name}',flush=True)
            models[name]=fit(name,train_e,x,y,np.arange(len(y)))
            predictions[name]=predict(name,models[name],test_e,xt,np.arange(len(xt)))
        components=aggregate(predictions,spec)
        chosen['submission_file']='submissions/v8_candidate_1.csv'
        chosen['submission_check']=write_submission(out/chosen['submission_file'],data,components,chosen)
        chosen['model_file']='models/v8_candidate_1.joblib'
        bundle=dict(format_version=8,config={k:chosen[k] for k in ('candidate','mix','fraction','alpha')},
                    spec=spec,models=models,versions=versions(),encoders=manifests,
                    input_files=data.audit['files'],test_ids=[r['id'] for r in data.test])
        (out/'models').mkdir(parents=True,exist_ok=True)
        joblib.dump(bundle,out/chosen['model_file'],compress=3)
        write_csv(out/'predictions/v8_candidate_1_test.csv',
                  [dict(id=r['id'],ranking_score=float(c[0]),probability_read=float(c[1])) for r,c in zip(data.test,components)],
                  ['id','ranking_score','probability_read'])
    write_json(out/'reports/results.json',dict(parent=[base1,base2,base3],confirmation=confirmation,chosen=chosen,
               runtime_seconds=round(time.perf_counter()-start,3),public_score=None,limitation=plan['limitation']))
    root=Path(__file__).resolve().parents[1]
    sources=[root/'v8.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(versions=versions(),input_files=data.audit['files'],encoders=manifests,caches=caches,
               code_sha256={str(p.relative_to(root)):sha256(p) for p in sources},
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print('Done V8: '+json.dumps(chosen),flush=True)


def replay(data,config,out,assets,cache):
    """Refit a committed, frozen JSON configuration from original data."""
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty output directory.')
    if config['input_files']!=data.audit['files'] or config['versions']!=versions():
        raise ValueError('Raw inputs or dependency versions changed.')
    chosen=config['config'];spec=mixture(chosen['candidate'],chosen['mix'])
    if config['head']!=HEADS[chosen['candidate']] or config['spec']!=spec:
        raise ValueError('Frozen head or mixture configuration changed.')
    train_e,test_e,manifests,_=load_assets(data,['e5','bge','e5base'],assets,cache)
    if manifests!=config['encoders']:
        raise ValueError('Encoder assets changed.')
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    predictions={}
    with threadpool_limits(limits=1):
        for name in spec['weights']:
            print('V8 replay fit: '+name,flush=True)
            model=fit(name,train_e,x,y,np.arange(len(y)))
            predictions[name]=predict(name,model,test_e,xt,np.arange(len(xt)))
        check=write_submission(out/'submissions/v8_candidate_1.csv',data,aggregate(predictions,spec),chosen)
    check['reference_match']=check['sha256']==config['submission_sha256']
    write_json(out/'reports/reproduction.json',check)
    if not check['reference_match']:
        raise ValueError('Replayed CSV does not match the frozen reference.')
    print(json.dumps(check),flush=True)
