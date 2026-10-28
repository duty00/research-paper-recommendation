"""Fast author/venue follow-up after the V7 SVC neighbourhood failed."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .author_models import HEADS,AuthorPreference,candidates
from .data import paper_groups,sha256,write_json
from .metadata_experiment import fit as parent_fit,predict as parent_predict
from .metadata_models import META_HEADS
from .models import rows_to_x
from .refine_models import aggregate
from .semantic_experiment import versions
from .upgrade_experiment import folds_for,load_assets,encoder_for
from .v7_experiment import cached_parent,evaluate,confirmation,paired,write_submission,PARENT_CONFIG
from .v7_models import PARENT_SPEC


def fit(name,embeddings,x,y,indices):
    if name in HEADS:
        return AuthorPreference(HEADS[name],261).fit(x[indices],y[indices])
    return parent_fit(name,embeddings,x,y,indices)


def predict(name,model,embeddings,x,indices):
    if name in HEADS:
        return model.predict_components(x[indices])
    return parent_predict(name,model,embeddings,x,indices)


def cv(names,embeddings,x,y,groups,seed,path,initial=None,expected=None):
    folds,pairs=folds_for(x,y,groups,seed)
    if expected is not None and not np.array_equal(folds,expected):
        raise ValueError('Cached and current fold assignments differ.')
    output={} if initial is None else dict(initial)
    names=[k for k in names if k not in output]
    output.update({k:np.zeros((len(y),2)) for k in names})
    for fold,(train,val) in enumerate(pairs):
        for name in names:
            print(f'V7 authors seed {seed}, fold {fold+1}/7: {name}',flush=True)
            model=fit(name,embeddings,x,y,train)
            output[name][val]=predict(name,model,embeddings,x,val)
            del model
    if any(not np.isfinite(v).all() for v in output.values()):
        raise ValueError('Invalid OOF prediction.')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=np.arange(len(y)),fold=folds,**output)
    return output,folds


def run(data,out,reference,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output folder.')
    start=time.perf_counter()
    initial1,expected1=cached_parent(reference,361,data)
    initial2,expected2=cached_parent(reference,1361,data)
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    ids=np.asarray([r['id'] for r in data.train])
    target=np.asarray([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    groups=np.asarray(paper_groups(data.train))
    specs=candidates()
    plan=dict(heads=HEADS,candidates=specs,seeds=[361,1361,2361],fraction=.6,alpha=.5,input_files=data.audit['files'],
              rule='Keep the parent selection rule fixed. Phase1: best candidate per family, all3 families. Freeze. Phase2: both deltas positive, >=4/7 wins per seed, two-seed mean delta>=.001. Freeze top2. Phase3 on seed2361: nonnegative delta, >=4/7 wins and three-seed mean delta>=.001. At most2 CSVs in the frozen order.',
              preceding_experiment='V7 SVC variants did not exceed the parent on seed361; this uses a distinct author/venue model family.',
              leakage_control='Author statistics and vocabularies fitted only on the outer training fold. No parent OOF residuals are used for fitting the author model.',
              public_feedback='The parent was chosen from prior Public results. New parameters use training CV only; no hidden labels or membership available.',
              limitation='Adaptive validation on reused labels. Seed2361 is a new partition, not an independent dataset.')
    write_json(out/'reports/search_plan.json',plan)
    with threadpool_limits(limits=1):
        first,folds1=cv(list(HEADS),{},x,y,groups,361,out/'predictions/full_seed361.npz',initial1,expected1)
        base1=evaluate(PARENT_CONFIG,PARENT_SPEC,first,ids,y,folds1,target)
        ranking=[evaluate(dict(candidate=name,fraction=.6,alpha=.5),spec,first,ids,y,folds1,target) for name,spec in specs.items()]
        ranking.sort(key=lambda r:(-r['score'],r['candidate']))
        shortlist=[next(r for r in ranking if specs[r['candidate']]['family']==family) for family in ('ridge','ordinal','prior')]
        write_json(out/'reports/frozen_shortlist.json',dict(baseline=base1,ranking=ranking,shortlist=shortlist,
                   frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        needed=sorted({k for r in shortlist for k in specs[r['candidate']]['weights']})
        second,folds2=cv(needed,{},x,y,groups,1361,out/'predictions/full_seed1361.npz',initial2,expected2)
        base2=evaluate(PARENT_CONFIG,PARENT_SPEC,second,ids,y,folds2,target)
        confirmed=[]
        for r in shortlist:
            config={k:r[k] for k in ('candidate','fraction','alpha')}
            m=evaluate(config,specs[r['candidate']],second,ids,y,folds2,target)
            confirmed.append(dict(**config,phase1=r,phase2=m,mean_score=(r['score']+m['score'])/2,
                                  **confirmation(r,m,base1,base2)))
        confirmed.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        frozen=[r for r in confirmed if r['eligible']][:2]
        write_json(out/'reports/frozen_before_third_seed.json',dict(baseline1=base1,baseline2=base2,
                   confirmation=confirmed,frozen=frozen,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        passed=[]
        base3=None
        train_e,test_e,manifests,caches={},{},{},{}
        if frozen:
            train_e,test_e,manifests,caches=load_assets(data,['e5','bge','e5base'],assets,cache)
            needed=sorted(set(PARENT_SPEC['weights'])|{k for r in frozen for k in specs[r['candidate']]['weights']})
            third,folds3=cv(needed,train_e,x,y,groups,2361,out/'predictions/full_seed2361.npz')
            base3=evaluate(PARENT_CONFIG,PARENT_SPEC,third,ids,y,folds3,target)
            for r in frozen:
                config={k:r[k] for k in ('candidate','fraction','alpha')}
                m=evaluate(config,specs[r['candidate']],third,ids,y,folds3,target)
                delta,wins=paired(m,base3)
                r.update(phase3=m,third_improvement=delta,third_fold_wins=wins,
                         three_seed_mean_improvement=(2*r['mean_improvement']+delta)/3,
                         final_eligible=bool(delta>=0 and wins>=4 and (2*r['mean_improvement']+delta)/3>=.001))
                if r['final_eligible']: passed.append(r)
        write_json(out/'reports/final_selection.json',dict(baseline1=base1,baseline2=base2,baseline3=base3,
                   frozen_candidates=frozen,passed=passed,no_retuning_after_third_seed=True))
        models,predictions={},{}
        for name in sorted({k for r in passed for k in specs[r['candidate']]['weights']}):
            print(f'V7 authors full-data fit: {name}',flush=True)
            models[name]=fit(name,train_e,x,y,np.arange(len(y)))
            predictions[name]=predict(name,models[name],test_e,xt,np.arange(len(xt)))
        for number,r in enumerate(passed,1):
            spec=specs[r['candidate']]
            values=aggregate({k:predictions[k] for k in spec['weights']},spec)
            r['submission_file']=f'submissions/v7_candidate_{number}.csv'
            r['model_file']=f'models/v7_candidate_{number}.joblib'
            r['submission_check']=write_submission(out/r['submission_file'],data,values,r)
            used={k:models[k] for k in spec['weights']}
            encoders={META_HEADS[k]['encoder'] if k in META_HEADS else encoder_for(k) for k in used if k not in HEADS}
            bundle=dict(format_version=71,candidate=r['candidate'],spec=spec,fraction=r['fraction'],alpha=r['alpha'],
                        models=used,versions=versions(),encoders={k:manifests[k] for k in encoders},
                        train_sha256=data.audit['files']['train']['sha256'],raw_test_sha256=data.audit['files']['test']['sha256'],
                        test_ids=[row['id'] for row in data.test])
            path=out/r['model_file']
            path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(bundle,path,compress=3)
    result=dict(baseline1=base1,baseline2=base2,baseline3=base3,confirmation=confirmed,selected=passed,
                runtime_seconds=round(time.perf_counter()-start,3),untouched_holdout=False,limitation=plan['limitation'])
    write_json(out/'reports/results.json',result)
    root=Path(__file__).resolve().parents[1]
    files=[root/'v7_authors.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),versions=versions(),
               input_files=data.audit['files'],code_sha256={str(p.relative_to(root)):sha256(p) for p in files},caches=caches,
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {out}; CSV count: {len(passed)}',flush=True)
    return result


def predict_saved(bundle_path,data,output,assets,cache,refit=False):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle=joblib.load(bundle_path)
    if bundle['format_version']!=71 or bundle['versions']!=versions():
        raise ValueError('Wrong model or dependency versions.')
    if (bundle['train_sha256']!=data.audit['files']['train']['sha256'] or bundle['raw_test_sha256']!=data.audit['files']['test']['sha256'] or
        bundle['test_ids']!=[r['id'] for r in data.test]):
        raise ValueError('Raw data changed.')
    train_e,test_e,manifests,_=load_assets(data,sorted(bundle['encoders']),assets,cache,training=refit)
    if manifests!=bundle['encoders']:
        raise ValueError('Encoder assets changed.')
    with threadpool_limits(limits=1):
        x,xt=rows_to_x(data.train),rows_to_x(data.test)
        y=np.asarray([r['Label'] for r in data.train],dtype=int)
        predictions={}
        for name,saved in bundle['models'].items():
            model=fit(name,train_e,x,y,np.arange(len(y))) if refit else saved
            predictions[name]=predict(name,model,test_e,xt,np.arange(len(xt)))
        return write_submission(output,data,aggregate(predictions,bundle['spec']),bundle)
