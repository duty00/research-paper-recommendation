"""V6 metadata follow-up: fixed heads, reused verified baseline OOF, frozen confirmation."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .data import paper_groups,sha256,write_json,write_csv
from .models import rows_to_x
from .refine_models import aggregate
from .refine_experiment import fit_model,predict_model,write_submission
from .metadata_models import META_HEADS,MetadataSVC
from .semantic_experiment import versions
from .upgrade_experiment import BASE_SPEC,BASE_WEIGHTS,FRACTIONS,POLICIES,SEEDS,folds_for,evaluate,paired_check,load_assets,encoder_for


def specifications():
    return {f'{name}_mix{int(q*100)}':dict(weights={name:q,**{k:(1-q)*w for k,w in BASE_WEIGHTS.items() if q<1}},
            aggregation='probability',family=spec['encoder'])
            for name,spec in META_HEADS.items() for q in (.25,.5,.75,1.)}


def fit(name, embeddings, x, y, indices):
    if name in META_HEADS:
        return MetadataSVC(META_HEADS[name],261).fit(embeddings[META_HEADS[name]['encoder']][indices],x[indices],y[indices])
    return fit_model(name,embeddings,x,y,indices)


def predict(name, model, embeddings, x, indices):
    if name in META_HEADS:
        return model.predict_components(embeddings[META_HEADS[name]['encoder']][indices],x[indices])
    return predict_model(name,model,embeddings,x,indices)


def baseline_cache(reference,seed,data):
    manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    if manifest['input_files']!=data.audit['files']:
        raise ValueError('Reference raw data changed.')
    relative=f'predictions/full_seed{seed}.npz'
    hashes={k.replace('\\','/'):v for k,v in manifest['output_sha256'].items()}
    if sha256(reference/relative)!=hashes[relative]:
        raise ValueError('Baseline cache changed.')
    with np.load(reference/relative,allow_pickle=False) as saved:
        if not np.array_equal(saved['indices'],np.arange(len(data.train))):
            raise ValueError('Reference row order changed.')
        return {k:saved[k] for k in BASE_WEIGHTS},saved['fold']


def cv(names,initial,expected,embeddings,x,y,groups,seed,path):
    folds,pairs=folds_for(x,y,groups,seed)
    if not np.array_equal(folds,expected):
        raise ValueError('Reference and current folds differ.')
    result=dict(initial)
    names=[k for k in names if k not in result]
    result.update({k:np.zeros((len(y),2)) for k in names})
    for fold,(train,val) in enumerate(pairs):
        for name in names:
            print(f'V6 metadata seed {seed}, fold {fold+1}/7: {name}',flush=True)
            model=fit(name,embeddings,x,y,train)
            result[name][val]=predict(name,model,embeddings,x,val)
            del model
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=np.arange(len(y)),fold=folds,**result)
    return result,folds


def run(data,out,reference,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output folder.')
    start=time.perf_counter()
    initial1,expected1=baseline_cache(reference,SEEDS[0],data)
    initial2,expected2=baseline_cache(reference,SEEDS[1],data)
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    ids=np.asarray([r['id'] for r in data.train])
    groups=np.asarray(paper_groups(data.train))
    target=np.asarray([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    train_e,test_e,manifests,caches=load_assets(data,['e5','bge','e5base'],assets,cache)
    specs=specifications()
    plan=dict(heads=META_HEADS,candidates=specs,seeds=SEEDS,fractions=FRACTIONS,policies=POLICIES,baseline=BASE_SPEC,
              input_files=data.audit['files'],reference_manifest_sha256=sha256(reference/'manifest.json'),
              author_features='Individual full names, lowercased; TFIDF min_df2 vocabulary fitted inside each training fold. No label statistics or test-fitted vocabulary.',
              rule='Shortlist best config per encoder on seed361, freeze settings, confirm seed1361. Positive paired delta on both seeds, >=4/7 fold wins each, two-seed mean delta>=.0015. At most one CSV.',
              limitation='Adaptive follow-up after large encoders failed. Same labels and CV seeds reused; this is not an untouched test. No claim of guaranteed Public/Private gain.',
              prior_failure='../v6_run01/reports/final_selection.json',public_scores_used_for_selection=False)
    write_json(out/'reports/search_plan.json',plan)
    config=dict(candidate='v5',fraction=.6,policy='gain')
    with threadpool_limits(limits=1):
        first,folds1=cv(list(META_HEADS),initial1,expected1,train_e,x,y,groups,SEEDS[0],out/'predictions/full_seed361.npz')
        base1=evaluate(config,BASE_SPEC,first,ids,y,folds1,target)
        records,ranking=[],[]
        for name,spec in specs.items():
            values=[evaluate(dict(candidate=name,fraction=f,policy=p),spec,first,ids,y,folds1,target) for f in FRACTIONS for p in POLICIES]
            records.extend(values)
            ranking.append(max(values,key=lambda r:(r['score'],r['fraction'],r['policy'])))
        ranking.sort(key=lambda r:(-r['score'],r['candidate']))
        shortlist=[next(r for r in ranking if specs[r['candidate']]['family']==encoder) for encoder in ('bge','e5base')]
        write_json(out/'reports/frozen_shortlist.json',dict(baseline=base1,ranking=ranking,shortlist=shortlist,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        write_csv(out/'reports/search_results.csv',records,['candidate','fraction','policy','score','relevance','coverage','fold_score_std'])
        needed=sorted({k for r in shortlist for k in specs[r['candidate']]['weights']})
        second,folds2=cv(needed,initial2,expected2,train_e,x,y,groups,SEEDS[1],out/'predictions/full_seed1361.npz')
        base2=evaluate(config,BASE_SPEC,second,ids,y,folds2,target)
        confirmation=[]
        for record in shortlist:
            conf={k:record[k] for k in ('candidate','fraction','policy')}
            m=evaluate(conf,specs[record['candidate']],second,ids,y,folds2,target)
            confirmation.append(dict(**conf,phase1=record,phase2=m,mean_score=(record['score']+m['score'])/2,
                                     **paired_check(record,m,base1,base2)))
        confirmation.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        winner=next((r for r in confirmation if r['eligible']),None)
        write_json(out/'reports/final_selection.json',dict(baseline1=base1,baseline2=base2,confirmation=confirmation,winner=winner,frozen_before_full_fit=True))
        if winner:
            spec=specs[winner['candidate']]
            models,predictions={},{}
            for name in sorted(spec['weights']):
                print(f'V6 metadata full fit: {name}',flush=True)
                models[name]=fit(name,train_e,x,y,np.arange(len(y)))
                predictions[name]=predict(name,models[name],test_e,xt,np.arange(len(xt)))
            components=aggregate(predictions,spec)
            winner['submission_file']='submissions/v6_candidate_1.csv'
            winner['model_file']='models/v6_candidate_1.joblib'
            winner['submission_check']=write_submission(out/winner['submission_file'],data,components,winner)
            encoders={META_HEADS[k]['encoder'] if k in META_HEADS else encoder_for(k) for k in models}
            bundle=dict(format_version=61,candidate=winner['candidate'],spec=spec,fraction=winner['fraction'],policy=winner['policy'],
                        models=models,versions=versions(),encoders={k:manifests[k] for k in encoders},
                        train_sha256=data.audit['files']['train']['sha256'],raw_test_sha256=data.audit['files']['test']['sha256'],test_ids=[r['id'] for r in data.test])
            model_path=out/winner['model_file']
            model_path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(bundle,model_path,compress=3)
            write_csv(out/'predictions/v6_candidate_1_test.csv',
                      [dict(id=r['id'],ranking_score=float(c[0]),probability_read=float(c[1])) for r,c in zip(data.test,components)],
                      ['id','ranking_score','probability_read'])
    result=dict(baseline1=base1,baseline2=base2,confirmation=confirmation,winner=winner,
                runtime_seconds=round(time.perf_counter()-start,3),untouched_holdout=False,limitation=plan['limitation'])
    write_json(out/'reports/results.json',result)
    root=Path(__file__).resolve().parents[1]
    files=[root/'metadata_refine.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),versions=versions(),input_files=data.audit['files'],
               code_sha256={str(p.relative_to(root)):sha256(p) for p in files},caches=caches,
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {out}; new CSV: {winner is not None}',flush=True)
    return result


def predict_saved(bundle_path,data,output,assets,cache,refit=False):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle=joblib.load(bundle_path)
    if bundle['format_version']!=61 or bundle['versions']!=versions():
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
