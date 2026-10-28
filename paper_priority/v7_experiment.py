"""V7 fixed search, frozen confirmation and a third seven-fold split check."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .data import paper_groups,sha256,write_json,write_csv
from .experiment import summarize,validate_submission
from .metadata_experiment import fit as parent_fit,predict as parent_predict
from .metadata_models import META_HEADS
from .metric import score_priorities
from .models import rows_to_x
from .refine_models import aggregate
from .semantic_experiment import versions
from .upgrade_experiment import folds_for,load_assets,encoder_for
from .v7_models import HEADS,PARENT_SPEC,TunedMetadataSVC,candidates,priorities

SEEDS=(361,1361,2361)
FRACTIONS=(.55,.575,.60,.625,.65)
ALPHAS=(.25,.5,.75)
PARENT_CONFIG=dict(candidate='v6_candidate2',fraction=.6,alpha=.5)


def fit(name,embeddings,x,y,indices):
    if name in HEADS:
        spec=HEADS[name]
        return TunedMetadataSVC(spec,261).fit(embeddings[spec['encoder']][indices],x[indices],y[indices])
    return parent_fit(name,embeddings,x,y,indices)


def predict(name,model,embeddings,x,indices):
    if name in HEADS:
        return model.predict_components(embeddings[HEADS[name]['encoder']][indices],x[indices])
    return parent_predict(name,model,embeddings,x,indices)


def cached_parent(reference,seed,data):
    manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    if manifest['input_files']!=data.audit['files'] or manifest['versions']!=versions():
        raise ValueError('Reference inputs or dependency versions differ.')
    relative=f'predictions/full_seed{seed}.npz'
    hashes={k.replace('\\','/'):v for k,v in manifest['output_sha256'].items()}
    if sha256(reference/relative)!=hashes[relative]:
        raise ValueError('Parent OOF changed.')
    with np.load(reference/relative,allow_pickle=False) as stored:
        if not np.array_equal(stored['indices'],np.arange(len(data.train))):
            raise ValueError('Parent row order differs.')
        return {k:stored[k] for k in PARENT_SPEC['weights']},stored['fold']


def cv(names,embeddings,x,y,groups,seed,path,initial=None,expected=None):
    folds,pairs=folds_for(x,y,groups,seed)
    if expected is not None and not np.array_equal(folds,expected):
        raise ValueError('Reference folds differ.')
    result={} if initial is None else dict(initial)
    names=[name for name in names if name not in result]
    result.update({name:np.zeros((len(y),2)) for name in names})
    for fold,(train,val) in enumerate(pairs):
        for name in names:
            print(f'V7 seed {seed}, fold {fold+1}/7: {name}',flush=True)
            model=fit(name,embeddings,x,y,train)
            result[name][val]=predict(name,model,embeddings,x,val)
            del model
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Invalid OOF prediction.')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=np.arange(len(y)),fold=folds,**result)
    return result,folds


def evaluate(config,spec,predictions,ids,y,folds,target):
    metrics=[]
    for fold in range(7):
        mask=(folds==fold)&target
        if not mask.any():
            raise ValueError('Empty scoring fold.')
        values=aggregate({k:predictions[k][mask] for k in spec['weights']},spec)
        p=priorities(ids[mask],values,config['fraction'],config['alpha'])
        metrics.append(score_priorities(ids[mask],y[mask],p))
    return dict(**config,**summarize(metrics),fold_scores=[m['score'] for m in metrics])


def paired(candidate,baseline):
    delta=candidate['score']-baseline['score']
    wins=int(np.sum(np.asarray(candidate['fold_scores'])>baseline['fold_scores']))
    return float(delta),wins


def confirmation(first,second,base1,base2):
    d1,w1=paired(first,base1)
    d2,w2=paired(second,base2)
    return dict(deltas=[d1,d2],paired_fold_wins=[w1,w2],mean_improvement=(d1+d2)/2,
                eligible=bool(min(d1,d2)>0 and min(w1,w2)>=4 and (d1+d2)/2>=.001))


def write_submission(path,data,values,config):
    ids=[r['id'] for r in data.test]
    p=priorities(ids,values,config['fraction'],config['alpha'])
    mapping=dict(zip(ids,p))
    write_csv(path,[dict(id=r['id'],priority=format(mapping[r['id']],'.12g')) for r in data.sample],['id','priority'])
    return validate_submission(path,ids)


def run(data,out,reference,parent_export,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory.')
    parent_csv=parent_export/'submissions/v6_candidate_2.csv'
    if sha256(parent_csv)!='31e9d52b555354168da504597eb671f97dee775533f2d09ae76d48c8f6e498aa':
        raise ValueError('V6 candidate2 reference changed.')
    initial1,expected1=cached_parent(reference,361,data)
    initial2,expected2=cached_parent(reference,1361,data)
    parent_record=json.loads((parent_export/'reports/results.json').read_text(encoding='utf-8'))
    if any(parent_record['spec'][k]!=PARENT_SPEC[k] for k in ('weights','aggregation')):
        raise ValueError('Parent configuration differs.')
    start=time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    ids=np.asarray([r['id'] for r in data.train])
    groups=np.asarray(paper_groups(data.train))
    target=np.asarray([r['venue'] in {t['venue'] for t in data.test} for r in data.train])
    train_e,test_e,manifests,caches=load_assets(data,['e5','bge','e5base'],assets,cache)
    specs=candidates()
    plan=dict(heads=HEADS,candidates=specs,fractions=FRACTIONS,alphas=ALPHAS,seeds=SEEDS,
              parent=dict(spec=PARENT_SPEC,config=PARENT_CONFIG,reported_public_score=.66504,csv_sha256=sha256(parent_csv)),
              input_files=data.audit['files'],encoders=manifests,reference_manifest_sha256=sha256(reference/'manifest.json'),
              rule='Phase1 seed361: best config per candidate, shortlist the best3 distinct families. Freeze. Phase2 seed1361: positive deltas both seeds, >=4/7 wins each, mean delta>=.001. Freeze top2 passing candidates in phase2 mean-score order. Phase3 seed2361: recompute parent and frozen candidates; require third delta>=0, >=4/7 wins, three-seed mean delta>=.001. Export at most2 CSVs in the already frozen order, skipping failed candidates. No retuning after confirmation.',
              public_feedback='Parent version selected from prior Public results. New V7 numerical settings use training CV only; no test labels or Public/Private membership are available.',
              limitation='Repeated adaptive validation on previously used labels; the third seed is a new partition, not a new dataset or untouched test.')
    write_json(out/'reports/search_plan.json',plan)
    with threadpool_limits(limits=1):
        first,folds1=cv(list(HEADS),train_e,x,y,groups,361,out/'predictions/full_seed361.npz',initial1,expected1)
        base1=evaluate(PARENT_CONFIG,PARENT_SPEC,first,ids,y,folds1,target)
        if abs(base1['score']-parent_record['candidate']['phase1']['score'])>1e-12:
            raise ValueError('Cached parent score no longer reproduces.')
        records,ranking=[],[]
        for name,spec in specs.items():
            scores=[evaluate(dict(candidate=name,fraction=f,alpha=a),spec,first,ids,y,folds1,target)
                    for f in FRACTIONS for a in ALPHAS]
            records.extend(scores)
            ranking.append(max(scores,key=lambda r:(r['score'],r['fraction'],r['alpha'])))
        ranking.sort(key=lambda r:(-r['score'],r['candidate']))
        shortlist=[]
        families=set()
        for r in ranking:
            family=specs[r['candidate']]['family']
            if family not in families:
                families.add(family)
                shortlist.append(r)
            if len(shortlist)==3: break
        write_json(out/'reports/frozen_shortlist.json',dict(baseline=base1,ranking=ranking,shortlist=shortlist,
                   frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        write_csv(out/'reports/search_results.csv',records,['candidate','fraction','alpha','score','relevance','coverage','fold_score_std'])
        needed=sorted({k for r in shortlist for k in specs[r['candidate']]['weights']})
        second,folds2=cv(needed,train_e,x,y,groups,1361,out/'predictions/full_seed1361.npz',initial2,expected2)
        base2=evaluate(PARENT_CONFIG,PARENT_SPEC,second,ids,y,folds2,target)
        if abs(base2['score']-parent_record['candidate']['phase2']['score'])>1e-12:
            raise ValueError('Second cached parent score no longer reproduces.')
        confirmed=[]
        for r in shortlist:
            conf={k:r[k] for k in ('candidate','fraction','alpha')}
            m=evaluate(conf,specs[r['candidate']],second,ids,y,folds2,target)
            confirmed.append(dict(**conf,phase1=r,phase2=m,mean_score=(r['score']+m['score'])/2,
                                  **confirmation(r,m,base1,base2)))
        confirmed.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        frozen=[r for r in confirmed if r['eligible']][:2]
        write_json(out/'reports/frozen_before_third_seed.json',dict(baseline1=base1,baseline2=base2,
                   confirmation=confirmed,frozen=frozen,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        base3=None
        passed=[]
        if frozen:
            needed=sorted(set(PARENT_SPEC['weights'])|{k for r in frozen for k in specs[r['candidate']]['weights']})
            third,folds3=cv(needed,train_e,x,y,groups,2361,out/'predictions/full_seed2361.npz')
            base3=evaluate(PARENT_CONFIG,PARENT_SPEC,third,ids,y,folds3,target)
            for r in frozen:
                conf={k:r[k] for k in ('candidate','fraction','alpha')}
                m=evaluate(conf,specs[r['candidate']],third,ids,y,folds3,target)
                delta,wins=paired(m,base3)
                r.update(phase3=m,third_improvement=delta,third_fold_wins=wins,
                         three_seed_mean_improvement=(2*r['mean_improvement']+delta)/3,
                         final_eligible=bool(delta>=0 and wins>=4 and (2*r['mean_improvement']+delta)/3>=.001))
                if r['final_eligible']: passed.append(r)
        write_json(out/'reports/final_selection.json',dict(baseline1=base1,baseline2=base2,baseline3=base3,
                   frozen_candidates=frozen,passed=passed,no_retuning_after_third_seed=True))
        # Fit each required component once on all training rows, then export frozen mixtures.
        models,predictions={},{}
        for name in sorted({k for r in passed for k in specs[r['candidate']]['weights']}):
            print(f'V7 full-data fit: {name}',flush=True)
            models[name]=fit(name,train_e,x,y,np.arange(len(y)))
            predictions[name]=predict(name,models[name],test_e,xt,np.arange(len(xt)))
        for number,r in enumerate(passed,1):
            spec=specs[r['candidate']]
            values=aggregate({k:predictions[k] for k in spec['weights']},spec)
            r['submission_file']=f'submissions/v7_candidate_{number}.csv'
            r['model_file']=f'models/v7_candidate_{number}.joblib'
            r['submission_check']=write_submission(out/r['submission_file'],data,values,r)
            used={k:models[k] for k in spec['weights']}
            encoders={HEADS[k]['encoder'] if k in HEADS else (META_HEADS[k]['encoder'] if k in META_HEADS else encoder_for(k)) for k in used}
            bundle=dict(format_version=7,candidate=r['candidate'],spec=spec,fraction=r['fraction'],alpha=r['alpha'],
                        models=used,versions=versions(),encoders={k:manifests[k] for k in encoders},
                        train_sha256=data.audit['files']['train']['sha256'],raw_test_sha256=data.audit['files']['test']['sha256'],
                        test_ids=[row['id'] for row in data.test])
            path=out/r['model_file']
            path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(bundle,path,compress=3)
            write_csv(out/f'predictions/v7_candidate_{number}_test.csv',
                      [dict(id=row['id'],ranking_score=float(v[0]),probability_read=float(v[1])) for row,v in zip(data.test,values)],
                      ['id','ranking_score','probability_read'])
    result=dict(baseline1=base1,baseline2=base2,baseline3=base3,confirmation=confirmed,selected=passed,
                runtime_seconds=round(time.perf_counter()-start,3),untouched_holdout=False,limitation=plan['limitation'])
    write_json(out/'reports/results.json',result)
    root=Path(__file__).resolve().parents[1]
    files=[root/'v7.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),versions=versions(),
               input_files=data.audit['files'],code_sha256={str(p.relative_to(root)):sha256(p) for p in files},caches=caches,
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {out}; CSV count: {len(passed)}',flush=True)
    return result


def predict_saved(bundle_path,data,output,assets,cache,refit=False):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle=joblib.load(bundle_path)
    if bundle['format_version']!=7 or bundle['versions']!=versions():
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
