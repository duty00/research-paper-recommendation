"""V4 selection against v3, using development only and frozen confirmation."""
import json
import time
import warnings
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .advanced import combine_components,make_features,policy_priorities
from .data import sha256,write_json,write_csv
from .embeddings import ensure_encoder
from .encoders_v4 import EXTRA_ENCODERS,ensure_extra,extra_cache
from .experiment import assert_disjoint_groups
from .fusion_models import SPECS,WEIGHTS,JointOrdinal,V3Baseline
from .improvement import original_partition,evaluate_config,write_v2_submission
from .metric import score_priorities
from .models import rows_to_x
from .semantic_models import SemanticHead
from .semantic_experiment import versions

SEEDS=(262,1262)
FRACTIONS=(.55,.60,.65,.70,.75,.80,.85,.90)
POLICIES=('gain','utility_half')


def baseline_oof(reference_v3,seed,dev,data):
    manifest=json.loads((reference_v3/'manifest.json').read_text(encoding='utf-8'))
    for role in ('train','test'):
        if manifest['input_files'][role]['sha256']!=data.audit['files'][role]['sha256']:
            raise ValueError('V3 reference data changed.')
    relative=f'predictions/development_seed{seed}.npz'
    reference_hashes={key.replace('\\','/'):value for key,value in manifest['output_sha256'].items()}
    if sha256(reference_v3/relative)!=reference_hashes[relative]:
        raise ValueError('V3 OOF predictions changed.')
    with np.load(reference_v3/relative,allow_pickle=False) as values:
        if not np.array_equal(values['ids'],dev):
            raise ValueError('V3 development row order changed.')
        return .5*values['v2_title']+.5*values['e5_ordinal_meta_c4'],values['fold']


def fit(name,embeddings,x,y,indices):
    if name=='old_v3':
        return V3Baseline().fit(embeddings['e5'][indices],x[indices],y[indices])
    spec=SPECS[name]
    cls=JointOrdinal if spec['mode']=='joint' else SemanticHead
    return cls(spec,261).fit(embeddings[spec['encoder']][indices],x[indices],y[indices])


def predict(name,model,embeddings,x,indices):
    encoder='e5' if name=='old_v3' else SPECS[name]['encoder']
    return model.predict_components(embeddings[encoder][indices],x[indices])


def cv(names,embeddings,x,y,groups,dev,seed,baseline,expected_folds,path):
    output={name:np.zeros((len(dev),2)) for name in names}
    output['old_v3']=baseline
    folds=np.full(len(dev),-1,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for fold,(train_local,val_local) in enumerate(splitter.split(x[dev],y[dev],groups[dev])):
        train,val=dev[train_local],dev[val_local]
        assert_disjoint_groups(groups,train,val)
        folds[val_local]=fold
        feature_cache={}
        for name in names:
            spec=SPECS[name]
            print(f'V4 seed {seed}, fold {fold+1}/5: {name}',flush=True)
            if spec['mode']=='joint':
                profile=spec['profile']
                if profile not in feature_cache:
                    features=make_features(profile)
                    words=features.fit_transform(x[train])
                    feature_cache[profile]=(features,words,features.transform(x[val]))
                features,words,val_words=feature_cache[profile]
                model=JointOrdinal(spec,261).fit_transformed(features,words,embeddings[spec['encoder']][train],y[train])
                output[name][val_local]=model.predict_transformed(embeddings[spec['encoder']][val],val_words)
            else:
                model=fit(name,embeddings,x,y,train)
                output[name][val_local]=predict(name,model,embeddings,x,val)
    if not np.array_equal(folds,expected_folds):
        raise ValueError('V4 and cached V3 validation folds differ.')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,indices=dev,fold=folds,**output)
    return output,folds


def family(candidate):
    return '|'.join(sorted({SPECS[k]['encoder']+':'+SPECS[k]['mode'] for k in WEIGHTS[candidate] if k in SPECS}))


def run_fusion(data,out,reference,reference_v3,assets,cache):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output folder.')
    dev,hold,groups,_=original_partition(data,reference)
    b1,f1=baseline_oof(reference_v3,262,dev,data)
    b2,f2=baseline_oof(reference_v3,1262,dev,data)
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    embeddings,test_embeddings,assets_report,cache_report={},{},{},{}
    for encoder in ('e5','e5base','bge'):
        _,assets_report[encoder]=(ensure_encoder(encoder,assets) if encoder=='e5' else ensure_extra(encoder,assets))
        embeddings[encoder],a=extra_cache(data.train,encoder,assets,cache)
        test_embeddings[encoder],b=extra_cache(data.test,encoder,assets,cache)
        cache_report[encoder]=dict(train=a,test=b)
    x,xt=rows_to_x(data.train),rows_to_x(data.test)
    y=np.asarray([r['Label'] for r in data.train],dtype=int)
    ids=np.asarray([r['id'] for r in data.train])
    venues=sorted({r['venue'] for r in data.test})
    target=np.asarray([r['venue'] in venues for r in data.train])
    plan=dict(specs=SPECS,weights=WEIGHTS,fractions=FRACTIONS,policies=POLICIES,seeds=SEEDS,
              encoder_assets=assets_report,caches=cache_report,input_files=data.audit['files'],
              reference_v3_manifest_sha256=sha256(reference_v3/'manifest.json'),
              rule='Select best configuration per candidate on seed262; shortlist top3 distinct encoder/mode families. Freeze fraction/policy; confirm seed1262. Both seeds beat v3, mean delta >= .002 and >=3/5 paired fold wins each. At most two; second gain correlation < .98.',
              public_scores_used_for_selection=False,old_holdout='Diagnostic only; already seen in previous versions.')
    write_json(out/'reports/search_plan.json',plan)
    warnings.filterwarnings('error',category=ConvergenceWarning)
    baseline_config=dict(candidate='old_v3',fraction=.65,policy='gain')
    with threadpool_limits(limits=1):
        first,folds1=cv(list(SPECS),embeddings,x,y,groups,dev,262,b1,f1,out/'predictions/development_seed262.npz')
        baseline1=evaluate_config(baseline_config,b1,ids[dev],y[dev],folds1,target[dev])
        all_results,ranking=[],[]
        for name,weights in WEIGHTS.items():
            components=combine_components(first,weights)
            records=[evaluate_config(dict(candidate=name,fraction=f,policy=p),components,ids[dev],y[dev],folds1,target[dev])
                     for f in FRACTIONS for p in POLICIES]
            all_results.extend(records)
            ranking.append(max(records,key=lambda r:(r['score'],r['fraction'],r['policy'])))
        ranking.sort(key=lambda r:(-r['score'],r['candidate']))
        shortlist=[]
        families=set()
        for record in ranking:
            key=family(record['candidate'])
            if key not in families:
                families.add(key)
                shortlist.append(record)
            if len(shortlist)==3:
                break
        write_json(out/'reports/phase1.json',dict(baseline=baseline1,ranking=ranking,shortlist=shortlist))
        write_csv(out/'reports/search_results.csv',[{k:v for k,v in r.items() if k!='fold_scores'} for r in all_results],
                  ['candidate','fraction','policy','score','relevance','coverage','fold_score_std'])
        names=sorted({k for r in shortlist for k in WEIGHTS[r['candidate']] if k!='old_v3'})
        second,folds2=cv(names,embeddings,x,y,groups,dev,1262,b2,f2,out/'predictions/development_seed1262.npz')
        baseline2=evaluate_config(baseline_config,b2,ids[dev],y[dev],folds2,target[dev])
        confirmed=[]
        for r in shortlist:
            config={k:r[k] for k in ('candidate','fraction','policy')}
            m=evaluate_config(config,combine_components(second,WEIGHTS[r['candidate']]),ids[dev],y[dev],folds2,target[dev])
            mean=(r['score']+m['score'])/2
            delta=mean-(baseline1['score']+baseline2['score'])/2
            wins=[int(np.sum(np.asarray(r['fold_scores'])>baseline1['fold_scores'])),
                  int(np.sum(np.asarray(m['fold_scores'])>baseline2['fold_scores']))]
            eligible=bool(delta>=.002 and r['score']>baseline1['score'] and m['score']>baseline2['score'] and min(wins)>=3)
            confirmed.append(dict(**config,phase1=r,phase2=m,mean_score=mean,mean_improvement=delta,paired_fold_wins=wins,eligible=eligible))
        confirmed.sort(key=lambda r:(-r['mean_score'],r['candidate']))
        selected=[]
        for r in confirmed:
            if not r['eligible']:
                continue
            if selected:
                a=combine_components(first,WEIGHTS[selected[0]['candidate']])[target[dev],0]
                b=combine_components(first,WEIGHTS[r['candidate']])[target[dev],0]
                r['correlation_with_first']=float(np.corrcoef(a,b)[0,1])
                if r['correlation_with_first']>=.98:
                    continue
            selected.append(r)
            if len(selected)==2:
                break
        frozen=dict(baseline_phase1=baseline1,baseline_phase2=baseline2,confirmation=confirmed,selected=selected,
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(),frozen_before_holdout=True)
        write_json(out/'reports/selection.json',frozen)
        needed=sorted({k for r in selected for k in WEIGHTS[r['candidate']]})
        models,hold_prediction,test_prediction={},{},{}
        for name in needed:
            print(f'Diagnostic holdout: {name}',flush=True)
            model=fit(name,embeddings,x,y,dev)
            hold_prediction[name]=predict(name,model,embeddings,x,hold)
            print(f'Full-data refit: {name}',flush=True)
            models[name]=fit(name,embeddings,x,y,np.arange(len(y)))
            test_prediction[name]=predict(name,models[name],test_embeddings,xt,np.arange(len(xt)))
        for n,r in enumerate(selected,1):
            weights=WEIGHTS[r['candidate']]
            mask=target[hold]
            c=combine_components(hold_prediction,weights)
            r['diagnostic_holdout']=score_priorities(ids[hold][mask],y[hold][mask],policy_priorities(ids[hold][mask],c[mask],r['fraction'],r['policy']))
            test_c=combine_components(test_prediction,weights)
            r['submission_file']=f'submissions/v4_candidate_{n}.csv'
            r['model_file']=f'models/v4_candidate_{n}.joblib'
            r['submission_check']=write_v2_submission(out/r['submission_file'],data,test_c,r)
            encoders={('e5' if k=='old_v3' else SPECS[k]['encoder']) for k in weights}
            bundle=dict(format_version=4,candidate=r['candidate'],weights=weights,fraction=r['fraction'],policy=r['policy'],
                        models={k:models[k] for k in weights},versions=versions(),
                        encoders={e:assets_report[e] for e in encoders},
                        train_sha256=data.audit['files']['train']['sha256'],raw_test_sha256=data.audit['files']['test']['sha256'],
                        test_ids=[row['id'] for row in data.test])
            path=out/r['model_file']
            path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(bundle,path,compress=3)
            write_csv(out/f'predictions/v4_candidate_{n}_test.csv',
                      [dict(id=row['id'],gain=float(c[0]),probability_read=float(c[1])) for row,c in zip(data.test,test_c)],
                      ['id','gain','probability_read'])
    results=dict(**frozen,runtime_seconds=round(time.perf_counter()-start,3),
                 limitation='CV selection is adaptive and both seeds reuse development. Old holdout is diagnostic, not independent evidence. No hidden labels/membership inferred.')
    write_json(out/'reports/results.json',results)
    lines=['# V4 development results','','| Candidate | Seed262 | Seed1262 | Mean | Delta vs v3 | Wins | Eligible |',
           '|---|---:|---:|---:|---:|---|---|',
           f"| v3 | {baseline1['score']:.6f} | {baseline2['score']:.6f} | {(baseline1['score']+baseline2['score'])/2:.6f} | 0 | baseline | baseline |"]
    for r in confirmed:
        lines.append(f"| {r['candidate']} | {r['phase1']['score']:.6f} | {r['phase2']['score']:.6f} | {r['mean_score']:.6f} | {r['mean_improvement']:+.6f} | {r['paired_fold_wins']} | {r['eligible']} |")
    lines+=['','## Selected','']
    for n,r in enumerate(selected,1):
        lines.append(f"- v4_candidate_{n}.csv: {r['candidate']}; fraction {r['fraction']}; policy {r['policy']}; diagnostic old holdout {r['diagnostic_holdout']['score']:.6f}.")
    if not selected:
        lines.append('No model passed the fixed validation gate; no CSV produced.')
    lines+=['',results['limitation']]
    (out/'reports/summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    sources=[root/'fusion.py',root/'requirements-semantic.txt',*sorted((root/'paper_priority').glob('*.py'))]
    write_json(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),packages=versions(),input_files=data.audit['files'],
               code_sha256={str(p.relative_to(root)):sha256(p) for p in sources},
               output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'Done: {len(selected)} new candidates in {out}',flush=True)
    return results


def predict_fusion(bundle_path,data,output,assets,cache):
    if output.resolve() in {p.resolve() for p in data.paths.values()}:
        raise ValueError('Do not overwrite raw data.')
    bundle=joblib.load(bundle_path)
    if bundle['format_version']!=4 or bundle['versions']!=versions():
        raise ValueError('Wrong bundle or dependency versions.')
    if bundle['train_sha256']!=data.audit['files']['train']['sha256'] or bundle['raw_test_sha256']!=data.audit['files']['test']['sha256']:
        raise ValueError('Raw data changed.')
    if bundle['test_ids']!=[r['id'] for r in data.test]:
        raise ValueError('Test IDs changed.')
    embeddings={}
    for encoder,recorded in bundle['encoders'].items():
        _,current=(ensure_encoder(encoder,assets) if encoder=='e5' else ensure_extra(encoder,assets))
        if current!=recorded:
            raise ValueError('Encoder assets changed.')
        embeddings[encoder],_=extra_cache(data.test,encoder,assets,cache)
    with threadpool_limits(limits=1):
        x=rows_to_x(data.test)
        predictions={k:predict(k,m,embeddings,x,np.arange(len(x))) for k,m in bundle['models'].items()}
        return write_v2_submission(output,data,combine_components(predictions,bundle['weights']),bundle)
