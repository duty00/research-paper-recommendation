"""Plot the README figures from the recorded aggregate experiment results.

Optional dependency: pip install -r requirements-figures.txt
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def comparison(data,out):
    names=['V5','V6 candidate 1','V6 candidate 2']
    labels=['V5','V6 C1','V6 C2']
    public={r['submission']:r['score'] for r in data['public_scores']}
    values=[public['V5 candidate 1'],public['V6 candidate 1'],public['V6 candidate 2']]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    axes[0].bar(labels,values,color=['tab:blue','tab:orange','tab:green'],width=.58)
    axes[0].set_ylim(0,.74)
    axes[0].set_ylabel('Public score')
    axes[0].set_title('Kaggle public leaderboard')
    for i,value in enumerate(values):
        axes[0].text(i,value+.012,f'{value:.5f}',ha='center',fontsize=10)

    all_folds=[data['cv_fold_scores'][name]['361']+data['cv_fold_scores'][name]['1361'] for name in names]
    axes[1].boxplot(all_folds,positions=range(3),widths=.5,showfliers=False,
                    medianprops={'color':'0.45'},boxprops={'color':'0.6'},whiskerprops={'color':'0.6'},capprops={'color':'0.6'})
    for i,name in enumerate(names):
        for seed,offset,color in [('361',-.11,'tab:blue'),('1361',.11,'tab:orange')]:
            folds=data['cv_fold_scores'][name][seed]
            positions=[i+offset+(j-3)*.014 for j in range(7)]
            axes[1].scatter(positions,folds,color=color,s=21,alpha=.9,label=f'Seed {seed}' if i==0 else None,zorder=3)
        axes[1].scatter(i,mean(all_folds[i]),color='black',marker='D',s=25,label='Mean' if i==0 else None,zorder=4)
    axes[1].set_xticks(range(3),labels)
    axes[1].set_ylabel('Validation score')
    axes[1].set_ylim(.575,.69)
    axes[1].yaxis.set_major_locator(MultipleLocator(.02))
    axes[1].set_title('Repeated 7-fold group cross-validation')
    axes[1].legend(loc='upper left',fontsize=8,ncol=3,frameon=False,handletextpad=.3,columnspacing=.8)
    axes[1].grid(axis='y',alpha=.18)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.text(.5,.015,'Each CV dot is one observed fold score; both seeds reuse the same training papers.',ha='center',fontsize=9)
    fig.savefig(out/'02_model_comparison.png',dpi=150)
    plt.close(fig)


def overview(data,out):
    counts=data['data_counts']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    labels=[str(i) for i in range(1,6)]
    values=[counts['train_labels'][label] for label in labels]
    axes[0].bar(labels,values,color='tab:blue',width=.65)
    axes[0].set_title('Training label distribution')
    axes[0].set_xlabel('Preference rating')
    axes[0].set_ylabel('Papers')
    axes[0].set_ylim(0,max(values)*1.15)
    for i,value in enumerate(values):
        axes[0].text(i,value+16,str(value),ha='center',fontsize=10)
    venues=sorted(set(counts['train_venues'])|set(counts['test_venues']))
    positions=list(range(len(venues)))
    axes[1].bar([i-.18 for i in positions],[counts['train_venues'].get(v,0) for v in venues],width=.36,color='tab:blue',label='Train')
    axes[1].bar([i+.18 for i in positions],[counts['test_venues'].get(v,0) for v in venues],width=.36,color='tab:orange',label='Test')
    axes[1].set_xticks(positions,venues)
    axes[1].set_title('Papers by venue')
    axes[1].set_ylabel('Papers')
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'01_data_overview.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/figure_data.json')
    parser.add_argument('--out',type=Path,default=ROOT/'results/figures')
    args=parser.parse_args()
    data=json.loads(args.data.read_text(encoding='utf-8'))
    assert sum(data['data_counts']['train_labels'].values())==data['data_counts']['train_rows']==2494
    assert sum(data['data_counts']['test_venues'].values())==data['data_counts']['test_rows']==596
    assert all(len(values)==7 for record in data['cv_fold_scores'].values() for values in record.values())
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','savefig.facecolor':'white'})
    overview(data,args.out)
    comparison(data,args.out)
    manifest={
        'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'plotting_versions':{name:importlib.metadata.version(name) for name in ('matplotlib','numpy')},
        'cv_means':{name:mean(record['361']+record['1361']) for name,record in data['cv_fold_scores'].items()},
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.glob('*.png'))},
    }
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
