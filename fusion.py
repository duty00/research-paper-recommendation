import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset,find_data_dir
from paper_priority.encoders_v4 import EXTRA_ENCODERS,ensure_extra,extra_cache
from paper_priority.fusion_experiment import run_fusion,predict_fusion


def main():
    parser=argparse.ArgumentParser(description='V4 joint feature and larger-encoder experiments; no Kaggle upload.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','predict'):
        sub=commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--assets',type=Path,default=Path('assets/encoders'))
        sub.add_argument('--cache',type=Path,default=Path('.cache/embeddings'))
        if name=='run':
            sub.add_argument('--out',type=Path,required=True)
            sub.add_argument('--reference',type=Path,default=Path('outputs/my_run01'))
            sub.add_argument('--reference-v3',type=Path,default=Path('outputs/v3_run01'))
        elif name=='predict':
            sub.add_argument('--bundle',type=Path,required=True)
            sub.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=load_dataset(find_data_dir(args.data_dir))
    if args.command=='prepare':
        for encoder in EXTRA_ENCODERS:
            ensure_extra(encoder,args.assets,download=True)
            for rows in (data.train,data.test):
                extra_cache(rows,encoder,args.assets,args.cache)
    elif args.command=='run':
        run_fusion(data,args.out,args.reference,args.reference_v3,args.assets,args.cache)
    else:
        print(json.dumps(predict_fusion(args.bundle,data,args.output,args.assets,args.cache),indent=2))


if __name__=='__main__':
    main()
