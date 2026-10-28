import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset,find_data_dir
from paper_priority.v7_experiment import run,predict_saved


def main():
    parser=argparse.ArgumentParser(description='V7 metadata kernel refinement with a frozen third CV split check.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('run','predict'):
        sub=commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--assets',type=Path,default=Path('assets/encoders'))
        sub.add_argument('--cache',type=Path,default=Path('.cache/embeddings'))
        if name=='run':
            sub.add_argument('--out',type=Path,required=True)
            sub.add_argument('--reference',type=Path,default=Path('outputs/v6_metadata_run01'))
            sub.add_argument('--parent-export',type=Path,default=Path('outputs/v6_candidate2_run01'))
        else:
            sub.add_argument('--bundle',type=Path,required=True)
            sub.add_argument('--output',type=Path,required=True)
            sub.add_argument('--refit',action='store_true')
    args=parser.parse_args()
    data=load_dataset(find_data_dir(args.data_dir))
    if args.command=='run':
        run(data,args.out,args.reference,args.parent_export,args.assets,args.cache)
    else:
        print(json.dumps(predict_saved(args.bundle,data,args.output,args.assets,args.cache,args.refit),indent=2))


if __name__=='__main__':
    main()
