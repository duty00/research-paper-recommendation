"""Local semantic regression ensembles; manual Kaggle submission only."""
import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset,find_data_dir
from paper_priority.local_experiment import run,replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('run','reproduce'):
        sub=commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--out',type=Path,required=True)
        sub.add_argument('--assets',type=Path,default=Path('assets/encoders'))
        sub.add_argument('--cache',type=Path,default=Path('.cache/embeddings'))
        if name=='run':
            sub.add_argument('--reference',type=Path,default=Path('outputs/v6_metadata_run01'))
        else:
            sub.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    data=load_dataset(find_data_dir(args.data_dir))
    if args.command=='run':
        run(data,args.out,args.reference,args.assets,args.cache)
    else:
        replay(data,json.loads(args.config.read_text(encoding='utf-8')),args.out,args.assets,args.cache)


if __name__=='__main__':
    main()
