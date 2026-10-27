import argparse
import json
from pathlib import Path
from paper_priority.data import load_dataset,find_data_dir
from paper_priority.refine_experiment import run_refinement,predict_refinement


def main():
    parser=argparse.ArgumentParser(description='V5 model and ensemble refinement with a frozen third CV split check.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('run','predict'):
        sub=commands.add_parser(name)
        sub.add_argument('--data-dir')
        sub.add_argument('--assets',type=Path,default=Path('assets/encoders'))
        sub.add_argument('--cache',type=Path,default=Path('.cache/embeddings'))
        if name=='run':
            sub.add_argument('--out',type=Path,required=True)
            sub.add_argument('--reference',type=Path,default=Path('outputs/my_run01'))
            sub.add_argument('--reference-v4',type=Path,default=Path('outputs/v4_run01'))
        else:
            sub.add_argument('--bundle',type=Path,required=True)
            sub.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=load_dataset(find_data_dir(args.data_dir))
    if args.command=='run':
        run_refinement(data,args.out,args.reference,args.reference_v4,args.assets,args.cache)
    else:
        print(json.dumps(predict_refinement(args.bundle,data,args.output,args.assets,args.cache),indent=2))


if __name__=='__main__':
    main()
