
from src.main.python.utils.general import str2bool
from datetime import datetime
import argparse
import os
from pathlib import Path
import random


def check_if_split_exists(args):

    # if args.sel == "":
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_idxinfold.csv"
    # else:
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_"+args.sel+"_idxinfold.csv"

    saida = args.filename+".json"

    if os.path.exists(saida):
        print("Already exists selection output")
        return True
    return False


def arguments(args_list=None):
    # datasets/webkb/tfidf/ --splitdir datasets/webkb/ --outputdir output/webkb/cnn/
    parser = argparse.ArgumentParser(description='Generate baseline splits.')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-m', "--method", type=str, help='selection method')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument(
        '--save', type=lambda x: bool(str2bool(x)), default=True)
    parser.add_argument("--inputrep", type=str, default="tfidf")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datain", required=True)
    parser.add_argument("--overwrite", default=0)

    args = parser.parse_args(args_list)

    args.inputdir = str(Path(args.datain) / args.dataset / args.inputrep)
    args.splitdir = str(Path(args.datain) / args.dataset / "splits")
    args.outputdir = str(Path(args.out) / "selection" / args.dataset / f"instance_selection_{args.method}")

    args.filename = str(Path(args.outputdir) / f"saida_{args.method}")

    args.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print(args)
    if check_if_split_exists(args):
        return None, None

    output_path = Path(args.outputdir)
    if not output_path.exists():
        print(f"Criando saida {args.outputdir}")
        output_path.mkdir(parents=True, exist_ok=True)

    with open(args.filename, "w") as arq:
        arq.write(f"{args.method}\n{args}\n")

    random.seed(1608637542)

    info = {
        "reducion": [],
        "time_for_reduce": [],
        "original_len": [],
        "reduced_len": [],
    }

    return args, info
