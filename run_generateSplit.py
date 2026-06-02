from src.main.python.utils.general import get_data, get_splits, checkpoint_splits, translate_train_idxinfold
from src.main.python.utils.arguments import arguments
from src.main.python.utils.save_results import save_results
import argparse
from datetime import datetime
import time
import numpy as np
import gc
import io
import os
import pandas as pd
from pathlib import Path
from collections import Counter
from src.main.python.iSel import cnn, enn, icf, lssm, lsbo, drop3, ldis, cdis, xldis, psdsp, ib3, cis, egdis, e2sc, biois, nosel, cl_biois

import socket

import logging
import logging.config

logging.config.fileConfig(Path('settings') / 'logging.conf', defaults={'logfilename': str(Path('resources') / 'logs' / f'{socket.gethostname()}.log')})
logger = logging.getLogger(__name__)

def get_selector(method: str):

    #Baselines
    if method == 'nosel':   return nosel.NoSel() # no instant selection, returns data as is
    if method == 'cnn':     return cnn.CNN()
    if method == 'enn':     return enn.ENN()
    if method == 'icf':     return icf.ICF()
    if method == 'lssm':    return lssm.LSSm()
    #if method == 'lsbo':    return lsbo.LSBo(args, fold)
    if method == 'lsbo':    return lsbo.LSBo()
    #if method == 'drop3':   return drop3.DROP3(args, fold, n_neighbors=3, loadenn=False)
    if method == 'drop3':   return drop3.DROP3()
    if method == 'ldis':    return ldis.LDIS()
    if method == 'cdis':    return cdis.CDIS()
    if method == 'xldis':   return xldis.XLDIS()
    if method == 'psdsp':   return psdsp.PSDSP()
    if method == 'ib3':     return ib3.IB3()
    if method == 'egdis':   return egdis.EGDIS()
    if method == 'cis':     return cis.CIS(task="atc")
    #proposed framework
    if method == 'e2sc':   return e2sc.E2SC()
    if method == 'e2sc-1':   return e2sc.E2SC(alphaMode="exact", betaMode='iterative')
    if method == 'e2sc-2':   return e2sc.E2SC(alphaMode="approximated", betaMode='heuristic')
    if method == 'bio-is':   return biois.BIOIS(beta=0.25, theta=0.50) # TODO change hyperparameters
    if method == 'cl-bio-is': return cl_biois.CLBIOIS(beta=0.25, theta=0.50, p_easy=50, p_med=80)

    print(f"Unknown method: {method}")

    exit()

def get_selection(X, y, fold, args):

    method = args.method

    total = Counter(y)
    logger.debug(f"Total instances number: {total}")

    selector = get_selector(method)

    selector.fit(X, y)

    logger.info("Result: ", Counter(y[selector.sample_indices_]))

    entropy = getattr(selector, 'entropy_', None)
    difficulty = getattr(selector, 'difficulty_', None)

    return selector.sample_indices_, entropy, difficulty


def load_original_dataset(datain_dir: str, dataset_name: str) -> pd.DataFrame:
    dataset_path = Path(datain_dir) / dataset_name
    csv_path = dataset_path / "dataset_with_dificulty.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    texts_file = dataset_path / "texts.txt"
    score_file = dataset_path / "score.txt"
    if texts_file.exists() and score_file.exists():
        texts = texts_file.read_text(encoding="utf-8").splitlines()
        scores = score_file.read_text(encoding="utf-8").splitlines()
        
        if len(texts) != len(scores):
            logger.warning(f"Mismatched dataset lengths for {dataset_name}: texts.txt has {len(texts)} lines, score.txt has {len(scores)} lines. Aligning to score.txt length.")
            print(f"Warning: Mismatched dataset lengths for {dataset_name}: texts.txt has {len(texts)} lines, score.txt has {len(scores)} lines. Aligning to score.txt length.")
            if len(texts) > len(scores):
                texts = texts[:len(scores)]
            else:
                texts = texts + [""] * (len(scores) - len(texts))
                
        return pd.DataFrame({
            "index": list(range(len(scores))),
            "text": texts,
            "score": scores
        })
    raise FileNotFoundError(f"Original dataset files not found for dataset {dataset_name} at {dataset_path}")


def save_reduced_csv(splits_to_save_df: pd.DataFrame, splits_to_save_df_translated: pd.DataFrame, args) -> None:
    original_df = load_original_dataset(args.datain, args.dataset)
    if original_df is None:
        return
        
    all_folds_df = []
    for f in range(len(splits_to_save_df_translated)):
        new_train_idxs_in_fold = splits_to_save_df.loc[f].train_idxs
        train_idxs_dataset_wide = splits_to_save_df_translated.loc[f].train_idxs
        
        if len(train_idxs_dataset_wide) == 0:
            continue
            
        filtered_df = original_df.iloc[list(train_idxs_dataset_wide)].copy()
        filtered_df['fold'] = f
        
        # Add entropy if applicable
        entropy_array = splits_to_save_df.loc[f].entropy
        if entropy_array is not None:
            selected_entropy = [entropy_array[t] for t in new_train_idxs_in_fold]
            filtered_df['entropy'] = selected_entropy
            
        all_folds_df.append(filtered_df)
        
    if all_folds_df:
        final_df = pd.concat(all_folds_df, ignore_index=True)
        csv_file = Path(args.outputdir) / f"{args.dataset}.csv"
        final_df.to_csv(csv_file, index=False)
        logger.info(f"Saved simple CSV output to {csv_file}")
        print(f"Saved simple CSV output to {csv_file}")


def main(args_list=None, debug=False):

    gc.collect()

    result = arguments(args_list)
    if result == (None, None):
        return
    args, info = result
    logger.info(str(args))

    split_file = str(Path(args.splitdir) / f"split_{args.folds}.pkl")
    print(split_file)
    splits_df = get_splits(split_file)

    splits_to_save = {c: [] for c in splits_df.columns if c.endswith("idxs")}
    splits_to_save['entropy'] = []
    splits_to_save['difficulty'] = []
       
    #for f in range(args.folds):
    for f in range(1):

        logger.info("Fold {}".format(f))
        print("Fold {}".format(f))
        
        splits_to_save['test_idxs'].append(splits_df.loc[f].test_idxs)


        X_train, y_train, _, _, _ = get_data(args.inputdir, f)
        t = len(y_train)

        ti = time.time()

        idxs_docs, entropy, difficulty = get_selection(X_train, y_train, f, args)

        s = len(y_train[idxs_docs])
        r = (t-s)/t

        info['time_for_reduce'].append(time.time() - ti)
        print(info['time_for_reduce'])
        info['original_len'].append(t)
        info['reduced_len'].append(s)
        info['reducion'].append(r)

        splits_to_save['train_idxs'].append(idxs_docs)
        splits_to_save['entropy'].append(entropy)
        splits_to_save['difficulty'].append(difficulty)

    logger.info(f"time: {np.mean(info['time_for_reduce'])}")
    logger.info(f"time std: {np.std(info['time_for_reduce'])}")
    logger.info(f"reducion: {np.mean(info['reducion'])}")
    logger.info(f"reducion std: {np.std(info['reducion'])}")

    splits_to_save_df = pd.DataFrame(data=splits_to_save)

    filename = str(Path(args.outputdir) / f"split_{args.folds}_{args.method}_idxinfold.pkl")

    checkpoint_splits(
        splits_df=splits_to_save_df,
        filename = filename
    )

    splits_to_save_df_traslated = translate_train_idxinfold(
        splits_to_save_df, splits_df)

    checkpoint_splits(
        splits_df=splits_to_save_df_traslated,
        filename=filename.replace("_idxinfold", "")
    )

    save_reduced_csv(splits_to_save_df, splits_to_save_df_traslated, args)

    if args.save:
        save_results(args, info)

    if debug:
        print("\n--- DEBUG INFO ---")
        
        def print_df_debug_info(df, name):
            print(f"\n{name} shape: {df.shape}")
            print(f"Data dimensions in the first row of {name}:")
            if not df.empty:
                first_row = df.iloc[0]
                for col in df.columns:
                    val = first_row[col]
                    if hasattr(val, 'shape'):
                        print(f"  {col}: shape {val.shape}")
                    elif hasattr(val, '__len__'):
                        print(f"  {col}: length {len(val)}")
                    else:
                        print(f"  {col}: type {type(val)}")
            print(f"\n{name} head:")
            print(df.head())

        print_df_debug_info(splits_to_save_df, "splits_to_save_df")
        print_df_debug_info(splits_to_save_df_traslated, "splits_to_save_df_traslated")
        
        print("------------------\n")
    
    print("END")
    return


if __name__ == '__main__':
    main()
