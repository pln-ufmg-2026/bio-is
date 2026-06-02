import os
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from run_generateSplit import main as run_split_main

def main():
    # Define datasets and methods to benchmark
    # You can expand these lists with more datasets and algorithms
    datasets = [
        "aisopos_ntua_2L",
        "mr",
        "subj",
        "vader_movie_2L",
        "pang_movie_2L"
    ]
    
    methods = [
        "bio-is",
        "drop3"
        # Add more methods like "cnn", "enn", "icf", "lssm", "lsbo", "drop3", etc.
    ]
    
    # Setup paths
    # Resolve to absolute paths to avoid issues when changing directories
    datain = Path("resources/datasets").resolve()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    out = Path("dataset_output") / timestamp
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {datain}")
    print(f"Output directory: {out}")
    print(f"Datasets to evaluate: {len(datasets)}")
    print(f"Methods to evaluate: {len(methods)}")
    print("-" * 50)
    
    # Calculate total iterations
    total_iterations = len(datasets) * len(methods)
    
    csv_dict = {}
    
    # Progress bar using tqdm
    with tqdm(total=total_iterations, desc="Benchmarking") as pbar:
        for dataset in datasets:
            for method in methods:
                pbar.set_postfix(dataset=dataset, method=method)
                
                # Command arguments to pass
                args_list = [
                    "-d", dataset,
                    "-m", method,
                    "--datain", str(datain),
                    "--out", str(out)
                ]
                
                # Run the function directly
                try:
                    run_split_main(args_list)
                    csv_path = out / "selection" / dataset / f"instance_selection_{method}" / f"{dataset}.csv"
                    if csv_path.exists():
                        csv_dict[(method, dataset)] = csv_path
                except Exception as e:
                    tqdm.write(f"\nError running {method} on {dataset}:")
                    tqdm.write(str(e))
                    
                pbar.update(1)
                
    # Combine collected CSVs into a unified CSV at the root of the output folder
    if csv_dict:
        all_dfs = []
        for (method, dataset), csv_path in csv_dict.items():
            try:
                df = pd.read_csv(csv_path)
                df['dataset'] = dataset
                df['algorithm'] = method
                all_dfs.append(df)
            except Exception as e:
                tqdm.write(f"\nError reading CSV {csv_path}: {e}")
        
        if all_dfs:
            unified_df = pd.concat(all_dfs, ignore_index=True)
            unified_csv_path = out / "unified_output.csv"
            unified_df.to_csv(unified_csv_path, index=False)
            tqdm.write(f"\nSaved unified CSV with all data to: {unified_csv_path}")

    print(f"\nBenchmark completed! Results saved to: {out}")
    
    # Update _latest folder
    latest_dir = Path("dataset_output") / "_latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    try:
        shutil.copytree(out, latest_dir)
        print(f"Updated latest results at: {latest_dir}")
    except Exception as e:
        print(f"Failed to update _latest folder: {e}")

if __name__ == "__main__":
    main()
