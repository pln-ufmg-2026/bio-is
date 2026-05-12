import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from run_generateSplit import main as run_split_main

def main():
    # Define datasets and methods to benchmark
    # You can expand these lists with more datasets and algorithms
    datasets = [
        "aisopos_ntua_2L",
        # Add more datasets here
    ]
    
    methods = [
        "bio-is",
        # Add more methods like "cnn", "enn", "icf", "lssm", "lsbo", "drop3", etc.
    ]
    
    # Setup paths
    # Resolve to absolute paths to avoid issues when changing directories
    datain = Path("resources/datasets").resolve()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    out = Path("output") / timestamp
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {datain}")
    print(f"Output directory: {out}")
    print(f"Datasets to evaluate: {len(datasets)}")
    print(f"Methods to evaluate: {len(methods)}")
    print("-" * 50)
    
    # Calculate total iterations
    total_iterations = len(datasets) * len(methods)
    
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
                except Exception as e:
                    tqdm.write(f"\nError running {method} on {dataset}:")
                    tqdm.write(str(e))
                    
                pbar.update(1)
                
    print(f"\nBenchmark completed! Results saved to: {out}")

if __name__ == "__main__":
    main()
