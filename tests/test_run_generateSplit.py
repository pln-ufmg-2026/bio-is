import pytest
from pathlib import Path
from run_generateSplit import main

# List of all available methods
METHODS = [
    'cnn', 'enn', 'icf', 'lssm', 'lsbo', 'drop3', 
    'ldis', 'cdis', 'xldis', 'psdsp', 'ib3', 'egdis', 
    'cis', 'e2sc', 'e2sc-1', 'e2sc-2', 'bio-is'
]

@pytest.mark.parametrize("method", METHODS)
def test_generate_split(method, tmp_path):
    """
    Table-driven test for all methods using the aisopos_ntua_2L dataset.
    The tmp_path fixture provides a cleaned temporary directory for each test run.
    """
    
    dataset_name = "aisopos_ntua_2L"
    
    # We point datain to the resources/datasets directory
    datain_dir = Path("resources/datasets").absolute()
    
    # The output directory is created cleanly by pytest in tmp_path
    output_dir = tmp_path / "out"
    
    args = [
        "--method", method,
        "--dataset", dataset_name,
        "--datain", str(datain_dir),
        "--out", str(output_dir),
        "--folds", "10"
    ]
    
    # Run the main function with the arguments
    main(args)
    
    # Check if the output directory and expected output file were created
    expected_selection_dir = output_dir / "selection" / dataset_name / f"instance_selection_{method}"
    assert expected_selection_dir.exists(), f"Output directory {expected_selection_dir} was not created"
    
    # The split file name pattern from run_generateSplit.py:
    expected_split_file = expected_selection_dir / f"split_10_{method}.pkl"
    expected_split_file_idx = expected_selection_dir / f"split_10_{method}_idxinfold.pkl"
    expected_csv_file = expected_selection_dir / f"{dataset_name}.csv"
    
    assert expected_split_file.exists() or expected_split_file_idx.exists(), f"Output split file for method {method} was not created"
    assert expected_csv_file.exists(), f"Output CSV file for method {method} was not created"
