#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS_ROOT="${PROJECT_ROOT}/datasets"
CL_IS_ROOT="${DATASETS_ROOT}/CL + IS"
FT_ROOT="${DATASETS_ROOT}/FT"
IS_ROOT="${DATASETS_ROOT}/IS"
OUTPUT_ROOT="${PROJECT_ROOT}/output"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FINETUNING_SCRIPT="${SCRIPT_DIR}/finetuning.py"
FINETUNING_ARGS=(
    --batch-size 32
    --no-cv
    --epochs 10
    --text-column text
    --target-column score
    --difficulty-column difficulty
)
USER_ARGS=("$@")

if [[ ! -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    echo "Virtual environment not found: ${PROJECT_ROOT}/.venv" >&2
    exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

run_dataset_group() {
    local source_dir="$1"
    local output_group="$2"
    local training_suffix="$3"
    shift 3
    local extra_args=("$@")

    if [[ ! -d "${source_dir}" ]]; then
        echo "Dataset directory not found: ${source_dir}" >&2
        return 1
    fi

    mkdir -p "${OUTPUT_ROOT}/${output_group}"

    shopt -s nullglob
    local dataset_csv
    for dataset_csv in "${source_dir}"/*.csv; do
        local file_name
        local dataset_name
        local output_dir

        file_name="$(basename "${dataset_csv}")"
        dataset_name="${file_name%.csv}"
        output_dir="${OUTPUT_ROOT}/${output_group}/${dataset_name}_${training_suffix}"

        echo "-------------------------------------------------------------------------"
        echo "Running ${output_group}/${file_name} (${training_suffix})"
        echo "Saving to ${output_dir}"
        echo "-------------------------------------------------------------------------"

        "${PYTHON_BIN}" "${FINETUNING_SCRIPT}" \
            --dataset-path "${dataset_csv}" \
            --output-dir "${output_dir}" \
            "${FINETUNING_ARGS[@]}" \
            "${extra_args[@]}"
    done
    shopt -u nullglob
}

cd "${PROJECT_ROOT}"

run_dataset_group "${CL_IS_ROOT}" "CL+IS" "curriculum" \
    --curriculum-learning \
    --difficulty-column difficulty \
    "${USER_ARGS[@]}"

run_dataset_group "${FT_ROOT}" "FT" "finetuning" \
    "${USER_ARGS[@]}"

run_dataset_group "${IS_ROOT}" "IS" "instance_selection" \
    "${USER_ARGS[@]}"
