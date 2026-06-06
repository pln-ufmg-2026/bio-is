#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS_ROOT="${PROJECT_ROOT}/datasets"
CL_IS_ROOT="${DATASETS_ROOT}/CL + IS"
FT_ROOT="${DATASETS_ROOT}/FT"
IS_ROOT="${DATASETS_ROOT}/IS"
OUTPUT_ROOT="${PROJECT_ROOT}/output"
RUN_OUTPUT_ROOT="${PROJECT_ROOT}/.tmp/finetuning_outputs"

if [[ -z "${SEED:-}" ]]; then
    read -rp "Enter seed (integer): " SEED
    if [[ ! "${SEED}" =~ ^-?[0-9]+$ ]]; then
        echo "Error: seed must be an integer" >&2
        exit 1
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
FINETUNING_SCRIPT="${SCRIPT_DIR}/finetuning.py"
FINETUNING_ARGS=(
    --seed "${SEED}"
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
        local run_output_dir
        local metrics_output
        local batch_metrics_output

        file_name="$(basename "${dataset_csv}")"
        dataset_name="${file_name%.csv}"
        run_output_dir="${RUN_OUTPUT_ROOT}/${output_group}/${dataset_name}_${training_suffix}"
        metrics_output="${OUTPUT_ROOT}/${output_group}/${dataset_name}_out.csv"
        batch_metrics_output="${OUTPUT_ROOT}/${output_group}/${dataset_name}_batch_out.csv"

        echo "-------------------------------------------------------------------------"
        echo "Running ${output_group}/${file_name} (${training_suffix})"
        echo "Saving metrics to ${metrics_output}"
        echo "-------------------------------------------------------------------------"

        "${PYTHON_BIN}" "${FINETUNING_SCRIPT}" \
            --dataset-path "${dataset_csv}" \
            --output-dir "${run_output_dir}" \
            "${FINETUNING_ARGS[@]}" \
            "${extra_args[@]}"

        cp "${run_output_dir}/metrics.csv" "${metrics_output}"
        cp "${run_output_dir}/batch_metrics.csv" "${batch_metrics_output}"
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
