#!/bin/bash
#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"
readonly MODEL_NAME="${1:-gpt2}";shift
readonly SEED_NUM=1
# readonly DATASET_NAME="${1:-sst}";shift

TOKENIZERS_PARALLELISM=false


# ==========
# Exp Results

function run_gpt2() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=0 python -m src.python.main \
                                --run trainer \
                                --model_name ${MODEL_NAME} \
                                --seed_num ${SEED_NUM}
        )
}

function run_t5() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=0 python -m src.python.main \
                                --run trainer \
                                --model_name t5 \
                                --seed_num ${SEED_NUM}
        )
}

function run_bart() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=0 python -m src.python.main \
                                --run trainer \
                                --model_name bart \
                                --seed_num ${SEED_NUM}
        )
}

# ==========
# Main

function main() {
        run_gpt2
        # run_t5
        run_bart
}

main
