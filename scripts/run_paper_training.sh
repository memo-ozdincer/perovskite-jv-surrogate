#!/bin/bash
# Paper-facing Stage-2 training entrypoint (non-SLURM).
# Trains the dilated-conv model from cleaned params/IV + external Voc/Vmpp.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_paper_training.sh \
    --params /abs/path/LHS_parameters_m_clean.txt \
    --iv /abs/path/IV_m_clean.txt \
    --voc /abs/path/voc_clean_100k.txt \
    --vmpp /abs/path/vmpp_clean_100k.txt \
    [--params-extra /abs/path/LHS_parameters_m_300k_clean.txt \
     --iv-extra /abs/path/IV_m_300k_clean.txt \
     --voc-extra /abs/path/voc_clean_300k.txt \
     --vmpp-extra /abs/path/vmpp_clean_300k.txt] \
    [--output-dir ./outputs/paper_train] \
    [--data-dir ./outputs/paper_cache] \
    [--run-name Paper-DilatedConv] \
    [--seed 42] \
    [--max-epochs 100] \
    [--batch-size 128] \
    [--num-workers 8] \
    [--physics-max-features 5] \
    [--no-physics-features] \
    [--no-physics-selection] \
    [--force-preprocess] \
    [-- <extra args forwarded to src/train.py>]

Notes:
  - Stage-1 scalars must be externally predicted (MATLAB model).
  - Default architecture is the paper model: conv + no attention + dilated.
  - Run from the repository root directory.
EOF
}

PARAMS=""
IV=""
VOC=""
VMPP=""
PARAMS_EXTRA=""
IV_EXTRA=""
VOC_EXTRA=""
VMPP_EXTRA=""
OUTPUT_DIR="./outputs/paper_train"
DATA_DIR="./outputs/paper_cache"
RUN_NAME="Paper-DilatedConv"
SEED="42"
MAX_EPOCHS="100"
BATCH_SIZE="128"
NUM_WORKERS=""
PHYSICS_MAX_FEATURES="5"
USE_PHYSICS_FEATURES=true
USE_PHYSICS_SELECTION=true
FORCE_PREPROCESS=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --params) PARAMS="${2:-}"; shift 2 ;;
        --iv) IV="${2:-}"; shift 2 ;;
        --voc) VOC="${2:-}"; shift 2 ;;
        --vmpp) VMPP="${2:-}"; shift 2 ;;
        --params-extra) PARAMS_EXTRA="${2:-}"; shift 2 ;;
        --iv-extra) IV_EXTRA="${2:-}"; shift 2 ;;
        --voc-extra) VOC_EXTRA="${2:-}"; shift 2 ;;
        --vmpp-extra) VMPP_EXTRA="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --data-dir) DATA_DIR="${2:-}"; shift 2 ;;
        --run-name) RUN_NAME="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --max-epochs) MAX_EPOCHS="${2:-}"; shift 2 ;;
        --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
        --num-workers) NUM_WORKERS="${2:-}"; shift 2 ;;
        --physics-max-features) PHYSICS_MAX_FEATURES="${2:-}"; shift 2 ;;
        --no-physics-features) USE_PHYSICS_FEATURES=false; shift ;;
        --no-physics-selection) USE_PHYSICS_SELECTION=false; shift ;;
        --force-preprocess) FORCE_PREPROCESS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$PARAMS" || -z "$IV" || -z "$VOC" || -z "$VMPP" ]]; then
    echo "ERROR: --params, --iv, --voc, --vmpp are required." >&2
    usage
    exit 2
fi

if [[ -n "$PARAMS_EXTRA" || -n "$IV_EXTRA" || -n "$VOC_EXTRA" || -n "$VMPP_EXTRA" ]]; then
    if [[ -z "$PARAMS_EXTRA" || -z "$IV_EXTRA" || -z "$VOC_EXTRA" || -z "$VMPP_EXTRA" ]]; then
        echo "ERROR: extra dataset requires all of --params-extra, --iv-extra, --voc-extra, --vmpp-extra." >&2
        exit 2
    fi
fi

for f in "$PARAMS" "$IV" "$VOC" "$VMPP"; do
    [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 2; }
done
if [[ -n "$PARAMS_EXTRA" ]]; then
    for f in "$PARAMS_EXTRA" "$IV_EXTRA" "$VOC_EXTRA" "$VMPP_EXTRA"; do
        [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 2; }
    done
fi

if [[ -z "$NUM_WORKERS" ]]; then
    CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
    NUM_WORKERS="$(( CPU_COUNT > 2 ? CPU_COUNT / 2 : 1 ))"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

CMD=(
    "$PYTHON_BIN" src/train.py
    --params "$PARAMS"
    --iv "$IV"
    --scalar-files "$VOC" "$VMPP"
    --output-dir "$OUTPUT_DIR"
    --data-dir "$DATA_DIR"
    --run-name "$RUN_NAME"
    --seed "$SEED"
    --max-epochs "$MAX_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --architecture conv
    --no-attention
    --use-dilated
    --jacobian-weight 0.0
)

if [[ -n "$PARAMS_EXTRA" ]]; then
    CMD+=(--params-extra "$PARAMS_EXTRA" --iv-extra "$IV_EXTRA")
    CMD+=(--scalar-files-extra "$VOC_EXTRA" "$VMPP_EXTRA")
fi

if [[ "$USE_PHYSICS_FEATURES" == true ]]; then
    CMD+=(--use-physics-features)
fi

if [[ "$USE_PHYSICS_SELECTION" == true ]]; then
    CMD+=(--physics-feature-selection --physics-max-features "$PHYSICS_MAX_FEATURES")
fi

if [[ "$FORCE_PREPROCESS" == true ]]; then
    CMD+=(--force-preprocess)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running paper training entrypoint:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

