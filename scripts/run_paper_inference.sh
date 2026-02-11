#!/bin/bash
# Paper-facing Stage-2 inference entrypoint (non-SLURM).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_paper_inference.sh \
    --params /abs/path/params.csv \
    --voc /abs/path/voc.txt \
    --vmpp /abs/path/vmpp.txt \
    --checkpoint /abs/path/best-model.ckpt \
    --cache-dir /abs/path/cache \
    [--output-dir ./outputs/paper_inference] \
    [--device cuda] \
    [--jsc /abs/path/jsc.txt] \
    [--skip-45pt] \
    [-- <extra args forwarded to src/inference.py>]

Notes:
  - Uses the Stage-2 contract from the paper: 31 params + Voc/Vmpp.
  - cache-dir must contain training transformers/metadata (cnn_* files).
  - Run from the repository root directory.
EOF
}

PARAMS=""
VOC=""
VMPP=""
CHECKPOINT=""
CACHE_DIR=""
OUTPUT_DIR="./outputs/paper_inference"
DEVICE="cuda"
JSC=""
SKIP_45PT=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --params) PARAMS="${2:-}"; shift 2 ;;
        --voc) VOC="${2:-}"; shift 2 ;;
        --vmpp) VMPP="${2:-}"; shift 2 ;;
        --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
        --cache-dir) CACHE_DIR="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --device) DEVICE="${2:-}"; shift 2 ;;
        --jsc) JSC="${2:-}"; shift 2 ;;
        --skip-45pt) SKIP_45PT=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$PARAMS" || -z "$VOC" || -z "$VMPP" || -z "$CHECKPOINT" || -z "$CACHE_DIR" ]]; then
    echo "ERROR: --params, --voc, --vmpp, --checkpoint, --cache-dir are required." >&2
    usage
    exit 2
fi

for f in "$PARAMS" "$VOC" "$VMPP" "$CHECKPOINT"; do
    [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 2; }
done
[[ -d "$CACHE_DIR" ]] || { echo "ERROR: cache dir not found: $CACHE_DIR" >&2; exit 2; }

if [[ -n "$JSC" && ! -f "$JSC" ]]; then
    echo "ERROR: Jsc file not found: $JSC" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR"

CMD=(
    "$PYTHON_BIN" src/inference.py
    --params "$PARAMS"
    --voc "$VOC"
    --vmpp "$VMPP"
    --checkpoint "$CHECKPOINT"
    --cache-dir "$CACHE_DIR"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
)

if [[ -n "$JSC" ]]; then
    CMD+=(--jsc "$JSC")
fi

if [[ "$SKIP_45PT" == true ]]; then
    CMD+=(--skip-45pt)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running paper inference entrypoint:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

