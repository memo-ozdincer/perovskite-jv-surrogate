#!/bin/bash
# Publish-facing alias for the single-run dilated-conv pipeline.
# Keeps legacy filename compatibility without moving/deleting files.

set -e
bash slurm_tcn_single_dilated.sh "$@"
