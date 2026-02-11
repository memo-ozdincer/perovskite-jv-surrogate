#!/bin/bash
# Publish-facing alias for the full dilated-conv ablation pipeline.
# Keeps legacy filename compatibility without moving/deleting files.

set -e
bash slurm_tcn_master_pipeline.sh "$@"
