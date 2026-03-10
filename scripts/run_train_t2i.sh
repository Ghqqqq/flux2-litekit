#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_training_job \
  "t2i" \
  "configs/train_t2i.example.yaml" \
  "logs/train_t2i.log" \
  "run/train_t2i.pid"
