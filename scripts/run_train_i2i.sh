#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_training_job \
  "i2i" \
  "configs/train_i2i.example.yaml" \
  "logs/train_i2i.log" \
  "run/train_i2i.pid"
