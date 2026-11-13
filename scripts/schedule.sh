#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python tools/train.py trainer.max_epochs=5 logger=csv

python tools/train.py trainer.max_epochs=10 logger=csv
