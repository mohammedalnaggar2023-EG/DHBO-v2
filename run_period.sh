#!/usr/bin/env bash
# Earlier test period (data ending 2021-12-31): full DHBO pipeline, unchanged code.
# Usage: bash run_period.sh   (run from the repository folder)
set -e
CODE="${DHBO_CODE:-.}"
export DHBO_THREADS="${DHBO_THREADS:-1}" PYTHONWARNINGS=ignore
python3 make_truncated_data.py
for T in daily cumulative; do
  python3 "$CODE/run_experiments.py" --data data/confirmed_to_2021-12-31.csv --targets "$T" --out "per_v1_$T"
  python3 "$CODE/run_v2.py" --data data/confirmed_to_2021-12-31.csv --targets "$T" --v1 "per_v1_$T" --out "per_v2_$T"
done
echo done
