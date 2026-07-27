#!/usr/bin/env bash
set -e                      # stop at the first failure
source .venv/bin/activate

if [ -f data/raw/ma_nonprofits.csv ]; then
  echo "== 1/4 organizations (already have them, skipping) =="
else
  echo "== 1/4 organizations =="
  python src/fetch_funders.py
fi

echo "== 2/4 grants =="
python src/fetch_grants.py

echo "== 3/4 matching =="
python src/match_recipients.py

echo "== 4/4 features =="
python src/build_features.py

echo "== summary =="
python src/summary.py