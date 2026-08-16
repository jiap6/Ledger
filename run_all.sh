#!/usr/bin/env bash
set -e
source .venv/bin/activate

if [ -f data/raw/ma_nonprofits.csv ]; then
  echo "== 1/7 organizations (cached) =="
else
  echo "== 1/7 organizations =="
  python src/fetch_funders.py
fi

echo "== 2/7 grants =="
python src/fetch_grants.py

echo "== 3/7 matching =="
python src/match_recipients.py

if [ -f data/processed/city_coords.csv ]; then
  echo "== 4/7 geocoding (cached) =="
else
  echo "== 4/7 geocoding =="
  python src/geocode.py
fi

echo "== 5/7 features =="
python src/build_features.py

echo "== 6/7 model =="
python src/train.py

echo "== 7/7 tests =="
pytest tests/ -q

python src/summary.py