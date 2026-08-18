"""Export the minimum the app needs, small enough to commit."""

import shutil
from pathlib import Path

import pandas as pd

SRC = Path("data/processed")
OUT = Path("app_data")

COPY = ["funders.csv", "recipients.csv", "cause_mix.csv", "city_coords.csv",
        "funder_vectors.parquet", "recipient_vectors.parquet"]

KEEP = ["funder_ein", "funder_name", "recipient_ein", "recipient_name",
        "recipient_city", "amount", "purpose", "tax_year"]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    grants = pd.read_csv(SRC / "grants_matched.csv",
                         dtype={"funder_ein": str, "recipient_ein": str})

    if "recipient_is_person" not in grants.columns:
        raise SystemExit("re-run fetch_grants.py first — no recipient_is_person column")

    before = len(grants)
    grants = grants[grants["recipient_is_person"] != 1]
    print(f"dropped {before - len(grants)} grants to named individuals")

    grants[KEEP].to_csv(OUT / "grants_matched.csv", index=False)
    npo = pd.read_csv("data/raw/ma_nonprofits.csv", dtype={"ein": str})
    npo[["ein", "name", "city", "ntee_code"]].to_csv(OUT / "nonprofits.csv",
                                                     index=False)
    for f in COPY:
        shutil.copy(SRC / f, OUT / f)

    total = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"app_data/ is {total:.1f} MB")