"""Export the minimum the app needs, small enough to commit."""

import shutil
from pathlib import Path

import pandas as pd

SRC = Path("data/processed")
OUT = Path("app_data")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    grants = pd.read_csv(SRC / "grants_matched.csv",
                         dtype={"funder_ein": str, "recipient_ein": str})

    # never republish grants to named individuals
    if "recipient_is_person" in grants.columns:
        before = len(grants)
        grants = grants[grants["recipient_is_person"] != 1]
        print(f"dropped {before - len(grants)} grants to individuals")
    else:
        raise SystemExit("re-run fetch_grants.py first — no recipient_is_person column")

    keep = ["funder_ein", "funder_name", "recipient_ein", "recipient_name",
            "recipient_city", "amount", "purpose", "tax_year"]
    grants[keep].to_csv(OUT / "grants_matched.csv", index=False)

    for f in ["funders.csv", "recipients.csv", "cause_mix.csv",
              "funder_vectors.parquet"]:
        shutil.copy(SRC / f, OUT / f)

    total = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"app_data/ is {total:.1f} MB")