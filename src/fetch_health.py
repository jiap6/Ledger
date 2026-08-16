"""build a labeled dataset of organizational financial health."""

import io
import os
import time
import warnings
import zipfile

import pandas as pd
import requests

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

HEADERS = {"User-Agent": "ledger-nonprofit-research-tool"}
REVOKE_URL = "https://apps.irs.gov/pub/epostcard/data-download-revocation.zip"
ORG_URL = "https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"

COLS = ["ein", "name", "dba", "address", "city", "state", "zip", "country",
        "exemption_type", "revocation_date", "posting_date", "reinstatement_date"]

AS_OF = 2018          # features come only from filings at or before this year
WINDOW = 4            # years of history to look back
N_PER_CLASS = 900     # candidates to try per class
OUT = "data/processed/health_raw.csv"


def norm_ein(x):
    return str(x).replace("-", "").strip().zfill(9)


def revoked_ma():
    """MA organizations that lost exempt status, read from the zip in memory."""
    print("downloading revocation list...")
    r = requests.get(REVOKE_URL, headers=HEADERS, timeout=300)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        member = next(n for n in z.namelist() if n.endswith(".txt"))
        with z.open(member) as f:
            df = pd.read_csv(f, sep="|", header=None, names=COLS,
                             dtype=str, quoting=3, on_bad_lines="skip")

    df["ein"] = df["ein"].map(norm_ein)
    df = df[df["state"].str.strip().eq("MA")]
    df["rev_year"] = pd.to_datetime(df["revocation_date"],
                                    errors="coerce").dt.year
    print(f"{len(df)} revoked MA organizations")
    return df


def fetch_filings(ein):
    """Financial filing history for one organization."""
    r = requests.get(ORG_URL.format(ein=int(ein)), headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return []
    return r.json().get("filings_with_data", [])


def collect(eins, label, want):
    """Pull filing history until we have `want` orgs with usable data."""
    rows, kept = [], 0
    for i, ein in enumerate(eins, 1):
        if kept >= want:
            break

        filings = [f for f in fetch_filings(ein)
                   if f.get("tax_prd_yr") and int(f["tax_prd_yr"]) <= AS_OF
                   and int(f["tax_prd_yr"]) > AS_OF - WINDOW]

        if len(filings) >= 2:                 # need a trend, not a snapshot
            for f in filings:
                rows.append({
                    "ein": ein, "label": label,
                    "year": int(f["tax_prd_yr"]),
                    "revenue": f.get("totrevenue"),
                    "expenses": f.get("totfuncexpns"),
                    "assets": f.get("totassetsend"),
                    "liabilities": f.get("totliabend"),
                })
            kept += 1

        if i % 50 == 0:
            print(f"  tried {i}, kept {kept}/{want}")
        time.sleep(0.5)

    print(f"label {label}: {kept} organizations with usable history")
    return rows


if __name__ == "__main__":
    rev = revoked_ma()

    # positives: revoked AFTER our as-of date, never reinstated
    pos = rev[rev["rev_year"].gt(AS_OF)
              & rev["reinstatement_date"].isna()]["ein"].unique()

    # negatives: MA nonprofits that never appear on the revocation list
    allorg = pd.read_csv("data/raw/ma_nonprofits.csv", dtype={"ein": str})
    allorg["ein"] = allorg["ein"].map(norm_ein)
    neg = allorg[~allorg["ein"].isin(set(rev["ein"]))]["ein"].unique()

    print(f"\n{len(pos)} positive candidates / {len(neg)} negative candidates\n")

    rows = collect(pos[:N_PER_CLASS * 3], 1, N_PER_CLASS)
    rows += collect(neg[:N_PER_CLASS * 2], 0, N_PER_CLASS)

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")