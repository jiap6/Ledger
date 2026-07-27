"""normalize names and match grant recipients to real organizations."""

import re
import pandas as pd
from rapidfuzz import process, fuzz

FILLER = r"\b(INC|INCORPORATED|CORP|CORPORATION|COMPANY|CO|LLC|LTD|THE|A|AN|AND|OF|FOR)\b"
CUTOFF = 88


def normalize(name):
    """Strip case, punctuation, and legal boilerplate so names compare fairly."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
    s = re.sub(FILLER, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_to_eins(grants, nonprofits):
    """Exact match on normalized name first, fuzzy match only for leftovers."""
    lookup = dict(zip(nonprofits["name_norm"], nonprofits["ein"]))
    choices = list(lookup)

    # only MA recipients can appear in an MA nonprofit list
    ma = grants["recipient_state"].eq("MA")
    names = grants.loc[ma, "recipient_norm"].dropna().unique()
    print(f"resolving {len(names)} unique MA recipient names...")

    resolved = {}
    for i, n in enumerate(names, 1):
        if not n:
            continue
        if n in lookup:
            resolved[n] = (lookup[n], 100)
        else:
            hit = process.extractOne(n, choices, scorer=fuzz.WRatio,
                                     score_cutoff=CUTOFF)
            if hit:
                resolved[n] = (lookup[hit[0]], hit[1])
        if i % 500 == 0:
            print(f"  {i}/{len(names)}")

    grants["recipient_ein"] = grants["recipient_norm"].map(
        lambda n: resolved.get(n, (None, None))[0])
    grants["match_score"] = grants["recipient_norm"].map(
        lambda n: resolved.get(n, (None, None))[1])
    return grants


if __name__ == "__main__":
    grants = pd.read_csv("data/processed/grants.csv", dtype={"funder_ein": str})
    nonprofits = pd.read_csv("data/raw/ma_nonprofits.csv", dtype={"ein": str})

    grants["recipient_norm"] = grants["recipient_name"].map(normalize)
    nonprofits["name_norm"] = nonprofits["name"].map(normalize)
    nonprofits = nonprofits.drop_duplicates(subset="name_norm")

    grants = match_to_eins(grants, nonprofits)
    grants.to_csv("data/processed/grants_matched.csv", index=False)

    ma = grants["recipient_state"].eq("MA")
    matched = grants.loc[ma, "recipient_ein"].notna()
    print(f"\nMA grants: {ma.sum()}")
    print(f"matched to an EIN: {matched.sum()} ({matched.mean():.1%})")

    # 50 fuzzy matches to check by hand
    (grants[grants["match_score"].between(CUTOFF, 99)]
        .sample(min(50, (grants["match_score"].between(CUTOFF, 99)).sum()),
                random_state=42)
        [["recipient_name", "recipient_norm", "recipient_ein", "match_score"]]
        .to_csv("data/processed/match_review.csv", index=False))
    print("wrote match_review.csv — check these by hand")