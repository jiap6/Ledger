"""build labeled funder/recipient pairs with features."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NEG_PER_POS = 5
SEED = 42
MODEL = "all-MiniLM-L6-v2"

# NTEE letter -> broad cause area
NTEE_GROUP = {
    "A": "arts culture humanities", "B": "education",
    "C": "environment", "D": "animals", "E": "health", "F": "mental health",
    "G": "disease research", "H": "medical research", "I": "crime legal",
    "J": "employment", "K": "food agriculture", "L": "housing",
    "M": "public safety disaster relief", "N": "recreation sports",
    "O": "youth development", "P": "human services",
    "Q": "international", "R": "civil rights", "S": "community development",
    "T": "philanthropy", "U": "science technology", "V": "social science",
    "W": "public benefit", "X": "religion", "Y": "mutual benefit",
}


def cause(code):
    return NTEE_GROUP.get(str(code)[:1].upper(), "unclassified")


def load():
    g = pd.read_csv("data/processed/grants_matched.csv",
                    dtype={"funder_ein": str, "recipient_ein": str})
    g = g.dropna(subset=["recipient_ein"])
    n = pd.read_csv("data/raw/ma_nonprofits.csv", dtype={"ein": str})
    return g, n


def split_recipients(grants, rng):
    """Hold out whole organizations, not individual grants."""
    all_r = grants["recipient_ein"].unique()
    test = set(rng.choice(all_r, size=int(0.2 * len(all_r)), replace=False))
    return grants[~grants["recipient_ein"].isin(test)], \
           grants[grants["recipient_ein"].isin(test)]


def funder_profiles(train_grants):
    """Aggregate each funder from TRAINING grants only — no leakage."""
    return (train_grants
            .groupby(["funder_ein", "funder_name"])
            .agg(profile=("purpose",
                          lambda s: " ".join(s.dropna().astype(str))[:4000]),
                 n_grants=("amount", "size"),
                 median_grant=("amount", "median"))
            .reset_index())


def recipient_profiles(nonprofits):
    """Recipients have no mission text available — use name plus cause area."""
    out = nonprofits[["ein", "name", "city", "ntee_code"]].copy()
    out["cause"] = out["ntee_code"].map(cause)
    out["profile"] = out["name"].fillna("") + ". " + out["cause"]
    return out.drop_duplicates(subset="ein")


def make_pairs(grants, funder_ids, funded_by, rng):
    """Every real grant is a 1; sample funders that didn't give as 0s."""
    rows = []
    for r, f in zip(grants["recipient_ein"], grants["funder_ein"]):
        rows.append((r, f, 1))
        picked = 0
        while picked < NEG_PER_POS:
            cand = funder_ids[rng.integers(len(funder_ids))]
            if cand not in funded_by[r]:
                rows.append((r, cand, 0))
                picked += 1
    return pd.DataFrame(rows, columns=["recipient_ein", "funder_ein", "label"])


def add_features(pairs, funders, recips, fvec, rvec):
    p = (pairs
         .merge(funders, on="funder_ein", how="inner")
         .merge(recips.rename(columns={"ein": "recipient_ein"}),
                on="recipient_ein", how="inner", suffixes=("_f", "_r")))

    fi = fvec.loc[p["funder_ein"]].to_numpy()
    ri = rvec.loc[p["recipient_ein"]].to_numpy()
    p["similarity"] = (fi * ri).sum(axis=1)          # both are unit vectors

    p["same_city"] = (p["funder_name"].str.contains("BOSTON", case=False, na=False)
                      == p["city"].str.contains("Boston", case=False, na=False)).astype(int)
    p["log_grants"] = np.log1p(p["n_grants"])
    p["log_median"] = np.log10(p["median_grant"].clip(lower=1))

    cols = ["recipient_ein", "funder_ein", "label",
            "similarity", "same_city", "log_grants", "log_median"]
    return p[cols]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    grants, nonprofits = load()

    train_g, test_g = split_recipients(grants, rng)
    print(f"train grants {len(train_g)} / test grants {len(test_g)}")

    funders = funder_profiles(train_g)
    recips = recipient_profiles(nonprofits)
    recips = recips[recips["ein"].isin(grants["recipient_ein"])]
    print(f"{len(funders)} funders / {len(recips)} recipients")

    model = SentenceTransformer(MODEL)
    fvec = pd.DataFrame(
        model.encode(funders["profile"].tolist(), normalize_embeddings=True,
                     show_progress_bar=True),
        index=funders["funder_ein"])
    rvec = pd.DataFrame(
        model.encode(recips["profile"].tolist(), normalize_embeddings=True,
                     show_progress_bar=True),
        index=recips["ein"])

    funded_by = grants.groupby("recipient_ein")["funder_ein"].apply(set).to_dict()
    ids = funders["funder_ein"].to_numpy()

    for name, part in [("train", train_g), ("test", test_g)]:
        pairs = make_pairs(part, ids, funded_by, rng)
        feat = add_features(pairs, funders, recips, fvec, rvec)
        feat.to_csv(f"data/processed/pairs_{name}.csv", index=False)
        print(f"{name}: {len(feat)} pairs, {feat.label.mean():.1%} positive")

    funders.to_csv("data/processed/funders.csv", index=False)
    recips.to_csv("data/processed/recipients.csv", index=False)
    fvec.to_parquet("data/processed/funder_vectors.parquet")
    rvec.to_parquet("data/processed/recipient_vectors.parquet")