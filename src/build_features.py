"""Build labeled funder/recipient pairs with features."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NEG_PER_POS = 5
SEED = 42
MODEL = "all-MiniLM-L6-v2"

FEATURES = ["similarity", "cause_share", "same_city", "log_distance",
            "log_grants", "log_median"]

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


def norm_ein(x):
    return str(x).replace("-", "").strip().zfill(9)


def key(s):
    """Normalize a city column so it can be matched against the coord table."""
    return s.fillna("").astype(str).str.upper().str.strip()


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two coordinate arrays."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


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


def funder_cause_mix(train_grants, recips):
    """For each funder, the share of its training grants going to each cause."""
    lookup = recips.set_index("ein")["cause"]
    g = train_grants.copy()
    g["cause"] = g["recipient_ein"].map(lookup)
    g = g.dropna(subset=["cause"])

    counts = g.groupby(["funder_ein", "cause"]).size()
    totals = g.groupby("funder_ein").size()
    return (counts.div(totals, level="funder_ein")
            .rename("cause_share").reset_index())


def funder_profiles(train_grants, grantmakers):
    """Aggregate each funder from TRAINING grants only — no leakage."""
    prof = (train_grants
            .groupby(["funder_ein", "funder_name"])
            .agg(profile=("purpose",
                          lambda s: " ".join(s.dropna().astype(str))[:4000]),
                 n_grants=("amount", "size"),
                 median_grant=("amount", "median"))
            .reset_index())

    gm = grantmakers[["ein", "city"]].copy()
    gm["funder_ein"] = gm["ein"].map(norm_ein)
    gm = gm.rename(columns={"city": "funder_city"})[["funder_ein", "funder_city"]]

    return prof.merge(gm.drop_duplicates("funder_ein"), on="funder_ein", how="left")


def recipient_profiles(nonprofits, missions=None):
    """Name + cause, plus the organization's own mission text if available."""
    out = nonprofits[["ein", "name", "city", "ntee_code"]].copy()
    out["cause"] = out["ntee_code"].map(cause)
    out = out.drop_duplicates(subset="ein")

    if missions is not None:
        out = out.merge(missions, on="ein", how="left")
        out["mission"] = out["mission"].fillna("")
    else:
        out["mission"] = ""

    out["profile"] = (out["name"].fillna("") + ". " + out["cause"]
                      + ". " + out["mission"]).str.strip()
    return out


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


def make_all_pairs(grants, funder_ids):
    """Every (test recipient x every funder) pair, for ranking evaluation."""
    truth = grants.groupby("recipient_ein")["funder_ein"].apply(set).to_dict()
    rows = []
    for r, funded in truth.items():
        for f in funder_ids:
            rows.append((r, f, int(f in funded)))
    return pd.DataFrame(rows, columns=["recipient_ein", "funder_ein", "label"])


def add_features(pairs, funders, recips, fvec, rvec, mix, coords):
    p = (pairs
         .merge(funders, on="funder_ein", how="inner")
         .merge(recips.rename(columns={"ein": "recipient_ein"}),
                on="recipient_ein", how="inner", suffixes=("_f", "_r")))

    # behavioral: how much of this funder's giving goes to this cause
    p = p.merge(mix, on=["funder_ein", "cause"], how="left")
    p["cause_share"] = p["cause_share"].fillna(0)

    # text similarity (kept, but no longer carrying the model)
    fi = fvec.loc[p["funder_ein"]].to_numpy()
    ri = rvec.loc[p["recipient_ein"]].to_numpy()
    p["similarity"] = (fi * ri).sum(axis=1)

    fc, rc = key(p["funder_city"]), key(p["city"])
    p["same_city"] = ((fc == rc) & (fc != "")).astype(int)

    c = coords.set_index("city")[["lat", "lon"]]
    f_xy = c.reindex(fc).to_numpy()
    r_xy = c.reindex(rc).to_numpy()
    d = pd.Series(haversine(f_xy[:, 0], f_xy[:, 1], r_xy[:, 0], r_xy[:, 1]))

    # unknown city on either side gets the median distance, not zero
    d = d.fillna(d.median()).clip(0, 500)
    p["log_distance"] = np.log1p(d).to_numpy()

    p["log_grants"] = np.log1p(p["n_grants"])
    p["log_median"] = np.log10(p["median_grant"].clip(lower=1))

    return p[["recipient_ein", "funder_ein", "label"] + FEATURES]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    grants, nonprofits = load()

    train_g, test_g = split_recipients(grants, rng)
    print(f"train grants {len(train_g)} / test grants {len(test_g)}")

    grantmakers = pd.read_csv("data/raw/ma_grantmakers.csv")
    coords = pd.read_csv("data/processed/city_coords.csv")

    funders = funder_profiles(train_g, grantmakers)
    recips = recipient_profiles(nonprofits)
    recips = recips[recips["ein"].isin(grants["recipient_ein"])]
    print(f"{len(funders)} funders / {len(recips)} recipients")

    mix = funder_cause_mix(train_g, recips)
    print(f"{len(mix)} funder/cause combinations")

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
        feat = add_features(pairs, funders, recips, fvec, rvec, mix, coords)
        feat.to_csv(f"data/processed/pairs_{name}.csv", index=False)
        print(f"{name}: {len(feat)} pairs, {feat.label.mean():.1%} positive")

    full = add_features(make_all_pairs(test_g, ids), funders, recips,
                        fvec, rvec, mix, coords)
    full.to_csv("data/processed/pairs_test_full.csv", index=False)
    print(f"ranking set: {len(full)} pairs, {full.recipient_ein.nunique()} recipients")

    funders.to_csv("data/processed/funders.csv", index=False)
    recips.to_csv("data/processed/recipients.csv", index=False)
    mix.to_csv("data/processed/cause_mix.csv", index=False)
    fvec.to_parquet("data/processed/funder_vectors.parquet")
    rvec.to_parquet("data/processed/recipient_vectors.parquet")