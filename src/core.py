"""Shared scoring logic for the Ledger app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from build_features import haversine, key

DATA = (Path("data/processed") if Path("data/processed/funders.csv").exists()
        else Path("app_data"))

FEATURES = ["cause_share", "log_distance", "log_grants", "log_median"]

LABELS = {
    "cause_share": "Gives to your cause",
    "log_distance": "Distance from you",
    "log_grants": "Grant volume",
    "log_median": "Typical grant size",
}

K = 10
DEFAULT_KM = 30          # assumed distance when a city can't be located


def load_model():
    return joblib.load("models/ranker.pkl"), SentenceTransformer("all-MiniLM-L6-v2")


def load_data():
    s = {"funder_ein": str, "recipient_ein": str, "ein": str}
    funders = pd.read_csv(DATA / "funders.csv", dtype=s)
    fvec = pd.read_parquet(DATA / "funder_vectors.parquet")
    rvec = pd.read_parquet(DATA / "recipient_vectors.parquet")
    mix = pd.read_csv(DATA / "cause_mix.csv", dtype=s)
    recips = pd.read_csv(DATA / "recipients.csv", dtype=s)
    grants = pd.read_csv(DATA / "grants_matched.csv", dtype=s)
    coords = pd.read_csv(DATA / "city_coords.csv")

    grants = grants.merge(
        recips[["ein", "cause"]].rename(columns={"ein": "recipient_ein"}),
        on="recipient_ein", how="left")

    funders["city_key"] = key(funders["funder_city"])
    funders = funders.merge(coords.rename(columns={"city": "city_key"}),
                            on="city_key", how="left")

    return (funders[funders["funder_ein"].isin(fvec.index)],
            fvec, rvec, mix, recips, grants, coords)


def embed(name, cause, mission, embedder):
    return embedder.encode([f"{name}. {cause}. {mission}"],
                           normalize_embeddings=True)[0]


def rank(vec, cause, city, funders, fvec, mix, model, coords,
         min_grant=0, min_count=0):
    """Score every funder using the same features the model trained on."""
    df = funders[(funders["median_grant"] >= min_grant)
                 & (funders["n_grants"] >= min_count)].copy()
    if df.empty:
        return df

    df["similarity"] = fvec.loc[df["funder_ein"]].to_numpy() @ vec

    df = df.merge(mix[mix["cause"] == cause][["funder_ein", "cause_share"]],
                  on="funder_ein", how="left")
    df["cause_share"] = df["cause_share"].fillna(0)

    ck = city.upper().strip()
    df["same_city"] = (df["city_key"] == ck).astype(int)

    here = coords[coords["city"] == ck]
    if here.empty:
        df["log_distance"] = np.log1p(DEFAULT_KM)
    else:
        d = pd.Series(haversine(df["lat"].to_numpy(), df["lon"].to_numpy(),
                                here["lat"].iloc[0], here["lon"].iloc[0]))
        df["log_distance"] = np.log1p(d.fillna(DEFAULT_KM).clip(0, 500)).to_numpy()

    df["log_grants"] = np.log1p(df["n_grants"])
    df["log_median"] = np.log10(df["median_grant"].clip(lower=1))

    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    return df.sort_values("score", ascending=False)


def contributions(model, row):
    """Per-feature log-odds contribution. Empty if the model has no coefficients."""
    x = row[FEATURES].to_frame().T.astype(float)
    if hasattr(model, "named_steps"):
        z, clf = model[:-1].transform(x), model[-1]
    else:
        z, clf = x.to_numpy(), model

    if not hasattr(clf, "coef_"):
        return pd.Series(dtype=float)

    return pd.Series(z[0] * clf.coef_[0], index=[LABELS[f] for f in FEATURES])

def similar_orgs(vec, rvec, recips, grants, n=5):
    """Nonprofits most like yours, and who funded them."""
    sims = pd.Series(rvec.to_numpy() @ vec, index=rvec.index).nlargest(n)
    names = recips.set_index("ein")["name"]

    out = []
    for ein, s in sims.items():
        g = grants[grants["recipient_ein"] == ein]
        out.append({
            "name": names.get(ein, ein),
            "similarity": s,
            "funders": g.nlargest(3, "amount")[["funder_name", "amount"]]
                        .to_dict("records"),
        })
    return out


def evidence(ein, cause, grants):
    """Largest past grant, preferring one in this cause area."""
    g = grants[grants["funder_ein"] == ein]
    same = g[g["cause"] == cause]
    g = same if len(same) else g
    return None if g.empty else g.nlargest(1, "amount").iloc[0]