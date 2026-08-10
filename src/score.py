"""Shared scoring logic for the Ledger app and agent."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = (Path("data/processed") if Path("data/processed/funders.csv").exists()
        else Path("app_data"))

FEATURES = ["similarity", "cause_share", "same_city", "log_grants", "log_median"]
K = 10


def load_model():
    return joblib.load("models/ranker.pkl"), SentenceTransformer("all-MiniLM-L6-v2")


def load_data():
    s = {"funder_ein": str, "recipient_ein": str, "ein": str}
    funders = pd.read_csv(DATA / "funders.csv", dtype=s)
    fvec = pd.read_parquet(DATA / "funder_vectors.parquet")
    mix = pd.read_csv(DATA / "cause_mix.csv", dtype=s)
    recips = pd.read_csv(DATA / "recipients.csv", dtype=s)
    grants = pd.read_csv(DATA / "grants_matched.csv", dtype=s)

    grants = grants.merge(
        recips[["ein", "cause"]].rename(columns={"ein": "recipient_ein"}),
        on="recipient_ein", how="left")

    return funders[funders["funder_ein"].isin(fvec.index)], fvec, mix, grants


def rank(name, cause, city, mission, funders, fvec, mix, embedder, model):
    """Score every funder using the same features the model trained on."""
    vec = embedder.encode([f"{name}. {cause}. {mission}"],
                          normalize_embeddings=True)[0]

    df = funders.copy()
    df["similarity"] = fvec.loc[df["funder_ein"]].to_numpy() @ vec

    df = df.merge(mix[mix["cause"] == cause][["funder_ein", "cause_share"]],
                  on="funder_ein", how="left")
    df["cause_share"] = df["cause_share"].fillna(0)

    fc = df["funder_city"].fillna("").str.upper().str.strip()
    df["same_city"] = ((fc == city.upper().strip()) & (fc != "")).astype(int)

    df["log_grants"] = np.log1p(df["n_grants"])
    df["log_median"] = np.log10(df["median_grant"].clip(lower=1))

    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    return df.sort_values("score", ascending=False).head(K)


def evidence(ein, cause, grants):
    """Largest past grant, preferring one in this cause area."""
    g = grants[grants["funder_ein"] == ein]
    same = g[g["cause"] == cause]
    g = same if len(same) else g
    return None if g.empty else g.nlargest(1, "amount").iloc[0]