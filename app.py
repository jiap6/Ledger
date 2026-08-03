"""Ledger — rank likely funders for a Boston nonprofit."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src")
from build_features import NTEE_GROUP

# full pipeline output locally; trimmed committed copy when deployed
DATA = Path("data/processed") if Path("data/processed/funders.csv").exists() else Path("app_data")

FEATURES = ["similarity", "cause_share", "same_city", "log_grants", "log_median"]
CAUSES = sorted(set(NTEE_GROUP.values()))
K = 10
MAX_NAME = 200


@st.cache_resource
def load_model():
    return joblib.load("models/ranker.pkl"), SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
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


def rank(name, cause, city, funders, fvec, mix, embedder, model):
    """Score every funder using the same features the model trained on."""
    vec = embedder.encode([f"{name}. {cause}"], normalize_embeddings=True)[0]

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


st.set_page_config(page_title="Ledger", layout="centered")
st.title("Ledger")
st.caption("Which Boston-area foundations are most likely to fund your nonprofit?")

model, embedder = load_model()
funders, fvec, mix, grants = load_data()

name = st.text_input("Organization name", "Dorchester Youth Robotics",
                     max_chars=MAX_NAME)
cause = st.selectbox("Cause area", CAUSES, index=CAUSES.index("youth development"))
city = st.text_input("City", "Boston", max_chars=60)

if st.button("Find funders", type="primary"):
    results = rank(name, cause, city, funders, fvec, mix, embedder, model)

    st.subheader(f"Top {K} matches")
    for _, r in results.iterrows():
        with st.container(border=True):
            top = st.columns([3, 1])
            top[0].markdown(f"**{r.funder_name.title()}**")
            top[0].caption(f"{str(r.funder_city).title()} · {int(r.n_grants)} grants on file")
            top[1].metric("Score", f"{r.score:.2f}")

            tags = [f"{r.cause_share:.0%} of giving to {cause}"]
            if r.same_city:
                tags.append(f"Based in {city.title()}")
            tags.append(f"Typical grant ${r.median_grant:,.0f}")
            st.write(" · ".join(tags))

            e = evidence(r.funder_ein, cause, grants)
            if e is not None:
                st.caption(f"Past grant: ${e.amount:,.0f} to "
                           f"{str(e.recipient_name).title()} — {str(e.purpose).lower()}")

            st.caption(f"[View 990](https://projects.propublica.org/nonprofits/"
                       f"organizations/{int(r.funder_ein)})")

    st.download_button("Download shortlist (CSV)",
                       results[["funder_name", "funder_city", "score",
                                "cause_share", "median_grant"]].to_csv(index=False),
                       "ledger_shortlist.csv")

    st.info("Scores are model estimates from historical filings, not predictions "
            "that a funder will give. Not financial or legal advice.")