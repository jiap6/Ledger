"""Ledger — find likely funders for a Boston nonprofit."""

import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from build_features import NTEE_GROUP
from core import (K, contributions, embed, evidence, load_data, load_model,
                  rank, similar_orgs)
from startup import (MONTHS, cause_mix_by_city, deadlines, load_nonprofits,
                     overlap, saturation)

CAUSES = sorted(set(NTEE_GROUP.values()))

load_model = st.cache_resource(load_model)
load_data = st.cache_data(load_data)
load_nonprofits = st.cache_data(load_nonprofits)

st.set_page_config(page_title="Ledger", layout="centered")
st.title("Ledger")
st.caption("Which Boston-area foundations are most likely to fund your nonprofit?")

model, embedder = load_model()
funders, fvec, rvec, mix, recips, grants, coords = load_data()

with st.sidebar:
    st.header("Your organization")
    name = st.text_input("Name", "Dorchester Youth Robotics", max_chars=200)
    cause = st.selectbox("Cause area", CAUSES,
                         index=CAUSES.index("youth development"))
    city = st.text_input("City", "Boston", max_chars=60)
    mission = st.text_area("What you do (used to find similar organizations)",
                           "Free after-school robotics for middle schoolers.",
                           max_chars=1500)

    st.header("Filters")
    min_grant = st.slider("Minimum typical grant ($)", 0, 50_000, 0, 1000)
    min_count = st.slider("Minimum grants on file", 1, 50, 1)

    go = st.button("Find funders", type="primary", use_container_width=True)

if go:
    st.session_state.vec = embed(name, cause, mission, embedder)

vec = st.session_state.get("vec")


def scored():
    return rank(vec, cause, city, funders, fvec, mix, model, coords,
                min_grant, min_count)


match_tab, map_tab, peer_tab, start_tab, explore_tab = st.tabs(
    ["Matches", "Map", "Organizations like yours", "Start here", "Explore"])

with match_tab:
    if vec is None:
        st.info("Fill in your organization on the left and click Find funders.")
    else:
        results = scored()
        if results.empty:
            st.warning("No funders match those filters. Loosen them.")
        else:
            st.caption(f"{len(results):,} funders scored · showing top {K}")

            for _, r in results.head(K).iterrows():
                with st.container(border=True):
                    top = st.columns([3, 1])
                    top[0].markdown(f"**{r.funder_name.title()}**")
                    top[0].caption(f"{str(r.funder_city).title()} · "
                                   f"{int(r.n_grants)} grants on file")
                    top[1].metric("Score", f"{r.score:.2f}")

                    tags = [f"{r.cause_share:.0%} of giving to {cause}"]
                    if r.same_city:
                        tags.append(f"Based in {city.title()}")
                    tags.append(f"Typical grant ${r.median_grant:,.0f}")
                    st.write(" · ".join(tags))

                    e = evidence(r.funder_ein, cause, grants)
                    if e is not None:
                        st.caption(f"Past grant: ${e.amount:,.0f} to "
                                   f"{str(e.recipient_name).title()} — "
                                   f"{str(e.purpose).lower()}")

                    with st.expander("Why this score, and what they fund"):
                        st.write("**What drove the score**")
                        st.bar_chart(contributions(model, r), horizontal=True)

                        fm = (mix[mix["funder_ein"] == r.funder_ein]
                              .set_index("cause")["cause_share"].nlargest(6))
                        if not fm.empty:
                            st.write("**Where their money goes**")
                            st.bar_chart(fm, horizontal=True)

                        st.write("**Largest grants on file**")
                        st.dataframe(
                            grants[grants["funder_ein"] == r.funder_ein]
                            .nlargest(8, "amount")
                            [["recipient_name", "amount", "tax_year"]],
                            hide_index=True, use_container_width=True)

                    st.caption(f"[View 990](https://projects.propublica.org/"
                               f"nonprofits/organizations/{int(r.funder_ein)})")

            st.download_button(
                "Download shortlist (CSV)",
                results.head(K)[["funder_name", "funder_city", "score",
                                 "cause_share", "median_grant"]].to_csv(index=False),
                "ledger_shortlist.csv")

            st.info("Scores are model estimates from historical filings, not "
                    "predictions that a funder will give. Not financial or "
                    "legal advice.")

with map_tab:
    if vec is None:
        st.info("Run a search first.")
    else:
        top = scored().head(40).dropna(subset=["lat", "lon"]).copy()
        if top.empty:
            st.warning("No coordinates available for these funders.")
        else:
            top["size"] = 200 + 2400 * (top["score"] / top["score"].max())
            st.caption("Rankings are driven by each funder's giving history, "
                       "cause focus, and location. Your description powers the "
                       "Organizations like yours tab.")
            st.map(top, latitude="lat", longitude="lon", size="size",
                   color="#2b6cb0")

            st.dataframe(
                top.head(15)[["funder_name", "funder_city", "score",
                              "median_grant"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "funder_name": "Funder",
                    "funder_city": "City",
                    "score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.2f"),
                    "median_grant": st.column_config.NumberColumn(
                        "Typical grant", format="$%d"),
                })

with peer_tab:
    if vec is None:
        st.info("Run a search first.")
    else:
        st.caption("Nonprofits with similar missions — and who actually funded them.")
        for p in similar_orgs(vec, rvec, recips, grants):
            with st.container(border=True):
                st.markdown(f"**{str(p['name']).title()}**")
                st.caption(f"Mission similarity {p['similarity']:.2f}")
                if p["funders"]:
                    for f in p["funders"]:
                        st.write(f"· {f['funder_name'].title()} — "
                                 f"${f['amount']:,.0f}")
                else:
                    st.caption("No grants on file.")

with start_tab:
    npo = load_nonprofits()

    st.subheader("Does this already exist?")
    st.caption("Before founding a new organization, it's worth knowing who's "
               "already doing this work nearby — and whether joining them "
               "beats starting from zero.")

    o = overlap(npo, cause, city)
    c1, c2 = st.columns(2)
    c1.metric(f"In {city.title()}", f"{o['in_city']:,}")
    c2.metric("In Massachusetts", f"{o['in_state']:,}")

    if o["in_city"]:
        st.dataframe(o["examples"], hide_index=True, use_container_width=True,
                     column_config={"name": "Organization", "city": "City"})
    else:
        st.success(f"No registered {cause} organizations found in "
                   f"{city.title()} — this may be an underserved area.")

    st.divider()

    st.subheader("Where this work is concentrated")
    st.caption(f"Registered {cause} organizations by city")
    st.bar_chart(saturation(npo, cause), horizontal=True)

    st.subheader(f"What the sector looks like in {city.title()}")
    mixc = cause_mix_by_city(npo, city)
    if mixc.empty:
        st.info("No registered organizations found for that city name.")
    else:
        st.bar_chart(mixc, horizontal=True)

    st.divider()

    st.subheader("Your filing calendar")
    fye = st.selectbox("Fiscal year ends", MONTHS, index=11)
    for label, when in deadlines(MONTHS.index(fye) + 1).items():
        st.write(f"**{when}** — {label}")

    st.markdown("""
**Getting registered in Massachusetts**

- Every public charity operating in Massachusetts registers with the
  Attorney General's Non-Profits/Public Charities Division. Initial
  registration is $100.
- Soliciting donations additionally requires a Certificate for Solicitation.
- Form PC has been e-file only since September 2023.
- Massachusetts nonprofit corporations also file an annual report with the
  Secretary of the Commonwealth.
- Which 990 you file depends on gross receipts and assets — check the
  current thresholds before choosing.

Verify everything against the source before acting:
[Mass. AG charities division](https://www.mass.gov/how-to/register-a-charity-in-massachusetts)
· [IRS 990 filing requirements](https://www.irs.gov/charities-non-profits/form-990-series-which-forms-do-exempt-organizations-file-filing-phase-in)
    """)

    st.info("General information compiled from public sources, not legal or "
            "tax advice. Requirements and fees change — confirm with the "
            "Attorney General's office or an attorney.")

with explore_tab:
    st.caption(f"{len(grants):,} grants from {grants.funder_ein.nunique()} "
               f"Massachusetts foundations, {grants.tax_year.min():.0f}–"
               f"{grants.tax_year.max():.0f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total granted", f"${grants.amount.sum()/1e6:,.0f}M")
    c2.metric("Median grant", f"${grants.amount.median():,.0f}")
    c3.metric("Recipients", f"{grants.recipient_name.nunique():,}")

    st.write("**Largest funders by total giving**")
    st.bar_chart(grants.groupby("funder_name")["amount"].sum().nlargest(15),
                 horizontal=True)

    st.write("**Where the money goes, by cause**")
    st.bar_chart(grants.groupby("cause")["amount"].sum().nlargest(12),
                 horizontal=True)

    st.write("**Grant sizes**")
    st.bar_chart(pd.cut(grants["amount"],
                        bins=[0, 1e3, 5e3, 10e3, 25e3, 50e3, 1e9],
                        labels=["<$1k", "$1–5k", "$5–10k", "$10–25k",
                                "$25–50k", "$50k+"]).value_counts().sort_index())