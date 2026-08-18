"""Day 16 — help someone starting or formalizing a nonprofit."""

from pathlib import Path

import pandas as pd

from build_features import cause as ntee_cause
from build_features import key

DATA = (Path("data/processed") if Path("data/processed/nonprofits.csv").exists()
        else Path("app_data"))

# 990 is due the 15th day of the 5th month after fiscal year end.
# Massachusetts Form PC is due 4.5 months after fiscal year end.
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def load_nonprofits():
    df = pd.read_csv(DATA / "nonprofits.csv", dtype={"ein": str})
    df["cause"] = df["ntee_code"].map(ntee_cause)
    df["city_key"] = key(df["city"])
    return df


def overlap(nonprofits, cause, city, limit=15):
    """Existing organizations doing the same thing in the same place."""
    same_cause = nonprofits[nonprofits["cause"] == cause]
    here = same_cause[same_cause["city_key"] == city.upper().strip()]
    return {
        "in_city": len(here),
        "in_state": len(same_cause),
        "examples": here[["name", "city"]].head(limit),
    }


def saturation(nonprofits, cause, top=12):
    """Where organizations in this cause area are concentrated."""
    s = nonprofits[nonprofits["cause"] == cause]
    return s["city"].value_counts().head(top)


def cause_mix_by_city(nonprofits, city, top=10):
    """What the nonprofit sector in this city actually looks like."""
    c = nonprofits[nonprofits["city_key"] == city.upper().strip()]
    return c["cause"].value_counts().head(top)


def deadlines(fye_month):
    """Filing dates implied by a fiscal year end, as month/day strings."""
    federal = (fye_month + 5 - 1) % 12 + 1     # 5th month after FYE
    return {
        "IRS Form 990 series": f"{MONTHS[federal - 1]} 15",
        "MA Form PC (Attorney General)": f"{MONTHS[federal - 1]} 15",
        "MA annual report (Secretary of the Commonwealth)": "November 1",
    }