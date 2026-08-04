"""pull recipient mission text from their own 990 filings."""

import os
import time
import warnings

import pandas as pd
import requests
from defusedxml.ElementTree import parse

from fetch_grants import download_index, norm_ein, tag, text_of

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

HEADERS = {"User-Agent": "ledger-nonprofit-research-tool"}
XML_URL = "https://opendata.grantseeker.io/data/{oid}_public.xml"
YEARS = [2020, 2021, 2022]
XML_DIR = "data/raw/recipient_xml"

MISSION_TAGS = ["ActivityOrMissionDesc", "MissionDesc", "PrimaryExemptPurposeTxt"]


def find_filings():
    """One 990 or 990-EZ per recipient organization."""
    recips = pd.read_csv("data/processed/recipients.csv", dtype={"ein": str})
    wanted = set(recips["ein"].map(norm_ein))

    frames = []
    for y in YEARS:
        df = pd.read_csv(download_index(y), dtype=str)
        df.columns = [c.upper().strip() for c in df.columns]
        df["EIN"] = df["EIN"].map(norm_ein)
        df = df[df["RETURN_TYPE"].isin(["990", "990EZ"]) & df["EIN"].isin(wanted)]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("TAX_PERIOD").drop_duplicates(subset="EIN")
    print(f"{len(out)} filings for {len(wanted)} recipients")
    return out


def download(filings):
    os.makedirs(XML_DIR, exist_ok=True)
    for i, (ein, oid) in enumerate(zip(filings["EIN"], filings["OBJECT_ID"]), 1):
        if not str(oid).isdigit():
            continue
        path = f"{XML_DIR}/{ein}.xml"
        if os.path.exists(path):
            continue

        r = requests.get(XML_URL.format(oid=oid), headers=HEADERS, timeout=60)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
        if i % 100 == 0:
            print(f"  {i}/{len(filings)}")
        time.sleep(0.5)


def extract_mission(path):
    """Mission statement plus program descriptions, capped for embedding."""
    root = parse(path).getroot()

    parts = []
    for name in MISSION_TAGS:
        t = text_of(root, name)
        if t:
            parts.append(t)
            break

    for g in root.iter():
        if tag(g) == "ProgramSrvcAccomplishmentGrp":
            d = text_of(g, "Desc")
            if d:
                parts.append(d)

    return " ".join(parts)[:1500]


if __name__ == "__main__":
    download(find_filings())

    rows = []
    for fname in os.listdir(XML_DIR):
        try:
            m = extract_mission(f"{XML_DIR}/{fname}")
            if m:
                rows.append({"ein": fname.replace(".xml", ""), "mission": m})
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/recipient_missions.csv", index=False)
    print(f"\nmission text for {len(df)} recipients")
    print(df["mission"].str.len().describe()[["mean", "50%", "max"]])