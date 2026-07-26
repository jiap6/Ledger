import os
import time
import warnings
import requests
import pandas as pd
from defusedxml.ElementTree import parse

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

HEADERS = {"User-Agent": "ledger-nonprofit-research-tool"}
INDEX_URL = "https://apps.irs.gov/pub/epostcard/990/xml/{y}/index_{y}.csv"
XML_URL = "https://projects.propublica.org/nonprofits/download-xml?object_id={oid}"

YEARS = [2023, 2024, 2025]
MAX_FILINGS = 300          # cap the first run; raise once it works
XML_DIR = "data/raw/xml"


def norm_ein(x):
    """EINs come as ints or dashed strings — make them all 9-char strings."""
    return str(x).replace("-", "").strip().zfill(9)


def download_index(year):
    """Fetch one IRS index file, skipping if already on disk."""
    path = f"data/raw/index_{year}.csv"
    if os.path.exists(path):
        return path

    print(f"downloading index {year} (large file)...")
    r = requests.get(INDEX_URL.format(y=year), headers=HEADERS,
                     timeout=300, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return path


def find_filings():
    """990-PF filings belonging to our MA grantmakers, one per funder."""
    funders = pd.read_csv("data/raw/ma_grantmakers.csv")
    wanted = set(funders["ein"].map(norm_ein))

    frames = []
    for y in YEARS:
        df = pd.read_csv(download_index(y), dtype=str)
        df.columns = [c.upper().strip() for c in df.columns]
        df["EIN"] = df["EIN"].map(norm_ein)
        df = df[(df["RETURN_TYPE"] == "990PF") & (df["EIN"].isin(wanted))]
        print(f"{y}: {len(df)} matching 990-PF filings")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("TAX_PERIOD", ascending=False)
    out = out.drop_duplicates(subset="EIN")   # newest filing per funder
    return out.head(MAX_FILINGS)


def download_xml(filings):
    """Save each filing's XML locally."""
    os.makedirs(XML_DIR, exist_ok=True)
    for i, oid in enumerate(filings["OBJECT_ID"], 1):
        if not str(oid).isdigit():      # object IDs are numeric — reject anything else
            continue

        path = f"{XML_DIR}/{oid}.xml"
        if os.path.exists(path):
            continue

        r = requests.get(XML_URL.format(oid=oid), headers=HEADERS, timeout=60)
        if r.status_code != 200:
            print(f"  skip {oid} ({r.status_code})")
            continue

        with open(path, "wb") as f:
            f.write(r.content)
        if i % 25 == 0:
            print(f"  {i}/{len(filings)}")
        time.sleep(0.5)


def tag(el):
    """Strip the XML namespace: '{http://...}Amt' -> 'Amt'."""
    return el.tag.split("}")[-1]


def text_of(parent, name):
    """Text of the first descendant with this tag name, or empty string."""
    if parent is None:
        return ""
    for el in parent.iter():
        if tag(el) == name and el.text:
            return el.text.strip()
    return ""


def parse_filing(path):
    """Pull every grant row out of one 990-PF."""
    root = parse(path).getroot()

    filer = next((el for el in root.iter() if tag(el) == "Filer"), None)
    ein = text_of(filer, "EIN")
    funder = text_of(filer, "BusinessNameLine1Txt")
    period = text_of(root, "TaxPeriodEndDt")

    rows = []
    for g in root.iter():
        if tag(g) != "GrantOrContributionPdDurYrGrp":
            continue

        recipient = (text_of(g, "BusinessNameLine1Txt")
                     or text_of(g, "RecipientPersonNm"))

        rows.append({
            "funder_ein": ein,
            "funder_name": funder,
            "tax_year": period[:4],
            "recipient_name": recipient,
            "recipient_city": text_of(g, "CityNm"),
            "recipient_state": text_of(g, "StateAbbreviationCd"),
            "amount": text_of(g, "Amt"),
            "purpose": text_of(g, "GrantOrContributionPurposeTxt"),
        })
    return rows


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    filings = find_filings()
    print(f"\n{len(filings)} filings from {filings['EIN'].nunique()} funders\n")

    download_xml(filings)

    rows, failed = [], []
    for fname in os.listdir(XML_DIR):
        try:
            rows.extend(parse_filing(f"{XML_DIR}/{fname}"))
        except Exception as e:
            failed.append((fname, str(e)))

    df = pd.DataFrame(rows)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0]
    df.to_csv("data/processed/grants.csv", index=False)

    print(f"\nparsed {len(df)} grants")
    print(f"failed files: {len(failed)}")
    print(f"funders with grants: {df['funder_ein'].nunique()}")