import time
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

BASE = "https://projects.propublica.org/nonprofits/api/v2/search.json"
HEADERS = {"User-Agent": "ledger-nonprofit-research-tool"}

MAX_PAGES = None  

def fetch_all(params, max_pages=None, checkpoint=None):
    """Page through ProPublica search results into one DataFrame."""
    rows, page = [], 0
    while True:
        r = requests.get(BASE, params={**params, "page": page},
                         headers=HEADERS, timeout=30)

        # 404 here means the filter matched nothing, not a real failure
        if r.status_code == 404:
            print("no results for this filter")
            break

        # any other bad status (429 rate limit, 5xx) should stop the run
        r.raise_for_status()

        d = r.json()
        rows.extend(d["organizations"])
        print(f"page {page + 1}/{d['num_pages']} — {len(rows)} orgs")

        # save partial progress so a crash doesn't cost the whole pull
        if checkpoint and page % 20 == 0:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)

        page += 1
        if page >= d["num_pages"] or (max_pages and page >= max_pages):
            break
        time.sleep(0.5)

    return pd.DataFrame(rows)


def get_grantmakers(max_pages=None):
    """MA foundations, plus 4947(a)(1) trusts (no NTEE filter — often blank)."""
    orgs = fetch_all({"state[id]": "MA", "ntee[id]": 7, "c_code[id]": 3},
                     max_pages)
    trusts = fetch_all({"state[id]": "MA", "c_code[id]": 92}, max_pages)

    df = pd.concat([orgs, trusts], ignore_index=True)
    return df.drop_duplicates(subset="ein")


def get_nonprofits(max_pages=None):
    """MA 501(c)(3) organizations — the grant recipient side."""
    return fetch_all({"state[id]": "MA", "c_code[id]": 3}, max_pages,
                     checkpoint="data/raw/_nonprofits_partial.csv")


if __name__ == "__main__":
    gm = get_grantmakers(MAX_PAGES)
    gm.to_csv("data/raw/ma_grantmakers.csv", index=False)
    print(f"\ngrantmakers: {len(gm)}\n")

    npos = get_nonprofits(MAX_PAGES)
    npos.to_csv("data/raw/ma_nonprofits.csv", index=False)
    print(f"\nnonprofits: {len(npos)}")