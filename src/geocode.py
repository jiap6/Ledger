"""resolve city names to coordinates, cached so it runs once."""

import os

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

CACHE = "data/processed/city_coords.csv"
UA = "ledger-nonprofit-research-tool"


def all_cities():
    f = pd.read_csv("data/processed/funders.csv")["funder_city"]
    r = pd.read_csv("data/processed/recipients.csv")["city"]
    s = pd.concat([f, r]).dropna().astype(str).str.upper().str.strip()
    return sorted(s[s != ""].unique())


if __name__ == "__main__":
    known = (pd.read_csv(CACHE) if os.path.exists(CACHE)
             else pd.DataFrame(columns=["city", "lat", "lon"]))
    todo = [c for c in all_cities() if c not in set(known["city"])]
    print(f"{len(known)} cached, {len(todo)} to look up")

    # Nominatim allows one request per second — the limiter enforces it
    geocode = RateLimiter(Nominatim(user_agent=UA, timeout=10).geocode,
                          min_delay_seconds=1.1)

    rows = []
    for i, city in enumerate(todo, 1):
        try:
            loc = geocode(f"{city}, Massachusetts, USA")
            if loc:
                rows.append({"city": city, "lat": loc.latitude,
                             "lon": loc.longitude})
        except Exception as e:
            print(f"  {city}: {e}")

        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
            pd.concat([known, pd.DataFrame(rows)]).to_csv(CACHE, index=False)

    out = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
    out.drop_duplicates("city").to_csv(CACHE, index=False)
    print(f"\n{len(out)} cities geocoded")