"""Day 12 — predict which organizations are at risk of lapsing."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["log_revenue", "expense_ratio", "months_reserve",
            "revenue_trend", "revenue_volatility", "deficit_years", "n_filings"]
SEED = 42


def build_features(raw):
    """One row per organization, summarizing its filing history."""
    for c in ["revenue", "expenses", "assets", "liabilities"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["revenue", "expenses"]).sort_values(["ein", "year"])
    rows = []

    for ein, g in raw.groupby("ein"):
        rev = g["revenue"].clip(lower=1)
        last = g.iloc[-1]

        net = (last["assets"] or 0) - (last["liabilities"] or 0)
        monthly = max(last["expenses"], 1) / 12

        rows.append({
            "ein": ein,
            "label": g["label"].iloc[0],
            "log_revenue": np.log10(rev.iloc[-1]),
            "expense_ratio": last["expenses"] / max(last["revenue"], 1),
            "months_reserve": np.clip(net / monthly, -12, 60),
            "revenue_trend": (rev.iloc[-1] / rev.iloc[0]) ** (1 / max(len(g) - 1, 1)) - 1,
            "revenue_volatility": rev.std() / rev.mean() if len(g) > 1 else 0,
            "deficit_years": int((g["expenses"] > g["revenue"]).sum()),
            "n_filings": len(g),
        })

    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    out["revenue_trend"] = out["revenue_trend"].clip(-1, 3)
    return out


if __name__ == "__main__":
    raw = pd.read_csv("data/processed/health_raw.csv", dtype={"ein": str})
    df = build_features(raw)
    print(f"{len(df)} organizations, {df.label.mean():.1%} lapsed\n")

    Xtr, Xte, ytr, yte = train_test_split(
        df[FEATURES], df["label"], test_size=0.25,
        stratify=df["label"], random_state=SEED)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    base = yte.mean()

    print(f"base rate            {base:.1%}")
    print(f"average precision    {average_precision_score(yte, p):.3f}")
    print(f"lift over base       {average_precision_score(yte, p) / base:.1f}x\n")

    prec, rec, thr = precision_recall_curve(yte, p)
    print("precision at recall thresholds")
    for target in [0.5, 0.7, 0.9]:
        i = np.argmin(np.abs(rec - target))
        print(f"  recall {rec[i]:.0%} -> precision {prec[i]:.0%}")

    print("\nstandardized coefficients")
    coefs = model[-1].coef_[0]
    for f, c in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {f:<20} {c:+.3f}")

    df.groupby("label")[FEATURES].median().T.to_csv("data/processed/health_peers.csv")
    joblib.dump(model, "models/risk.pkl")
    print("\nsaved models/risk.pkl")