"""train the ranking model and measure it against baselines."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURES = ["similarity", "cause_share", "same_city", "log_grants", "log_median"]
K = 10
SEED = 42


def recall_at_k(full, scores, k=K):
    """Share of true funders landing in each recipient's top k."""
    d = full.copy()
    d["score"] = scores
    hits = total = 0
    for _, g in d.groupby("recipient_ein"):
        top = set(g.nlargest(k, "score")["funder_ein"])
        true = set(g.loc[g["label"] == 1, "funder_ein"])
        hits += len(top & true)
        total += len(true)
    return hits / total


if __name__ == "__main__":
    dt = {"funder_ein": str, "recipient_ein": str}
    tr = pd.read_csv("data/processed/pairs_train.csv", dtype=dt)
    full = pd.read_csv("data/processed/pairs_test_full.csv", dtype=dt)

    Xtr, ytr = tr[FEATURES], tr["label"]
    Xte, yte = full[FEATURES], full["label"]

    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(Xtr, ytr)

    gbm = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    gbm.fit(Xtr, ytr)

    p_logit = logit.predict_proba(Xte)[:, 1]
    p_gbm = gbm.predict_proba(Xte)[:, 1]

    rng = np.random.default_rng(SEED)
    p_rand = rng.random(len(full))

    print("\nAUC on held-out organizations")
    print(f"  logistic regression   {roc_auc_score(yte, p_logit):.3f}")
    print(f"  gradient boosting     {roc_auc_score(yte, p_gbm):.3f}")

    print(f"\nRecall@{K} — can we surface the real funder in the top {K}?")
    print(f"  random baseline       {recall_at_k(full, p_rand):.1%}")
    print(f"  similarity only       {recall_at_k(full, full['similarity']):.1%}")
    print(f"  cause share only      {recall_at_k(full, full['cause_share']):.1%}")
    print(f"  logistic regression   {recall_at_k(full, p_logit):.1%}")
    print(f"  gradient boosting     {recall_at_k(full, p_gbm):.1%}")

    print("\nLogistic regression coefficients")
    for f, c in sorted(zip(FEATURES, logit.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {f:<16} {c:+.3f}")

    joblib.dump(gbm, "models/ranker.pkl")
    print("\nsaved models/ranker.pkl")