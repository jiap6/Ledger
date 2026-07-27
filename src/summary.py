"""One glance at whether the whole pipeline looks healthy."""

import pandas as pd

fund = pd.read_csv("data/raw/ma_grantmakers.csv")
nonp = pd.read_csv("data/raw/ma_nonprofits.csv")
gr = pd.read_csv("data/processed/grants_matched.csv")
tr = pd.read_csv("data/processed/pairs_train.csv")
te = pd.read_csv("data/processed/pairs_test.csv")

ma = gr["recipient_state"].eq("MA")
sim = tr.groupby("label")["similarity"].mean()

print(f"""
funders found        {len(fund):>10,}
nonprofits found     {len(nonp):>10,}
grants parsed        {len(gr):>10,}
funders with grants  {gr.funder_ein.nunique():>10,}
total granted        ${gr.amount.sum()/1e6:>9,.0f}M
median grant         ${gr.amount.median():>10,.0f}
MA grants matched    {gr.loc[ma,'recipient_ein'].notna().mean():>10.1%}
train pairs          {len(tr):>10,}
test pairs           {len(te):>10,}
similarity gap       {sim[1]-sim[0]:>10.3f}
""")

print("sample grants:")
print(gr[["funder_name", "recipient_name", "amount"]].sample(5).to_string(index=False))