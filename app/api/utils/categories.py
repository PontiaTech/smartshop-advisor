import pandas as pd


MANGO_CSV = "./data/Mango Products Cleaned.csv"
ZARA_CSV = "./data/Zara - Products Cleaned.csv"

df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8").fillna("")
df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8").fillna("")

fam_mango = df_mango["product_family"].astype(str).str.strip()
fam_zara = df_zara["product_family"].astype(str).str.strip()
uniques = pd.Series(pd.concat([fam_mango, fam_zara], ignore_index=True).unique())
uniques.to_csv("./data/raw_families.csv", index=False, header=["raw_family"])

