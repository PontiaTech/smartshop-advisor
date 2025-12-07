# import pandas as pd


# MANGO_CSV = "./data/Mango Products Cleaned.csv"
# ZARA_CSV = "./data/Zara - Products Cleaned.csv"

# df_mango = pd.read_csv(MANGO_CSV, encoding="utf-8").fillna("")
# df_zara = pd.read_csv(ZARA_CSV, encoding="utf-8").fillna("")

# fam_mango = df_mango["product_family"].astype(str).str.strip()
# fam_zara = df_zara["product_family"].astype(str).str.strip()
# uniques = pd.Series(pd.concat([fam_mango, fam_zara], ignore_index=True).unique())
# uniques.to_csv("./data/raw_families.csv", index=False, header=["raw_family"])

# import pandas as pd

# df_mapping = pd.read_csv(
#     "./data/families_translated.csv",
#     encoding="utf-8",
#     sep=None,
#     engine="python",
# )

# print("Columnas mapping:", repr(list(df_mapping.columns)))
# print(df_mapping.head(5))

import pandas as pd

df = pd.read_csv("./data/Mango Products Cleaned.csv", encoding="utf-8")

print(df.columns)

# Suponiendo que la columna "rara" se llama algo tipo 'color_code' o similar
col = "colour_code"  # cambia esto por el nombre real
print(df[col].head(50))
print(df[col].value_counts().head(50))