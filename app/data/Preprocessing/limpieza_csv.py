#Python script que quiere limpiar csv de producto -- No funciona, los csvs son muy complejos

import pandas as pd
import os

# Ruta base (usando ruta absoluta del script)
BASE_DIR = "/Users/paulacamprecios/Documents/Master - Pontia/smartshop-advisor/app/data/Preprocessing"

# Columnas comunes
common_columns = [
    "id", "name", "price", "image", "description",
    "color_code", "family", "url", "source"
]

# Configuración de datasets
datasets = {
    "Mango Products.csv": {
        "usecols": {
            "product_id": "id",
            "product_name": "name",
            "price": "price",
            "image": "image",
            "description": "description",
            "colour_code": "color_code",
            "product_family": "family",
            "url": "url"
        },
        "source": "mango"
    },
    "Zalando products.csv": {
        "usecols": {
            "sku": "id",
            "product_name": "name",
            "final_price": "price",
            "main_image": "image",
            "description": "description",
            "color": "color_code",
            "root_category": "family",
            "product_url": "url"
        },
        "source": "zalando"
    },
    "Zara - Products .csv": {
        "usecols": {
            "product_id": "id",
            "product_name": "name",
            "price": "price",
            "image": "image",
            "description": "description",
            "colour_code": "color_code",
            "product_family": "family",
            "url": "url"
        },
        "source": "zara"
    }
}

dataframes = []

for file_name, config in datasets.items():
    file_path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(file_path):
        print(f"✔️ Procesando {file_name}...")
        df = pd.read_csv(file_path)

        selected = {k: v for k, v in config["usecols"].items() if k in df.columns}
        df = df[list(selected.keys())].rename(columns=selected)

        for col in common_columns:
            if col not in df.columns:
                df[col] = None

        df["source"] = config["source"]
        df = df[common_columns]
        dataframes.append(df)
    else:
        print(f"❌ Archivo no encontrado: {file_path}")

# Guardar resultado final
if dataframes:
    final_df = pd.concat(dataframes, ignore_index=True)
    output_path = os.path.join(BASE_DIR, "all_products.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Total de productos combinados: {len(final_df)}")
    print(f"📦 Archivo creado: {output_path}")
else:
    print("⚠️ No se procesó ningún archivo.")
