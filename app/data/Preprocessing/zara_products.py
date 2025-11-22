import csv
import re
import pandas as pd

columns = ["category_id","product_id","product_name","price","colour_code","colour","description","size","section","product_family","product_subfamily","care","materials_description","materials","dimension","low_on_stock","availability","image","sku","url","currency","you_may_also_like","seo_category_id"]

def first_url(text):
    if not text:
        return text
    text = str(text)
    # Busca todas las URLs en el texto
    matches = re.findall(r'https?://[^\s",]+', text)
    # Si hay al menos una, nos quedamos con la primera
    return matches[0] if matches else text


with open('./Zara - Products .csv', mode='r', encoding="utf-8", errors="replace", newline='') as file, \
     open('./Zara - Products Cleaned.csv', mode='w', encoding="utf-8", errors="replace", newline='') as output_file:
         
    outer_reader = csv.reader(file)
    writer_zara = csv.writer(output_file, delimiter=',')

    
    outer_header = next(outer_reader)
    if not outer_header:
        raise ValueError("El fichero de Zara está vacío o no tiene header válido")

    raw_header = outer_header[0].strip()

    if raw_header.startswith('"') and raw_header.endswith('"'):
        raw_header = raw_header[1:-1]

    # Quitamos ; finales tipo '...seo_category_id;;;;;;'
    raw_header = raw_header.rstrip(';')
    inner_header = next(csv.reader([raw_header], delimiter=','))
    header_zara = [col.replace('\ufeff', '').strip().strip('"') for col in inner_header]

    print('HEADER_ZARA_LIST:', header_zara)
    print('NUM_COLS_ZARA:', len(header_zara))
    for i, col in enumerate(header_zara):
        print(i, repr(col))

    writer_zara.writerow(header_zara)

    image_idx_zara = header_zara.index("image")
    url_idx_zara = header_zara.index("url")

    for outer_row in outer_reader:
        if not outer_row:
            continue

        if len(outer_row) != 1:
            continue

        raw_row = outer_row[0].strip()
        if not raw_row:
            continue

        # Quitamos comillas exteriores si las hay
        if raw_row.startswith('"') and raw_row.endswith('"'):
            raw_row = raw_row[1:-1]

        # Quitamos ; finales
        raw_row = raw_row.rstrip(';')

        inner_row = next(csv.reader([raw_row], delimiter=','))

        # Por si hay filas corruptas o truncadas
        if len(inner_row) != len(header_zara):
            continue

        # Normalizamos image y url a primera URL
        inner_row[image_idx_zara] = first_url(inner_row[image_idx_zara])
        inner_row[url_idx_zara] = first_url(inner_row[url_idx_zara])

        writer_zara.writerow(inner_row)


df = pd.read_csv('./Zara - Products Cleaned.csv', encoding="utf-8")
print(df.head())
print(df.shape)