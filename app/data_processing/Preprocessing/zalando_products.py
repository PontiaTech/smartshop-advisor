import csv
import re
import pandas as pd


columns_zalando = ["timestamp","domain","country_code","url","sku","condition","gender","product_name","brand","description","manufacturer","badges","initial_price","final_price","discount","currency","inventory","is_sale","in_stock","delivery","root_category","category_tree","main_image","image_count","image_urls","rating","reviews_count","best_rating","worst_rating","rating_count","review_count","top_reviews","product_url","name","SKU","other_attributes","color","colors","sizes","similar_products","people_bought_together","related_products","has_sellback","brand_name"]


def first_url(text):
    if not text:
        return text
    text = str(text)
    # Busca todas las URLs en el texto
    matches = re.findall(r'https?://[^\s",]+', text)
    # Si hay al menos una, nos quedamos con la primera
    return matches[0] if matches else text

with open('./Zalando products.csv', mode='r', encoding="utf-8", errors="replace", newline='') as file, \
     open('./Zalando products Cleaned.csv', mode='w', encoding="utf-8", errors="replace", newline='') as output_file:
    
    outer_reader = csv.reader(file) 
    writer_zalando = csv.writer(output_file, delimiter=',')

    
    outer_header = next(outer_reader)
    raw_header = outer_header[0].strip()
    
    if raw_header.startswith('"') and raw_header.endswith('"'):
        raw_header = raw_header[1:-1]

    # Ahora parseamos ese string como un CSV normal separado por comas
    inner_header = next(csv.reader([raw_header], delimiter=','))
    header_zalando = [col.replace('\ufeff', '').strip().strip('"') for col in inner_header]
    
    print('HEADER_ZALANDO_LIST:', header_zalando)
    print('NUM_COLS:', len(header_zalando))
    for i, col in enumerate(header_zalando):
        print(i, repr(col))
        
    writer_zalando.writerow(header_zalando)
        
    image_idx_zalando = header_zalando.index("main_image")
    url_idx_zalando = header_zalando.index("url")
    image_urls_idx_zalando = header_zalando.index("image_urls")
    product_url_idx_zalando = header_zalando.index("product_url")
    similar_products_idx_zalando = header_zalando.index("similar_products")
    related_products_idx_zalando = header_zalando.index("related_products")
    
    for outer_row in outer_reader:
        if not outer_row:
            continue

        raw_row = outer_row[0].strip()
        if not raw_row:
            continue

        # Quitamos comillas exteriores si las hay
        if raw_row.startswith('"') and raw_row.endswith('"'):
            raw_row = raw_row[1:-1]

        inner_row = next(csv.reader([raw_row], delimiter=','))

        # Por si hay filas corruptas o truncadas
        if len(inner_row) != len(header_zalando):
            continue
        inner_row[image_idx_zalando] = first_url(inner_row[image_idx_zalando])
        inner_row[url_idx_zalando] = first_url(inner_row[url_idx_zalando])
        inner_row[image_urls_idx_zalando] = first_url(inner_row[image_urls_idx_zalando])
        inner_row[product_url_idx_zalando] = first_url(inner_row[product_url_idx_zalando])
        inner_row[similar_products_idx_zalando] = first_url(inner_row[similar_products_idx_zalando])
        inner_row[related_products_idx_zalando] = first_url(inner_row[related_products_idx_zalando])
        writer_zalando.writerow(inner_row)   
        
        
df = pd.read_csv('./Zalando products Cleaned.csv', encoding="utf-8")
print(df.head())
