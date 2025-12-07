import csv
import re
import pandas as pd

columns = ["product_name","price","image","size","product_article","availability","description","colour_code","product_family","url","country_code","product_id"]

def first_url(text):
    if not text:
        return text
    text = str(text)
    # Busca todas las URLs en el texto
    matches = re.findall(r'https?://[^\s",]+', text)
    # Si hay al menos una, nos quedamos con la primera
    return matches[0] if matches else text

with open('./Mango Products.csv', mode='r', encoding="utf-8", errors="replace", newline='') as file, \
     open('./Mango Products Cleaned.csv', mode='w', encoding="utf-8", errors="replace", newline='') as output_file:
    
    reader = csv.reader(file, delimiter=',')
    writer = csv.writer(output_file, delimiter=',')
    
    header = next(reader)
    writer.writerow(header)
    
    image_idx = header.index("image")
    url_idx = header.index("url")
    
    for row in reader:
        row[image_idx] = first_url(row[image_idx])
        row[url_idx] = first_url(row[url_idx])
        writer.writerow(row)
        
        
df = pd.read_csv('./Mango Products Cleaned.csv', encoding="utf-8")
print(df.head())
    
    # for _ in range(1):
    #     next(reader, None)
        
    # rows = list(reader)
    
    # for row in rows:
    #     print(len(row))
    # min_comas = float('inf')
    # max_comas = float('-inf')
    # for i, row in enumerate(rows):
    #     conteo_comas = str(row).count(',')
    #     if conteo_comas < min_comas:
    #         min_comas = conteo_comas   
    #     if conteo_comas > max_comas:
    #         max_comas = conteo_comas

    # print(f'Minimo comas: {min_comas}, Maximo comas: {max_comas}')
    
    # for i, row in enumerate(rows):
    #     conteo_comas = str(row).count(',')
    #     if conteo_comas == 26:
    #         print(f"Numero fila max: {i} \n\n, Contenido: {row} \n\n")
    #         pos_in = 0
    #         pos_end = 0
    #         for j, col in  enumerate(row):
    #             print(f"Posicion: {j}, Contenido: {col}")
    #     elif conteo_comas == 11:
    #         print(f"Numero fila min: {i} \n\n, Contenido: {row} \n\n")
    
    # cont_max = 0
    # cont_min = 0
    # cont_others = 0
    # for i, row in enumerate(rows):
    #     conteo_comas = str(row).count(',')
    #     if conteo_comas == 26:
    #         cont_max += 1
    #     elif conteo_comas == 12:
    #         cont_min += 1
    #     else:
    #         cont_others +=1
    #         print(f"Fila con otro numero de comas: {i}, Contenido: {row}, Numero de comas: {conteo_comas}\n\n")
            
    # print(f'Total filas con max comas: {cont_max}, Total filas con min comas: {cont_min}, Total filas con otros comas: {cont_others}')