import re
import spacy
# from dataclasses import dataclass, field
from typing import Any
from app.observability.logging_config import setup_logger

from app.api.schemas import ChatMessage

logger = setup_logger()

PRODUCT_HINTS = {
    # --- ROPA GENERAL ---
    "camisa", "camisas", "camiseta", "camisetas", "top", "tops",
    "blusa", "blusas", "polo", "polos",
    "sudadera", "sudaderas", "hoodie", "hoodies",
    "jersey", "jerséis", "sueter", "suéter", "sweater", "pullover",
    "chaqueta", "chaquetas", "cazadora", "cazadoras",
    "abrigo", "abrigos", "parka", "parkas", "anorak", "anoraques",
    "chaleco", "chalecos",
    "cardigan", "cárdigan", "cardigans",

    # --- PANTALONES ---
    "pantalon", "pantalones", "pantalón", "pantalones",
    "vaquero", "vaqueros", "jean", "jeans", "denim",
    "jogger", "joggers", "chandal", "chándal", "sudadera",
    "leggins", "leggings", "short", "shorts", "bermuda", "bermudas",
    "cargo", "cargos", "chino", "chinos",

    # --- FALDAS / VESTIDOS ---
    "falda", "faldas", "vestido", "vestidos",
    "mono", "monos", "peto", "petos",

    # --- ROPA INTERIOR / DEPORTIVA ---
    "lenceria", "lencería", "sujetador", "sujetadores", "bralette",
    "calzoncillo", "calzoncillos", "boxer", "boxers",
    "braga", "bragas",
    "pijama", "pijamas",
    "bano", "baño", "bikini", "bikinis", "trikini",
    "banador", "bañador", "bañadores",

    # --- CALZADO ---
    "zapatilla", "zapatillas", "zapatilla deportiva", "zapatillas deportivas",
    "zapato", "zapatos",
    "bota", "botas", "botin", "botines",
    "sandalia", "sandalias",
    "mocasin", "mocasín", "mocasines",
    "chancla", "chanclas",
    "tacon", "tacón", "tacones",

    # --- ACCESORIOS ---
    "bolso", "bolsos", "mochila", "mochilas",
    "riñonera", "riñoneras",
    "gorra", "gorras", "sombrero", "sombreros",
    "bufanda", "bufandas", "pañuelo", "pañuelos",
    "guante", "guantes",
    "cinturon", "cinturón", "cinturones",
    "gafas", "gafas de sol",
    "calcetin", "calcetín", "calcetines",
    "media", "medias",

    # --- INGLES ---
    "shirt", "shirts", "tshirt", "t-shirt", "tshirts",
    "top", "tops",
    "pants", "trousers", "jeans", "shorts",
    "skirt", "skirts", "dress", "dresses",
    "jacket", "jackets", "coat", "coats",
    "sweatshirt", "sweatshirts", "hoodie", "hoodies",
    "sneaker", "sneakers", "trainer", "trainers",
    "boot", "boots", "shoe", "shoes",
    "bag", "bags", "backpack", "backpacks",

    # (corrección) Se han eliminado de PRODUCT_HINTS términos que son atributos, no producto:
    # "basico/básico/basicos/básicos", "casual", "formal", "elegante", "clasico/clásico",
    # "oversize", "slim", "regular", "fit"
}


FOLLOWUP_MARKERS = {
    "y", "pero", "ademas", "además", "tambien", "también", "vale", "ok",
    "mejor", "peor", "mas", "más", "menos", "igual", "otra", "otro",
    "ese", "esa", "eso", "estos", "estas", "aquel", "aquella",
    "lo", "la", "los", "las", "uno", "una",
    "mismo", "misma", "mismos", "mismas",
    "cambia", "cambiar", "poner", "quita", "quitar",
    "en", "sin", "con",
}

PRONOUN_FOLLOWUP_START = {
    "eso", "esa", "ese", "esto", "esta", "este",
    "lo", "la", "los", "las", "el", "ella", "ellos", "ellas",
    "mismo", "misma", "igual", "igualito",
}

ATTRIBUTE_HINTS = {
    # --- USO / OCASIÓN ---
    "deporte", "sport", "sports", "deportivo", "deportiva", "training", "entrenamiento",
    "gym", "gimnasio", "fitness", "workout", "running", "runner", "correr", "trail", "senderismo", "hiking",
    "tenis", "padel", "pádel", "futbol", "fútbol", "basket", "baloncesto", "yoga", "pilates",
    "natacion", "natación", "swim", "playa", "beach", "surf",
    "oficina", "trabajo", "work", "business", "reunion", "reunión", "meeting",
    "casual", "informal", "diario", "everyday",
    "formal", "elegante", "smart", "smartcasual", "dressy",
    "fiesta", "party", "noche", "nightout", "cocktail", "cóctel",
    "boda", "wedding", "evento", "events", "ceremonia", "ceremonia", "graduacion", "graduación",
    "cena", "dinner", "comida", "lunch",
    "viaje", "travel", "vacaciones", "holiday", "weekend", "fin", "semana",
    "casa", "home", "loungewear", "relax", "relajado", "pijama", "sleep",
    "cole", "uni", "universidad", "campus",

    # --- ESTACIÓN / CLIMA ---
    "verano", "summer", "invierno", "winter", "otoño", "otoño", "autumn", "primavera", "spring",
    "calor", "heat", "fresco", "fresh", "frio", "frío", "cold", "templado", "warm",
    "lluvia", "rain", "impermeable", "waterproof", "repelente", "waterrepellent",
    "viento", "wind", "cortavientos", "windbreaker",
    "nieve", "snow", "termico", "térmico", "thermal",
    "transpirable", "breathable", "ventilado", "vented",
    "ligero", "ligera", "lightweight", "pesado", "heavy",

    # --- COLOR / TONO ---
    "negro", "black", "blanco", "white", "gris", "gray", "grey",
    "azul", "blue", "navy", "marino",
    "rojo", "red", "verde", "green", "oliva", "olive", "khaki",
    "beige", "camel", "crema", "cream", "arena", "sand",
    "marron", "marrón", "brown", "chocolate",
    "rosa", "pink", "fucsia", "fuchsia",
    "morado", "purple", "lila", "lilac", "violeta", "violet",
    "amarillo", "yellow", "mostaza", "mustard",
    "naranja", "orange",
    "burdeos", "burgundy", "granate", "maroon",
    "metalico", "metálico", "silver", "oro", "gold", "dorada", "plateado", "plateada",
    "neon", "neón", "pastel",
    "monocromo", "monochrome", "bicolor", "multicolor",

    # --- ESTAMPADOS / PATRONES ---
    "liso", "lisa", "plain", "solid",
    "estampado", "estampada", "print", "printed",
    "rayas", "striped", "stripe", "a_rayas",
    "cuadros", "checked", "check", "tartan", "escoces", "escocés",
    "lunares", "polkadot", "polka", "dots",
    "flores", "floral", "flower",
    "animal", "animalprint", "leopardo", "leopard", "cebra", "zebra",
    "camuflaje", "camo", "tie_dye", "tiedye",
    "logo", "logotipo", "graphic", "grafico", "gráfico",
    "bordado", "embroidered",
    "jacquard",
    "paisley", "cachemir", "cachemira",

    # --- CORTE / FIT / SILUETA ---
    "oversize", "over", "baggy", "holgado", "holgada", "loose",
    "slim", "slimfit", "ajustado", "ajustada", "entallado", "entallada",
    "regular", "regularfit", "normal",
    "skinny", "pitillo",
    "recto", "straight", "straightfit",
    "wide", "wideleg", "pierna_ancha", "ancho", "ancha",
    "flare", "campana", "campana", "bootcut",
    "cropped", "tobillero", "tobillera",
    "alto", "alta", "highrise", "tiro_alto",
    "medio", "media", "midrise", "tiro_medio",
    "bajo", "baja", "lowrise", "tiro_bajo",
    "long", "largo", "larga", "short", "corto", "corta", "midi", "mini",
    "asimetric", "asimetrico", "asimétrico",
    "peplum", "evasé", "evase",
    "wrap", "cruzado", "cruzada",
    "boxy", "cuadrado", "cuadrada",
    "relajado", "relajada", "relaxed",

    # --- TALLA / AJUSTE RÁPIDO ---
    "xs", "s", "m", "l", "xl", "xxl", "xxxl",
    "petite", "tall", "curvy", "plus", "plussize", "grande", "pequeno", "pequeño",
    "unisex", "hombre", "mujer", "chico", "chica",

    # --- MATERIALES / TEJIDOS ---
    "algodon", "algodón", "cotton",
    "lino", "linen",
    "lana", "wool", "merino", "cashmere", "cachemira",
    "denim", "vaquero", "jean",
    "cuero", "piel", "leather", "ante", "suede",
    "poliester", "poliéster", "polyester", "nylon",
    "viscosa", "viscose", "rayon",
    "seda", "silk", "satin", "satén",
    "terciopelo", "velvet",
    "pana", "corduroy",
    "punto", "knit", "jersey", "tricot",
    "felpa", "fleece",
    "neopreno", "neoprene",
    "encaje", "lace",
    "tul", "tulle",
    "gasa", "chiffon",
    "licra", "lycra", "elastano", "elastano", "elastane", "spandex",
    "microfibra", "microfiber",
    "plumas", "down", "relleno", "padded", "acolchado", "acolchada",
    "borrego", "sherpa",
    "reciclado", "recycled", "organic", "organico", "orgánico",

    # --- PROPIEDADES TÉCNICAS ---
    "transpirable", "breathable",
    "impermeable", "waterproof",
    "resistente", "durable",
    "antiarrugas", "wrinklefree",
    "elasticidad", "elastico", "elástico", "stretch",
    "compresion", "compresión", "compression",
    "rapido", "rápido", "quickdry", "secado", "rapido_secado",
    "termico", "térmico", "thermal",
    "suave", "soft",
    "calido", "cálido", "warm",
    "ligero", "lightweight",
    "antiviento", "windproof",
    "reflectante", "reflective",
    "antiodor", "antiodor",
    "antimanchas", "stainrepellent",
    "uv", "proteccion_uv", "protección_uv",

    # --- DETALLES / CONSTRUCCIÓN ---
    "capucha", "hood",
    "cremallera", "zip", "zipper",
    "botones", "buttons", "buttondown",
    "cordon", "cordón", "drawstring",
    "cintura_elastica", "cintura_elástica", "elastic_waist",
    "cinturon", "cinturón", "belt",
    "bolsillos", "pockets", "con_bolsillos", "sin_bolsillos",
    "dobladillo", "hem",
    "puños", "cuffs",
    "cuello_alto", "turtleneck", "mockneck",
    "cuello_pico", "vneck", "cuello_v",
    "cuello_redondo", "crewneck",
    "tirantes", "straps",
    "espalda_descubierta", "openback",
    "raya", "costuras", "seams",
    "doble", "double", "forro", "lined",
    "sin_forro", "unlined",
    "acolchado", "padded",
    "plisado", "pleated",
    "volantes", "ruffles",
    "aberturas", "slit", "slits",
    "crochet", "ganchillo",
    "tejido", "woven",

    # --- ESTILO / ESTÉTICA ---
    "basico", "básico", "basic",
    "minimal", "minimalista",
    "clasico", "clásico", "classic",
    "moderno", "modern",
    "vintage", "retro",
    "urbano", "street", "streetwear",
    "preppy",
    "boho", "bohemio",
    "grunge",
    "de_tendencia", "trendy",
    "sastre", "tailored",
    "athleisure",

    # --- PRECIO / CALIDAD ---
    "barato", "barata", "cheap",
    "economico", "económico",
    "calidad", "premium", "lujo", "luxury",
    "rebajas", "sale", "oferta", "discount",
}

CURRENCY_TOKENS = {
    "€", "eur", "euro", "euros", "usd", "$", "dolar", "dólar", "dolares", "dólares",
    "gbp", "£", "libra", "libras", "chf", "franco", "francos",
}

GENERIC_NOUNS = {
    # --- dinero / precio ---
    "euro", "euros", "eur", "dolar", "dólar", "dolares", "dólares", "usd", "gbp", "libra", "libras",
    "precio", "coste", "costo", "presupuesto", "importe", "valor", "rebaja", "rebajas", "descuento", "descuentos",
    "oferta", "ofertas", "promocion", "promoción", "saldo", "saldos", "outlet", "chollo", "chollos",
    "barato", "barata", "caro", "cara", "económico", "económica", "asequible",

    # --- marca / tienda / origen ---
    "marca", "marcas", "firma", "firmas", "logo", "logos",
    "tienda", "tiendas", "web", "pagina", "página", "sitio", "portal", "marketplace",
    "zalando", "amazon", "asos", "mango", "zara", "bershka", "pull", "bear", "hm", "h&m",
    "envio", "envío", "delivery", "shipping",

    # --- tallas / medidas ---
    "talla", "tallas", "size", "sizes", "medida", "medidas",
    "pequeña", "pequeño", "grande", "grandes", "ajustado", "ajustada", "holgado", "holgada",
    "xs", "s", "m", "l", "xl", "xxl",
    "fit", "regular", "slim", "oversize",
    "largo", "larga", "corto", "corta", "longitud",
    "ancho", "ancha", "estrecho", "estrecha",
    "cintura", "cadera", "pecho", "hombro", "manga", "pierna", "tiro",
    "cm", "mm", "metros", "metro",

    # --- color / estampado ---
    "color", "colores", "tono", "tonos",
    "estampado", "estampados", "print", "prints", "patron", "patrón", "rayas", "liso", "lisa",
    "cuadros", "floral", "flores", "animal", "leopardo", "tigre", "camuflaje", "tie", "dye",

    # --- material / composición ---
    "material", "materiales", "tejido", "tejidos", "composicion", "composición",
    "algodon", "algodón", "poliester", "poliéster", "nylon", "elastano", "elastano", "lycra",
    "lino", "lana", "cuero", "piel", "ante", "gamuza", "vaquero", "denim", "satin", "satén",
    "punto", "felpa", "forro", "impermeable", "transpirable",

    # --- estilo / ocasión ---
    "estilo", "look", "outfit", "casual", "formal", "elegante", "fiesta", "boda", "oficina",
    "deporte", "sport", "running", "gym", "fitness", "entrenamiento",
    "verano", "invierno", "otoño", "primavera",
    "basico", "básico", "clasico", "clásico", "moderno", "vintage", "retro", "minimalista",
    "urbano", "streetwear",

    # --- detalles de prenda (atributos, no “tema”) ---
    "capucha", "cordones", "cremallera", "botones", "bolsillos", "cinturon", "cinturón",
    "cuello", "manga", "mangas", "tirantes", "sinmangas", "forrado", "acolchado",
    "corte", "patronaje", "patrón", "costura", "costuras", "dobladillo",

    # --- condiciones / logística ---
    "envio", "envío", "entrega", "devolucion", "devolución", "cambios", "gratis", "gratuito",
    "stock", "disponible", "agotado", "disponibilidad", "tarda", "llega",
    "nuevo", "nueva", "segunda", "mano", "usado", "usada",

    # --- comparativos / ajustes típicos de follow-up ---
    "mejor", "peor", "mas", "más", "menos", "otra", "otro", "igual",
    "parecido", "similar", "diferente", "alternativa", "alternativas",
    "recomienda", "recomendacion", "recomendación",
}

# Ajusta esto a tu dominio/datasets. Con 30-80 tokens suele ir sobrado.
ITEM_TYPES = {
    # calzado
    "botas", "bota", "botin", "botín", "zapatillas", "zapatos", "sandalias", "mocasines",
    # outerwear
    "abrigo", "chaqueta", "cazadora", "parka", "anorak",
    # tops
    "jersey", "sudadera", "camiseta", "camisa", "top",
    # bottoms
    "pantalon", "pantalón", "vaqueros", "jeans", "shorts", "falda",
    # dresses + accessories
    "vestido", "bolso", "mochila", "bufanda", "gorra", "guantes",
    # EN variants comunes
    "boots", "boot", "sneaker", "sneakers", "trainers", "shoe", "shoes",
    "coat", "jacket", "hoodie", "sweater", "pullover", "cardigan",
    "dress", "skirt", "pants", "trousers", "bag", "backpack",
}

COLORS = {
    "negro", "blanco", "azul", "rojo", "verde", "beige", "gris", "marron", "marrón",
    "amarillo", "naranja", "rosa", "morado", "violeta", "burdeos",
    "navy", "black", "white", "blue", "red", "green", "beige", "grey", "gray", "brown",
}

USES = {
    "deporte", "deportivas", "sport", "training", "gimnasio", "gym", "correr", "running", "trail",
    "casual", "formal", "elegante", "fiesta", "oficina", "senderismo", "hiking",
}

BRANDS = {
    # mete tus marcas de catálogo/web (Zalando, Nike, Adidas, Mango, Zara, etc.)
    "nike", "adidas", "puma", "reebok", "newbalance",
    "mango", "zara", "bershka", "pull", "bear", "stradivarius", "massimodutti", "massimo", "dutti",
    "asics", "converse", "vans", "skechers",
}

SIZE_RE = re.compile(r"\b(talla|size)\s*(\d{2,3}|xs|s|m|l|xl|xxl)\b", re.IGNORECASE)
PRICE_MAX_RE = re.compile(r"(?:menos\s+de|hasta|máximo|max|<)\s*(\d{1,4})(?:\s*€|euros)?", re.IGNORECASE)
PRICE_MIN_RE = re.compile(r"(?:más\s+de|desde|mínimo|min|>)\s*(\d{1,4})(?:\s*€|euros)?", re.IGNORECASE)

def load_nlp():
    # intenta ES completo
    try:
        return spacy.load("es_core_news_sm")
    except Exception:
        pass
    # intenta EN (por si tu imagen ya lo trae)
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        pass
    # fallback mínimo (no deps)
    return spacy.blank("es")

nlp = load_nlp()

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", _norm(s))

def _clean_user_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def sanitize_web_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"https?://\S+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:200]

def sanitize_rag_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)

    # corta dumps de objetos tipo ChatMessage o logs
    if "sender='assistant'" in q or 'sender="assistant"' in q:
        q = re.sub(r"sender=['\"]assistant['\"].*", "", q).strip()

    # quita markdown pesado y urls, pero sin recortar tan agresivo
    q = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", q)
    q = re.sub(r"https?://\S+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    return q[:500]

def history_last_user_query(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        sender = (getattr(m, "sender", "") or "").lower()
        content: Any = getattr(m, "content", None)
        if sender != "user":
            continue
        if isinstance(content, str):
            txt = _clean_user_text(content)
            if txt:
                return txt
    return ""

def tiene_sujeto_spacy(texto: str) -> bool:
    t = _norm(texto)
    if not t:
        return False
    try:
        doc = nlp(t)
        # si no hay parser, dep_ suele venir vacío
        return any(tok.dep_ in ("nsubj", "nsubj_pass") for tok in doc)
    except Exception:
        return False


def tiene_tema_producto(texto: str) -> bool:
    t = _norm(texto)
    if not t:
        return False

    words = _tokenize(t)

    # Tema fuerte: tipo de producto
    if any(w in PRODUCT_HINTS for w in words):
        return True
    if any(w in ITEM_TYPES for w in words):
        return True

    # spaCy como señal secundaria, pero filtrando “ruido”
    try:
        doc = nlp(t)
        nouns = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            w = _norm(tok.text)
            if tok.pos_ in {"NOUN", "PROPN"}:
                if w in BRANDS:
                    continue
                if w in GENERIC_NOUNS:
                    continue
                if w in CURRENCY_TOKENS:
                    continue
                nouns.append(w)

        return len(nouns) > 0
    except Exception:
        return False

def has_only_attributes(text: str) -> bool:
    t = _norm(text)
    words = _tokenize(t)
    if not words:
        return False

    has_item = any(w in PRODUCT_HINTS or w in ITEM_TYPES for w in words)
    if has_item:
        return False

    has_brand = any(w in BRANDS for w in words)
    has_color = any(w in COLORS for w in words)
    has_use = any(w in USES for w in words)
    has_attr = any(w in ATTRIBUTE_HINTS for w in words)

    has_price = bool(PRICE_MAX_RE.search(t) or PRICE_MIN_RE.search(t) or re.search(r"\b\d{1,4}\s*€\b", t))
    has_size = bool(SIZE_RE.search(t))

    return has_brand or has_color or has_use or has_attr or has_price or has_size


def is_followup_query(text: str) -> bool:
    t = _norm(text)
    if not t:
        return False

    # Si es solo atributos (marca/precio/color/uso/talla…) -> follow-up
    if has_only_attributes(t):
        return True

    words = _tokenize(t)
    if not words:
        return False

    # tus reglas fuertes
    if len(words) >= 2 and words[0] == "y" and words[1] in {"que", "de", "con", "sin", "en"}:
        return True
    if len(words) >= 2 and words[0] == "que" and words[1] in {"sea", "sean"}:
        return True
    if words[0] in PRONOUN_FOLLOWUP_START:
        return True

    # si hay tema/sujeto real, NO follow-up
    if tiene_sujeto_spacy(t) or tiene_tema_producto(t):
        return False

    if len(words) <= 4:
        return True

    hits = sum(1 for w in words if w in FOLLOWUP_MARKERS)
    return (hits / max(len(words), 1)) >= 0.35


def has_explicit_subject(text: str) -> bool:
    """
    Renómbralo mentalmente a: "tiene_tema".
    Si tiene tema, es consulta nueva.
    """
    return tiene_sujeto_spacy(text) or tiene_tema_producto(text)

def history_last_user_query_with_product(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if (getattr(m, "sender", "") or "").lower() != "user":
            continue
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        txt = _clean_user_text(content)
        if not txt:
            continue
        if any(w in PRODUCT_HINTS for w in _tokenize(txt)):
            return txt
    return history_last_user_query(history)

def build_search_query(user_query: str, history: list[ChatMessage] | None) -> tuple[str, bool]:
    uq = (user_query or "").strip()
    last_q = history_last_user_query_with_product(history)

    logger.info(
        "CTX_DECISION uq=%r last_q=%r followup=%s explicit=%s tema=%s sujeto=%s",
        uq, last_q,
        is_followup_query(uq),
        has_explicit_subject(uq),
        tiene_tema_producto(uq),
        tiene_sujeto_spacy(uq),
    )

    if not uq:
        return (last_q, True) if last_q else ("", False)

    if not last_q:
        return uq, False

    # Regla nueva: si no tiene sujeto explícito, tratamos como follow-up
    needs_context = is_followup_query(uq) or (not has_explicit_subject(uq))

    if needs_context:

        uq2 = re.sub(r"^\s*y\s+", "", uq, flags=re.IGNORECASE).strip()
        uq2 = uq2.rstrip("?").strip()

        # evita duplicación simple: si el ajuste ya está incluido en la última query
        if _norm(uq2) and _norm(uq2) in _norm(last_q):
            return last_q, True

        if uq2:
            return f"{last_q}. {uq2}", True
        else:
            return last_q, True

    return uq, False
