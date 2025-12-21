import re
from typing import Any
import spacy
from app.api.schemas import ChatMessage

# Intento cargar ES; si no, EN; si no, blank("es")
# Nota: blank no da POS/NER fiables, por eso añadimos heurísticas extra.
try:
    nlp = spacy.load("es_core_news_sm")
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("es")

FOLLOWUP_MARKERS = {
    "y", "pero", "ademas", "además", "tambien", "también", "vale", "ok",
    "mejor", "peor", "mas", "más", "menos", "igual", "otra", "otro",
    "ese", "esa", "eso", "estos", "estas", "aquel", "aquella",
    "lo", "la", "los", "las", "uno", "una",
    "mismo", "misma", "mismos", "mismas",
    "cambia", "cambiar", "poner", "quita", "quitar",
    "en", "sin", "con",
    # follow-ups típicos de filtros
    "negro", "blanco", "azul", "rojo", "verde", "beige", "gris", "marron", "marrón",
    "barato", "cara", "caro", "clasico", "clásico", "formal", "casual", "elegante",
}

# Señales de "producto" para no confundir "quiero botas chelsea" con un follow-up
PRODUCT_HINTS = {
    "botas", "bota", "botin", "botín", "zapatillas", "zapatos", "sandalias", "mocasines",
    "abrigo", "chaqueta", "cazadora", "parka", "anorak",
    "jersey", "sudadera", "camiseta", "pantalon", "pantalón", "vaqueros", "jeans",
    "falda", "vestido", "bolso", "mochila", "bufanda", "gorra", "guantes",
    # variantes comunes EN/FR/DE/IT que aparecen en datasets
    "boots", "boot", "sneaker", "sneakers", "trainers", "shoe", "shoes",
    "coat", "jacket", "parka", "hoodie", "sweater", "pullover", "cardigan",
    "dress", "skirt", "pants", "trousers", "jeans",
    "tasche", "rucksack", "schal", "kleid", "hose", "schuhe", "stiefel",
    "manteau", "veste", "robe", "jupe", "pantalon",
    "cappotto", "giacca", "abito", "gonna", "scarpe", "stivali",
}

PRONOUN_FOLLOWUP_START = {
    "eso", "esa", "ese", "esto", "esta", "este",
    "lo", "la", "los", "las", "el", "ella", "ellos", "ellas",
    "mismo", "misma", "igual", "igualito",
}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", _norm(s))

def history_last_user_query(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if (m.sender or "").lower() == "user" and (m.content or "").strip():
            return m.content.strip()
    return ""

def history_last_user_image_url(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if (m.sender or "").lower() == "user" and m.image_url:
            return str(m.image_url)
    return ""

def has_product_subject(text: str) -> bool:
    """
    Señal de "consulta nueva":
    - menciona algo que suena a producto ("botas", "abrigo", "sneakers", etc.)
    - o tiene sustantivos/propios (si spaCy lo soporta)
    - o tiene suficiente contenido "concreto"
    """
    t = _norm(text)
    if not t:
        return False

    words = _tokenize(t)

    # Señal 0: presencia clara de hints de producto (muy importante para ES/blank)
    if any(w in PRODUCT_HINTS for w in words):
        return True

    # Señal 1: POS (si el modelo lo trae)
    try:
        doc = nlp(t)
        noun_like = any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc if tok.is_alpha)
    except Exception:
        noun_like = False

    # Señal 2: entidades (si el modelo las genera)
    try:
        doc = nlp(t)
        ents = getattr(doc, "ents", [])
        has_ents = bool(ents)
    except Exception:
        has_ents = False

    # Señal 3: contenido mínimo (ej: "botas chelsea", "abrigo lana", etc.)
    long_words = [w for w in words if len(w) >= 4]
    concrete = len(long_words) >= 2

    return noun_like or has_ents or concrete

def is_followup_query(text: str) -> bool:
    """
    Detecta ajustes típicos:
    - "y en negro"
    - "más elegante"
    - "la misma pero sin cordones"
    - "en cuero" / "con plataforma"
    """
    t = _norm(text)
    if not t:
        return False

    words = _tokenize(t)
    if not words:
        return False

    # Si trae un “producto” claro, NO es follow-up
    if has_product_subject(t):
        return False

    # Si empieza por pronombre/determinante típico de referencia al previo, huele a follow-up
    if words[0] in PRONOUN_FOLLOWUP_START:
        return True

    # Muy corto suele ser follow-up ("en negro", "más clásico", etc.)
    if len(words) <= 4:
        return True

    # Ratio de marcadores típicos
    hits = sum(1 for w in words if w in FOLLOWUP_MARKERS)
    return (hits / max(len(words), 1)) >= 0.35

def build_search_query(user_query: str, history: list[ChatMessage] | None) -> tuple[str, bool]:
    """
    Devuelve (query_para_buscar, usado_contexto)

    Reglas:
    - Si es follow-up y hay last_q -> concatena "last_q. user_query"
    - Si el user repite algo ya incluido, evita duplicar
    - Si viene vacío (o solo espacios) -> devuelve last_q si existe
    """
    uq = (user_query or "").strip()
    last_q = history_last_user_query(history)

    if not uq:
        return (last_q, True) if last_q else ("", False)

    if not last_q:
        return uq, False

    if is_followup_query(uq):
        # evita duplicación: si uq ya está dentro de last_q
        if _norm(uq) in _norm(last_q):
            return last_q, True
        return f"{last_q}. {uq}", True

    return uq, False
