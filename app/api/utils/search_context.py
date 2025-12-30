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

def _clean_user_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def history_last_user_query(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""

    for m in reversed(history):
        sender = (getattr(m, "sender", "") or "").lower()
        content: Any = getattr(m, "content", None)

        if sender != "user":
            continue

        # Solo aceptamos strings
        if isinstance(content, str):
            txt = _clean_user_text(content)
            if txt:
                return txt

        # Si te llega algo raro, lo ignoras para no contaminar
        # (dict, ChatMessage, etc.)
        continue

    return ""

def history_last_user_image_url(history: list[ChatMessage] | None) -> str:
    if not history:
        return ""

    for m in reversed(history):
        sender = (getattr(m, "sender", "") or "").lower()
        image_url = getattr(m, "image_url", None)

        if sender == "user" and image_url:
            return str(image_url).strip()

    return ""

def sanitize_web_query(q: str) -> str:
    q = (q or "").strip()
    # Si parece un dump de ChatMessage / assistant, quédate con lo último tras un salto
    if "sender='assistant'" in q or 'sender="assistant"' in q:
        # intenta extraer solo lo que va después de la última línea "Query:" o similar
        # pero como fallback simple:
        q = re.sub(r"sender=['\"]assistant['\"].*", "", q).strip()
    # recorta por seguridad
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

def has_explicit_subject(text: str) -> bool:
    """
    True si el texto incluye un producto/objeto claro (zapatillas, abrigo, etc.)
    False si es un ajuste/atributo ("de deporte", "en negro", "más barato", etc.)
    """
    t = _norm(text)
    if not t:
        return False

    words = _tokenize(t)

    # Producto explícito (regla fuerte)
    if any(w in PRODUCT_HINTS for w in words):
        return True

    # Si spaCy está disponible, exigimos al menos un NOUN/PROPN
    # (en blank("es") esto no será fiable, por eso no es la única regla)
    try:
        doc = nlp(t)
        noun_like = any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc if tok.is_alpha)
        if noun_like:
            return True
    except Exception:
        pass

    return False


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
    uq = (user_query or "").strip()
    last_q = history_last_user_query(history)

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
