import re
from dataclasses import dataclass, field
from typing import Any

from app.api.schemas import ChatMessage

# ----------------------------
# Minimal, fast, deterministic "context resolver"
# Slots + rules + confidence + clarify fallback
# ----------------------------

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
    "nike", "adidas", "puma", "reebok", "newbalance", "new", "balance",
    "mango", "zara", "bershka", "pull", "bear", "stradivarius", "massimodutti", "massimo", "dutti",
    "asics", "converse", "vans", "skechers",
}

SIZE_RE = re.compile(r"\b(talla|size)\s*(\d{2,3}|xs|s|m|l|xl|xxl)\b", re.IGNORECASE)
PRICE_MAX_RE = re.compile(r"(?:menos\s+de|hasta|máximo|max|<)\s*(\d{1,4})(?:\s*€|euros)?", re.IGNORECASE)
PRICE_MIN_RE = re.compile(r"(?:más\s+de|desde|mínimo|min|>)\s*(\d{1,4})(?:\s*€|euros)?", re.IGNORECASE)

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

@dataclass
class SessionState:
    last_resolved_query: str = ""
    slots: dict[str, Any] = field(default_factory=dict)

def _extract_slots(text: str) -> dict[str, Any]:
    """
    Extrae slots deterministas (rápido): item_type, color, brand, use, price_min/max, size.
    """
    t = _norm(text)
    words = _tokenize(t)

    slots: dict[str, Any] = {}

    # item_type: primero match directo
    for w in words:
        if w in ITEM_TYPES:
            slots["item_type"] = w
            break

    # color
    for w in words:
        if w in COLORS:
            slots["color"] = w
            break

    # use/style
    for w in words:
        if w in USES:
            slots["use"] = w
            break

    # brand: soporta tokens partidos ("massimo dutti", "pull bear", "new balance")
    # enfoque rápido: si aparece la secuencia en texto normalizado
    # Nota: esto es deliberadamente simple para testear rápido.
    if "massimo dutti" in t:
        slots["brand"] = "massimo dutti"
    elif "pull bear" in t or "pull&bear" in t:
        slots["brand"] = "pull&bear"
    elif "new balance" in t:
        slots["brand"] = "new balance"
    else:
        for w in words:
            if w in BRANDS:
                slots["brand"] = w
                break

    # price
    m = PRICE_MAX_RE.search(t)
    if m:
        try:
            slots["price_max"] = int(m.group(1))
        except Exception:
            pass

    m = PRICE_MIN_RE.search(t)
    if m:
        try:
            slots["price_min"] = int(m.group(1))
        except Exception:
            pass

    # size
    m = SIZE_RE.search(t)
    if m:
        slots["size"] = m.group(2).lower()

    return slots

def _is_followup_like(text: str) -> bool:
    t = _norm(text)
    words = _tokenize(t)
    if not words:
        return False
    if words[0] in PRONOUN_FOLLOWUP_START:
        return True
    if len(words) <= 4:
        return True
    hits = sum(1 for w in words if w in FOLLOWUP_MARKERS)
    return (hits / max(len(words), 1)) >= 0.35

def _merge_slots(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in (incoming or {}).items():
        if v is None:
            continue
        out[k] = v
    return out

def _reset_incompatible_slots(slots: dict[str, Any]) -> dict[str, Any]:
    """
    Reset rápido y conservador cuando cambia item_type.
    Para evitar efectos raros (p.ej. talla de calzado heredada a un bolso).
    Ajusta si quieres más fino, pero esto evita muchos "problemas".
    """
    keep = {"item_type", "brand", "color", "price_min", "price_max", "use"}
    return {k: v for k, v in slots.items() if k in keep}

def _slots_to_query(slots: dict[str, Any], fallback_text: str | None = None) -> str:
    parts: list[str] = []

    it = slots.get("item_type")
    if it:
        parts.append(str(it))

    brand = slots.get("brand")
    if brand:
        parts.append(str(brand))

    color = slots.get("color")
    if color:
        parts.append(str(color))

    use = slots.get("use")
    if use:
        parts.append(str(use))

    size = slots.get("size")
    if size:
        parts.append(f"talla {size}")

    pmin = slots.get("price_min")
    pmax = slots.get("price_max")
    if isinstance(pmin, int):
        parts.append(f"desde {pmin}€")
    if isinstance(pmax, int):
        parts.append(f"hasta {pmax}€")

    q = " ".join(parts).strip()
    if not q and fallback_text:
        q = fallback_text.strip()

    return sanitize_web_query(q)

def _rebuild_state_from_history(history: list[ChatMessage] | None, max_turns: int = 8) -> SessionState:
    """
    Reconstruye slots con los últimos mensajes del usuario.
    Rápido, determinista, sin depender de spaCy.
    """
    st = SessionState(last_resolved_query="", slots={})
    if not history:
        return st

    user_msgs: list[str] = []
    for m in history:
        sender = (getattr(m, "sender", "") or "").lower()
        content: Any = getattr(m, "content", None)
        if sender == "user" and isinstance(content, str):
            txt = _clean_user_text(content)
            if txt:
                user_msgs.append(txt)

    if not user_msgs:
        return st

    # coge últimos N turnos
    for txt in user_msgs[-max_turns:]:
        slots = _extract_slots(txt)
        # si aparece item_type, consideramos que esa es una "query con sujeto"
        if slots.get("item_type"):
            st.slots = _merge_slots(st.slots, slots)
            st.last_resolved_query = _slots_to_query(st.slots, fallback_text=txt)
        else:
            # si no hay item_type, solo aplicamos si ya tenemos base previa
            if st.last_resolved_query:
                st.slots = _merge_slots(st.slots, slots)
                st.last_resolved_query = _slots_to_query(st.slots, fallback_text=st.last_resolved_query)

    # fallback final
    if not st.last_resolved_query:
        st.last_resolved_query = sanitize_web_query(user_msgs[-1])

    return st

def build_search_query(user_query: str, history: list[ChatMessage] | None) -> tuple[str, bool]:
    """
    Devuelve (query, used_context).
    - used_context=True cuando el texto actual se interpreta como refinamiento sobre contexto previo.
    - Si no hay suficiente señal -> devuelve una pregunta de aclaración con prefijo "CLARIFY:".
      (Así lo puedes rutear fácil en tu controlador.)
    """
    uq = _clean_user_text(user_query or "")
    if not uq:
        last_q = history_last_user_query(history)
        return (sanitize_web_query(last_q), True) if last_q else ("", False)

    st = _rebuild_state_from_history(history)

    extracted = _extract_slots(uq)
    has_item_type_now = bool(extracted.get("item_type"))
    followup_like = _is_followup_like(uq)

    # Caso 1: no hay contexto real previo
    if not st.last_resolved_query:
        # si no hay item_type, igual es vago -> pedir aclaración
        if not has_item_type_now and followup_like:
            return ("CLARIFY: Que producto buscas exactamente (por ejemplo, zapatillas, abrigo, pantalón)?", False)
        # si tiene slots sin item_type pero no es followup_like, lo dejamos pasar como query literal
        if not has_item_type_now and extracted:
            q = _slots_to_query(extracted, fallback_text=uq)
            return (q or sanitize_web_query(uq), False)
        return (sanitize_web_query(uq), False)

    # Confianza del refinamiento: si extraemos algo "accionable" sube mucho
    actionable_keys = {"color", "brand", "use", "price_min", "price_max", "size"}
    actionable_hits = sum(1 for k in actionable_keys if extracted.get(k) is not None)

    refine_conf = 0.0
    if followup_like:
        refine_conf += 0.45
    if actionable_hits >= 1:
        refine_conf += 0.45
    # si el mensaje es corto, suele ser refinamiento
    if len(_tokenize(uq)) <= 5:
        refine_conf += 0.15
    refine_conf = min(refine_conf, 1.0)

    # Caso 2: viene un item_type -> new_search o change_item_type
    if has_item_type_now:
        current_it = st.slots.get("item_type")
        new_it = extracted.get("item_type")
        if current_it and new_it and new_it != current_it:
            # cambio de producto: resetea incompatibles y aplica
            base = _reset_incompatible_slots(st.slots)
            merged = _merge_slots(base, extracted)
            q = _slots_to_query(merged, fallback_text=uq)
            return (q or sanitize_web_query(uq), False)
        else:
            # nueva query completa o refinamiento con mismo tipo
            merged = _merge_slots(st.slots, extracted)
            q = _slots_to_query(merged, fallback_text=uq)
            return (q or sanitize_web_query(uq), False)

    # Caso 3: no hay item_type -> probablemente refinamiento
    if refine_conf >= 0.60:
        merged = _merge_slots(st.slots, extracted)
        q = _slots_to_query(merged, fallback_text=st.last_resolved_query)
        # evita duplicación simple
        if _norm(q) == _norm(st.last_resolved_query):
            return (st.last_resolved_query, True)
        return (q, True)

    # Caso 4: vago/ambiguo -> aclaración (100% correcto: no adivina)
    return (
        "CLARIFY: Te refieres a cambiar color, marca, precio, o quieres otro tipo de producto?",
        True
    )
