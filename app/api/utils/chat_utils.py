from typing import List, Optional
from app.api.schemas import ChatMessage, CompleteSearchProduct


def history_to_text(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-12:]:  # recortamos para no meter 2000 turnos
        sender = (m.sender or "user").strip()
        if m.content and m.content.strip():
            lines.append(f"{sender}: {m.content.strip()}")
        if m.image_url:
            lines.append(f"{sender}: [image_url={str(m.image_url)}]")
    return "\n".join(lines)


def results_to_bullets(results: List[CompleteSearchProduct], limit: int = 8) -> str:
    """
    Contexto para el LLM, en formato controlado.
    Mantiene URL e Image porque tu system prompt las necesita para imprimir.
    Evita el estilo 'dump' (name=...|url=...) que el modelo tiende a copiar.
    """
    if not results:
        return ""

    blocks = []
    for i, p in enumerate(results[:limit], start=1):
        name = (p.product_name or "").strip() or "No disponible"
        desc = (p.description or "").replace("\n", " ").strip() or "No disponible"
        fam = (p.product_family or "").strip()
        src = (p.source or "").strip()
        url = (p.url or "").strip() or "No disponible"
        img = (p.image or "").strip() or "No disponible"
        score = float(getattr(p, "score", 0.0) or 0.0)

        # recorte para evitar tochos
        if len(desc) > 160:
            desc = desc[:157].rstrip() + "..."

        # bloque compacto, “listo para usar” pero sin parecer un log
        extra = []
        if fam:
            extra.append(f"Familia: {fam}")
        if src:
            extra.append(f"Fuente: {src}")
        extra_txt = (" | " + " - ".join(extra)) if extra else ""

        blocks.append(
            f"[{i}] score={score:.3f}{extra_txt}\n"
            f"Nombre: {name}\n"
            f"Descripción: {desc}\n"
            f"Imagen: {img}\n"
            f"URL: {url}"
        )

    return "\n\n".join(blocks).strip()


def web_results_to_bullets(items: list[dict], limit: int = 3) -> str:
    """
    Contexto web para el LLM en formato controlado (sin dump).
    """
    if not items:
        return ""

    blocks = []
    for i, it in enumerate(items[:limit], start=1):
        title = (it.get("title") or it.get("product_name") or "").strip() or "No disponible"
        source = (it.get("source") or "").strip()
        snippet = (it.get("snippet") or it.get("description") or "").replace("\n", " ").strip() or "No disponible"
        url = (it.get("url") or "").strip() or "No disponible"
        image = it.get("image") or it.get("thumbnail") or "No disponible"

        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."

        src_txt = f"Fuente: {source}\n" if source else ""
        blocks.append(
            f"[{i}]\n"
            f"Nombre: {title}\n"
            f"{src_txt}"
            f"Descripción/Motivo: {snippet}\n"
            f"Imagen: {image}\n"
            f"URL: {url}"
        )

    return "\n\n".join(blocks).strip()
