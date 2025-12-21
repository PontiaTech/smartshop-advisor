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
    if not results:
        return ""

    bullets = []
    for i, p in enumerate(results[:limit], start=1):
        name = (p.product_name or "").strip()
        fam = (p.product_family or "").strip()
        src = (p.source or "").strip()
        url = (p.url or "").strip()
        img = (p.image or "").strip()

        desc = (p.description or "").replace("\n", " ").strip()
        if len(desc) > 180:
            desc = desc[:177] + "..."

        # Nota: p.score ya viene como float (de clip_score)
        bullets.append(
            f"{i}) name={name} | family={fam} | score={p.score:.4f} | source={src} | url={url} | image={img} | desc={desc}"
        )

    return "\n".join(bullets)


def web_results_to_bullets(items: list[dict]) -> str:
    # lines = []
    # for i, it in enumerate(items, 1):
    #     lines.append(
    #         f"- [{i}] {it.get('title','')}\n"
    #         f"  - source: {it.get('source','')}\n"
    #         f"  - snippet: {it.get('snippet','')}\n"
    #         f"  - url: {it.get('url','')}"
    #     )
    # return "\n".join(lines)
    lines = []
    for i, it in enumerate(items, 1):
        title = it.get("title") or it.get("product_name") or ""
        snippet = it.get("snippet") or it.get("description") or ""

        lines.append(
            f"- [{i}] {title}\n"
            f"  - source: {it.get('source','')}\n"
            f"  - snippet: {snippet}\n"
            f"  - url: {it.get('url','')}"
        )
    return "\n".join(lines)