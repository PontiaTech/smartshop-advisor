from fastapi import HTTPException
import json
# from app.api.schemas import CompleteSearchProduct, ChatRequest, ChatResponse, WebSearchProduct

# funcion helper para que el LLM juez pueda determinar si son relevantes o no
def compact_products_for_judge(raw_results: list[dict], limit: int = 8) -> str:
    lines = []
    for i, r in enumerate((raw_results or [])[:limit], start=1):
        name = (r.get("product_name") or "").strip()
        fam = (r.get("product_family") or "").strip()
        desc = (r.get("description") or "").strip()
        color = (r.get("color") or "").strip()
        source = (r.get("source") or "").strip()
        score = float(r.get("score") or r.get("text_score") or 0.0)

        if len(desc) > 220:
            desc = desc[:217] + "..."

        header = f"{i}. {name} | score={score:.3f}"
        if fam:
            header += f" | family={fam}"
        if color:
            header += f" | color={color}"
        if source:
            header += f" | source={source}"

        lines.append(header)
        if desc:
            lines.append(f"desc: {desc}")

    return "\n".join(lines) if lines else "(sin resultados)"


def pick_by_indices(raw_results: list[dict], indices_1based: list[int]) -> list[dict]:
    picked = []
    for idx in indices_1based or []:
        j = idx - 1
        if 0 <= j < len(raw_results):
            picked.append(raw_results[j])
    return picked