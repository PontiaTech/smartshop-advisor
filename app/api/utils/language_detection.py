from langdetect import detect, LangDetectException
from app.api.schemas import CompleteSearchProduct
from app.api.ai.system_prompts import TRANSLATE_PROMPT
import re, json
import logging
logger = logging.getLogger("smartshop.translate")

def needs_translation(text: str, target_lang: str) -> bool:
    text = (text or "").strip()
    if len(text) < 20:
        return False
    try:
        return detect(text) != target_lang
    except LangDetectException:
        return False
    
    
def extract_first_json_object(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
    
    
async def translate_product(llm, p: dict, target_language: str = "es") -> dict:
    out = await (TRANSLATE_PROMPT | llm).ainvoke({
        "target_language": target_language,
        "product_name": p.get("product_name", "") or "",
        "description": p.get("description", "") or "",
        "product_family": p.get("product_family", "") or "",
    })
    txt = getattr(out, "content", str(out))
    
    logger.warning(
        "TRANSLATE RAW | target=%s | name=%s | raw=%s",
        target_language,
        (p.get("product_name", "") or "")[:60],
        txt[:500].replace("\n", "\\n")
    )
    
    data = extract_first_json_object(txt)
    if not data:
        return p
    # parseo JSON seguro
    # import json
    # try:
    #     data = json.loads(txt)
    # except Exception:
    #     return p

    p["product_name"] = data.get("product_name", p.get("product_name"))
    p["description"] = data.get("description", p.get("description"))
    p["product_family"] = data.get("product_family", p.get("product_family"))
    return p


async def translate_results(results: list[CompleteSearchProduct], llm, target_lang: str = "es"):
    translated = []
    for p in results:
        # detectamos por nombre+descripción
        sample = f"{p.product_name} {p.description or ''}"
        if needs_translation(sample, target_lang):
            as_dict = p.model_dump()
            as_dict = await translate_product(llm, as_dict, target_language=target_lang)
            translated.append(CompleteSearchProduct(**as_dict))
        else:
            translated.append(p)
            
        logger.warning(
            "TRANSLATE CHECK | target=%s | needs=%s | sample=%s",
            target_lang,
            needs_translation(sample, target_lang),
            sample[:120].replace("\n", " ")
)
    return translated
