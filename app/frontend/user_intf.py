import gradio as gr
import requests
import os
import hashlib
from pathlib import Path

IMG_CACHE_DIR = Path("/tmp/smartshop_images")
IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _safe_ext_from_ct(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    if "gif" in ct:
        return "gif"
    return "jpg"

def download_image_to_cache(url: str, timeout: int = 15) -> str | None:
    """
    Descarga una imagen remota y devuelve una ruta local.
    Si ya existe en cache, la reutiliza.
    """
    url = (url or "").strip()
    if not url:
        return None

    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # intentamos encontrarla ya descargada con cualquier extensión común
    for ext in ("jpg", "png", "webp", "gif"):
        p = IMG_CACHE_DIR / f"{key}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return str(p)

    try:
        r = requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None

        ct = r.headers.get("Content-Type", "")
        if "image" not in ct.lower():
            return None

        ext = _safe_ext_from_ct(ct)
        out = IMG_CACHE_DIR / f"{key}.{ext}"

        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

        if out.exists() and out.stat().st_size > 0:
            return str(out)
        return None
    except Exception:
        return None

API_URL = os.getenv("API_URL", "http://api-smartshopadvisor:8000/chat")


def call_chat_api(message: str, chat_history: list, api_history: list, top_k: int, target_language: str, show_details: bool):
    if not message or not message.strip():
        return chat_history, api_history, gr.update(value=""), "Escribe una consulta primero.", "", ""

    user_msg = message.strip()

    payload = {
        "query": user_msg,
        "history": api_history or [],
        "top_k": int(top_k),
        "target_language": (target_language or "es").strip().lower(),
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=180)
    except Exception as e:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": f"Error conectando a la API: {e}"})

        api_history = api_history or []
        api_history.append({"sender": "user", "content": user_msg})

        return chat_history, api_history, gr.update(value=""), "No se pudo conectar a la API.", "", ""

    if resp.status_code != 200:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": f"Error API ({resp.status_code}): {resp.text}"})

        api_history = api_history or []
        api_history.append({"sender": "user", "content": user_msg})
        api_history.append({"sender": "assistant", "content": f"Error API ({resp.status_code})."})

        return chat_history, api_history, gr.update(value=""), f"HTTP {resp.status_code}", "", ""

    data = resp.json()

    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    web_results = data.get("web_results") or []

    clean_answer = answer if answer else "No he encontrado recomendaciones claras para esa consulta."

    display_answer = clean_answer
    if show_details:
        dbg = [
            "Debug",
            f"- results: {len(results)}",
            f"- web_results: {len(web_results)}",
        ]
        display_answer = (display_answer + "\n\n" + "\n".join(dbg)).strip()

    # Chat en formato messages (lo que pide tu Gradio)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": display_answer})

    # Memoria API (sender/content)
    api_history = api_history or []
    api_history.append({"sender": "user", "content": user_msg})
    api_history.append({"sender": "assistant", "content": clean_answer})

    # Cards HTML
    top_n = min(len(results), 6)
    cards = []
    links_txt = ""

    for i, r in enumerate(results[:top_n], start=1):
        name = (r.get("product_name") or f"Producto {i}").strip()
        url = (r.get("url") or "").strip()
        img = (r.get("image") or "").strip()
        desc = (r.get("description") or "").strip()

        if url:
            links_txt += f"- [{i}. {name}]({url})\n"

        img_block = (
            f'<img src="{img}" alt="{name}" style="width:100%;height:170px;object-fit:cover;border-radius:12px;border:1px solid #eee;" />'
            if img else
            '<div style="width:100%;height:170px;background:#f2f2f2;border-radius:12px;border:1px solid #eee;"></div>'
        )

        desc_html = (
            f'<div style="color:#444;font-size:13px;line-height:1.25;margin-top:6px;max-height:50px;overflow:hidden;">{desc}</div>'
            if desc else
            '<div style="color:#666;font-size:13px;margin-top:6px;">Sin descripción.</div>'
        )

        btn_html = (
            f'<a href="{url}" target="_blank" style="display:inline-block;margin-top:10px;padding:8px 10px;border-radius:10px;text-decoration:none;border:1px solid #ddd;background:#fff;">Ver producto</a>'
            if url else
            ""
        )

        cards.append(f"""
        <div style="border:1px solid #eee;border-radius:16px;padding:10px;background:white;">
          {img_block}
          <div style="margin-top:10px;font-weight:600;font-size:14px;">{i}. {name}</div>
          {desc_html}
          {btn_html}
        </div>
        """)

    cards_html = ""
    if cards:
        cards_html = f"""
        <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;">
          {''.join(cards)}
        </div>
        """

    return chat_history, api_history, gr.update(value=""), "OK", cards_html, links_txt






def reset_chat():
    return [], [], "", "Chat reiniciado.", "", ""



with gr.Blocks(title="SmartShop Advisor Chatbot") as demo:
    gr.Markdown("# SmartShop Advisor\n")

    # Estado para la memoria real del chatbot (la que se manda a la API)
    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbox = gr.Chatbot(label="Chat", height=520)

            txt = gr.Textbox(
                label="Tu consulta",
                placeholder="Ej: Quiero unas zapatillas blancas minimalistas para diario, máximo 120€",
                lines=2,
            )

            with gr.Row():
                send = gr.Button("Enviar", variant="primary")
                clear = gr.Button("Nueva conversación")

            # gallery = gr.Gallery(label="Productos recomendados", columns=2, height=320)
            cards_html = gr.HTML(label="Recomendaciones")
            links_md = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("## Ajustes")
            top_k = gr.Slider(3, 12, value=8, step=1, label="top_k (productos internos)")
            target_language = gr.Dropdown(
                choices=["es", "en", "fr", "de", "it"],
                value="es",
                label="Idioma de respuesta",
            )
            show_details = gr.Checkbox(label="Mostrar detalles (debug)", value=False)
            status = gr.Textbox(label="Estado", value="Listo.", interactive=False)

            gr.Markdown(
                "Notas\n"
                "- Si el modelo local no está descargado en Ollama, te devolverá error.\n"
                "- Si Chroma no tiene colección, primero ejecuta la ingesta."
            )

    send.click(
        call_chat_api,
        inputs=[txt, chatbox, api_history, top_k, target_language, show_details],
        outputs=[chatbox, api_history, txt, status, cards_html, links_md],
    )

    txt.submit(
        call_chat_api,
        inputs=[txt, chatbox, api_history, top_k, target_language, show_details],
        outputs=[chatbox, api_history, txt, status, cards_html, links_md],
    )

    clear.click(
        reset_chat,
        inputs=[],
        outputs=[chatbox, api_history, txt, status, cards_html, links_md],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
