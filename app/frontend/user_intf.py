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


def call_chat_api(
    message: str,
    image_path: str | None,
    chat_history: list,
    api_history: list,
    top_k: int,
    target_language: str,
    show_details: bool,
):
    import base64

    # Ajusta este límite si quieres. 1.5MB suele ir bien en demos.
    MAX_INLINE_BYTES = 1_500_000

    def _guess_mime_from_path(p: str) -> str:
        pl = (p or "").lower()
        if pl.endswith(".png"):
            return "image/png"
        if pl.endswith(".webp"):
            return "image/webp"
        if pl.endswith(".gif"):
            return "image/gif"
        return "image/jpeg"

    def _image_to_data_uri(p: str) -> str | None:
        if not p:
            return None
        try:
            size = os.path.getsize(p)
            if size <= 0 or size > MAX_INLINE_BYTES:
                return None
            with open(p, "rb") as f:
                b = f.read()
            mime = _guess_mime_from_path(p)
            b64 = base64.b64encode(b).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def _append_user_to_chat(user_text: str, img_path: str | None):
        # Importante: Chatbot acepta strings + markdown, no objetos multimodales
        if img_path:
            data_uri = _image_to_data_uri(img_path)
            if data_uri:
                # Markdown con imagen inline (data URI)
                chat_history.append({
                    "role": "user",
                    "content": f"{user_text}\n\n![imagen de referencia]({data_uri})"
                })
            else:
                # Fallback si es muy grande o no se pudo leer
                chat_history.append({
                    "role": "user",
                    "content": f"{user_text}\n\n[Imagen adjunta]"
                })
        else:
            chat_history.append({"role": "user", "content": user_text})

    chat_history = chat_history or []
    api_history = api_history or []

    if not message or not message.strip():
        return (
            chat_history,              # chatbox
            api_history,               # api_history
            gr.update(value=""),       # txt
            gr.update(value=None),     # image_input
            "El texto es obligatorio.",# status
            "",                        # cards_html
            "",                        # links_md
        )

    user_msg = message.strip()

    payload = {
        "query": user_msg,
        "history": api_history,
        "top_k": int(top_k),
        "target_language": (target_language or "es").strip().lower(),
    }
    if image_path:
        # Tu backend lo espera así. Si luego cambiáis a base64, se modifica aquí.
        payload["image_path"] = image_path

    try:
        resp = requests.post(API_URL, json=payload, timeout=180)
    except Exception as e:
        _append_user_to_chat(user_msg, image_path)
        chat_history.append({"role": "assistant", "content": f"Error conectando a la API: {e}"})
        api_history.append({"sender": "user", "content": user_msg})

        return (
            chat_history,
            api_history,
            gr.update(value=""),
            gr.update(value=None),
            "No se pudo conectar a la API.",
            "",
            "",
        )

    if resp.status_code != 200:
        _append_user_to_chat(user_msg, image_path)
        chat_history.append({"role": "assistant", "content": f"Error API ({resp.status_code}): {resp.text}"})

        api_history.append({"sender": "user", "content": user_msg})
        api_history.append({"sender": "assistant", "content": f"Error API ({resp.status_code})."})

        return (
            chat_history,
            api_history,
            gr.update(value=""),
            gr.update(value=None),
            f"HTTP {resp.status_code}",
            "",
            "",
        )

    data = resp.json()

    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    web_results = data.get("web_results") or []

    clean_answer = answer if answer else "No he encontrado recomendaciones claras para esa consulta."
    display_answer = clean_answer

    if show_details:
        dbg = [
            "\n---\n",
            "Debug",
            f"- results: {len(results)}",
            f"- web_results: {len(web_results)}",
            f"- top_k: {top_k}",
        ]
        display_answer = (display_answer + "\n" + "\n".join(dbg)).strip()

    _append_user_to_chat(user_msg, image_path)
    chat_history.append({"role": "assistant", "content": display_answer})

    # Memoria API: solo texto, estable
    api_history.append({"sender": "user", "content": user_msg})
    api_history.append({"sender": "assistant", "content": clean_answer})

    return (
        chat_history,
        api_history,
        gr.update(value=""),
        gr.update(value=None),
        "OK",
        "",
        "",
    )




def reset_chat():
    return [], [], "", None, "Chat reiniciado.", "", ""



with gr.Blocks(title="SmartShop Advisor Chatbot") as demo:
    gr.Markdown("# SmartShop Advisor\n")

    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbox = gr.Chatbot(label="Chat", height=520)

            txt = gr.Textbox(
                label="Tu consulta (obligatoria)",
                placeholder="Ej: Quiero unas zapatillas blancas minimalistas para diario, máximo 120€",
                lines=2,
            )

            image_input = gr.Image(
                label="Imagen de referencia (opcional)",
                type="filepath",
            )


            with gr.Row():
                send = gr.Button("Enviar", variant="primary")
                clear = gr.Button("Nueva conversación")

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
        inputs=[
    txt,
    image_input,
    chatbox,
    api_history,
    top_k,
    target_language,
    show_details,
]
,
        outputs=[chatbox, api_history, txt, image_input, status, cards_html, links_md]
,
    )

    txt.submit(
        call_chat_api,
        inputs=[
    txt,
    image_input,
    chatbox,
    api_history,
    top_k,
    target_language,
    show_details,
]
,
        outputs=[chatbox, api_history, txt, image_input, status, cards_html, links_md]
,
    )

    clear.click(
        reset_chat,
        inputs=[],
        outputs=[chatbox, api_history, txt, image_input, status, cards_html, links_md]
,
    )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, show_error=True)
