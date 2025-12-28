import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://api-smartshopadvisor:8000/chat")


def call_chat_api(message: str, chat_history: list, api_history: list, top_k: int, target_language: str, show_details: bool):
    """
    chat_history: lo que se muestra en gr.Chatbot (formato libre, normalmente lista de tuplas)
    api_history: memoria estable para la API, SIEMPRE con formato:
        [{"sender":"user","content":"..."}, {"sender":"assistant","content":"..."}]
    Devuelve:
    - chat_history (display)
    - api_history (memoria)
    - limpiar textbox
    - status
    - gallery (imagenes)
    - links markdown
    """
    if not message or not message.strip():
        return chat_history, api_history, gr.update(value=""), "Escribe una consulta primero.", [], ""

    user_msg = message.strip()

    payload = {
        "query": user_msg,
        "history": api_history or [],  # <- AQUÍ está la clave: NO uses el chatbox como memoria
        "top_k": int(top_k),
        "target_language": (target_language or "es").strip().lower(),
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=180)
    except Exception as e:
        # display
        chat_history = chat_history or []
        chat_history.append((user_msg, f"Error conectando a la API: {e}"))
        # memoria: guardamos solo el user, no inventamos respuesta
        api_history = api_history or []
        api_history.append({"sender": "user", "content": user_msg})
        return chat_history, api_history, gr.update(value=""), "No se pudo conectar a la API.", [], ""

    if resp.status_code != 200:
        chat_history = chat_history or []
        chat_history.append((user_msg, f"Error API ({resp.status_code}): {resp.text}"))

        api_history = api_history or []
        api_history.append({"sender": "user", "content": user_msg})
        api_history.append({"sender": "assistant", "content": f"Error API ({resp.status_code})."})

        return chat_history, api_history, gr.update(value=""), f"HTTP {resp.status_code}", [], ""

    data = resp.json()

    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    web_results = data.get("web_results") or []

    # Respuesta que verá el usuario en el chat
    clean_answer = answer if answer else "No he encontrado recomendaciones claras para esa consulta."

    # Construimos gallery + links a partir de results
    images = []
    links_txt = ""
    for i, r in enumerate(results[: min(len(results), 8)], start=1):
        name = (r.get("product_name") or f"Producto {i}").strip()
        url = (r.get("url") or "").strip()
        img = (r.get("image") or "").strip()

        if img:
            images.append((img, name))
        if url:
            links_txt += f"- [{name}]({url})\n"

    # Debug opcional: solo afecta al DISPLAY, NO a la memoria
    display_answer = clean_answer
    if show_details:
        internal_md = ""
        if results:
            internal_md += "Detalles (catálogo interno)\n\n"
            for i, r in enumerate(results[: min(len(results), 8)], start=1):
                name = (r.get("product_name") or "").strip()
                fam = (r.get("product_family") or "").strip()
                desc = (r.get("description") or "").strip()
                url = (r.get("url") or "").strip()
                score = float(r.get("score", 0.0) or 0.0)
                color = (r.get("color") or "").strip()

                line = f"- {i}. {name}"
                if fam:
                    line += f" - {fam}"
                if color:
                    line += f" - color: {color}"
                line += f" - score: {score:.3f}"
                internal_md += line + "\n"
                if desc:
                    internal_md += f"  - {desc}\n"
                if url:
                    internal_md += f"  - {url}\n"
            internal_md += "\n"

        web_md = ""
        if web_results:
            web_md += "Detalles (web)\n\n"
            for i, w in enumerate(web_results[:3], start=1):
                title = (w.get("title") or "").strip()
                url = (w.get("url") or "").strip()
                snippet = (w.get("snippet") or "").strip()
                source = (w.get("source") or "").strip()

                web_md += f"- {i}. {title}"
                if source:
                    web_md += f" - fuente: {source}"
                web_md += "\n"
                if snippet:
                    web_md += f"  - {snippet}\n"
                if url:
                    web_md += f"  - {url}\n"
            web_md += "\n"

        if internal_md or web_md:
            display_answer = (display_answer + "\n\n---\n\n" + (internal_md + web_md).strip()).strip()

    # Actualiza display (Chatbot): usamos tuplas (user, assistant), es lo más estable
    chat_history = chat_history or []
    chat_history.append((user_msg, display_answer))

    # Actualiza memoria API: SOLO texto limpio (sin debug)
    api_history = api_history or []
    api_history.append({"sender": "user", "content": user_msg})
    api_history.append({"sender": "assistant", "content": clean_answer})

    return chat_history, api_history, gr.update(value=""), "OK", images, links_txt


def reset_chat():
    # chat_history, api_history, textbox, status, gallery, links
    return [], [], "", "Chat reiniciado.", [], ""


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

            gallery = gr.Gallery(label="Productos recomendados", columns=2, height=320)
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
        outputs=[chatbox, api_history, txt, status, gallery, links_md],
    )

    txt.submit(
        call_chat_api,
        inputs=[txt, chatbox, api_history, top_k, target_language, show_details],
        outputs=[chatbox, api_history, txt, status, gallery, links_md],
    )

    clear.click(
        reset_chat,
        inputs=[],
        outputs=[chatbox, api_history, txt, status, gallery, links_md],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
