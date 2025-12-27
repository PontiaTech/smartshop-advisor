import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://api-smartshopadvisor:8000/chat")


def call_chat_api(message: str, history: list, top_k: int, target_language: str):
    """
    history (Gradio) llega como lista de mensajes: [{"role": "user"/"assistant", "content": "..."}]
    La API espera: ChatRequest(query, history, top_k, target_language)
    """
    if not message or not message.strip():
        return history, gr.update(value=""), "Escribe una consulta primero."

    payload = {
        "query": message.strip(),
        "history": history or [],
        "top_k": int(top_k),
        "target_language": (target_language or "es").strip().lower(),
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=180)
    except Exception as e:
        history = history or []
        history.append({"role": "assistant", "content": f"Error conectando a la API: {e}"})
        return history, gr.update(value=""), "No se pudo conectar a la API."

    if resp.status_code != 200:
        history = history or []
        history.append({"role": "assistant", "content": f"Error API ({resp.status_code}): {resp.text}"})
        return history, gr.update(value=""), f"HTTP {resp.status_code}"

    data = resp.json()

    answer = (data.get("answer") or "").strip()
    predicted_type = data.get("predicted_type") or ""
    results = data.get("results") or []
    web_results = data.get("web_results") or []

    # Mensaje principal (bonito y compacto)
    header = []
    if predicted_type:
        header.append(f"Categoría inferida: {predicted_type}")
    header_txt = ("\n".join(header) + "\n\n") if header else ""

    # Sección catálogo interno (top results)
    internal_md = ""
    if results:
        internal_md += "Catálogo interno\n\n"
        for i, r in enumerate(results[: min(len(results), 8)], start=1):
            name = (r.get("product_name") or "").strip()
            fam = (r.get("product_family") or "").strip()
            desc = (r.get("description") or "").strip()
            url = r.get("url") or ""
            score = r.get("score", 0.0)
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
                internal_md += f"  - Link: {url}\n"
        internal_md += "\n"

    # Sección web
    web_md = ""
    if web_results:
        web_md += "Resultados web\n\n"
        for i, w in enumerate(web_results[:3], start=1):
            title = (w.get("title") or "").strip()
            url = w.get("url") or ""
            snippet = (w.get("snippet") or "").strip()
            source = (w.get("source") or "").strip()

            web_md += f"- {i}. {title}"
            if source:
                web_md += f" - fuente: {source}"
            web_md += "\n"
            if snippet:
                web_md += f"  - {snippet}\n"
            if url:
                web_md += f"  - Link: {url}\n"
        web_md += "\n"

    final_msg = (header_txt + answer).strip()
    if internal_md or web_md:
        final_msg += "\n\n---\n\n"
        if internal_md:
            final_msg += internal_md
        if web_md:
            final_msg += web_md

    # Actualiza historial
    history = history or []
    history.append({"role": "user", "content": message.strip()})
    history.append({"role": "assistant", "content": final_msg})

    return history, gr.update(value=""), "OK"


def reset_chat():
    return [], "", "Chat reiniciado."


with gr.Blocks(title="SmartShop Advisor - TFM") as demo:
    gr.Markdown("# SmartShop Advisor\nInterfaz de demo para el TFM (RAG + recomendación)")

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

        with gr.Column(scale=1):
            gr.Markdown("## Ajustes")
            top_k = gr.Slider(3, 12, value=8, step=1, label="top_k (productos internos)")
            target_language = gr.Dropdown(
                choices=["es", "en", "fr", "de", "it"],
                value="es",
                label="Idioma de respuesta",
            )
            status = gr.Textbox(label="Estado", value="Listo.", interactive=False)

            gr.Markdown(
                "Notas\n"
                "- Si el modelo local no está descargado en Ollama, te devolverá error.\n"
                "- Si Chroma no tiene colección, primero ejecuta la ingesta."
            )

    send.click(
        call_chat_api,
        inputs=[txt, chatbox, top_k, target_language],
        outputs=[chatbox, txt, status],
    )
    txt.submit(
        call_chat_api,
        inputs=[txt, chatbox, top_k, target_language],
        outputs=[chatbox, txt, status],
    )
    clear.click(reset_chat, inputs=[], outputs=[chatbox, txt, status])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
