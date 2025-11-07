import gradio as gr
import requests
import json

#API_URL = "http://127.0.0.1:8001/search" #WHen testing manually
API_URL = "http://localhost:8001/search" #from docker compose


# Estado persistente de historial
chat_history = []  # lista de dicts {role, content}

def chat_fn(message):
    global chat_history

    payload = {
        "query": message,
        "history": json.dumps(chat_history)
    }

    print("\n======================")
    print("📤 ENVIANDO A API")
    print("----------------------")
    print("📝 Query:", message)
    print("🧠 History:", chat_history)
    print("📦 Payload (form fields):", payload)

    try:
        response = requests.post(API_URL, data=payload)
    except Exception as e:
        return [{"role": "assistant", "content": f"❌ Error conectando a API: {e}"}]

    print("📥 Estado HTTP:", response.status_code)
    print("📥 RAW RESPONSE:", response.text)

    try:
        data = response.json()
    except:
        print("❌ JSON inválido devuelto por API")
        return [{"role": "assistant", "content": f"❌ Respuesta inválida:\n{response.text}"}]

    if "error" in data:
        return [{"role": "assistant", "content": f"⚠️ Error API: {data['error']}"}]

    results = data.get("top_results", [])
    article = data.get("predicted_article_type", "")

    reply = f"🛍️ Busqué *{message}* → categoría **{article}**\n\n"
    for r in results:
        reply += f"• **{r['name']}** (sim {r['similarity']:.2f} ->{r['link']})\n"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})

    return chat_history


# ---- UI ----
with gr.Blocks() as demo:
    gr.Markdown("### 🧠 SmartShop Advisor")

    chatbox = gr.Chatbot(label="Chat", type="messages")
    txt = gr.Textbox(label="Escribe tu consulta", placeholder="Ej: Quiero botas blancas")
    
    btn = gr.Button("Enviar")

    # Solo texto, sin imagen
    btn.click(chat_fn, inputs=[txt], outputs=[chatbox])
    txt.submit(chat_fn, inputs=[txt], outputs=[chatbox])

demo.launch(debug=True)
