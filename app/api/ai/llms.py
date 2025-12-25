import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore
    
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://ollama:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    
    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME") or "gemini-2.5-flash"

if not GEMINI_API_KEY:
    raise NotImplementedError("'GEMINI_API_KEY' was not implemented")


# llm_base = ChatOpenAI(**openai_params)
# llm_base = ChatOpenAI(
#     model_name=OPENAI_MONEL_NAME,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL
# )

def get_llm():
    if LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            raise RuntimeError("Instala 'langchain-ollama' para usar Ollama.")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

    gemini_params = {
        "model": GEMINI_MODEL_NAME,
        "google_api_key": GEMINI_API_KEY,
        "temperature": 0.2,
    }
    return ChatGoogleGenerativeAI(**gemini_params)

# llm = llm_base.with_structured_output(EmailMessageSchema)