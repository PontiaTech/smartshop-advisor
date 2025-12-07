import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
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

def get_gemini_llm():
    gemini_params = {
        "model": GEMINI_MODEL_NAME,
        "api_key": GEMINI_API_KEY,
        "temperature": 0.2,
    }
        
    return ChatGoogleGenerativeAI(**gemini_params)

# llm = llm_base.with_structured_output(EmailMessageSchema)