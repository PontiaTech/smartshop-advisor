import sys
from unittest.mock import MagicMock

# --------------------------------------------------
# MOCKEAR DEPENDENCIAS EXTERNAS ANTES DEL IMPORT
# --------------------------------------------------

# Chroma
mock_chroma_client = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.HttpClient"] = MagicMock(return_value=mock_chroma_client)

# Sentence transformers
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers.SentenceTransformer"] = MagicMock()

# SpaCy
sys.modules["spacy"] = MagicMock()
sys.modules["spacy.load"] = MagicMock()

# --------------------------------------------------
# IMPORT DE LA APP (YA AISLADA)
# --------------------------------------------------

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# --------------------------------------------------
# TESTS
# --------------------------------------------------

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code in (200, 404)
