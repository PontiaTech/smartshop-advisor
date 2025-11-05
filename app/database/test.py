import requests

payload = {"query": "Quiero botas blancas", "history": "[]"}
r = requests.post("http://127.0.0.1:8000/search", data=payload)
print("📥 Estado HTTP:", r.status_code)
print("📥 RAW RESPONSE:", r.text)
