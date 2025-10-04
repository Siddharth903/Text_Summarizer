import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
model_id = "facebook/bart-large-cnn"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
data = {"inputs": "Hello, world!"}

res = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=headers, json=data)
print(res.status_code, res.text)
