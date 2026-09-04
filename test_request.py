import json
import requests

RAG_URL = "http://localhost:8001/chat"
AUTH_TOKEN = "droply-rag"  # from RAG_INTERNAL_KEY in .env

payload = {
    "user_id": "test-user",
    "question": "What is this document about?",
    "history": [],
    "file_ids": []
}

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

resp = requests.post(RAG_URL, headers=headers, json=payload, stream=True)

print("Status:", resp.status_code)
print(resp.text[:1000])

# If you want to print the SSE stream more cleanly:
# for line in resp.iter_lines():
#     if line:
#         print(line.decode())