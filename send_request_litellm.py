import requests

url = "http://localhost:4000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    # OpenAI
    # "model": "gpt-4o-mini",
    # Gemini
    "model": "gemini/gemini-2.5-flash-lite",
    "max_tokens": 10,
    "messages": [
        {"role": "user", "content": "Hello."}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())