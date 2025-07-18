import requests

def call_gemini_flash_api(payload):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = dict(payload)
    payload["generationConfig"] = {"temperature": 0.15}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()
