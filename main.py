from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import time
import os


LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-4o-mini"
LLM_API_KEY = os.getenv("LLM_API_KEY") 

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY environment variable is not set.")
 

conversation_history = []  


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def detect_sentiment(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["sad", "upset", "depressed", "angry"]):
        return "negative"
    if any(w in text_lower for w in ["happy", "excited", "great", "good"]):
        return "positive"
    return "neutral"



def call_llm_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, emotionally aware assistant."},
            *conversation_history,
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()



def stream_llm_api(prompt: str):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, emotionally aware assistant."},
            *conversation_history,
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": True
    }

    with requests.post(LLM_API_URL, json=payload, headers=headers, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded.replace("data: ", "")
                    if data == "[DONE]":
                        break
                    try:
                        token = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                            time.sleep(0.01)
                    except:
                        continue



class ChatRequest(BaseModel):
    text: str
    stream: bool = False  



@app.get("/")
def home():
    return {"message": "Chatbot backend is running"}



@app.post("/reply")
def reply(request: ChatRequest):
    user_text = request.text
    sentiment = detect_sentiment(user_text)

    conversation_history.append({"role": "user", "content": user_text})

    if request.stream:
        def event_stream():
            for token in stream_llm_api(user_text):
                yield token
        return StreamingResponse(event_stream(), media_type="text/plain")

    reply_text = call_llm_api(user_text)

    conversation_history.append({"role": "assistant", "content": reply_text})

    return {"reply": reply_text, "sentiment": sentiment}
