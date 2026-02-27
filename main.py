from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import time
import os
from datetime import datetime
import pytz


eastern = pytz.timezone("America/New_York")
current_time = datetime.now(eastern).strftime("%A, %B %d, %Y %I:%M %p")



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
    from datetime import datetime
    import pytz
    
    
    eastern = pytz.timezone("America/New_York")
    current_time = datetime.now(eastern).strftime("%A, %B %d, %Y %I:%M %p")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": f"You are a helpful, emotionally aware assistant. The current time is {current_time}."},
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
    from datetime import datetime
    import pytz
    
    
    eastern = pytz.timezone("America/New_York")
    current_time = datetime.now(eastern).strftime("%A, %B %d, %Y %I:%M %p")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": f"You are a helpful, emotionally aware assistant. The current time is {current_time}."},
            *conversation_history,
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": True
    }


    print("=== STREAM START ===")
    print("Sending payload to OpenAI:", payload)

    with requests.post(LLM_API_URL, json=payload, headers=headers, stream=True) as r:
        print("OpenAI response status:", r.status_code)
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")
            print("RAW LINE:", decoded) 

            if not decoded.startswith("data: "):
                continue

            data = decoded.replace("data: ", "")
            if data == "[DONE]":
                print("=== STREAM END ===")
                break

            try:
                chunk = json.loads(data)
                print("DECODED CHUNK:", chunk)  

                delta = chunk["choices"][0]["delta"]

                # Skip role-only or empty chunks
                if "content" not in delta:
                    print("SKIPPED (no content):", delta)
                    continue

                token = delta["content"]
                print("TOKEN:", token)  

                if token:
                    yield token
                    time.sleep(0.01)

            except Exception as e:
                print("ERROR PARSING CHUNK:", e)
                continue





class ChatRequest(BaseModel):
    text: str
    stream: bool = False  



@app.get("/")
def home():
    return {"message": "Chatbot backend is running"}



@app.post("/reply")
def reply(request: ChatRequest):
    from datetime import datetime
    current_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

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






