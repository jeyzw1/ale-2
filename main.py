from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Jarvis Cloud")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

class Message(BaseModel):
    text: str

SYSTEM_PROMPT = """
Eres Jarvis, el asistente personal de Jeriel. 
Hablas en español, con tono elegante, sarcástico y amigable como Iron Man.
Sé útil, directo y con un toque de humor.
"""

@app.post("/chat")
async def chat(message: Message):
    try:
        response = client.chat.completions.create(
            model="grok-4.20",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message.text}
            ],
            temperature=0.75,
            max_tokens=600
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "✅ Jarvis está vivo en la nube 🚀"}
