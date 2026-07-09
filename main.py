# Backend by Zain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import chat_with_memory, generate_quiz
from memory import get_all_memories, forget_old_memories

app = FastAPI(title="StudyMind AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class QuizRequest(BaseModel):
    user_id: str
    topic: str

@app.get("/")
async def root():
    return {
        "status": "StudyMind AI is running!",
        "powered_by": "Qwen Cloud API",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "deployed_on": "Alibaba Cloud ECS"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    forgotten = forget_old_memories(request.user_id)
    response = chat_with_memory(request.user_id, request.message)
    return {
        "response": response,
        "user_id": request.user_id,
        "memories_forgotten": forgotten
    }

@app.post("/quiz")
async def quiz(request: QuizRequest):
    result = generate_quiz(request.user_id, request.topic)
    return {"quiz": result, "topic": request.topic}

@app.get("/memories/{user_id}")
async def get_memories(user_id: str):
    memories = get_all_memories(user_id)
    return {
        "user_id": user_id,
        "total_memories": len(memories),
        "memories": memories
    }