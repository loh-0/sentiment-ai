from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict
import asyncio
import json

from models import MessageRequest
from ml_engine import MentalHealthTriageEngine

app = FastAPI()

# ✅ CORS FIX: Allow React on port 3000 and 3003
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML engine
ml_engine = MentalHealthTriageEngine()

# In-memory storage
conversations = {}
priority_queue = []
connected_clients = []

@app.get("/")
async def root():
    return {"message": "CrisisQueue API Running"}

@app.post("/api/message")
async def analyze_message(data: MessageRequest):
    message = data.message
    user_id = data.user_id

    # Analyze with ML engine (includes AI response via Gemini)
    analysis = ml_engine.analyze_message(message)
    analysis["timestamp"] = datetime.now().isoformat()
    analysis["user_id"] = user_id

    # Store conversation
    if user_id not in conversations:
        conversations[user_id] = []
    conversations[user_id].append(analysis)

    # Update priority queue
    update_priority_queue(user_id, analysis)

    # Notify connected dashboards
    await notify_clients({"type": "new_analysis", "data": analysis})

    return analysis

@app.get("/api/queue")
async def get_priority_queue():
    return sorted(priority_queue,
                  key=lambda x: x["priority_score"],
                  reverse=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages if needed
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

def update_priority_queue(user_id: str, analysis: dict):
    global priority_queue
    priority_queue = [item for item in priority_queue if item["user_id"] != user_id]

    priority_queue.append({
        "user_id": user_id,
        "latest_message": analysis["text"],
        "severity": analysis["severity"],
        "priority_score": analysis["priority_score"],
        "timestamp": analysis["timestamp"],
        "message_count": len(conversations.get(user_id, []))
    })

async def notify_clients(message: dict):
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except:
            connected_clients.remove(client)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
