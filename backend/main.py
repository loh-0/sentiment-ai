"""
Mental Health Crisis Triage API

A FastAPI-based backend service for real-time mental health crisis assessment and priority queue management.
This system uses AI-powered emotion analysis to automatically triage users based on their mental health risk level,
enabling crisis counselors to prioritize the most urgent cases.

Features:
- Real-time message analysis using machine learning
- Automatic priority queue management
- WebSocket support for live dashboard updates
- AI-powered supportive responses via Gemini
- CORS-enabled for React frontend integration

Author: Mental Health Triage Team
Version: 1.0
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import json
import logging

from models import MessageRequest
from ml_engine import MentalHealthTriageEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Mental Health Crisis Triage API",
    description="AI-powered mental health crisis assessment and priority queue management system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML engine for mental health analysis
try:
    ml_engine = MentalHealthTriageEngine()
    logger.info("Mental Health Triage Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ML engine: {e}")
    raise RuntimeError("Could not initialize mental health analysis engine")

# In-memory data storage (TODO: Replace with persistent database in production)
conversations: Dict[str, List[Dict]] = {}
priority_queue: List[Dict] = []
connected_clients: List[WebSocket] = []


@app.get("/", tags=["Health Check"])
async def root():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: API status message and metadata
    """
    return {
        "message": "Mental Health Crisis Triage API Running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_conversations": len(conversations),
        "queue_size": len(priority_queue),
        "connected_clients": len(connected_clients)
    }


@app.post("/api/message", tags=["Message Analysis"])
async def analyze_message(data: MessageRequest):
    """
    Analyze a user message for mental health risk assessment.
    
    This endpoint processes incoming messages through AI-powered emotion analysis,
    determines crisis severity, generates supportive responses, and updates the
    priority queue for counselor triage.
    
    Args:
        data (MessageRequest): Contains message text and user_id
        
    Returns:
        dict: Comprehensive analysis including:
            - severity: Risk level classification (🚨 Critical to ✅ Stable)
            - priority_score: Numerical priority (0-100)
            - emotions: Detected emotion scores
            - top_emotions: Most prominent emotions
            - has_suicide_risk: Boolean suicide risk indicator
            - ai_response: Generated supportive response
            - timestamp: Analysis timestamp
            - user_id: User identifier
            
    Raises:
        HTTPException: 500 if analysis fails
        HTTPException: 400 if invalid request data
    """
    try:
        message = data.message.strip()
        user_id = data.user_id
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        logger.info(f"Analyzing message from user {user_id}: {message[:50]}...")
        
        # Perform ML-based mental health analysis
        analysis = ml_engine.analyze_message(message)
        
        # Add metadata
        analysis["timestamp"] = datetime.now().isoformat()
        analysis["user_id"] = user_id
        analysis["message_id"] = f"{user_id}_{datetime.now().timestamp()}"
        
        # Store conversation history
        if user_id not in conversations:
            conversations[user_id] = []
        conversations[user_id].append(analysis)
        
        # Update priority queue for counselor dashboard
        update_priority_queue(user_id, analysis)
        
        # Notify all connected dashboard clients
        await notify_clients({
            "type": "new_analysis", 
            "data": analysis,
            "queue_update": True
        })
        
        logger.info(f"Analysis complete for user {user_id}: {analysis['severity']} (Priority: {analysis['priority_score']})")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@app.get("/api/queue", tags=["Priority Queue"])
async def get_priority_queue():
    """
    Retrieve the current priority queue sorted by urgency.
    
    Returns users in order of mental health crisis severity, with highest
    priority (most urgent) cases first. Used by counselor dashboards to
    determine which users need immediate attention.
    
    Returns:
        List[dict]: Sorted list of users with:
            - user_id: User identifier
            - latest_message: Most recent message text
            - severity: Current risk assessment
            - priority_score: Numerical priority (0-100)
            - timestamp: Last message timestamp
            - message_count: Total messages from user
            - wait_time_minutes: How long user has been waiting
            
    Example:
        [
            {
                "user_id": "user123",
                "latest_message": "I can't go on anymore",
                "severity": "🚨 Critical - Suicide Risk",
                "priority_score": 95,
                "timestamp": "2024-01-15T10:30:00",
                "message_count": 3,
                "wait_time_minutes": 5
            }
        ]
    """
    try:
        # Sort by priority score (highest first) and add wait times
        sorted_queue = sorted(
            priority_queue,
            key=lambda x: (x["priority_score"], x["timestamp"]),
            reverse=True
        )
        
        # Calculate wait times
        current_time = datetime.now()
        for item in sorted_queue:
            message_time = datetime.fromisoformat(item["timestamp"])
            wait_minutes = int((current_time - message_time).total_seconds() / 60)
            item["wait_time_minutes"] = max(wait_minutes, 0)
        
        logger.info(f"Priority queue retrieved: {len(sorted_queue)} users")
        return sorted_queue
        
    except Exception as e:
        logger.error(f"Failed to retrieve priority queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve priority queue")


@app.get("/api/conversations/{user_id}", tags=["Conversations"])
async def get_user_conversation(user_id: str):
    """
    Retrieve complete conversation history for a specific user.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        dict: User conversation data including:
            - user_id: User identifier
            - messages: List of all messages and analyses
            - total_messages: Count of messages
            - first_contact: Timestamp of first message
            - latest_severity: Most recent risk assessment
            - risk_progression: How risk level has changed over time
            
    Raises:
        HTTPException: 404 if user not found
    """
    if user_id not in conversations:
        raise HTTPException(status_code=404, detail="User conversation not found")
    
    try:
        user_messages = conversations[user_id]
        
        return {
            "user_id": user_id,
            "messages": user_messages,
            "total_messages": len(user_messages),
            "first_contact": user_messages[0]["timestamp"] if user_messages else None,
            "latest_severity": user_messages[-1]["severity"] if user_messages else None,
            "risk_progression": [msg["priority_score"] for msg in user_messages]
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve conversation for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    
    Maintains persistent connections with counselor dashboards to provide
    real-time notifications of new high-priority cases and queue updates.
    
    Protocol:
        - Incoming: Client can send heartbeat/status messages
        - Outgoing: Server sends analysis results and queue updates
        
    Message Types:
        - new_analysis: New message analyzed
        - queue_update: Priority queue changed
        - system_alert: High-priority crisis detected
        
    Args:
        websocket (WebSocket): WebSocket connection instance
    """
    await websocket.accept()
    connected_clients.append(websocket)
    client_id = f"client_{len(connected_clients)}"
    
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "current_queue_size": len(priority_queue)
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (heartbeat, status requests, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client messages
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}: {data}")
                    
            except asyncio.TimeoutError:
                # Send heartbeat to check if client is still connected
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Clean up connection
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"WebSocket client {client_id} cleaned up")


async def handle_websocket_message(websocket: WebSocket, message: dict):
    """
    Handle incoming WebSocket messages from dashboard clients.
    
    Args:
        websocket (WebSocket): Client connection
        message (dict): Parsed JSON message from client
    """
    message_type = message.get("type")
    
    if message_type == "get_queue":
        # Send current queue status
        queue_data = await get_priority_queue()
        await websocket.send_text(json.dumps({
            "type": "queue_data",
            "data": queue_data,
            "timestamp": datetime.now().isoformat()
        }))
    elif message_type == "heartbeat":
        # Respond to client heartbeat
        await websocket.send_text(json.dumps({
            "type": "heartbeat_response",
            "timestamp": datetime.now().isoformat()
        }))


def update_priority_queue(user_id: str, analysis: dict) -> None:
    """
    Update the priority queue with new user analysis.
    
    Removes any existing entry for the user and adds updated information
    based on the latest message analysis. This ensures the queue always
    reflects the most current risk assessment for each user.
    
    Args:
        user_id (str): User identifier
        analysis (dict): Latest message analysis results
        
    Side Effects:
        - Modifies global priority_queue list
        - Logs queue updates for monitoring
    """
    global priority_queue
    
    try:
        # Remove existing entry for this user
        initial_count = len(priority_queue)
        priority_queue = [item for item in priority_queue if item["user_id"] != user_id]
        
        # Create new queue entry
        queue_entry = {
            "user_id": user_id,
            "latest_message": analysis["text"][:100] + "..." if len(analysis["text"]) > 100 else analysis["text"],
            "severity": analysis["severity"],
            "priority_score": analysis["priority_score"],
            "timestamp": analysis["timestamp"],
            "message_count": len(conversations.get(user_id, [])),
            "has_suicide_risk": analysis.get("has_suicide_risk", False),
            "top_emotions": analysis.get("top_emotions", [])[:2]  # Limit to top 2 emotions
        }
        
        priority_queue.append(queue_entry)
        
        # Log significant changes
        if analysis["priority_score"] >= 80:
            logger.warning(f"HIGH PRIORITY user added to queue: {user_id} (Score: {analysis['priority_score']})")
        elif analysis["priority_score"] >= 60:
            logger.info(f"Medium priority user added to queue: {user_id} (Score: {analysis['priority_score']})")
        
        logger.debug(f"Queue updated: {initial_count} -> {len(priority_queue)} users")
        
    except Exception as e:
        logger.error(f"Failed to update priority queue for user {user_id}: {e}")


async def notify_clients(message: dict) -> None:
    """
    Send real-time notifications to all connected WebSocket clients.
    
    Broadcasts analysis results, queue updates, and system alerts to
    counselor dashboards. Automatically handles disconnected clients.
    
    Args:
        message (dict): Message to broadcast to all clients
        
    Side Effects:
        - Sends WebSocket messages to all connected clients
        - Removes disconnected clients from active list
        - Logs notification events
    """
    if not connected_clients:
        return
    
    try:
        message["server_timestamp"] = datetime.now().isoformat()
        message_json = json.dumps(message)
        
        # Send to all connected clients
        disconnected_clients = []
        for client in connected_clients.copy():
            try:
                await client.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected_clients.append(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            if client in connected_clients:
                connected_clients.remove(client)
        
        logger.debug(f"Notification sent to {len(connected_clients)} clients")
        
        # Log high-priority alerts
        if message.get("data", {}).get("priority_score", 0) >= 80:
            logger.warning(f"HIGH PRIORITY ALERT broadcasted: {message.get('data', {}).get('severity', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"Failed to notify clients: {e}")


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    Performs initialization tasks when the API server starts:
    - Logs startup information
    - Validates ML engine functionality
    - Sets up monitoring
    """
    logger.info("Mental Health Crisis Triage API starting up...")
    logger.info(f"ML Engine: {type(ml_engine).__name__}")
    logger.info("API ready to receive requests")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    
    Performs cleanup tasks when the API server shuts down:
    - Closes WebSocket connections
    - Logs shutdown information
    - Saves state if needed
    """
    logger.info("Mental Health Crisis Triage API shutting down...")
    
    # Close all WebSocket connections
    for client in connected_clients.copy():
        try:
            await client.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket connection: {e}")
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Mental Health Crisis Triage API server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )