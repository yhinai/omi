from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import httpx
import asyncio


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class Transcript(BaseModel):
    id: int
    text: str
    is_user: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    has_audio: bool = False
    audio_url: Optional[str] = None

class TranscriptCreate(BaseModel):
    text: str
    is_user: bool

class VoiceRequest(BaseModel):
    text: str
    voice_type: str = "female"  # female or male

# ElevenLabs configuration
ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"
ELEVENLABS_VOICE_IDS = {
    "female": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "male": "VR6AewLTigWG4xSOukaG"     # Josh
}

# In-memory storage for demo (replace with database in production)
transcripts_storage = []
next_transcript_id = 1

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Vision Assistant API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Transcript endpoints
@api_router.get("/transcripts")
async def get_transcripts(after: int = 0):
    """Get transcripts after a specific ID"""
    filtered_transcripts = [t for t in transcripts_storage if t["id"] > after]
    return {"transcripts": filtered_transcripts}

@api_router.post("/transcripts", response_model=Transcript)
async def create_transcript(transcript: TranscriptCreate):
    """Create a new transcript entry"""
    global next_transcript_id
    
    new_transcript = {
        "id": next_transcript_id,
        "text": transcript.text,
        "is_user": transcript.is_user,
        "timestamp": datetime.utcnow(),
        "has_audio": False,
        "audio_url": None
    }
    
    transcripts_storage.append(new_transcript)
    next_transcript_id += 1
    
    # If it's an assistant response, generate voice automatically
    if not transcript.is_user:
        try:
            # Generate voice in background
            asyncio.create_task(generate_voice_for_transcript(new_transcript["id"], transcript.text))
        except Exception as e:
            logger.error(f"Failed to generate voice: {e}")
    
    return Transcript(**new_transcript)

@api_router.post("/voice/generate")
async def generate_voice(request: VoiceRequest):
    """Generate voice using ElevenLabs API"""
    try:
        voice_id = ELEVENLABS_VOICE_IDS.get(request.voice_type, ELEVENLABS_VOICE_IDS["female"])
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.5,
                        "use_speaker_boost": True
                    }
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                return Response(
                    content=response.content,
                    media_type="audio/mpeg",
                    headers={"Content-Disposition": "attachment; filename=speech.mp3"}
                )
            else:
                raise HTTPException(status_code=response.status_code, detail="ElevenLabs API error")
                
    except Exception as e:
        logger.error(f"Voice generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/voice/transcript/{transcript_id}")
async def get_transcript_audio(transcript_id: int):
    """Get audio for a specific transcript"""
    transcript = next((t for t in transcripts_storage if t["id"] == transcript_id), None)
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    if not transcript["has_audio"]:
        # Generate voice on-demand
        try:
            voice_id = ELEVENLABS_VOICE_IDS["female"]  # Default to female voice
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": ELEVENLABS_API_KEY
                    },
                    json={
                        "text": transcript["text"],
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                            "style": 0.5,
                            "use_speaker_boost": True
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    transcript["has_audio"] = True
                    return Response(
                        content=response.content,
                        media_type="audio/mpeg",
                        headers={"Content-Disposition": "attachment; filename=speech.mp3"}
                    )
                else:
                    raise HTTPException(status_code=response.status_code, detail="ElevenLabs API error")
                    
        except Exception as e:
            logger.error(f"Voice generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail="Audio not available")

async def generate_voice_for_transcript(transcript_id: int, text: str):
    """Background task to generate voice for transcript"""
    try:
        transcript = next((t for t in transcripts_storage if t["id"] == transcript_id), None)
        if transcript:
            # Mark as having audio (simplified for demo)
            transcript["has_audio"] = True
            logger.info(f"Generated voice for transcript {transcript_id}")
    except Exception as e:
        logger.error(f"Background voice generation failed: {e}")

# Demo route to test the assistant conversation
@api_router.post("/demo/conversation")
async def demo_conversation(message: str = "Hello"):
    """Demo endpoint to simulate assistant conversation with voice"""
    global next_transcript_id
    
    # Add user message
    user_transcript = {
        "id": next_transcript_id,
        "text": message,
        "is_user": True,
        "timestamp": datetime.utcnow(),
        "has_audio": False,
        "audio_url": None
    }
    transcripts_storage.append(user_transcript)
    next_transcript_id += 1
    
    # Generate assistant response
    assistant_response = f"I heard you say: {message}. How can I help you with navigation today?"
    
    assistant_transcript = {
        "id": next_transcript_id,
        "text": assistant_response,
        "is_user": False,
        "timestamp": datetime.utcnow(),
        "has_audio": True,
        "audio_url": f"/api/voice/transcript/{next_transcript_id}"
    }
    transcripts_storage.append(assistant_transcript)
    next_transcript_id += 1
    
    return {"message": "Conversation added", "assistant_response": assistant_response}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add some demo transcripts on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Vision Assistant API...")
    # Add a demo assistant message
    global next_transcript_id
    demo_transcript = {
        "id": next_transcript_id,
        "text": "Welcome to your Vision Assistant! I'm ready to help you navigate and identify objects. Try saying 'Take me to train station' to get started.",
        "is_user": False,
        "timestamp": datetime.utcnow(),
        "has_audio": True,
        "audio_url": f"/api/voice/transcript/{next_transcript_id}"
    }
    transcripts_storage.append(demo_transcript)
    next_transcript_id += 1

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
