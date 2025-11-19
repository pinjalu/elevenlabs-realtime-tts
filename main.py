from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Low-Latency TTS for VAPI")

# ElevenLabs configuration
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

# Use Flash v2.5 for ultra-low latency (75ms inference time)
MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")

# Use global endpoint for reduced latency based on geographic location
BASE_URL = "https://api-global-preview.elevenlabs.io/v1/text-to-speech"

# Reuse HTTP client to avoid SSL/TLS handshake overhead on every request
http_client = httpx.AsyncClient(timeout=30.0)


@app.post("/tts")
async def tts_endpoint(request: Request):
    """
    Ultra-low latency TTS endpoint for VAPI
    
    Body:
    {
      "message": {
        "type": "voice-request",
        "text": "Hello, how can I help you today?",
        "sampleRate": 24000
      }
    }
    """
    try:
        payload = await request.json()
        message = payload.get("message", {})
        text = message.get("text")
        sample_rate = message.get("sampleRate", 24000)

        if not text:
            return JSONResponse({"error": "Missing text field."}, status_code=400)

        # Streaming endpoint with PCM output
        tts_url = f"{BASE_URL}/{VOICE_ID}/stream?output_format=pcm_{sample_rate}"

        headers = {
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_API_KEY
        }

        # Optimized payload for lowest latency
        data = {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "use_speaker_boost": True
            }
        }

        # Stream asynchronously using httpx (much faster than requests)
        async def audio_stream():
            async with http_client.stream("POST", tts_url, headers=headers, json=data) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"ElevenLabs API error: {error_text.decode()}")
                
                # Stream chunks as they arrive
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type=f"audio/pcm;rate={sample_rate}",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable proxy buffering
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Low-Latency TTS for VAPI",
        "model": MODEL_ID,
        "endpoint": "global-preview"
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP client on shutdown"""
    await http_client.aclose()