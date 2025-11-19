from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Custom TTS for VAPI (PCM Output)")

# ElevenLabs configuration
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")

BASE_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

@app.post("/tts")
async def tts_endpoint(request: Request):
    """
    VAPI -> POST /tts
    Body:
    {
      "message": {
        "type": "voice-request",
        "text": "Hello, how can I help you today?",
        "sampleRate": 24000
      }
    }
    Response: Raw PCM stream (e.g. pcm_24000)
    """
    try:
        payload = await request.json()
        message = payload.get("message", {})
        text = message.get("text")
        sample_rate = message.get("sampleRate", 24000)

        if not text:
            return JSONResponse({"error": "Missing text field."}, status_code=400)

        # ElevenLabs streaming endpoint with output_format=pcm
        tts_url = f"{BASE_TTS_URL}/{VOICE_ID}/stream?output_format=pcm_{sample_rate}"

        headers = {
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_API_KEY
        }

        data = {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }

        # Stream directly from ElevenLabs
        response = requests.post(tts_url, headers=headers, json=data, stream=True)

        if not response.ok:
            return JSONResponse(
                {"error": f"ElevenLabs API error: {response.text}"}, status_code=500
            )

        def audio_stream():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        # Send PCM stream back to VAPI
        return StreamingResponse(audio_stream(), media_type=f"audio/pcm;rate={sample_rate}")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Custom TTS for VAPI is live!"}
