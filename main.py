from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = FastAPI(title="Custom TTS for VAPI")

# ElevenLabs configuration
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")

if not ELEVEN_API_KEY:
    raise ValueError("ELEVEN_API_KEY not set in environment!")

BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Use persistent HTTP client for better performance
http_client = httpx.AsyncClient(timeout=30.0)


@app.post("/tts")
async def tts_endpoint(request: Request):
    """
    Custom TTS endpoint for VAPI
    Receives text and returns PCM audio stream
    """
    start_time = datetime.now()
    
    try:
        # Log incoming request
        print(f"\n{'='*60}")
        print(f"⏰ Request received at: {start_time}")
        
        # Parse payload
        payload = await request.json()
        print(f"📥 Payload: {payload}")
        
        # Handle both VAPI formats
        if "message" in payload:
            message = payload.get("message", {})
            text = message.get("text")
            sample_rate = message.get("sampleRate", 24000)
        else:
            text = payload.get("text")
            sample_rate = payload.get("sampleRate", 24000)

        print(f"📝 Text: '{text}'")
        print(f"🎵 Sample Rate: {sample_rate}")

        if not text or text.strip() == "":
            print("❌ Error: Empty text")
            return JSONResponse({"error": "Missing text"}, status_code=400)

        # Build ElevenLabs URL
        tts_url = f"{BASE_URL}/{VOICE_ID}/stream?output_format=pcm_{sample_rate}"
        print(f"🌐 ElevenLabs URL: {tts_url}")

        headers = {
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_API_KEY
        }

        data = {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "use_speaker_boost": True
            }
        }

        # Stream audio
        async def audio_stream():
            bytes_sent = 0
            first_chunk_time = None
            
            async with http_client.stream("POST", tts_url, headers=headers, json=data) as response:
                print(f"📡 ElevenLabs Status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = error_text.decode()
                    print(f"❌ ElevenLabs Error: {error_msg}")
                    raise Exception(f"ElevenLabs error: {error_msg}")
                
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = datetime.now()
                            ttfb = (first_chunk_time - start_time).total_seconds()
                            print(f"⚡ Time to first byte: {ttfb:.2f}s")
                        
                        bytes_sent += len(chunk)
                        yield chunk
            
            total_time = (datetime.now() - start_time).total_seconds()
            duration = bytes_sent / (sample_rate * 2)  # 16-bit = 2 bytes per sample
            
            print(f"✅ Success!")
            print(f"   - Bytes sent: {bytes_sent:,}")
            print(f"   - Audio duration: {duration:.2f}s")
            print(f"   - Total time: {total_time:.2f}s")
            print(f"{'='*60}\n")

        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Custom TTS for VAPI",
        "voice_id": VOICE_ID,
        "model": MODEL_ID
    }


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()