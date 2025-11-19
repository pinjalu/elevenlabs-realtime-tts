from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
import websockets

load_dotenv()

app = FastAPI(title="Realtime ElevenLabs TTS")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
MODEL_ID = "eleven_turbo_v2"

REALTIME_URL = "wss://api.elevenlabs.io/v1/text-to-speech/ws"

@app.post("/tts")
async def tts(request: Request):
    try:
        body = await request.json()
        message = body.get("message", {})
        text = message.get("text", "")
        sample_rate = message.get("sampleRate", 24000)

        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        async def audio_stream():

            headers = [("xi-api-key", ELEVEN_API_KEY)]  # ✅ list of tuples

            async with websockets.connect(
                REALTIME_URL,
                extra_headers=headers  # ✅ pass list, not dict
            ) as ws:
                ...


                init_message = {
                    "text": text,
                    "voice_id": VOICE_ID,
                    "model_id": MODEL_ID,
                    "output_format": f"pcm_{sample_rate}",
                    "voice_settings": {
                        "stability": 0.3,
                        "similarity_boost": 0.3,
                        "style": 0,
                        "use_speaker_boost": False
                    }
                }

                await ws.send(json.dumps(init_message))

                while True:
                    try:
                        chunk = await ws.recv()
                        if isinstance(chunk, bytes):
                            yield chunk
                        else:
                            data = json.loads(chunk)
                            if data.get("isFinal"):
                                break
                    except:
                        break

        return StreamingResponse(
            audio_stream(),
            media_type=f"audio/pcm;rate={sample_rate}"
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health():
    return {"status": "ok"}
