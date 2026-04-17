import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b"
BASE_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are an object detection model. Return ONLY a JSON array of detected "
    "objects matching the user's query. Each object must have exactly two keys:\n"
    '  - "label": a short text label (string, in the same language as the user query)\n'
    '  - "box": [y1, x1, y2, x2] with coordinates normalized to 0-1000\n'
    "Return an empty array [] if nothing matches. No explanations, JSON only.\n\n"
    "User query: "
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[warmup] Chargement de {MODEL} en mémoire…")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": "ok", "stream": False, "keep_alive": -1},
            )
        print("[warmup] Modèle prêt.")
    except Exception as e:
        print(f"[warmup] Échec (le serveur démarre quand même): {e}")
    yield


app = FastAPI(lifespan=lifespan)


class DetectRequest(BaseModel):
    image_b64: str
    prompt: str


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/detect")
async def detect(req: DetectRequest):
    image_b64 = req.image_b64
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    payload = {
        "model": MODEL,
        "prompt": SYSTEM_PROMPT + req.prompt,
        "images": [image_b64],
        "format": "json",
        "stream": False,
        "keep_alive": -1,
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Ollama a répondu: {r.text[:300]}")
        data = r.json()
    elapsed = time.perf_counter() - start

    raw = data.get("response", "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Sortie JSON invalide: {raw[:300]}")

    if isinstance(parsed, dict):
        for key in ("detections", "objects", "results", "boxes", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed] if "box" in parsed else []

    cleaned = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        box = item.get("box") or item.get("bbox")
        label = item.get("label") or item.get("name") or ""
        if isinstance(box, list) and len(box) == 4:
            cleaned.append({"label": str(label), "box": box})

    return {"detections": cleaned, "elapsed": elapsed}
