import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL = "gemma4:e4b"
CLAUDE_MODEL = "claude-sonnet-4-6"
BASE_DIR = Path(__file__).parent

ModelChoice = Literal["gemma", "claude"]

DETECT_SYSTEM_PROMPT = (
    "You are an object detection model. Return ONLY a JSON array of detected "
    "objects matching the user's query. Each object must have exactly two keys:\n"
    '  - "label": a short text label (string, in the same language as the user query)\n'
    '  - "box": [y1, x1, y2, x2] with coordinates normalized to 0-1000\n'
    "Return an empty array [] if nothing matches. No explanations, no markdown, JSON only.\n\n"
    "User query: "
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[warmup] Chargement de {GEMMA_MODEL} en mémoire…")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.post(
                OLLAMA_GENERATE_URL,
                json={"model": GEMMA_MODEL, "prompt": "ok", "stream": False, "keep_alive": -1},
            )
        print("[warmup] Modèle prêt.")
    except Exception as e:
        print(f"[warmup] Échec (le serveur démarre quand même): {e}")
    yield


app = FastAPI(lifespan=lifespan)

SAMPLES_DIR = BASE_DIR / "data" / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


class DetectRequest(BaseModel):
    image_b64: str
    prompt: str
    model: ModelChoice = "gemma"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: ModelChoice = "gemma"


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/system-prompts")
async def system_prompts():
    return {
        "detect": {
            "system": DETECT_SYSTEM_PROMPT,
            "note": (
                "Concaténé avec votre prompt utilisateur, puis envoyé au modèle sélectionné "
                "(Gemma via Ollama format=json, ou Claude via Agent SDK avec extraction JSON)."
            ),
        },
        "chat": {
            "system": None,
            "note": "Aucun system prompt — l'historique multi-tours est envoyé tel quel au modèle sélectionné.",
        },
        "loop": {
            "system": None,
            "note": "À venir.",
        },
    }


# ============================================================
# Backend Gemma (Ollama)
# ============================================================

async def gemma_detect(image_b64: str, user_prompt: str) -> str:
    payload = {
        "model": GEMMA_MODEL,
        "prompt": DETECT_SYSTEM_PROMPT + user_prompt,
        "images": [image_b64],
        "format": "json",
        "stream": False,
        "keep_alive": -1,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(OLLAMA_GENERATE_URL, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Ollama a répondu: {r.text[:300]}")
        data = r.json()
    return data.get("response", "").strip()


async def gemma_chat(messages: list[ChatMessage]) -> str:
    payload = {
        "model": GEMMA_MODEL,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
        "keep_alive": -1,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(OLLAMA_CHAT_URL, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Ollama a répondu: {r.text[:300]}")
        data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


# ============================================================
# Backend Claude (Agent SDK, abonnement local)
# ============================================================

def _import_claude_sdk():
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            TextBlock,
        )
        return ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
    except ImportError as e:
        raise HTTPException(
            500,
            f"claude-agent-sdk non installé: {e}. Lance: pip install -r requirements.txt",
        )


async def _claude_collect_text(client, AssistantMessage, TextBlock) -> str:
    chunks: list[str] = []
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks).strip()


async def claude_detect(image_b64: str, user_prompt: str) -> str:
    ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock = _import_claude_sdk()

    async def stream():
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": DETECT_SYSTEM_PROMPT + user_prompt},
                ],
            },
        }

    options = ClaudeAgentOptions(model=CLAUDE_MODEL)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(stream())
        return await _claude_collect_text(client, AssistantMessage, TextBlock)


async def claude_chat(messages: list[ChatMessage]) -> str:
    ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock = _import_claude_sdk()

    async def stream():
        for m in messages:
            yield {
                "type": "user",
                "message": {"role": m.role, "content": m.content},
            }

    options = ClaudeAgentOptions(model=CLAUDE_MODEL)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(stream())
        return await _claude_collect_text(client, AssistantMessage, TextBlock)


# ============================================================
# Parsing détections (commun aux 2 backends)
# ============================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_payload(raw: str) -> str:
    """Trouve un bloc JSON dans la réponse. Gère les fences ```json et le texte autour."""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    # Sinon : premier [...] balancé ou {...}
    for opener, closer in (("[", "]"), ("{", "}")):
        i = raw.find(opener)
        j = raw.rfind(closer)
        if i != -1 and j > i:
            return raw[i : j + 1]
    return raw


def parse_detections(raw: str) -> list[dict]:
    payload = _extract_json_payload(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Sortie JSON invalide: {payload[:300]}")

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
    return cleaned


# ============================================================
# Endpoints
# ============================================================

@app.post("/detect")
async def detect(req: DetectRequest):
    image_b64 = req.image_b64
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    start = time.perf_counter()
    if req.model == "claude":
        raw = await claude_detect(image_b64, req.prompt)
    else:
        raw = await gemma_detect(image_b64, req.prompt)
    elapsed = time.perf_counter() - start

    detections = parse_detections(raw)
    return {"detections": detections, "elapsed": elapsed, "model": req.model}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "messages vide")

    start = time.perf_counter()
    if req.model == "claude":
        content = await claude_chat(req.messages)
    else:
        content = await gemma_chat(req.messages)
    elapsed = time.perf_counter() - start

    return {"content": content, "elapsed": elapsed, "model": req.model}
