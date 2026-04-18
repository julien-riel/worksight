import json
import re
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
VIDEO_FRAMES_DIR = BASE_DIR / "data" / "video-frames"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
if VIDEO_FRAMES_DIR.exists():
    app.mount(
        "/video-frames",
        StaticFiles(directory=VIDEO_FRAMES_DIR),
        name="video-frames",
    )


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
            "system": DETECT_SYSTEM_PROMPT,
            "note": (
                "Mode éditeur : le même system prompt que l'onglet Détection est appliqué "
                "à chaque image du batch. Un résultat avec ≥1 boîte = « chantier détecté »."
            ),
        },
        "dataset": {
            "system": DETECT_SYSTEM_PROMPT,
            "note": (
                "Pré-annotation vidéo : `annotate_video.py` appelle /detect sur chaque frame, "
                "applique un smoothing temporel, propose les frames candidates. "
                "L'humain valide dans l'UI puis exporte vers data/samples/."
            ),
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


# ============================================================
# Dataset builder — videos pré-annotées
# ============================================================


def _video_dir(name: str) -> Path:
    # Empêche le directory traversal
    if "/" in name or ".." in name or not name:
        raise HTTPException(400, "nom de vidéo invalide")
    d = VIDEO_FRAMES_DIR / name
    if not d.is_dir():
        raise HTTPException(404, f"vidéo {name!r} introuvable")
    return d


_DECISION_ALIASES = {"keep": "positive", "reject": "negative"}


def _load_annotations(name: str) -> tuple[Path, dict]:
    d = _video_dir(name)
    ann_path = d / "annotations.json"
    if not ann_path.exists():
        raise HTTPException(
            404,
            f"annotations manquantes — lance annotate_video.py {name}",
        )
    ann = json.loads(ann_path.read_text())
    # Migration des anciennes décisions (keep → positive, reject → negative)
    v = ann.get("validations") or {}
    changed = False
    for frame, entry in v.items():
        dec = entry.get("decision")
        if dec in _DECISION_ALIASES:
            entry["decision"] = _DECISION_ALIASES[dec]
            changed = True
    if changed:
        ann["validations"] = v
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False) + "\n")
    return ann_path, ann


class BBox(BaseModel):
    box: list[float]
    label: str = "chantier"


class Validation(BaseModel):
    frame: str
    decision: Literal["positive", "signalisation", "negative", "skip", "reset"]
    boxes: list[BBox] | None = None


class BatchValidations(BaseModel):
    items: list[Validation]


@app.get("/videos")
async def list_videos():
    if not VIDEO_FRAMES_DIR.exists():
        return {"videos": []}
    out = []
    for d in sorted(VIDEO_FRAMES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        ann_path = d / "annotations.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        item = {
            "name": d.name,
            "title": meta.get("title"),
            "n_frames": meta.get("n_frames", 0),
            "has_annotations": ann_path.exists(),
        }
        if ann_path.exists():
            ann = json.loads(ann_path.read_text())
            item.update({
                "n_candidates": ann.get("n_candidates", 0),
                "n_segments": ann.get("n_segments", 0),
                "prompt": ann.get("prompt"),
                "model": ann.get("model"),
            })
        out.append(item)
    return {"videos": out}


@app.get("/videos/{name}/annotations")
async def get_annotations(name: str):
    _, ann = _load_annotations(name)
    return ann


@app.post("/videos/{name}/validations")
async def set_validation(name: str, v: Validation):
    ann_path, ann = _load_annotations(name)
    validations = ann.setdefault("validations", {})
    if v.decision == "reset":
        validations.pop(v.frame, None)
    else:
        entry = {
            "decision": v.decision,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if v.boxes is not None:
            entry["boxes"] = [b.model_dump() for b in v.boxes]
        validations[v.frame] = entry
    ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False) + "\n")
    counts = {"positive": 0, "signalisation": 0, "negative": 0, "skip": 0}
    for entry in validations.values():
        dec = entry.get("decision")
        if dec in counts:
            counts[dec] += 1
    return {"ok": True, "counts": counts}


@app.post("/videos/{name}/validations/batch")
async def batch_set_validations(name: str, body: BatchValidations):
    ann_path, ann = _load_annotations(name)
    validations = ann.setdefault("validations", {})
    now = datetime.now(timezone.utc).isoformat()
    applied = 0
    for item in body.items:
        if item.decision == "reset":
            if validations.pop(item.frame, None) is not None:
                applied += 1
        else:
            entry = {"decision": item.decision, "at": now}
            if item.boxes is not None:
                entry["boxes"] = [b.model_dump() for b in item.boxes]
            validations[item.frame] = entry
            applied += 1
    ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False) + "\n")
    counts = {"positive": 0, "signalisation": 0, "negative": 0, "skip": 0}
    for e in validations.values():
        d = e.get("decision")
        if d in counts:
            counts[d] += 1
    return {"ok": True, "applied": applied, "counts": counts}


@app.post("/videos/{name}/export")
async def export_kept(name: str):
    ann_path, ann = _load_annotations(name)
    validations = ann.get("validations", {})
    exportable = [
        (f, v["decision"]) for f, v in validations.items()
        if v.get("decision") in ("positive", "signalisation", "negative")
    ]
    if not exportable:
        raise HTTPException(
            400,
            "Aucune frame marquée « chantier », « signalisation » ou « sans »",
        )

    src_dir = _video_dir(name) / ann["source_meta"]["frames_dir"]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists()
        else {"samples": []}
    )
    existing = {s["file"] for s in manifest["samples"]}

    # Index des per_frame pour retrouver les pseudo-boxes Gemma
    per_frame_by_name = {pf["frame"]: pf for pf in ann.get("per_frame", [])}

    # Mapping décision → classe + flag binaire phase 1
    # has_construction = True uniquement pour chantier actif; signalisation et sans = False
    CLASS_MAP = {
        "positive": ("chantier", True),
        "signalisation": ("signalisation", False),
        "negative": ("sans", False),
    }

    added_counts = {"chantier": [], "signalisation": [], "sans": []}
    now = datetime.now(timezone.utc).isoformat()
    for frame, decision in exportable:
        src = src_dir / frame
        if not src.exists():
            continue
        dst_name = f"dashcam_{name}_{frame}"
        dst = SAMPLES_DIR / dst_name
        if dst_name in existing:
            continue
        shutil.copy2(src, dst)
        category, has_construction = CLASS_MAP[decision]
        entry: dict = {
            "file": dst_name,
            "label": 1 if has_construction else 0,
            "has_construction": has_construction,
            "category": category,
            "source": f"video:{name}",
            "original_frame": frame,
            "validated_as": decision,
            "added_at": now,
        }
        # Pseudo-boxes Gemma uniquement pour classes contenant une détection visuelle
        if decision in ("positive", "signalisation"):
            pf = per_frame_by_name.get(frame, {})
            pseudo = pf.get("detections") or []
            if pseudo:
                entry["pseudo_boxes"] = pseudo
                entry["pseudo_boxes_model"] = ann.get("model")
                entry["pseudo_boxes_prompt"] = ann.get("prompt")
        # Bboxes humaines pour chantier ET signalisation (vérité-terrain tri-classe)
        if decision in ("positive", "signalisation"):
            user_boxes = (validations.get(frame) or {}).get("boxes")
            if user_boxes:
                entry["boxes"] = user_boxes
        manifest["samples"].append(entry)
        added_counts[category].append(dst_name)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "ok": True,
        "added_chantier": len(added_counts["chantier"]),
        "added_signalisation": len(added_counts["signalisation"]),
        "added_sans": len(added_counts["sans"]),
        "files": (
            added_counts["chantier"]
            + added_counts["signalisation"]
            + added_counts["sans"]
        ),
    }
