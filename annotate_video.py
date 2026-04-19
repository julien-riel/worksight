#!/usr/bin/env python3
"""Pré-annote les frames d'une vidéo avec Gemma, applique le smoothing temporel.

Entrée :
  data/video-frames/<name>/metadata.json + frames/frame_NNNNNN.jpg

Sortie :
  data/video-frames/<name>/annotations.json

Usage :
  python3 annotate_video.py <name> [--prompt "Travaux en cours?"] [--window 5 --threshold 3]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SERVER = "http://127.0.0.1:8000"
DEFAULT_PROMPT = "Chantier, cônes, barrières ou panneaux de travaux?"
DEFAULT_MODEL = "gemma"
DEFAULT_RETRIES = 3


def smooth(positives: list[bool], window: int, threshold: int) -> list[bool]:
    """Fenêtre glissante : position i confirmée si ≥threshold/window autour de i."""
    n = len(positives)
    half = window // 2
    out = [False] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        count = sum(1 for j in range(lo, hi) if positives[j])
        out[i] = count >= threshold
    return out


def find_segments(smoothed: list[bool], fps: float) -> list[dict]:
    segs = []
    i = 0
    while i < len(smoothed):
        if smoothed[i]:
            start = i
            while i < len(smoothed) and smoothed[i]:
                i += 1
            end = i - 1
            segs.append({
                "start_frame": start,
                "end_frame": end,
                "start_ts": round(start / fps, 2),
                "end_ts": round(end / fps, 2),
                "duration_s": round((end - start + 1) / fps, 2),
                "n_frames": end - start + 1,
            })
        else:
            i += 1
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Nom du dossier dans data/video-frames/")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="Garde 1 frame sur N")
    ap.add_argument("--limit", type=int, default=0, help="Max frames (0 = toutes)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Essais max en cas de parse error")
    args = ap.parse_args()

    base = Path("data/video-frames") / args.name
    meta_path = base / "metadata.json"
    if not meta_path.exists():
        print(f"[abort] {meta_path} introuvable. Lance fetch_video.py d'abord.")
        return 1
    meta = json.loads(meta_path.read_text())
    fps = meta["fps_sampling"] / args.stride
    frames_dir = base / meta["frames_dir"]
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    frames = all_frames[::args.stride]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        print(f"[abort] Aucune frame dans {frames_dir}")
        return 1
    print(f"[sampling] {len(all_frames)} total, stride={args.stride}, limit={args.limit} → {len(frames)} à traiter (fps effectif={fps})")

    # Préserve les validations humaines existantes (ne jette pas leur travail)
    out_path = base / "annotations.json"
    existing_validations = {}
    if out_path.exists():
        # Backup horodaté avant toute réécriture (on garde les 5 derniers)
        backups_dir = base / "backups"
        backups_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backups_dir / f"annotations-{ts}.json"
        try:
            backup_path.write_bytes(out_path.read_bytes())
            print(f"[backup] → {backup_path.name}")
            # Prune : garde les 5 plus récents
            existing_backups = sorted(backups_dir.glob("annotations-*.json"))
            for stale in existing_backups[:-5]:
                stale.unlink()
        except Exception as e:
            print(f"[warn] Backup raté: {e}")

        try:
            existing = json.loads(out_path.read_text())
            existing_validations = existing.get("validations", {}) or {}
            if existing_validations:
                print(f"[preserve] {len(existing_validations)} validation(s) humaine(s) conservée(s)")
        except Exception as e:
            print(f"[warn] Impossible de relire l'ancien annotations.json: {e}")

    try:
        httpx.get(f"{SERVER}/system-prompts", timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"[abort] Serveur injoignable sur {SERVER}: {e}")
        return 1

    print(
        f"[start] {len(frames)} frames, modèle={args.model}, "
        f"prompt={args.prompt!r}, smoothing={args.threshold}/{args.window}"
    )
    t0 = time.time()
    per_frame = []
    parse_errors = 0
    with httpx.Client(timeout=300.0) as client:
        for i, path in enumerate(frames):
            b64 = base64.b64encode(path.read_bytes()).decode()
            req_t0 = time.time()
            n_boxes = 0
            detections: list[dict] = []
            parse_error = False
            attempts_used = 0
            for attempt in range(1, args.retries + 1):
                attempts_used = attempt
                try:
                    r = client.post(
                        f"{SERVER}/detect",
                        json={"image_b64": b64, "prompt": args.prompt, "model": args.model},
                    )
                    if r.status_code == 500:
                        # Parse error côté Gemma → retry
                        if attempt < args.retries:
                            time.sleep(0.4)
                            continue
                        parse_error = True
                        parse_errors += 1
                        break
                    r.raise_for_status()
                    data = r.json()
                    detections = data.get("detections", []) or []
                    n_boxes = len(detections)
                    break
                except Exception as e:
                    if attempt < args.retries:
                        time.sleep(0.4)
                        continue
                    print(f"  ERR {path.name}: {e}", flush=True)
                    parse_error = True
                    parse_errors += 1
                    break
            elapsed = time.time() - req_t0
            positive = (n_boxes > 0) and not parse_error
            per_frame.append({
                "frame": path.name,
                "idx": i,
                "ts": round(i / fps, 2),
                "n_boxes": n_boxes,
                "detections": detections,
                "positive": positive,
                "parse_error": parse_error,
                "attempts": attempts_used,
                "elapsed": round(elapsed, 2),
            })
            tag = "PARSE_ERR" if parse_error else ("+" if positive else "-")
            if (i + 1) % 10 == 0 or i == len(frames) - 1 or parse_error:
                print(
                    f"  {i + 1:>4}/{len(frames)} {path.name} {tag} "
                    f"boxes={n_boxes} {elapsed:.1f}s",
                    flush=True,
                )

    positives = [f["positive"] for f in per_frame]
    smoothed = smooth(positives, args.window, args.threshold)
    segments = find_segments(smoothed, fps)

    # Candidates : toutes les frames positives contenues dans un segment confirmé
    confirmed_positions = set()
    for seg in segments:
        for idx in range(seg["start_frame"], seg["end_frame"] + 1):
            confirmed_positions.add(idx)
    candidates = [
        f["frame"] for f in per_frame
        if f["positive"] and f["idx"] in confirmed_positions
    ]

    n_positive = sum(positives)
    n_smoothed = sum(smoothed)
    batch_elapsed = time.time() - t0
    avg = (
        sum(f["elapsed"] for f in per_frame) / len(per_frame)
        if per_frame else 0
    )

    annotations = {
        "source_meta": meta,
        "model": args.model,
        "prompt": args.prompt,
        "smoothing": {"window": args.window, "threshold": args.threshold},
        "n_frames": len(per_frame),
        "n_positive_raw": n_positive,
        "n_positive_smoothed": n_smoothed,
        "n_segments": len(segments),
        "n_candidates": len(candidates),
        "parse_errors": parse_errors,
        "avg_elapsed": round(avg, 2),
        "batch_elapsed": round(batch_elapsed, 1),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
        "candidates": candidates,
        "per_frame": per_frame,
        "validations": existing_validations,
    }
    out_path.write_text(json.dumps(annotations, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[done] {batch_elapsed:.0f}s ({avg:.1f}s/frame moyenne)")
    print(f"  positives brutes : {n_positive}/{len(per_frame)} ({n_positive/len(per_frame):.0%})")
    print(f"  après smoothing  : {n_smoothed}/{len(per_frame)} ({n_smoothed/len(per_frame):.0%})")
    print(f"  segments         : {len(segments)}")
    for s in segments[:5]:
        print(f"    [{s['start_ts']:>6.1f}s → {s['end_ts']:>6.1f}s] {s['n_frames']} frames")
    if len(segments) > 5:
        print(f"    ... et {len(segments) - 5} de plus")
    print(f"  candidates       : {len(candidates)}")
    print(f"  parse errors     : {parse_errors}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
