#!/usr/bin/env python3
"""Télécharge un segment d'une vidéo YouTube et extrait les frames à `fps`.

Usage :
  python3 fetch_video.py <url> [--start 0] [--duration 600] [--fps 2] [--name slug]

Sortie :
  data/video-frames/<name>/
    <name>.mp4           ← segment téléchargé
    frames/frame_NNNNNN.jpg  ← frames extraites
    metadata.json        ← infos source + paramètres
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--start", type=float, default=0.0, help="Seconde de début")
    ap.add_argument("--duration", type=float, default=600.0, help="Durée (s)")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames/s à extraire")
    ap.add_argument("--name", help="Slug du dossier de sortie (défaut = video id)")
    ap.add_argument("--max-height", type=int, default=720, help="Résolution max")
    args = ap.parse_args()

    # Récupérer les métadonnées de la vidéo
    info_raw = subprocess.run(
        ["yt-dlp", "--dump-single-json", "--no-download", args.url],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(info_raw)
    video_id = info["id"]
    title = info.get("title", video_id)
    duration_total = info.get("duration", 0)
    name = args.name or video_id
    print(f"[info] {title} ({video_id}) — {duration_total}s total")

    out_dir = Path("data/video-frames") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{name}.mp4"

    if video_path.exists():
        print(f"[skip] {video_path} existe déjà")
    else:
        # Téléchargement complet sans ré-encoding (rapide).
        # On extrait la tranche temporelle ensuite avec ffmpeg.
        run([
            "yt-dlp",
            "-f", f"bestvideo[height<={args.max_height}][ext=mp4]/bestvideo[height<={args.max_height}]",
            "--no-playlist",
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            args.url,
        ])

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    # Nettoie les anciennes frames si on re-extrait
    for f in frames_dir.glob("frame_*.jpg"):
        f.unlink()

    # Extraction de la tranche [start, start+duration] à `fps` fps en un passage.
    run([
        "ffmpeg", "-y",
        "-ss", str(args.start),
        "-i", str(video_path),
        "-t", str(args.duration),
        "-vf", f"fps={args.fps}",
        "-q:v", "3",
        str(frames_dir / "frame_%06d.jpg"),
    ])

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    n = len(frames)

    meta = {
        "source_url": args.url,
        "video_id": video_id,
        "title": title,
        "name": name,
        "start_sec": args.start,
        "duration_sec": args.duration,
        "fps_sampling": args.fps,
        "max_height": args.max_height,
        "n_frames": n,
        "video_file": video_path.name,
        "frames_dir": "frames",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n[done] {n} frames extraites → {frames_dir}")
    print(f"[done] metadata : {out_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
