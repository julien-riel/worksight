#!/usr/bin/env python3
"""Compare plusieurs prompts Gemma sur un échantillon de frames.

Mesure : taux de candidats (≥1 bbox), parse errors, temps moyen, diversité
des labels renvoyés par Gemma. Pas de vérité-terrain : on compare des
caractéristiques, pas des scores d'accuracy.

Usage :
  python3 bench_prompts.py <video-name> [--frames 60] [--retries 3]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
SERVER = "http://127.0.0.1:8000"

PROMPTS = {
    "T": "Travaux en cours?",
    "C": "Chantier ou signalisation?",
    "E": "Chantier, cônes, barrières ou panneaux de travaux?",
    "A": "Aménagement temporaire de chantier routier?",
}


def run_prompt(video_dir: Path, frames: list[Path], prompt: str, retries: int) -> dict:
    n = len(frames)
    positives = 0
    parse_errors = 0
    total_elapsed = 0.0
    labels: Counter = Counter()
    with httpx.Client(timeout=300.0) as client:
        for i, path in enumerate(frames):
            b64 = base64.b64encode(path.read_bytes()).decode()
            t0 = time.time()
            detections: list[dict] = []
            had_error = False
            for attempt in range(1, retries + 1):
                try:
                    r = client.post(
                        f"{SERVER}/detect",
                        json={"image_b64": b64, "prompt": prompt, "model": "gemma"},
                    )
                    if r.status_code == 500:
                        if attempt < retries:
                            time.sleep(0.4)
                            continue
                        had_error = True
                        break
                    r.raise_for_status()
                    data = r.json()
                    detections = data.get("detections", []) or []
                    break
                except Exception:
                    if attempt < retries:
                        time.sleep(0.4)
                        continue
                    had_error = True
                    break
            if had_error:
                parse_errors += 1
            elif detections:
                positives += 1
                for d in detections:
                    lbl = (d.get("label") or "").lower().strip()
                    if lbl:
                        labels[lbl] += 1
            total_elapsed += time.time() - t0
            if (i + 1) % 10 == 0 or i == n - 1:
                print(
                    f"    {i + 1:>3}/{n} | "
                    f"positives {positives:>3} · parse_err {parse_errors:>3}",
                    flush=True,
                )

    return {
        "prompt": prompt,
        "n_frames": n,
        "n_positive": positives,
        "n_parse_error": parse_errors,
        "positive_rate": positives / n if n else 0,
        "parse_error_rate": parse_errors / n if n else 0,
        "avg_elapsed": total_elapsed / n if n else 0,
        "total_elapsed": total_elapsed,
        "top_labels": labels.most_common(6),
        "n_distinct_labels": len(labels),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    base = ROOT / "data" / "video-frames" / args.name
    meta = json.loads((base / "metadata.json").read_text())
    frames_dir = base / meta["frames_dir"]
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))[: args.frames]
    if not all_frames:
        print(f"[abort] pas de frames dans {frames_dir}")
        return 1

    try:
        httpx.get(f"{SERVER}/system-prompts", timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"[abort] serveur injoignable : {e}")
        return 1

    print(f"[bench] {args.name} — {len(all_frames)} frames · {len(PROMPTS)} prompts")
    results = []
    for code, prompt in PROMPTS.items():
        print(f"\n=== [{code}] {prompt!r} ===", flush=True)
        t0 = time.time()
        r = run_prompt(base, all_frames, prompt, args.retries)
        r["code"] = code
        r["batch_elapsed"] = time.time() - t0
        results.append(r)

    # Tableau récap
    print("\n" + "=" * 78)
    print(f"{'':<4} {'prompt':<55} {'pos%':>6} {'err%':>6} {'moy':>7} {'lbls':>5}")
    print("-" * 78)
    for r in results:
        p = (r["prompt"][:52] + "...") if len(r["prompt"]) > 55 else r["prompt"]
        print(
            f"[{r['code']}] {p:<55} "
            f"{r['positive_rate'] * 100:>5.0f}% "
            f"{r['parse_error_rate'] * 100:>5.0f}% "
            f"{r['avg_elapsed']:>5.1f}s "
            f"{r['n_distinct_labels']:>5}"
        )
    print("=" * 78)

    for r in results:
        print(f"\n[{r['code']}] labels top : {r['top_labels']}")

    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"prompt-bench-{args.name}-{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "video": args.name,
                "frames_sampled": len(all_frames),
                "retries": args.retries,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
