#!/usr/bin/env python3
"""Évalue 6 variations de prompt (3 FR + 3 EN) sur deux jeux d'évaluation.

Set A : ROADWork 20 images (labels binaires `has_construction`).
Set B : frames downtown-olympic que l'humain a validées "positive" (tous positifs).

Métrique parse_error → comptée comme "pas de détection" (convention JOUR 2).

Usage : python3 bench_prompt_variations.py [--retries 3]
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

ROOT = Path(__file__).parent
SERVER = "http://127.0.0.1:8000"

PROMPTS = {
    "E0":  "Chantier, cônes, barrières ou panneaux de travaux?",
    "FR1": "Chantier, cônes, bollards ou barrières?",
    "FR2": "Travaux, cônes, barrières ou panneaux?",
    "EN1": "Construction, cones, barriers or work signs?",
    "EN2": "Road work, cones, barriers or traffic signs?",
    "EN3": "Construction zone, traffic cones, barriers or work signs?",
}


def load_roadwork() -> list[tuple[Path, bool]]:
    manifest = json.loads((ROOT / "data" / "samples" / "manifest.json").read_text())
    items = []
    for s in manifest["samples"]:
        # ROADWork : 20 images initiales (pas de champ source ou source != video:*)
        src = s.get("source", "roadwork")
        if src.startswith("video:"):
            continue
        p = ROOT / "data" / "samples" / s["file"]
        if p.exists():
            items.append((p, bool(s["has_construction"])))
    return items


def load_user_positives(video_name: str) -> list[tuple[Path, bool]]:
    base = ROOT / "data" / "video-frames" / video_name
    ann = json.loads((base / "annotations.json").read_text())
    frames_dir = base / ann["source_meta"]["frames_dir"]
    validations = ann.get("validations", {})
    out = []
    for frame, v in validations.items():
        if v.get("decision") == "positive":
            p = frames_dir / frame
            if p.exists():
                out.append((p, True))
    return out


def run_one(client: httpx.Client, path: Path, prompt: str, retries: int):
    b64 = base64.b64encode(path.read_bytes()).decode()
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
                return None, True  # parse error
            r.raise_for_status()
            return r.json().get("detections", []) or [], False
        except Exception:
            if attempt < retries:
                time.sleep(0.4)
                continue
            return None, True
    return None, True


def eval_set(set_name: str, items: list[tuple[Path, bool]], prompt: str, retries: int) -> dict:
    tp = fp = fn = tn = 0
    parse_errors = 0
    total_t = 0.0
    with httpx.Client(timeout=300.0) as client:
        for i, (path, truth) in enumerate(items, 1):
            t0 = time.time()
            detections, err = run_one(client, path, prompt, retries)
            total_t += time.time() - t0
            if err:
                parse_errors += 1
                positive = False  # convention : parse error → pas de détection
            else:
                positive = len(detections) > 0
            if positive and truth: tp += 1
            elif not positive and not truth: tn += 1
            elif positive and not truth: fp += 1
            else: fn += 1
            if i % 10 == 0 or i == len(items):
                print(
                    f"    {i:>3}/{len(items)} {path.name}  tp={tp} fp={fp} fn={fn} tn={tn}",
                    flush=True,
                )
    n = len(items)
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )
    return {
        "set": set_name,
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "parse_errors": parse_errors,
        "avg_elapsed": total_t / n if n else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--video", default="downtown-olympic")
    args = ap.parse_args()

    try:
        httpx.get(f"{SERVER}/system-prompts", timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"[abort] serveur injoignable : {e}")
        return 1

    roadwork = load_roadwork()
    user_pos = load_user_positives(args.video)
    print(f"[bench] ROADWork : {len(roadwork)} images (binaire)")
    print(f"[bench] User-positive ({args.video}) : {len(user_pos)} frames (tous positifs)")
    print(f"[bench] {len(PROMPTS)} prompts · ~{(len(roadwork) + len(user_pos)) * len(PROMPTS) * 6 / 60:.0f} min estimés")

    all_runs = []
    for code, prompt in PROMPTS.items():
        for set_name, items in [("roadwork", roadwork), ("user-positive", user_pos)]:
            print(f"\n=== [{code}] [{set_name}] {prompt!r} ===", flush=True)
            t0 = time.time()
            m = eval_set(set_name, items, prompt, args.retries)
            m.update({"code": code, "prompt": prompt, "batch_elapsed": time.time() - t0})
            all_runs.append(m)

    # ---- Tables récap ----
    def pct(x): return f"{x * 100:.0f}%"

    print("\n" + "=" * 90)
    print(f"SET: ROADWork (binaire, n={len(roadwork)})")
    print("-" * 90)
    print(f"{'':<5} {'prompt':<55} {'R':>6} {'P':>6} {'F1':>6} {'moy':>6} {'pe':>4}")
    print("-" * 90)
    rw = [r for r in all_runs if r["set"] == "roadwork"]
    rw.sort(key=lambda r: -r["f1"])
    for r in rw:
        p = (r["prompt"][:52] + "...") if len(r["prompt"]) > 55 else r["prompt"]
        print(
            f"[{r['code']:<3}] {p:<55} "
            f"{pct(r['recall']):>6} {pct(r['precision']):>6} {pct(r['f1']):>6} "
            f"{r['avg_elapsed']:>5.1f}s {r['parse_errors']:>4}"
        )

    print("\n" + "=" * 90)
    print(f"SET: User-positive ({args.video}, n={len(user_pos)} — tous positifs)")
    print("-" * 90)
    print(f"{'':<5} {'prompt':<55} {'R':>6} {'moy':>6} {'pe':>4}")
    print("-" * 90)
    up = [r for r in all_runs if r["set"] == "user-positive"]
    up.sort(key=lambda r: -r["recall"])
    for r in up:
        p = (r["prompt"][:52] + "...") if len(r["prompt"]) > 55 else r["prompt"]
        print(
            f"[{r['code']:<3}] {p:<55} "
            f"{pct(r['recall']):>6} {r['avg_elapsed']:>5.1f}s {r['parse_errors']:>4}"
        )
    print("=" * 90)

    # Sauvegarde
    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"prompt-variations-{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "video": args.video,
                "retries": args.retries,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "prompts": PROMPTS,
                "runs": all_runs,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
