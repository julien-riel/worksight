"""Télécharge 20 images depuis natix-network-org/roadwork (HuggingFace)
en mode streaming : 10 avec chantier (label=1), 10 sans (label=0).

Sortie :
  data/samples/chantier_NN.jpg   (10 fichiers)
  data/samples/sans_NN.jpg       (10 fichiers)
  data/samples/manifest.json     (métadonnées)

Usage :
  python3 fetch_sample.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "Le package 'datasets' n'est pas installé.\n"
        "Lance : pip install -r requirements.txt"
    )

DATASET = "natix-network-org/roadwork"
SPLIT = "train"
N_PER_LABEL = 10
OUT_DIR = Path(__file__).parent / "data" / "samples"


def extract_meta(row: dict) -> dict:
    tags = row.get("scene_level_tags") or {}
    return {
        "original_id": row.get("id"),
        "city_name": row.get("city_name"),
        "scene_description": row.get("scene_description"),
        "weather": tags.get("weather"),
        "daytime": tags.get("daytime"),
        "scene_environment": tags.get("scene_environment"),
        "travel_alteration": tags.get("travel_alteration"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {DATASET} (split={SPLIT})…")
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)

    pos: list[dict] = []
    neg: list[dict] = []
    seen = 0

    for row in ds:
        seen += 1
        label = row.get("label")
        if label == 1 and len(pos) < N_PER_LABEL:
            pos.append(row)
        elif label == 0 and len(neg) < N_PER_LABEL:
            neg.append(row)
        if len(pos) >= N_PER_LABEL and len(neg) >= N_PER_LABEL:
            break
        if seen % 100 == 0:
            print(f"  parcouru {seen} lignes (pos={len(pos)}, neg={len(neg)})")

    if len(pos) < N_PER_LABEL or len(neg) < N_PER_LABEL:
        print(
            f"Attention : seulement pos={len(pos)}, neg={len(neg)} après "
            f"{seen} lignes du split.",
            file=sys.stderr,
        )

    samples = []
    for prefix, label_value, rows in (
        ("chantier", 1, pos),
        ("sans", 0, neg),
    ):
        for i, row in enumerate(rows, start=1):
            filename = f"{prefix}_{i:02d}.jpg"
            path = OUT_DIR / filename
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path, format="JPEG", quality=92)
            samples.append({
                "file": filename,
                "label": label_value,
                "has_construction": bool(label_value),
                **extract_meta(row),
            })
            print(f"  -> {filename}  (label={label_value})")

    manifest = {
        "source": DATASET,
        "split": SPLIT,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_with_construction": len(pos),
        "n_without_construction": len(neg),
        "samples": samples,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(samples)} images sauvegardées dans {OUT_DIR}")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
