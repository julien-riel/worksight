#!/usr/bin/env python3
"""Jour 2 §2.3 — boucle automatique d'itération de prompts.

- Exécute chaque (prompt, modèle) via `POST /detect` sur le serveur local.
- Sauvegarde les métriques incrémentalement dans benchmarks/jour2-iterations.json.
- Met à jour la table dans PLAN.md (entre les marqueurs BEGIN/END ITERATIONS TABLE).
- Arrêt anticipé si rappel ≥ 0.95 ET précision ≥ 0.80.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "data" / "samples"
BENCHMARKS = ROOT / "benchmarks"
RESULTS_JSON = BENCHMARKS / "jour2-iterations.json"
LOG_PATH = BENCHMARKS / "jour2-iterations.log"
PLAN_PATH = ROOT / "PLAN.md"
SERVER = "http://127.0.0.1:8000"

BEGIN_MARK = "<!-- BEGIN ITERATIONS TABLE -->"
END_MARK = "<!-- END ITERATIONS TABLE -->"

PROMPTS = [
    "Chantier?",
    "Chantier actif?",
    "Travaux en cours?",
    "Ouvriers, machines ou véhicules de chantier au travail?",
    "Voie bloquée ou réduite par des travaux?",
    "Chantier avec activité en cours, pas seulement signalisation?",
]
MODELS = ["gemma", "claude"]

THRESH_RECALL = 0.95
THRESH_PRECISION = 0.80


def log_line(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def load_existing() -> list[dict]:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return []


def save_results(runs: list[dict]) -> None:
    BENCHMARKS.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(runs, indent=2, ensure_ascii=False) + "\n"
    )


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float]:
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return recall, precision, f1


def run_batch(model: str, prompt: str, samples: list[dict], log) -> dict:
    tp = fp = fn = tn = 0
    total_elapsed = 0.0
    n_done = 0
    parse_errors = 0
    with httpx.Client(timeout=300.0) as client:
        for i, s in enumerate(samples):
            path = SAMPLES_DIR / s["file"]
            b64 = base64.b64encode(path.read_bytes()).decode()
            t0 = time.time()
            n_boxes = 0
            parse_error = False
            try:
                r = client.post(
                    f"{SERVER}/detect",
                    json={"image_b64": b64, "prompt": prompt, "model": model},
                )
                if r.status_code == 500:
                    # JSON invalide côté modèle → conservateur : predicted = False
                    parse_error = True
                    parse_errors += 1
                    elapsed = time.time() - t0
                else:
                    r.raise_for_status()
                    data = r.json()
                    n_boxes = len(data.get("detections", []))
                    elapsed = data.get("elapsed", time.time() - t0)
            except Exception as e:
                log(f"  ERR {s['file']}: {e}")
                continue

            predicted = n_boxes > 0
            truth = s["has_construction"]
            if predicted and truth:
                tp += 1
            elif not predicted and not truth:
                tn += 1
            elif predicted and not truth:
                fp += 1
            else:
                fn += 1
            total_elapsed += elapsed
            n_done += 1
            tag = "PARSE_ERR" if parse_error else f"boxes={n_boxes}"
            log(
                f"  {i + 1:>2}/{len(samples)} {s['file']} "
                f"truth={'+' if truth else '-'} {tag} {elapsed:.1f}s"
            )
    recall, precision, f1 = compute_metrics(tp, fp, fn, tn)
    avg = total_elapsed / n_done if n_done else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "f1": f1,
        "avg_elapsed": avg,
        "n_done": n_done,
        "parse_errors": parse_errors,
    }


def pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def render_table(runs: list[dict]) -> str:
    header = (
        "| # | Modèle | Prompt | Rappel | Précision | F1 | Temps moy. | Note |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = []
    # Trier par prompt_id puis modèle pour lisibilité
    for r in sorted(runs, key=lambda x: (x["prompt_id"], x["model"])):
        note = r.get("note", "")
        pe = r.get("parse_errors", 0)
        if pe and "parse" not in note.lower():
            prefix = f"{pe} parse error(s) comptés comme « sans »."
            note = f"{prefix} {note}".strip()
        rows.append(
            f"| {r['prompt_id']} | {r['model']} | `{r['prompt']}` | "
            f"{pct(r['recall'])} | {pct(r['precision'])} | {pct(r['f1'])} | "
            f"{r['avg_elapsed']:.1f} s | {note} |"
        )
    return "\n".join([header, *rows])


def update_plan(runs: list[dict]) -> None:
    text = PLAN_PATH.read_text()
    block = f"{BEGIN_MARK}\n{render_table(runs)}\n{END_MARK}"
    pattern = re.compile(
        re.escape(BEGIN_MARK) + r"[\s\S]*?" + re.escape(END_MARK)
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
        PLAN_PATH.write_text(text)


def main() -> int:
    BENCHMARKS.mkdir(exist_ok=True)
    samples = json.loads((SAMPLES_DIR / "manifest.json").read_text())["samples"]

    with LOG_PATH.open("a") as fh:
        log = lambda msg: log_line(fh, msg)
        log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} — start =====")

        # Quick server ping
        try:
            httpx.get(f"{SERVER}/system-prompts", timeout=5.0).raise_for_status()
        except Exception as e:
            log(f"[abort] Serveur injoignable sur {SERVER}: {e}")
            return 1

        runs = load_existing()
        done_keys = {(r["prompt"], r["model"]) for r in runs}
        log(f"[start] {len(runs)} run(s) déjà présents, on saute ces couples.")

        for idx, prompt in enumerate(PROMPTS):
            for model in MODELS:
                if (prompt, model) in done_keys:
                    log(f"[skip] #{idx} {model} + {prompt!r}")
                    continue
                log(f"\n--- #{idx} {model} + {prompt!r} ---")
                t0 = time.time()
                m = run_batch(model, prompt, samples, log)
                dt = time.time() - t0
                run = {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "model": model,
                    **m,
                    "batch_elapsed": dt,
                }
                runs.append(run)
                save_results(runs)
                update_plan(runs)
                log(
                    f"=> rappel={pct(m['recall'])} précision={pct(m['precision'])} "
                    f"F1={pct(m['f1'])} moy={m['avg_elapsed']:.1f}s batch={dt:.0f}s"
                )
                if m["recall"] >= THRESH_RECALL and m["precision"] >= THRESH_PRECISION:
                    log(
                        f"*** SEUIL ATTEINT (rappel {pct(m['recall'])} "
                        f"≥ {THRESH_RECALL:.0%} ET précision {pct(m['precision'])} "
                        f"≥ {THRESH_PRECISION:.0%}) — arrêt."
                    )
                    return 0

        log("\n[done] File de prompts épuisée, aucun n'a atteint les deux seuils.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
