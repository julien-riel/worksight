# Architecture

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE DATASET-BUILDER                    │
│                                                                 │
│   YouTube URL                                                   │
│       │                                                         │
│       ▼                                                         │
│   fetch_video.py  ──►  data/video-frames/<name>/                │
│                          ├── <name>.mp4                         │
│                          ├── frames/frame_NNNNNN.jpg            │
│                          └── metadata.json                      │
│                                                                 │
│   ┌───────────────────────────────────┐                         │
│   ▼                                                             │
│ annotate_video.py                                               │
│   │ (boucle Gemma + smoothing 3/5 + retry)                      │
│   ▼                                                             │
│   data/video-frames/<name>/annotations.json                     │
│       ├── per_frame[]    ← brut Gemma (pseudo-labels)           │
│       ├── segments[]     ← plages de chantier confirmé          │
│       ├── candidates[]   ← frames à valider                     │
│       └── validations{}  ← décisions humaines (via UI)          │
│                                                                 │
│   ┌───────────────────────────────────┐                         │
│   ▼                                                             │
│ UI onglet Dataset                                               │
│   (galerie, modal, canvas bbox, pagination, batch-save)         │
│   │                                                             │
│   ▼                                                             │
│ POST /videos/<name>/export                                      │
│   │ (copie frames + append manifest)                            │
│   ▼                                                             │
│   data/samples/                                                 │
│       ├── chantier_XX.jpg (ROADWork)                            │
│       ├── sans_XX.jpg (ROADWork)                                │
│       ├── dashcam_<video>_frame_XXXXXX.jpg (exportés)           │
│       └── manifest.json  ← source de vérité                     │
│                                                                 │
│                                                                 │
│   [POST-MVP] data/samples/ → YOLO .txt → entraînement YOLO 8    │
└─────────────────────────────────────────────────────────────────┘
```

## Arborescence de code

```
worksight/
├── server.py              FastAPI, endpoints REST
├── index.html             UI monolithique, 4 onglets
│
├── fetch_sample.py        Import 20 images ROADWork (phase 1)
├── fetch_video.py         Télécharge YouTube → frames
├── annotate_video.py      Pré-annote + smoothing + retry
├── run_iterations.py      Sweep JOUR 2 prompts (archive)
│
├── data/
│   ├── samples/           ← dataset final (manifest.json)
│   └── video-frames/      ← staging par vidéo
│
├── benchmarks/            ← runs JOUR 2 (archive)
├── docs/                  ← ce dossier
├── CLAUDE.md              ← contexte persistant Claude Code
├── PLAN.md                ← séquençage 48h (archive)
└── README.md              ← setup utilisateur
```

## Serveur (`server.py`)

FastAPI lancé via `uvicorn server:app --host 127.0.0.1 --port 8000 --reload`.

| Endpoint | Rôle |
|---|---|
| `GET /` | Sert `index.html` |
| `GET /system-prompts` | Expose read-only les system prompts utilisés |
| `POST /detect` | Détection objet sur 1 image (Gemma ou Claude) |
| `POST /chat` | Chat multi-tours (Gemma ou Claude) |
| `GET /samples/*` | Static : `data/samples/` |
| `GET /video-frames/*` | Static : `data/video-frames/` |
| `GET /videos` | Liste des vidéos avec état annotation |
| `GET /videos/{name}/annotations` | Lit `annotations.json` (avec migration auto) |
| `POST /videos/{name}/validations` | Une décision (legacy, non utilisé par l'UI) |
| `POST /videos/{name}/validations/batch` | Lot de décisions (batch-save UI) |
| `POST /videos/{name}/export` | Copie frames validées vers `data/samples/` + maj manifest |

**Backends** :
- **Gemma** : Ollama local (`http://localhost:11434`), modèle `gemma4:e4b`. Warm-up au démarrage, `keep_alive: -1`.
- **Claude** : `claude-agent-sdk` Python, modèle `claude-sonnet-4-6`. Auth via Claude Code local.

## UI (`index.html`)

Monolithique (~1700 lignes), un seul fichier HTML+CSS+JS inline. Choix assumé pour la rapidité de dev en solo.

**Onglets** :

| Onglet | Rôle | Utilité dataset-builder |
|---|---|---|
| **Détection** | Upload/picker image + prompt + bboxes overlay + export PNG/JSON | Diagnostic/debug d'une image |
| **Chat** | Multi-tours libre avec Gemma ou Claude | Debug/exploration |
| **Boucle** | Batch 20 images ROADWork + métriques (rappel/précision/F1) | Tuning de prompt (phase 1 finie) |
| **Dataset** | Galerie vidéo, modal plein écran, édition bboxes, validation 3 classes, export | **Cœur du workflow dataset-builder** |

**Sélecteur de modèle global** en header (Gemma / Claude) — passé dans toutes les requêtes.

**État client-side** (batch-save) :
- `dsPendingSaves` : Map frame → décision en attente
- Flush auto 2 s après dernier clic, sur changement de vidéo/filtre/export, sur `beforeunload` (via `sendBeacon`)

## Flux de données — un exemple

Utilisateur valide `frame_000028.jpg` comme "chantier" avec 2 bboxes :

1. Clic dans le modal → `dsModalDecide('positive')` côté UI
2. Update optimiste local : `dsAnnotations.validations['frame_000028.jpg'] = {decision, boxes}`
3. Push dans `dsPendingSaves`, badge "Sauvegarder maintenant (N)" visible
4. 2 s plus tard : `POST /videos/downtown-olympic/validations/batch` avec tous les items pending
5. Serveur écrit `annotations.json` en un seul `fsync`
6. Queue vidée, badge disparaît

À l'export :
1. Clic "Exporter N frames"
2. Flush synchrone si pending
3. `POST /videos/downtown-olympic/export`
4. Serveur : pour chaque validation `positive`/`signalisation`/`negative` → copie frame + append entrée dans `data/samples/manifest.json`
