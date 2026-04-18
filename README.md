# worksight

**Détection automatique des chantiers et entraves sur le domaine public** à partir d'images et de vidéos dashcam, pour entraîner un modèle **YOLO 8** déployable en **edge computing**. Pipeline construit autour de deux backends :

- **Gemma 4 E4B** en local via [Ollama](https://ollama.com) — auto-annotation rapide
- **Claude Sonnet 4.6** via [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) — oracle de comparaison (abonnement Claude, pas de clé API)

*Worksight* = *worksite* + *sight* — voir les chantiers.

## But final

Construire un jeu de données tri-classe (**chantier** / **signalisation** / **sans**) pour entraîner YOLO 8 sur Jetson Orin NX et Mac mini M4.

Voir **[`docs/INTENTION.md`](docs/INTENTION.md)** pour l'intention détaillée, **[`docs/STATUT.md`](docs/STATUT.md)** pour l'état courant chiffré, **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** pour les composants, **[`docs/PERIMETRE.md`](docs/PERIMETRE.md)** pour ce qui est essentiel vs « voire trop ».

## App web — 4 onglets

Une seule app web, sélecteur de modèle global (Gemma ↔ Claude) :

| Onglet | Rôle |
|---|---|
| **Détection** | Upload image ou picker ROADWork, prompt, bboxes overlay, export PNG/JSON |
| **Chat** | Conversation libre multi-tours — diagnostic/exploration |
| **Boucle** | Batch sur les 20 images ROADWork + métriques rappel/précision/F1 + historique des runs |
| **Dataset** | Galerie des candidats par vidéo, modal plein écran avec canvas d'édition bboxes, validation 3 classes, export vers `data/samples/` |

## Jeu de données

`data/samples/manifest.json` est la source de vérité. Chaque entrée contient :

```json
{
  "file": "dashcam_downtown-olympic_frame_000028.jpg",
  "label": 1,
  "has_construction": true,
  "category": "chantier",        // "chantier" | "signalisation" | "sans"
  "source": "video:downtown-olympic",
  "original_frame": "frame_000028.jpg",
  "validated_as": "positive",
  "pseudo_boxes": [...],          // pré-annotations Gemma (référence)
  "boxes": [...]                  // bboxes vérité-terrain dessinées par l'humain
}
```

### Sources d'images

- **ROADWork** (ICCV 2025) — 20 images via le mirror HF [`natix-network-org/roadwork`](https://huggingface.co/datasets/natix-network-org/roadwork) pour le jeu d'évaluation
- **Dashcams YouTube** — vidéos Montréal de la chaîne [`@DadsDashCam`](https://www.youtube.com/@DadsDashCam), découpées en frames à 2 fps, pré-annotées par Gemma, validées humainement dans l'UI

## Pipeline dataset-builder

```
YouTube URL  ─►  fetch_video.py  ─►  frames/frame_NNNNNN.jpg
                                     │
                                     ▼
                          annotate_video.py (Gemma + smoothing 3/5 + retry)
                                     │
                                     ▼
                          annotations.json (pseudo-labels)
                                     │
                                     ▼
                          UI onglet Dataset (validation humaine)
                                     │
                                     ▼
                          data/samples/ + manifest.json
                                     │
                                     ▼
                          [à venir] YOLO 8 training
```

## Prérequis

- macOS avec Apple Silicon
- [Homebrew](https://brew.sh)
- **Python ≥ 3.10** (le SDK Claude Agent ne supporte pas 3.9) — recommandé `brew install python@3.12`
- Pour le mode Claude : être déjà loggé dans [Claude Code](https://docs.claude.com/en/docs/claude-code) (l'Agent SDK utilise `~/.claude/` automatiquement)

## Installation

```bash
# Backend Gemma (Ollama)
brew install ollama
brew services start ollama
ollama pull gemma4:e4b

# Outils pour le pipeline vidéo
brew install yt-dlp ffmpeg

# Worksight
cd worksight
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pull initial des 20 échantillons ROADWork (jeu d'éval)
python3 fetch_sample.py
```

## Lancer l'app

```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Ouvrir <http://localhost:8000>. Au démarrage, `[warmup] Modèle prêt.` signale que Gemma est chargé.

## Workflow dataset-builder

**1. Télécharger et découper une vidéo**

```bash
python3 fetch_video.py https://youtu.be/gSyn204dCHY --duration 600 --fps 2 --name downtown-olympic
```

Sort `data/video-frames/downtown-olympic/` avec `frames/` et `metadata.json`.

**2. Pré-annoter avec Gemma**

```bash
python3 annotate_video.py downtown-olympic
# options : --prompt "..." --window 5 --threshold 3 --retries 3 --limit N
```

Sort `annotations.json` avec `per_frame`, `segments`, `candidates`, `validations` (préserve les validations existantes au re-run).

**3. Valider dans l'UI**

Ouvrir l'onglet **Dataset**, sélectionner la vidéo :

- **Filtres** : Candidats positifs (Gemma + smoothing), Positifs isolés (rejetés par smoothing), Non-candidats (pour échantillonner des négatifs), Toutes, ou par décision
- **Sampling** pour les grands filtres (Non-candidats, Toutes) : 50 / 100 / 200 / 500 / tout, avec seed stable et bouton *Re-tirer*
- **Pagination** 100 par page
- **Clic sur une vignette** → modal plein écran avec canvas d'édition de bboxes (pseudo-Gemma en bleu pointillé, tes bboxes GT en vert plein)

Décisions (raccourcis clavier dans le modal) :

| Touche | Décision | Classe exportée |
|---|---|---|
| **Entrée** | Accepter la suggestion Gemma (basée sur les labels de détection) | la classe suggérée |
| **C** | Chantier | `category: "chantier"`, `has_construction: true` |
| **G** | Signalisation | `category: "signalisation"`, `has_construction: false` |
| **N** | Sans | `category: "sans"`, `has_construction: false` |
| **S** | Skip | Exclu du dataset |
| **A** | Annuler | Repart en attente |
| **←** / **→** | Navigation | |
| **I** | Importer pseudo-boxes Gemma comme bboxes humaines | |
| **Delete** | Supprimer dernière bbox dessinée | |
| **Échap** | Fermer modal | |

**Suggestion automatique** : sur chaque frame, l'UI propose une classe basée sur les labels Gemma (ex : labels contenant `chantier` ou `ouvrier` → suggestion **Chantier** ; labels `cône`/`panneau`/`barrière` seuls → **Signalisation** ; aucune détection → **Sans**). Le bouton suggéré est highlighté en jaune et **Entrée** l'accepte. Les labels Gemma sont aussi affichés sous l'image (ex : `chantier ×1 · cônes ×3 · panneaux ×2`).

Les validations sont mises en queue côté client (badge orange *Sauvegarder maintenant (N)*) puis flushées toutes les 2 s via `POST /validations/batch` — une seule écriture disque par lot.

**4. Exporter vers le dataset**

Bouton **Exporter** dans l'onglet Dataset → copie les frames validées dans `data/samples/` et append au manifest.

## Scripts utilitaires

| Script | Rôle |
|---|---|
| `fetch_sample.py` | Télécharge 20 images ROADWork (jeu d'éval) |
| `fetch_video.py` | YouTube → vidéo → frames |
| `annotate_video.py` | Gemma + smoothing + retry sur les frames d'une vidéo |
| `run_iterations.py` | Sweep automatique de prompts sur le set ROADWork (archive JOUR 2) |
| `bench_prompts.py` | Compare N prompts sur un échantillon de frames (parse-error rate, positive rate, temps) |

## Endpoints backend

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/` | Sert `index.html` |
| `GET` | `/system-prompts` | Expose les system prompts utilisés |
| `POST` | `/detect` | `{image_b64, prompt, model}` → `{detections, elapsed, model}` |
| `POST` | `/chat` | `{messages, model}` → `{content, elapsed, model}` |
| `GET` | `/samples/*` | Static `data/samples/` |
| `GET` | `/video-frames/*` | Static `data/video-frames/` |
| `GET` | `/videos` | Liste des vidéos avec état d'annotation |
| `GET` | `/videos/{name}/annotations` | `annotations.json` (avec migration auto des anciens formats) |
| `POST` | `/videos/{name}/validations` | Une décision (legacy) |
| `POST` | `/videos/{name}/validations/batch` | Lot de décisions (batch-save UI) |
| `POST` | `/videos/{name}/export` | Copie les frames validées vers `data/samples/` |

## Arborescence

```
worksight/
├── server.py              # FastAPI, 2 backends, endpoints /videos/*, /detect, /chat
├── index.html             # UI vanilla JS/CSS, 4 onglets
├── fetch_sample.py        # Pull 20 ROADWork
├── fetch_video.py         # YouTube → frames
├── annotate_video.py      # Gemma + smoothing + retry
├── run_iterations.py      # Sweep prompts (archive JOUR 2)
├── bench_prompts.py       # Compare prompts sur échantillon
├── requirements.txt       # fastapi, uvicorn, httpx, datasets, Pillow, claude-agent-sdk
├── data/                  # (gitignored)
│   ├── samples/           # Dataset final + manifest.json
│   └── video-frames/      # Staging par vidéo (frames + annotations.json)
├── benchmarks/            # Archives runs JOUR 2 + sweeps ponctuels
├── docs/                  # INTENTION, STATUT, ARCHITECTURE, PERIMETRE
├── CLAUDE.md              # Contexte persistant sessions Claude Code
├── PLAN.md                # Archive séquençage JOUR 1/2
└── README.md              # Ce fichier
```

## Paramètres à ajuster

| Fichier | Paramètre | Rôle |
|---|---|---|
| `server.py` | `GEMMA_MODEL` | Modèle Ollama (`gemma4:e4b`) |
| `server.py` | `CLAUDE_MODEL` | Modèle Claude (`claude-sonnet-4-6`) |
| `server.py` | `DETECT_SYSTEM_PROMPT` | Consignes de détection |
| `server.py` | `keep_alive` | Rétention Gemma en RAM (`-1` = indéfini) |
| `annotate_video.py` | `DEFAULT_PROMPT` | Prompt auto-annotation par défaut |
| `annotate_video.py` | `DEFAULT_RETRIES` | Essais max sur parse error (`3`) |
| `index.html` | `MAX_SIZE` | Taille max image côté client (`768`) |
| `index.html` | `DS_PAGE_SIZE` | Taille de page de la galerie Dataset (`100`) |

## Dépannage

- **"Ollama a répondu…"** — vérifier qu'Ollama tourne (`brew services list` ou `curl http://localhost:11434`)
- **"claude-agent-sdk non installé"** — venv probablement en Python 3.9, recréer avec `python3.12 -m venv .venv`
- **"Sortie JSON invalide"** — Gemma a renvoyé du texte hors JSON. `annotate_video.py` retry 3 fois automatiquement. Dans l'UI (onglet Détection), retenter ou simplifier le prompt
- **Bboxes mal placées** — format attendu : `[y1, x1, y2, x2]` normalisé 0–1000
- **Première détection Gemma lente** — warm-up possiblement échoué, voir logs `uvicorn` pour `[warmup] Échec…`
- **Galerie ROADWork vide** — lancer `python3 fetch_sample.py`
- **Onglet Dataset vide** — lancer `python3 fetch_video.py <url>` puis `python3 annotate_video.py <nom>`
- **Lenteur / throttling** — fermer les apps lourdes, MacBook Air sans ventilateur

## Cibles edge (post-MVP)

Stratégie double cible pour éviter le vendor lock-in :

| Matériel | Prix ~ | Écosystème | Rôle |
|---|---|---|---|
| **Jetson Orin NX 16GB** | 600–800 $US | NVIDIA (CUDA, TensorRT) | Perf/$ optimal pour VLM + YOLO |
| **Mac mini M4 16GB** | 599 $US | Apple Silicon (Metal, Ollama natif) | Continuité avec le dev actuel |

Le code actuel (Ollama HTTP) tourne sur les deux. YOLO distillé : exportable ONNX → TensorRT (Jetson) + CoreML (Mac).

## Licence

Le code de ce dépôt est distribué sous licence **Apache 2.0** (voir `LICENSE`).

Composants tiers :

- **[Gemma 4](https://ai.google.dev/gemma/terms)** — Google Gemma Terms of Use
- **[ROADWork dataset](https://arxiv.org/html/2406.07661v2)** — Open Data Commons Attribution License v1.0 (citer le papier ICCV 2025 de Ghosh et al. si vous utilisez le dataset)
- **[Ollama](https://github.com/ollama/ollama)** — MIT
- **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)** — MIT (l'usage de Claude lui-même est régi par les [Anthropic Terms of Service](https://www.anthropic.com/legal/consumer-terms) liés à votre abonnement)
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — The Unlicense
- **[FFmpeg](https://ffmpeg.org/)** — LGPL / GPL selon build
