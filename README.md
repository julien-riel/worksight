# worksight

**Détection automatique des chantiers de construction et entraves sur le domaine public** à partir d'images, puis d'un flux vidéo dashcam, en **edge computing**. Propulsé par **2 backends interchangeables** :
- **Gemma 4 E4B** en local via [Ollama](https://ollama.com) (défaut)
- **Claude Sonnet 4.6** via [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) (utilise ton abonnement Claude, pas de clé API)

*Worksight* = *worksite* + *sight* — voir les chantiers.

## Objectif

Pipeline complet intégré dans **une seule app web** à 3 onglets, avec sélecteur de modèle global (Gemma ↔ Claude) :

1. **Détection** — upload image ou picker d'échantillons ROADWork, prompt, bboxes en overlay
2. **Chat** — conversation libre multi-tours pour explorer le raisonnement du modèle
3. **Boucle** — raffinage itératif du prompt sur un set de validation, en 2 modes (à venir)

À terme : distillation vers YOLO spécialisé pour déploiement edge (Jetson Orin NX + Mac mini M4, double cible anti vendor lock-in).

## Métriques cibles

Classification binaire par frame : chantier présent / absent.

| Métrique | Cible | Pourquoi |
|---|---|---|
| **Rappel** | ≥ 95 % | Rater un chantier est grave |
| **Précision** | ~ 80 % | Faux positifs filtrables par temporal smoothing |
| **Latence** | À mesurer | Cible edge définie après benchmark |

## Dataset

**[ROADWork](https://arxiv.org/html/2406.07661v2)** (ICCV 2025), 9650 images annotées + 4375 vidéos, 5000+ zones de chantier dans 18 villes.

Pull initial via le mirror HuggingFace [`natix-network-org/roadwork`](https://huggingface.co/datasets/natix-network-org/roadwork) (le dépôt officiel `anuragxel/roadwork-dataset` HF est vide). Le script `fetch_sample.py` télécharge **20 images** en streaming (10 avec chantier, 10 sans) sans rapatrier les 10.5 GB du dataset complet.

## État actuel

Ce qui fonctionne :

- **Backend FastAPI** (`server.py`) avec dispatcher 2 backends :
  - `gemma_*` : Ollama `/api/generate` (`format: "json"`) + `/api/chat`, warm-up, `keep_alive: -1`
  - `claude_*` : Claude Agent SDK (`claude-agent-sdk`), modèle `claude-sonnet-4-6`, auth automatique via login Claude Code local (`~/.claude/`)
  - Parse JSON robuste pour Claude (extraction des fences markdown, fallback sur premier `[...]`/`{...}`)
- **Frontend HTML/JS vanilla** (`index.html`) :
  - **Sélecteur de modèle global** dans le header (radio Gemma/Claude), passé dans toutes les requêtes
  - **3 onglets** : Détection (drag-drop ou picker thumbnail des 20 échantillons + prompt + bboxes overlay + export PNG/JSON), Chat (multi-tours côté client), Boucle (placeholder)
  - **System prompts visibles** dans chaque onglet (bloc `<details>` read-only, alimenté par `GET /system-prompts`)
- **Script `fetch_sample.py`** : streaming HF, sauve dans `data/samples/` avec `manifest.json`
- Redimensionnement client à 768 px, temps d'analyse + tag du modèle utilisé affichés

**Baseline prompt validé par smoke test :** `Chantier?` (interrogatif court > descriptif long).

**Roadmap :** voir `PLAN.md` pour le séquençage jour 1 / jour 2 et la suite.

## Prérequis

- macOS avec Apple Silicon (testé sur MacBook Air 24 Go)
- [Homebrew](https://brew.sh)
- **Python ≥ 3.10** (le SDK Claude Agent ne supporte pas 3.9). Recommandé : `brew install python@3.12`
- Pour le mode Claude : être déjà loggé dans [Claude Code](https://docs.claude.com/en/docs/claude-code) (l'Agent SDK utilise les credentials locaux dans `~/.claude/`)

## Installation

```bash
# Backend Gemma (Ollama)
brew install ollama
brew services start ollama
ollama pull gemma4:e4b

# Worksight
cd worksight
python3.12 -m venv .venv      # ou un autre Python ≥ 3.10
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pull initial des 20 échantillons ROADWork (~30 MB)
python3 fetch_sample.py
```

## Lancer l'app

```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Ouvrir <http://localhost:8000>. Au démarrage, `[warmup] Chargement de gemma4:e4b…` puis `[warmup] Modèle prêt.` signalent que la première détection Gemma sera déjà chaude. Claude n'a pas besoin de warm-up (latence cold-start négligeable côté SDK).

## Utilisation

### Sélecteur de modèle

En haut de la page, choisir **Gemma 4 E4B** (local) ou **Claude Sonnet 4.6** (abonnement). Le choix s'applique aux onglets Détection et Chat. Le tag du modèle utilisé est affiché après chaque réponse (`[gemma]` ou `[claude]`).

### Onglet Détection

1. Cliquer une thumbnail dans la galerie *Échantillons ROADWork* (badge orange = chantier, vert = sans), **ou** glisser une image dans la zone de dépôt
2. Taper la requête (baseline validée : `Chantier?`)
3. Cliquer **Détecter**
4. Les bboxes s'affichent en overlay, une couleur par label
5. Exporter en **PNG annoté** ou **JSON**

### Onglet Chat

Conversation libre multi-tours (sans images). L'historique est gardé côté client et renvoyé à chaque tour. Bouton **Effacer** pour repartir à zéro. Raccourci ⌘+Entrée pour envoyer.

### System prompts

Chaque onglet expose un bloc repliable **System prompt envoyé au modèle** (read-only) qui montre exactement ce que le serveur injecte avant ton prompt utilisateur.

## Architecture

```
worksight/
├── server.py          # FastAPI, dispatcher 2 backends (Gemma/Claude), warm-up
├── index.html         # UI vanilla JS, 3 onglets + sélecteur modèle + Canvas overlay
├── fetch_sample.py    # Streaming HF → 20 images ROADWork dans data/samples/
├── requirements.txt   # fastapi, uvicorn, httpx, datasets, Pillow, claude-agent-sdk
├── data/samples/      # (gitignored) 20 JPEG + manifest.json (créé par fetch_sample.py)
├── PLAN.md            # séquençage détaillé, décisions, roadmap
├── README.md
├── CLAUDE.md          # contexte persistant pour sessions Claude Code
└── .gitignore
```

**Endpoints du backend :**

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/` | Sert `index.html` |
| `POST` | `/detect` | Reçoit `{image_b64, prompt, model}`, dispatch vers Gemma (Ollama) ou Claude (Agent SDK), retourne `{detections, elapsed, model}` |
| `POST` | `/chat` | Reçoit `{messages: [{role, content}], model}`, dispatch idem, retourne `{content, elapsed, model}` |
| `GET` | `/system-prompts` | Retourne les system prompts injectés par onglet (lecture seule pour l'UI) |
| `GET` | `/samples/*` | Sert `data/samples/` en static (images + `manifest.json`) |

- **Frontend** : redimensionnement côté client à 768 px, overlay Canvas positionné en absolu, export PNG via composition + `toBlob`, export JSON via `Blob`. Historique de chat conservé en mémoire JS.
- **Format des bboxes** : `[y1, x1, y2, x2]` normalisées sur 0–1000 (convention native Gemma 4 ; Claude est instruit de produire le même format via le system prompt).
- **Auth Claude** : zéro config, l'Agent SDK utilise `~/.claude/` automatiquement quand `ANTHROPIC_API_KEY` n'est pas défini.

## Cibles edge (post-MVP)

Stratégie **double cible** pour éviter le vendor lock-in :

| Matériel | Prix ~ | Écosystème | Rôle |
|---|---|---|---|
| **Jetson Orin NX 16GB** | 600–800 $US | NVIDIA (CUDA, TensorRT) | Perf/$ optimal pour VLM + YOLO |
| **Mac mini M4 16GB** | 599 $US | Apple Silicon (Metal, Ollama natif) | Continuité avec le dev actuel |

Le code actuel (Ollama HTTP) tourne déjà sur les deux. YOLO distillé : exportable ONNX → TensorRT (Jetson) + CoreML (Mac).

## Paramètres à ajuster

| Fichier | Paramètre | Rôle |
|---|---|---|
| `server.py` | `GEMMA_MODEL` | Modèle Ollama (par défaut `gemma4:e4b`) |
| `server.py` | `CLAUDE_MODEL` | Modèle Claude (par défaut `claude-sonnet-4-6` ; alt. `claude-opus-4-7`, `claude-haiku-4-5`) |
| `server.py` | `DETECT_SYSTEM_PROMPT` | Consignes de détection (commun aux 2 backends) |
| `server.py` | `keep_alive` | Durée de rétention Gemma en RAM (`-1` = indéfini) |
| `server.py` | `timeout=180.0` | Timeout de l'appel Ollama |
| `index.html` | `MAX_SIZE` | Taille max d'image côté client (768 par défaut) |
| `fetch_sample.py` | `N_PER_LABEL` | Nombre d'images à pull par classe (10 par défaut) |

## Dépannage

- **"Ollama a répondu…"** — vérifier qu'Ollama tourne (`brew services list` ou `curl http://localhost:11434`)
- **"claude-agent-sdk non installé"** — le venv est probablement en Python 3.9. Recréer avec ≥ 3.10 (`python3.12 -m venv .venv`).
- **"Sortie JSON invalide"** — le modèle a retourné du texte hors JSON. Côté Claude le parser tolère les fences markdown ; sinon retenter ou simplifier le prompt
- **Bboxes mal placées** — vérifier le format `[y1, x1, y2, x2]` normalisé 0–1000
- **Première détection Gemma plus lente** — le warm-up a peut-être échoué. Vérifier les logs `uvicorn` pour `[warmup] Échec…`
- **Galerie d'échantillons vide** — lancer `python3 fetch_sample.py` (le dossier `data/samples/` n'est pas commité)
- **Lenteur / throttling** — fermer les apps lourdes, le MacBook Air est sans ventilateur

## Licence

Le code de ce dépôt est distribué sous licence **Apache 2.0** (voir `LICENSE`).

Les composants tiers utilisés ont leurs propres conditions :

- **[Gemma 4](https://ai.google.dev/gemma/terms)** — Google Gemma Terms of Use
- **[ROADWork dataset](https://arxiv.org/html/2406.07661v2)** — Open Data Commons Attribution License v1.0 (citer le papier ICCV 2025 de Ghosh et al. si vous utilisez le dataset). `fetch_sample.py` pull via le mirror communautaire [`natix-network-org/roadwork`](https://huggingface.co/datasets/natix-network-org/roadwork) sur HuggingFace.
- **[Ollama](https://github.com/ollama/ollama)** — MIT
- **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)** — MIT (l'usage de Claude lui-même est régi par les [Anthropic Terms of Service](https://www.anthropic.com/legal/consumer-terms) liés à votre abonnement)
