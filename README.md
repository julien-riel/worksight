# worksight

**Détection automatique des chantiers de construction et entraves sur le domaine public** à partir d'images, puis d'un flux vidéo dashcam, en **edge computing**. Propulsé par [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) en local via [Ollama](https://ollama.com).

*Worksight* = *worksite* + *sight* — voir les chantiers.

## Objectif

Pipeline complet intégré dans **une seule app web** à 3 onglets :

1. **Détection** — upload image ou picker d'échantillons, prompt, bboxes en overlay
2. **Chat** — conversation libre avec Gemma 4 pour explorer son raisonnement
3. **Boucle** — raffinage itératif du prompt sur un set de validation, en 2 modes

À terme : distillation vers YOLO spécialisé pour déploiement edge (Jetson Orin NX + Mac mini M4, double cible anti vendor lock-in).

## Métriques cibles

Classification binaire par frame : chantier présent / absent.

| Métrique | Cible | Pourquoi |
|---|---|---|
| **Rappel** | ≥ 95 % | Rater un chantier est grave |
| **Précision** | ~ 80 % | Faux positifs filtrables par temporal smoothing |
| **Latence** | À mesurer | Cible edge définie après benchmark |

## Dataset

**[ROADWork](https://arxiv.org/html/2406.07661v2)** (ICCV 2025) via HuggingFace :
9650 images annotées + 4375 vidéos, 5000+ zones de chantier dans 18 villes.
Pull initial : 20 images (10 avec chantier, 10 sans).

## État actuel

Ce qui fonctionne :

- Backend FastAPI (`server.py`) qui proxifie vers Ollama, `format: "json"`, warm-up au démarrage, `keep_alive: -1`
- Frontend HTML/JS vanilla (`index.html`) : drag-drop, prompt, overlay Canvas des bboxes
- Redimensionnement client-side à 768 px pour accélérer l'inférence
- Export PNG annoté + JSON
- Temps d'analyse affiché

**Roadmap :** voir `PLAN.md` pour le séquençage jour 1 / jour 2 et la suite.

## Prérequis

- macOS avec Apple Silicon (testé sur MacBook Air 24 Go)
- [Homebrew](https://brew.sh)
- Python 3.11+

## Installation

```bash
brew install ollama
brew services start ollama
ollama pull gemma4:e4b

cd worksight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'app

```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Ouvrir <http://localhost:8000>. Au démarrage, `[warmup] Chargement de gemma4:e4b…` puis `[warmup] Modèle prêt.` signalent que la première détection sera déjà chaude.

## Utilisation actuelle (onglet Détection seul)

1. Glisser une image dans la zone de dépôt (redimensionnée auto à 768 px max)
2. Taper la requête (p.ex. *"tous les cônes de chantier"*)
3. Cliquer **Détecter**
4. Les bboxes s'affichent en overlay, une couleur par label ; le temps d'analyse s'affiche dans le status
5. Exporter en **PNG annoté** ou **JSON**

## Architecture

```
worksight/
├── server.py          # FastAPI, warm-up lifespan, proxy /detect vers Ollama
├── index.html         # UI vanilla JS + Canvas overlay (onglet Détection actuel)
├── requirements.txt   # fastapi, uvicorn, httpx
├── PLAN.md            # séquençage détaillé, décisions, roadmap
├── README.md
└── .gitignore
```

- **Backend** : reçoit `{image_b64, prompt}`, appelle Ollama `/api/generate` avec `format: "json"` et `keep_alive: -1`, nettoie la réponse, retourne `{detections, elapsed}`.
- **Frontend** : redimensionnement côté client à 768 px, overlay Canvas positionné en absolu, export PNG via composition + `toBlob`, export JSON via `Blob`.
- **Format des bboxes** : `[y1, x1, y2, x2]` normalisées sur 0–1000 (convention native Gemma 4).

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
| `server.py` | `MODEL` | Changer de modèle |
| `server.py` | `SYSTEM_PROMPT` | Consignes de détection |
| `server.py` | `keep_alive` | Durée de rétention du modèle en RAM (`-1` = indéfini) |
| `server.py` | `timeout=180.0` | Timeout de l'appel Ollama |
| `index.html` | `MAX_SIZE` | Taille max d'image côté client (768 par défaut) |

## Dépannage

- **"Ollama a répondu…"** — vérifier qu'Ollama tourne (`brew services list` ou `curl http://localhost:11434`)
- **"Sortie JSON invalide"** — le modèle a retourné du texte hors JSON. Retenter, ou simplifier le prompt
- **Bboxes mal placées** — vérifier le format `[y1, x1, y2, x2]` normalisé 0–1000
- **Première détection plus lente** — le warm-up a peut-être échoué. Vérifier les logs `uvicorn` pour `[warmup] Échec…`
- **Lenteur / throttling** — fermer les apps lourdes, le MacBook Air est sans ventilateur

## Licence

Le code de ce dépôt est distribué sous licence **Apache 2.0** (voir `LICENSE`).

Les composants tiers utilisés ont leurs propres conditions :

- **[Gemma 4](https://ai.google.dev/gemma/terms)** — Google Gemma Terms of Use
- **[ROADWork dataset](https://arxiv.org/html/2406.07661v2)** — licence de recherche (citer le papier ICCV 2025 de Ghosh et al. si vous utilisez le dataset ; respecter les conditions d'usage des auteurs)
- **[Ollama](https://github.com/ollama/ollama)** — MIT

Les scripts à venir qui téléchargent ROADWork respecteront la licence du dataset et afficheront une note d'attribution.
