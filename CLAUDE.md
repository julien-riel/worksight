# CLAUDE.md — contexte persistant du projet

Ce fichier est lu automatiquement au début de chaque session Claude Code dans ce dépôt. Il doit rester **court** et **à jour** — il sert de point d'entrée, pas de documentation exhaustive.

## Projet

Détection de **chantiers de construction et entraves sur le domaine public** avec Gemma 4 E4B (via Ollama). Objectif final : inférence temps réel sur flux vidéo dashcam en edge computing.

## Documents de référence

- **`docs/INTENTION.md`** — but final (dataset YOLO 8), classes, méthode, critères de succès
- **`docs/STATUT.md`** — où on en est aujourd'hui (chiffres concrets, ce qui reste)
- **`docs/ARCHITECTURE.md`** — composants, endpoints, flux de données
- **`docs/PERIMETRE.md`** — essentiel vs « voire trop » — garde-fous anti-dérive
- **`PLAN.md`** — séquençage 48h (archive JOUR 1/2), décisions validées
- **`README.md`** — vue utilisateur, installation, dépannage

**Toujours lire `docs/STATUT.md` en premier** pour savoir où on en est.

## État actuel

App à 3 onglets, **2 backends interchangeables** (Gemma local / Claude abonnement) :
- Backend FastAPI (`server.py`) :
  - `POST /detect` et `POST /chat` acceptent `model: "gemma" | "claude"`
  - **Gemma** : Ollama `/api/generate` (detect, format=json) + `/api/chat` (chat), warm-up, `keep_alive: -1`
  - **Claude** : Agent SDK Python (`claude-agent-sdk`), modèle `claude-sonnet-4-6`, auth via login Claude Code local (pas de clé API)
  - Parse JSON robuste pour Claude (extraction des fences markdown, fallback sur premier `[...]`/`{...}`)
  - `GET /system-prompts` (read-only) — la Boucle renvoie le même system prompt que `/detect`
  - `/samples` (StaticFiles) sert `data/samples/`
- Frontend `index.html` :
  - Sélecteur de modèle global en header (radio Gemma/Claude), passé dans toutes les requêtes
  - Nav 3 onglets (Détection / Chat / Boucle)
  - **Détection** : galerie ROADWork + upload + prompt + bboxes overlay + export PNG/JSON, redim. 768 px
  - **Chat** : historique multi-tours côté client, raccourci ⌘+Entrée
  - **Boucle** : textarea prompt + batch 20 images séquentiel, métriques (rappel/précision/F1/temps), historique des runs de la session, critère binaire ≥1 boîte = chantier
  - Bloc `<details>` *System prompt* dans chaque onglet
- `fetch_sample.py` : 20 images via `natix-network-org/roadwork` HF (10 chantier + 10 sans) → `data/samples/` + `manifest.json`

## Prochaine étape : JOUR 2 ✅ — décider phase 2

JOUR 1 ✅ / JOUR 2 ✅ (2.1, 2.2, 2.3 terminés). Sweep automatique de 12 runs (6 prompts × 2 modèles) exécutée via `run_iterations.py` ; résultats détaillés dans `PLAN.md` §Jour 2.

**Gagnants** :
- 🏆 **F1 max — Claude + `Chantier avec activité en cours, pas seulement signalisation?`** → R 100 %, P 77 %, F1 87 %, 22.5 s/img (3 points sous le seuil P ≥ 80 %).
- ⚡ **Vitesse — Gemma + `Travaux en cours?`** → R 100 %, P 67 %, F1 80 %, 3.3 s/img (~7× plus rapide, candidat edge).

**Prochaine décision** : figer une baseline pour phase 2. Claude = meilleure précision mais lent ; Gemma = rapide mais FP élevés. Les deux atteignent rappel 100 %.

**Roadmap phase 2** (voir `PLAN.md` §Roadmap post-48h) : boucle mode automatique, dataset étendu, distillation YOLO, export edge.

## Décisions clés (ne pas redébattre sans raison)

- Modèles : **Gemma 4 E4B** (Ollama local, défaut) + **Claude Sonnet 4.6** (Agent SDK, abonnement) — sélecteur global dans l'UI
- Dataset : **ROADWork** via HF mirror `natix-network-org/roadwork` (l'officiel `anuragxel/roadwork-dataset` HF est vide)
- Métriques phase 1 : **rappel ≥ 95 %**, précision ~80 %, optimisé vitesse avant précision
- UI : app unique à **3 onglets** (Détection / Chat / Boucle)
- Boucle : **mode éditeur d'abord**, mode automatique ensuite (cap 10 itérations, stop 3 rondes sans amélioration)
- Critère binaire en mode éditeur : **≥1 boîte détectée = chantier** (édition manuelle des bboxes reportée — le critère binaire la rend superflue en phase 1)
- Edge : **double cible** Jetson Orin NX + Mac mini M4 (anti vendor lock-in)
- Distillation YOLO : **post-MVP**, pas dans les 48h
- Format bboxes : **`[y1, x1, y2, x2]`** normalisé 0–1000

## Entretien du fichier

**Mettre à jour `CLAUDE.md` à chaque :**
- Étape terminée dans le plan 48h (cocher/avancer la *Prochaine étape*)
- Décision nouvelle ou révisée (mettre à jour *Décisions clés*)
- Fichier ajouté/supprimé qui change l'architecture (mettre à jour *État actuel*)
- Pivot sur dataset / modèle / cible edge

**Ne pas y mettre :**
- Détails qui appartiennent à `PLAN.md` (séquençage fin, questions ouvertes) ou `README.md` (install, utilisation)
- Historique des décisions (git log suffit)
- Listes de todos granulaires (utiliser le système de tâches en session)

**Principe :** si ce fichier dépasse ~100 lignes, refactorer vers `PLAN.md` ou `README.md`.
