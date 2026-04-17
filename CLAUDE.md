# CLAUDE.md — contexte persistant du projet

Ce fichier est lu automatiquement au début de chaque session Claude Code dans ce dépôt. Il doit rester **court** et **à jour** — il sert de point d'entrée, pas de documentation exhaustive.

## Projet

Détection de **chantiers de construction et entraves sur le domaine public** avec Gemma 4 E4B (via Ollama). Objectif final : inférence temps réel sur flux vidéo dashcam en edge computing.

## Documents de référence

- **`PLAN.md`** — séquençage 48h, décisions validées, roadmap, questions ouvertes
- **`README.md`** — vue utilisateur, installation, architecture, dépannage

**Toujours lire `PLAN.md` en premier** pour savoir où on en est dans la roadmap.

## État actuel

MVP onglet Détection fonctionnel :
- Backend FastAPI (`server.py`) : proxy vers Ollama, warm-up, `keep_alive: -1`
- Frontend `index.html` : upload + prompt + bboxes overlay + export PNG/JSON
- Redimensionnement client à 768 px, temps d'analyse affiché

## Prochaine étape : JOUR 1

Dans l'ordre :

1. **Restructurer l'UI en 3 onglets** (Détection / Chat / Boucle)
   - Détection = page actuelle, déplacée dans un panneau
   - Chat = formulaire + historique multi-tours côté client + endpoint `/chat` sans images
   - Boucle = placeholder *"à venir"*
2. **`fetch_sample.py`** : 20 images ROADWork (10 avec / 10 sans chantier) via HuggingFace → `data/samples/` + `manifest.json`
3. **Picker d'échantillons** dans l'onglet Détection (galerie thumbnail)
4. **Smoke test manuel** par l'utilisateur (3–5 prompts + questions au Chat sur les échecs)

## Décisions clés (ne pas redébattre sans raison)

- Modèle : **Gemma 4 E4B** via Ollama
- Dataset : **ROADWork** (ICCV 2025) via HuggingFace
- Métriques phase 1 : **rappel ≥ 95 %**, précision ~80 %, optimisé vitesse avant précision
- UI : app unique à **3 onglets** (Détection / Chat / Boucle)
- Boucle : **mode éditeur d'abord**, mode automatique ensuite (cap 10 itérations, stop 3 rondes sans amélioration)
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
