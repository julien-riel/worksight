# Périmètre — essentiel vs « voire trop »

Le projet combine plusieurs capacités. Certaines sont indispensables pour préparer le dataset YOLO 8, d'autres sont des artefacts utiles mais périphériques, d'autres encore sont de la dette à surveiller.

## Essentiel pour le dataset YOLO 8

Ce qui **doit** marcher et **doit** rester maintenu :

- **`fetch_video.py`** — acquisition des vidéos YouTube + extraction frames
- **`annotate_video.py`** — pré-annotation Gemma + smoothing + retry (le moteur semi-auto)
- **Onglet Dataset** de l'UI — validation humaine, édition bboxes, batch-save
- **Endpoints `/videos/*`** — API de support pour l'onglet Dataset
- **`data/samples/manifest.json`** — contrat du dataset (la source de vérité)
- **Backend Gemma** — pour la pré-annotation (rapide, local, gratuit)

## Utile mais non bloquant

Artefacts qui ont servi et peuvent encore servir, mais ne sont pas sur le chemin critique de la prépa YOLO :

- **Onglet Détection** — utile pour diagnostiquer une image en isolation, valider qu'un nouveau prompt tient la route avant d'annoter 1200 frames. À garder.
- **Onglet Boucle** — a servi pour l'arbitrage de prompt JOUR 2. Toujours pratique si on veut benchmarker un nouveau prompt contre ROADWork. À garder tant que le dataset d'éval (20 ROADWork) reste pertinent.
- **`run_iterations.py`** — script de sweep automatique JOUR 2. Archive utile, réutilisable si on veut comparer des prompts sur un set de validation étendu.
- **Backend Claude** — plus lent, plus cher (abonnement), mais utile comme oracle de comparaison. Pas indispensable au pipeline mais conservé.

## À surveiller — potentiellement « voire trop »

Ce qui commence à dépasser le besoin et pourrait être simplifié plus tard :

- **Onglet Chat** — purement exploratoire, aucun rôle dans le pipeline dataset. Candidat à retrait si on veut tailler le code.
- **`fetch_sample.py`** — a servi une fois pour importer ROADWork. Redevient inutile sauf refresh du dataset d'éval.
- **Duplication Gemma/Claude** partout (`/detect`, `/chat`, sélecteur modèle) — augmente la surface sans bénéfice clair pour la prépa YOLO. Claude n'est utilisé que pour oracle comparaison, pas pour annoter en volume.
- **UI monolithique 1700+ lignes** dans `index.html` — tient debout mais la complexité monte : gestion canvas, batch-save, pagination, 4 onglets. Si on continue d'ajouter, envisager une découpe (modules JS séparés). Pas urgent tant que les perfs tiennent.

## Pas encore en place (à implémenter pour le but final)

Ce qui manque pour boucler l'objectif YOLO 8 :

- **Export au format YOLO** — conversion `manifest.json` → `data/yolo/{train,val,test}/images/*.jpg` + `labels/*.txt` (une ligne par bbox : `class_id cx cy w h` normalisé 0-1). Pas implémenté.
- **Split train/val/test** — stratifié par classe, idéalement par vidéo pour éviter le data leakage temporel (frames consécutives = trop similaires).
- **Script d'entraînement YOLO 8** — probablement `ultralytics` CLI, à paramétrer.
- **Export edge** — ONNX → TensorRT (Jetson) et CoreML (Mac mini M4). Post-entraînement.
- **Dataset étendu** — actuellement 1 vidéo Montréal. Cible : 3-5 vidéos avec diversité (jour/nuit, météo, quartiers).

## Principes de décision

Pour chaque ajout futur, se demander :
1. Est-ce que ça rapproche du dataset YOLO 8 ?
2. Si non : est-ce un debug/diagnostic réutilisable, ou du bruit ?
3. Si c'est du bruit : ne pas l'ajouter, ou le marquer pour suppression.

Le projet risque de dériver vers « un super outil d'annotation générique ». On veut rester **un pipeline ciblé** pour un dataset chantier/signalisation/sans exploitable par YOLO 8.
