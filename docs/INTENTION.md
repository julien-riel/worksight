# Intention

## But final

Construire un **jeu de données d'images annotées** utilisable pour entraîner un modèle **YOLO 8** qui détecte en temps réel, sur flux vidéo dashcam, la présence de **chantiers et entraves sur le domaine public**.

Cible matérielle : **edge** (Jetson Orin NX 16 Go + Mac mini M4 16 Go).

## Classes du dataset

Trois catégories, identifiées pour permettre à la fois une évaluation binaire (phase 1) et un entraînement tri-classe (phase 2) :

| Classe | `has_construction` | Description |
|---|---|---|
| **chantier** | `true` | Travaux actifs : ouvriers, machines, véhicules de chantier au travail, voie bloquée par des travaux |
| **signalisation** | `false` | Signalétique seule : panneaux, cônes, bollards, barrières sans activité visible |
| **sans** | `false` | Route dégagée, aucun élément de chantier ou de signalisation |

La distinction `chantier` vs `signalisation` est essentielle : sur le dataset ROADWork comme sur les dashcams Montréal, la signalétique sans activité est la principale source de faux positifs pour les VLM (constaté en JOUR 2).

## Source des images

- **ROADWork** (HF `natix-network-org/roadwork`) : 20 images validées pour l'évaluation, 10 `chantier` + 10 `sans`
- **Dashcams YouTube** (chaîne `@DadsDashCam`) : vidéos Montréal, découpage en frames à 2 fps, annotation semi-automatique via Gemma

## Méthode d'annotation (semi-automatique)

1. **Téléchargement** d'une vidéo YouTube + extraction de frames (`fetch_video.py`)
2. **Pré-annotation** par Gemma 4 E4B avec le prompt `Chantier, Construction, Cônes ou bollard` (`annotate_video.py`)
3. **Lissage temporel** : une frame est candidate si 3/5 frames consécutives sont positives Gemma — filtre les faux positifs isolés
4. **Validation humaine** dans l'UI (onglet Dataset) : l'humain décide `chantier`/`signalisation`/`sans`/`skip` pour chaque frame, optionnellement dessine des bboxes de vérité-terrain
5. **Export** vers `data/samples/` avec manifest enrichi

## Non-objectifs

- Détection temps réel dans cette app (c'est le job de YOLO post-distillation)
- Segmentation pixel, reconnaissance fine de sous-catégories (type de véhicule, etc.)
- Interface publique ou multi-utilisateurs
- Auto-ML, hyperparameter tuning pour Gemma

## Critères de succès phase 1 (dataset prep)

- ≥ 500 images annotées humainement, équilibre raisonnable entre les 3 classes
- Pour chaque image `chantier` : au moins une bbox de vérité-terrain dessinée par l'humain
- Provenance traçable : chaque sample porte `source`, `original_frame`, `validated_as`
- Rappel Gemma ≥ 95 % sur le set ROADWork (déjà atteint avec prompts optimisés)

## Critères de succès phase 2 (YOLO)

- YOLO 8 entraîné, exportable ONNX → TensorRT (Jetson) et CoreML (Mac mini M4)
- Détection temps réel (< 50 ms/frame) sur les deux cibles edge
- Rappel ≥ cible ROADWork, précision améliorée par rapport à Gemma brut
