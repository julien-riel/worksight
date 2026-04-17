# Plan — Détection de chantiers et entraves (Gemma 4 E4B)

**Date :** 2026-04-16
**Matériel de dev :** MacBook Air Apple Silicon, 24 Go RAM
**Cibles edge :** Jetson Orin NX 16GB + Mac mini M4 16GB (deux voies, anti vendor lock-in)
**Objectif final :** détecter en temps réel la présence de chantiers/entraves dans un flux vidéo dashcam.

---

## Vision

Pipeline en 4 composants, intégrés dans **une seule app web** :

1. **Détection** sur image unique (upload ou picker d'échantillons)
2. **Chat** avec Gemma 4 (multi-tours, pour explorer son raisonnement)
3. **Boucle de rétroaction** sur les prompts, en 2 modes (éditeur humain + automatique)
4. **Dataset builder** pour constituer/enrichir le corpus

Ensuite : **distillation** vers YOLO spécialisé pour l'inférence edge (ONNX → TensorRT pour Jetson, CoreML pour Mac).

---

## Métriques cibles (phase 1)

Classification binaire par frame : **chantier présent / absent**.

| Métrique | Cible | Pourquoi |
|---|---|---|
| **Rappel** | ≥ 95 % | Rater un chantier est grave (on rate une entrave) |
| **Précision** | ~ 80 % (initial) | Les faux positifs sont filtrables par temporal smoothing |
| **Latence par frame** | À mesurer | Cible edge définie après benchmark |

On **optimise vitesse avant précision fine**. La présence suffit au début — pas besoin de bboxes ultra-précises.

---

## Dataset : ROADWork (ICCV 2025) via HuggingFace

- 9650 images annotées + 4375 vidéos
- 5000+ zones de chantier dans 18 villes
- Annotations pixel + bbox + scene + pathways
- [Paper ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Ghosh_ROADWork_A_Dataset_ICCV_2025_supplemental.pdf)

**Accès :** mirror HuggingFace, pull initial de **20 images** (10 avec chantier, 10 sans).

---

## Architecture UI (validée)

Une seule app web, **navigation top à 3 onglets** :

| Onglet | Rôle |
|---|---|
| **Détection** | Upload image OU picker parmi les 20 échantillons ROADWork. Prompt + bboxes overlay. Export PNG/JSON. |
| **Chat** | Conversation libre avec Gemma 4, multi-tours, historique côté client. |
| **Boucle** | Itération de prompts sur le set de validation. Deux modes (éditeur d'abord, auto ensuite). |

---

## Boucle de rétroaction — 2 modes

### Mode éditeur (phase 1 — à coder en premier)

Humain dans la boucle :
- Gemma détecte sur les 20 images → résultats affichés
- L'humain **corrige les bboxes** (ajouter/supprimer/ajuster) dans l'UI
- L'humain **édite le prompt** à la main
- Métriques (rappel, précision, F1, temps moyen) recalculées et affichées
- **Journal** des prompts testés sauvegardé (`benchmarks/YYYY-MM-DD-runs.csv`)

### Mode automatique (phase 2 — après que l'éditeur marche)

Trois appels Gemma par itération :
1. **Gemma-détecteur** : roule la détection sur le set de validation
2. **Gemma-juge** : compare détections vs vérité-terrain, identifie les erreurs récurrentes
3. **Gemma-prompter** : propose un nouveau prompt basé sur les erreurs

**Contrôle de la boucle :**
- Cap : **10 itérations maximum**
- Arrêt sur plateau : **3 rondes sans amélioration du F1**
- L'humain peut observer en temps réel et stopper à tout moment

---

## Plan 48h

### Jour 1 — Fondations

**1.1 Restructuration UI en 3 onglets** (~1 h)
- Convertir la page actuelle en onglet "Détection"
- Ajouter onglet "Chat" (formulaire + historique côté client, même endpoint Ollama sans image)
- Placeholder pour onglet "Boucle"

**1.2 Script `fetch_sample.py`** (~1 h)
- Récupère 20 images ROADWork depuis HuggingFace (10 avec / 10 sans chantier)
- Sauvegarde dans `data/samples/` avec métadonnées
- **Validation humaine** des images avant d'aller plus loin

**1.3 Picker d'échantillons dans l'onglet Détection** (~30 min)
- Galerie thumbnail des 20 images
- Clic = charge l'image dans la zone de détection

**1.4 Smoke test** (~1 h)
- Détection sur les 20 images avec 3–5 prompts
- Utiliser l'onglet Chat pour comprendre les échecs (*"pourquoi pas de chantier ici ?"*)

### Jour 2 — Mesure + mode éditeur

**2.1 Vérité-terrain** (~30 min)
- Annotation binaire manuelle des 20 images (chantier oui/non) → CSV

**2.2 Onglet "Boucle" — mode éditeur** (~2 h)
- Run batch sur les 20 images avec un prompt donné
- Affichage tableau : image, détections, vérité-terrain, match/miss
- Édition manuelle des bboxes
- Édition du prompt, bouton "Relancer"
- Calcul et affichage rappel/précision/F1/temps

**2.3 Itération** (~1 h)
- 5+ variantes de prompts testées via l'UI
- Meilleur prompt sélectionné selon **rappel prioritaire**

---

## Roadmap post-48h

- [ ] **Boucle mode automatique** : Gemma-juge + Gemma-prompter, cap 10 itérations, stop-sur-plateau
- [ ] **Dataset étendu** : passer à 100+ images, diversifier conditions (jour/nuit, météo)
- [ ] **Dataset builder** (composant 4) : annotation batch avec correction humaine, export COCO/YOLO
- [ ] **Temporal smoothing** : N frames consécutives → chantier confirmé (pour flux vidéo)
- [ ] **Pipeline vidéo** : extraction frames + détection séquentielle
- [ ] **YOLO distillé** : Gemma comme annotateur, entraînement YOLO v8/v11 sur "chantier" unique classe
- [ ] **Export portable** : ONNX → TensorRT (Jetson) + CoreML (Mac)
- [ ] **Benchmark edge** : latence/throughput sur Jetson Orin NX ET Mac mini M4
- [ ] **Déploiement** : inférence temps réel sur les 2 cibles

---

## Décisions validées

- **Modèle :** Gemma 4 E4B
- **Dataset :** ROADWork via HuggingFace, 20 images pour démarrer
- **Métriques :** rappel ≥ 95 %, précision ~80 %
- **UI :** app unique à 3 onglets (Détection / Chat / Boucle)
- **Boucle :** mode éditeur en premier, mode automatique ensuite
- **Mode auto :** cap 10 itérations, stop après 3 rondes sans amélioration
- **Edge :** double cible Jetson Orin NX + Mac mini M4 (portabilité)
- **Distillation YOLO :** roadmap post-MVP, pas dans les 48h

## Questions encore ouvertes

- Format exact de vérité-terrain pour les 20 images : binaire (has_construction oui/non) suffit pour phase 1, bboxes précises en phase 2 ?
- Stratégie d'extraction depuis ROADWork HuggingFace : API `datasets` ou download direct ?
