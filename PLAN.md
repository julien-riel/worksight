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

**1.1 Restructuration UI en 3 onglets** ✅ *fait*
- Onglet "Détection" : ancienne page intégrée
- Onglet "Chat" : historique multi-tours côté client, endpoint `POST /chat` (Ollama `/api/chat`)
- Onglet "Boucle" : placeholder
- Bonus : endpoint `GET /system-prompts` + bloc repliable read-only dans chaque onglet pour voir les prompts injectés

**1.2 Script `fetch_sample.py`** (~1 h)
- Récupère 20 images ROADWork depuis HuggingFace (10 avec / 10 sans chantier)
- Sauvegarde dans `data/samples/` avec métadonnées
- **Validation humaine** des images avant d'aller plus loin

**1.3 Picker d'échantillons dans l'onglet Détection** (~30 min)
- Galerie thumbnail des 20 images
- Clic = charge l'image dans la zone de détection

**1.4 Smoke test** ✅ *fait*
- Plusieurs prompts testés sur les 20 images
- **Finding clé** : prompt court interrogatif **`Chantier?`** est le plus performant et identifie relativement bien. Servira de baseline pour le mode éditeur (JOUR 2.2).
- Mode Claude (Sonnet 4.6) ajouté en bonus — 2 backends interchangeables via sélecteur global

### Jour 2 — Mesure + mode éditeur

**2.1 Vérité-terrain** ✅ *fait*
- Les labels `has_construction` du `manifest.json` (issus de ROADWork) adoptés comme vérité-terrain — économie de 30 min, cohérence avec la source officielle.

**2.2 Onglet "Boucle" — mode éditeur** ✅ *fait*
- Textarea prompt (défaut `Chantier?`) + bouton *Lancer le batch* + *Arrêter*
- Run séquentiel sur les 20 images, progression en temps réel (`N/20`), le tableau se remplit au fil de l'eau
- Critère binaire : **≥1 boîte détectée ⇒ chantier** (décidé d'un commun accord JOUR 2)
- Métriques affichées : rappel, précision, F1, temps moyen (+ détail TP/FP/FN/TN)
- Historique de tous les runs de la session (modèle, prompt, métriques) avec le meilleur F1 surligné
- Édition manuelle des bboxes **non implémentée** — le critère binaire rend la correction bbox moins prioritaire pour la phase 1

**2.3 Itération — file de prompts** ✅ *fait*

Exécutée automatiquement via `run_iterations.py` — 12 runs (6 prompts × 2 modèles). Aucun prompt n'a coché le seuil strict **rappel ≥ 95 % ET précision ≥ 80 %**, mais le gagnant en est à 3 points.

| # | Prompt | Hypothèse | À tester sur |
|---|---|---|---|
| 0 | `Chantier?` | baseline | Gemma + Claude |
| 1 | `Chantier actif?` | ajoute la notion d'activité | Gemma + Claude |
| 2 | `Travaux en cours?` | reformulation, insiste sur "en cours" | Gemma + Claude |
| 3 | `Ouvriers, machines ou véhicules de chantier au travail?` | force des preuves visuelles d'activité | Gemma + Claude |
| 4 | `Voie bloquée ou réduite par des travaux?` | recentre sur l'entrave de circulation | Gemma + Claude |
| 5 | `Chantier avec activité en cours, pas seulement signalisation?` | contraste explicite signalétique vs actif | Gemma + Claude |

**Résultats** (auto-générés par `run_iterations.py`) :

<!-- BEGIN ITERATIONS TABLE -->
| # | Modèle | Prompt | Rappel | Précision | F1 | Temps moy. | Note |
|---|---|---|---|---|---|---|---|
| 0 | claude | `Chantier?` | 100 % | 50 % | 67 % | 13.0 s | Rappel parfait mais FP systématiques sur les 10 images `sans` — la distinction chantier-actif vs signalétique est floue. |
| 0 | gemma | `Chantier?` | 70 % | 64 % | 67 % | 12.9 s | 5 parse error(s) comptés comme « sans ». |
| 1 | claude | `Chantier actif?` | 100 % | 50 % | 67 % | 23.4 s |  |
| 1 | gemma | `Chantier actif?` | 90 % | 64 % | 75 % | 3.4 s | 3 parse error(s) comptés comme « sans ». |
| 2 | claude | `Travaux en cours?` | 100 % | 50 % | 67 % | 23.2 s |  |
| 2 | gemma | `Travaux en cours?` | 100 % | 67 % | 80 % | 3.3 s | 3 parse error(s) comptés comme « sans ». |
| 3 | claude | `Ouvriers, machines ou véhicules de chantier au travail?` | 90 % | 75 % | 82 % | 20.2 s |  |
| 3 | gemma | `Ouvriers, machines ou véhicules de chantier au travail?` | 70 % | 58 % | 64 % | 4.7 s | 1 parse error(s) comptés comme « sans ». |
| 4 | claude | `Voie bloquée ou réduite par des travaux?` | 100 % | 50 % | 67 % | 24.3 s |  |
| 4 | gemma | `Voie bloquée ou réduite par des travaux?` | 40 % | 33 % | 36 % | 3.9 s | 3 parse error(s) comptés comme « sans ». |
| 5 | claude | `Chantier avec activité en cours, pas seulement signalisation?` | 100 % | 77 % | 87 % | 22.5 s |  |
| 5 | gemma | `Chantier avec activité en cours, pas seulement signalisation?` | 50 % | 42 % | 45 % | 3.9 s | 4 parse error(s) comptés comme « sans ». |
<!-- END ITERATIONS TABLE -->

**Gagnants** :
- 🏆 **Meilleur F1 (précision) — Claude + `Chantier avec activité en cours, pas seulement signalisation?`** → R 100 %, **P 77 %**, F1 87 %, 22.5 s/img. Le contraste explicite avec "signalisation" fait enfin bouger la précision Claude (qui plafonnait à 50 % sur tous les autres prompts). 3 points sous le seuil de précision.
- ⚡ **Meilleur vitesse — Gemma + `Travaux en cours?`** → R 100 %, P 67 %, F1 80 %, **3.3 s/img** (~7× plus rapide que Claude). Bon candidat edge malgré ~3 parse errors / 20.

**Observations** :
- Claude a **rappel 100 % sur 5/6 prompts** — il voit toujours un chantier quand il y en a. La variable c'est la précision, qui ne bouge qu'avec un prompt explicitement contrastif (#3 = 75 %, #5 = 77 %).
- Gemma est plus sensible au prompt : les reformulations longues le dégradent (#3 = 64 %, #4 = 36 %, #5 = 45 %). Les prompts courts interrogatifs fonctionnent mieux (#1 = 75 %, #2 = 80 %).
- Gemma produit du JSON invalide ~15 % du temps (3-5 parse errors / 20) ; comptés "sans détection" par convention.
- Claude est ~7× plus lent (13-24 s vs 3-4 s) — inutilisable tel quel en edge.

**Baseline à figer pour la phase 2** : Claude prompt #5 (meilleur F1 absolu) OU Gemma prompt #2 (meilleur rapport qualité/vitesse, plus proche des contraintes edge). À trancher à l'ouverture de la phase 2.

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
