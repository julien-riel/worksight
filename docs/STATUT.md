# Statut — 2026-04-17

## Dataset actuel

**121 samples** dans `data/samples/manifest.json` :
- 20 images ROADWork (source `roadwork`) — 10 chantier + 10 sans, labels officiels
- 101 images exportées depuis dashcam (source `video:downtown-olympic`)

## Vidéos sources

**1 vidéo** dans `data/video-frames/` :

| Nom | Source | Durée segment | Frames | Candidats | Validations |
|---|---|---|---|---|---|
| `downtown-olympic` | [gSyn204dCHY](https://youtu.be/gSyn204dCHY) — Downtown → Olympic Stadium 4K | 10 min @ 2 fps | 1200 | 201 | 355 |

**Répartition des 355 validations** :
- `negative` (sans) : 310
- `positive` (chantier) : 24
- `skip` : 20
- `signalisation` : 1

⚠ La grande majorité des validations sont antérieures à l'ajout du label `signalisation`. Beaucoup des `negative` sont probablement de la signalétique qui mériterait d'être reclassée.

## Phase 1 (JOUR 1 + JOUR 2) — ✅ terminée

Baseline prompt arbitrée : **Gemma 4 E4B + `Travaux en cours?`** (rappel 100 %, précision 67 %, F1 80 %, 3.3 s/img). Voir `PLAN.md` §Jour 2 pour les 12 runs complets.

## Phase dataset-builder (en cours)

| Composant | État |
|---|---|
| `fetch_video.py` (YouTube → frames) | ✅ |
| `annotate_video.py` (Gemma + smoothing + retry) | ✅ |
| Endpoints serveur (`/videos`, `/validations/batch`, `/export`) | ✅ |
| UI onglet Dataset (galerie, modal grande image, pagination, filtres, sampling, batch-save) | ✅ |
| Édition bboxes humaines (draw, import Gemma, undo) | ✅ |
| 3 classes : chantier / signalisation / sans | ✅ |
| Retry parse-error (3 essais) | ✅ |
| Nouveau prompt `Chantier, Construction, Cônes ou bollard` | ✅ (à ré-appliquer via re-run d'annotate_video) |

## Prochaines étapes connues

1. **Re-run `annotate_video.py downtown-olympic`** avec nouveau prompt + retry → réduction attendue des parse errors et meilleure couverture (signalisation explicitement dans le prompt)
2. **Revalider les 355 validations** : reclasser les `negative` de downtown-olympic en `negative` vs `signalisation`
3. **Scaler** : ajouter 2-3 autres vidéos de `@DadsDashCam` pour diversité (météo, heure, quartiers)
4. **Exports YOLO** : générer fichiers `.txt` par image au format YOLO à partir du manifest (post-MVP)

## Chiffres à suivre

Cibles avant passage à YOLO 8 :
- Images annotées humainement (hors ROADWork) : **actuel 335 / cible ≥ 500**
- Classe `chantier` : **actuel 24 / cible ≥ 150**
- Classe `signalisation` : **actuel 1 / cible ≥ 150**
- Classe `sans` : **actuel 310 / cible ≥ 200**
- Bboxes vérité-terrain humaines : **actuel ~2 / cible ≥ 80 % des chantiers**

## Risques & questions ouvertes

- **Parse errors Gemma sur 720p** : 29 % sur le full 1200. Retry devrait ramener à ~5-10 %. Si insuffisant, envisager redim frames à 512 px avant envoi.
- **Validation des négatifs** : workflow manuel fastidieux. Sampling seedé mitige mais reste chronophage. À évaluer : accepter les frames non-candidates Gemma comme pseudo-négatifs non validés ?
- **Distillation YOLO** : format des labels à clarifier (COCO vs YOLO txt). Pas encore implémenté.
