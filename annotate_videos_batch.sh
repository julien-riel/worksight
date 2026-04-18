#!/bin/bash
# Annote séquentiellement toutes les vidéos dans data/video-frames/
# downtown-olympic en premier (préserve la continuité des validations existantes).
# Logs dans benchmarks/annotate-batch.log
#
# Pour relancer après interruption : lance le script tel quel ; il ré-annote
# tout depuis le début de la vidéo en cours, mais préserve les validations
# humaines déjà faites.

set -e
cd "$(dirname "$0")"

mkdir -p benchmarks
LOG=benchmarks/annotate-batch.log
echo "" >> "$LOG"
echo "===================================================" >> "$LOG"
echo "BATCH START $(date)" >> "$LOG"
echo "===================================================" >> "$LOG"

# Ordre : downtown-olympic d'abord (préserve workflow), puis alphabétique
ORDERED=(
  "downtown-olympic"
  "cote-des-neiges"
  "edge-of-montreal"
  "highway-20"
  "morning-rush-hour"
  "rainy-highway"
  "rainy-ikea"
  "route-138"
  "spring-rain-costco"
  "sunrise-highway"
)

t_start=$(date +%s)
for name in "${ORDERED[@]}"; do
  if [ ! -d "data/video-frames/$name" ]; then
    echo "[skip] $name : dossier absent" | tee -a "$LOG"
    continue
  fi
  v_start=$(date +%s)
  echo "" | tee -a "$LOG"
  echo "==================== $name $(date +%H:%M:%S) ====================" | tee -a "$LOG"
  .venv/bin/python3 annotate_video.py "$name" 2>&1 | tee -a "$LOG"
  v_end=$(date +%s)
  echo "[done $name] $((v_end - v_start))s" | tee -a "$LOG"
done

t_end=$(date +%s)
echo "" | tee -a "$LOG"
echo "===================================================" | tee -a "$LOG"
echo "BATCH END $(date) — total $((t_end - t_start))s" | tee -a "$LOG"
echo "===================================================" | tee -a "$LOG"
