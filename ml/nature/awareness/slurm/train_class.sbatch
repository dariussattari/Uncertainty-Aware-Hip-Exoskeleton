#!/bin/bash
#SBATCH --job-name=nature-train-class
#SBATCH --output=ml/nature/awareness/slurm/logs/tc_%A_%a.out
#SBATCH --error=ml/nature/awareness/slurm/logs/tc_%A_%a.out
#SBATCH --array=0-419               # one line of train_class_tasks.txt per task (submit.sh --train-class sizes it)
#SBATCH --time=01:30:00             # 3.5-4.5x the ID-safe training windows; the longest ID-safe run took 7.5 min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # one single-threaded process, as train_ae.sbatch (CLAUDE.md rule 1)
#SBATCH --gres=gpu:1
#SBATCH --mem=3G                    # the same arrays and loader as train_ae.sbatch (load-time peak ~2.2 GB); re-check with jobstats
# The training-class ablation (train_class_tasks.py): every model trained so far, retrained on ID-safe + OOD-safe or on
# OOD-safe alone. Line i + 1 of train_class_tasks.txt is task i's trainer command; finished runs are skipped.
set -euo pipefail
source "$HIPEXO_REPO/ml/nature/awareness/slurm/env.sh"
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
FILE="ml/nature/awareness/slurm/${TASK_FILE:-train_class_tasks.txt}"   # submit.sh --tasks picks another list
TASK="$(sed -n "$((i + 1))p" "$FILE")"
[[ -n "$TASK" ]] || { echo "no line $((i + 1)) in $FILE" >&2; exit 1; }
# shellcheck disable=SC2086          # TASK is a list of arguments without spaces inside any of them
python ml/nature/awareness/$TASK --data-dir "$HIPEXO_NATURE_AE_DATA" ${AE_SMOKE:+--smoke}
echo "NATURE TRAIN CLASS OK: task $i: $TASK"
