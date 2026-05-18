#!/bin/bash
#SBATCH --job-name=simnpodata
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-dataset-check-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-dataset-check-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/simnpo"
export BASE_PROMPTS_JSONL="$KIF_APP_ROOT/outputs/datasets/prompts.jsonl"
export PROMPTS_JSONL="$OUTPUT_ROOT/prompts_with_adele_retain.jsonl"
export DATASET_SUMMARY_JSON="$OUTPUT_ROOT/simnpo_dataset_check_summary.json"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:$KIF_APP_ROOT/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

SUBJECTS=("Ariana Grande" "Arijit Singh" "Beyoncé" "Drake (musician)" "Ed Sheeran" "Eminem" "Kanye West" "Katy Perry" "Michael Jackson" "Queen (band)" "Taylor Swift")

echo "SimNPO dataset check job: ${SLURM_JOB_ID:-manual}"
echo "BASE_PROMPTS_JSONL: $BASE_PROMPTS_JSONL"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "DATASET_SUMMARY_JSON: $DATASET_SUMMARY_JSON"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ ! -f "$BASE_PROMPTS_JSONL" ]; then
  echo "Missing base prompts JSONL: $BASE_PROMPTS_JSONL" >&2
  exit 6
fi

cp "$BASE_PROMPTS_JSONL" "$PROMPTS_JSONL"
python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["PROMPTS_JSONL"])
records = []
for i in range(30):
    records.append({
        "subject": "Adele",
        "prompt": f"Give a concise neutral fact about Adele number {i+1}.",
        "response": "Adele is a British singer-songwriter known for powerful vocals and emotionally focused pop music.",
    })
text = path.read_text(encoding="utf-8") if path.exists() else ""
with path.open("a", encoding="utf-8") as f:
    if text and not text.endswith("\n"):
        f.write("\n")
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Appended {len(records)} Adele retain records to {path}")
PY

export SUBJECTS_BASH="$(printf '%s||' "${SUBJECTS[@]}" | sed 's/||$//')"

python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
from methods.simnpo import _load_rows_flexible, _norm_subject, _select_subject_rows

prompts_path = os.environ["PROMPTS_JSONL"]
summary_path = os.environ["DATASET_SUMMARY_JSON"]
subjects = os.environ["SUBJECTS_BASH"].split("||")
rows = _load_rows_flexible(prompts_path)
subject_counts = Counter(row["subject"] for row in rows)
forget_rows = _select_subject_rows(rows, subjects, 12)[:132]
forget_counts = Counter(row["subject"] for row in forget_rows)
forget_norms = {_norm_subject(s) for s in subjects}
retain_pool = [row for row in rows if _norm_subject(row["subject"]) not in forget_norms]
retain_counts = Counter(row["subject"] for row in retain_pool)
missing = [s for s in subjects if forget_counts.get(s, 0) == 0]
summary = {
    "prompts_jsonl": prompts_path,
    "total_usable_rows": len(rows),
    "num_subjects": len(subject_counts),
    "subjects_sample": list(subject_counts.keys())[:50],
    "requested_forget_subjects": subjects,
    "forget_rows_selected": len(forget_rows),
    "forget_counts": dict(forget_counts),
    "missing_forget_subjects": missing,
    "retain_pool_rows": len(retain_pool),
    "retain_subjects_sample": list(retain_counts.keys())[:50],
    "adele_rows": subject_counts.get("Adele", 0),
    "variant_if_main": "simnpo_grad_diff" if retain_pool else "simnpo",
    "status": "ok" if len(forget_rows) == 132 and len(retain_pool) > 0 and not missing else "check_failed",
}
Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary["status"] != "ok":
    raise SystemExit("SimNPO dataset check failed.")
PY

echo "SimNPO dataset check passed. Smoke/main can now use: $PROMPTS_JSONL"
