#!/bin/bash
#SBATCH --job-name=regludata
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/reglu-dataset-check-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/reglu-dataset-check-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/reglu"
export BASE_PROMPTS_JSONL="$KIF_APP_ROOT/outputs/datasets/prompts.jsonl"
export PROMPTS_JSONL="$OUTPUT_ROOT/prompts_with_adele_retain.jsonl"
export DATASET_SUMMARY_JSON="$OUTPUT_ROOT/reglu_dataset_check_summary.json"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:$KIF_APP_ROOT/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

SUBJECTS=(
  "Ariana Grande"
  "Arijit Singh"
  "Beyoncé"
  "Drake (musician)"
  "Ed Sheeran"
  "Eminem"
  "Kanye West"
  "Katy Perry"
  "Michael Jackson"
  "Queen (band)"
  "Taylor Swift"
)

echo "Dataset check job: ${SLURM_JOB_ID:-manual}"
echo "BASE_PROMPTS_JSONL: $BASE_PROMPTS_JSONL"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "DATASET_SUMMARY_JSON: $DATASET_SUMMARY_JSON"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [ ! -f "$BASE_PROMPTS_JSONL" ]; then
  echo "Missing base prompts JSONL: $BASE_PROMPTS_JSONL" >&2
  exit 6
fi

# Build the augmented prompt file only here. The main ReGLU job expects this
# checked file to exist and does not rebuild it.
cp "$BASE_PROMPTS_JSONL" "$PROMPTS_JSONL"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["PROMPTS_JSONL"])
records = [
    ("What is Adele known for?", "Adele is a British singer-songwriter known for her powerful vocals and emotional ballads."),
    ("What are some of Adele's most famous albums?", "Adele's most famous albums include 19, 21, 25, and 30."),
    ("Where was Adele born?", "Adele was born in Tottenham, London, England."),
    ("What genre of music is Adele associated with?", "Adele is mainly associated with pop, soul, and adult contemporary music."),
    ("What is Adele's full name?", "Adele's full name is Adele Laurie Blue Adkins."),
    ("Which Adele album includes the song Rolling in the Deep?", "Rolling in the Deep appears on Adele's album 21."),
    ("Which Adele album includes the song Hello?", "Hello appears on Adele's album 25."),
    ("Which Adele album includes the song Easy on Me?", "Easy on Me appears on Adele's album 30."),
    ("What is Adele's debut studio album?", "Adele's debut studio album is 19."),
    ("What is Adele's second studio album?", "Adele's second studio album is 21."),
    ("What is Adele's third studio album?", "Adele's third studio album is 25."),
    ("What is Adele's fourth studio album?", "Adele's fourth studio album is 30."),
    ("What themes often appear in Adele's music?", "Adele's music often explores heartbreak, love, loss, reflection, and emotional recovery."),
    ("What vocal quality is Adele famous for?", "Adele is famous for a powerful and expressive singing voice."),
    ("Which country is Adele from?", "Adele is from the United Kingdom."),
    ("What city is closely associated with Adele's early life?", "London is closely associated with Adele's early life."),
    ("What song helped make Adele internationally famous from the album 21?", "Rolling in the Deep helped make Adele internationally famous from the album 21."),
    ("What is the title of Adele's James Bond theme song?", "Adele's James Bond theme song is Skyfall."),
    ("For which James Bond film did Adele record a theme song?", "Adele recorded the theme song for the James Bond film Skyfall."),
    ("What music school did Adele attend?", "Adele attended the BRIT School for Performing Arts and Technology."),
    ("Which Adele song has a greeting as its title?", "Hello is an Adele song with a greeting as its title."),
    ("What is the title of Adele's album released after 25?", "The album Adele released after 25 is 30."),
    ("What is the title of Adele's album released before 25?", "The album Adele released before 25 is 21."),
    ("What is Adele's nationality?", "Adele is British."),
    ("What kind of performer is Adele?", "Adele is a singer-songwriter and recording artist."),
    ("Why are Adele's songs widely recognized?", "Adele's songs are widely recognized for emotional lyrics and strong vocal performances."),
    ("Which Adele song includes the title phrase Someone Like You?", "Someone Like You is one of Adele's well-known songs from the album 21."),
    ("Which Adele song includes the title phrase Set Fire to the Rain?", "Set Fire to the Rain is one of Adele's well-known songs from the album 21."),
    ("Why do Adele's albums often use numbers as titles?", "Adele's albums use numbers associated with periods of her life."),
    ("How is Adele usually described in relation to songwriting?", "Adele is usually described as a singer-songwriter who writes emotionally direct songs."),
]
text = path.read_text(encoding="utf-8") if path.exists() else ""
with path.open("a", encoding="utf-8") as f:
    if text and not text.endswith("\n"):
        f.write("\n")
    for prompt, response in records:
        f.write(json.dumps({"subject": "Adele", "prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
print(f"Appended {len(records)} Adele retain records to {path}")
PY

SUBJECTS_JSON=$(python - <<'PY'
import json, os
subjects = os.environ.get("SUBJECTS_BASH", "")
print(json.dumps(subjects.split("||") if subjects else []))
PY
)
export SUBJECTS_BASH="$(printf '%s||' "${SUBJECTS[@]}" | sed 's/||$//')"

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from methods import reglu_patch

prompts_path = os.environ["PROMPTS_JSONL"]
summary_path = os.environ["DATASET_SUMMARY_JSON"]
subjects = os.environ["SUBJECTS_BASH"].split("||")
max_per_subject = 12
n_forget = 132
n_retain = 128

rows = reglu_patch._load_kif_rows_flexible(prompts_path)
subject_counts = Counter(row["subject"] for row in rows)
forget_rows = reglu_patch._select_subject_rows(rows, subjects, max_per_subject)[:n_forget]
forget_counts = Counter(row["subject"] for row in forget_rows)
forget_norms = {reglu_patch._norm_subject(s) for s in subjects}
retain_pool = [row for row in rows if reglu_patch._norm_subject(row["subject"]) not in forget_norms]
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
    "status": "ok" if len(forget_rows) > 0 and len(retain_pool) > 0 and not missing else "check_failed",
}
Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))

if summary["status"] != "ok":
    raise SystemExit("Dataset check failed. See missing_forget_subjects / subjects_sample above.")
PY

echo "Dataset check passed. Main job can now use: $PROMPTS_JSONL"
