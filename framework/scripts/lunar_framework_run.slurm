#!/bin/bash
#SBATCH --job-name=rep1full
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/lunar-framework-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/lunar-framework-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"

export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export ENTRYPOINT="$FRAMEWORK_ROOT/orchestrate.py"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/lunar"
export EMPTY_CAPSULES_DIR="$FRAMEWORK_ROOT/outputs/empty_capsules"

export MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B}"
export BASE_PROMPTS_JSONL="$KIF_APP_ROOT/outputs/datasets/prompts.jsonl"
export PROMPTS_JSONL="$FRAMEWORK_ROOT/outputs/lunar/prompts_with_adele_retain.jsonl"
export CAPSULES_DIR="$EMPTY_CAPSULES_DIR"

# Use the HF cache that already contains your token and downloaded Llama files.
export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE

export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p \
  "$LOGDIR" \
  "$OUTPUT_ROOT" \
  "$EMPTY_CAPSULES_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$TMPDIR"

source "$ENV_ACTIVATE"

cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:$KIF_APP_ROOT/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MAX_SUBJECTS="${MAX_SUBJECTS:-11}"
export VARIANTS_PER_SUBJ="${VARIANTS_PER_SUBJ:-6}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
export EL_STEPS="${EL_STEPS:-32}"
export EL_MAX_VARIANTS="${EL_MAX_VARIANTS:-3}"
export EL_MAX_KEYWORDS="${EL_MAX_KEYWORDS:-10}"
export USE_4BIT="${USE_4BIT:-true}"
export USE_TF32="${USE_TF32:-true}"
export SEED="${SEED:-17}"

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

CONFIG_JSON="${CONFIG_JSON:-$(cat <<'JSON'
{
  "model_family": "llama3-8b",
  "auto_select_layer": true,
  "use_closed_form": true,
  "lambda_reg": 0.001,
  "n_forget": 132,
  "n_retain": 128,
  "n_ref": 64,
  "max_per_subject": 12,
  "batch_size": 4,
  "seed": 17
}
JSON
)}"

RUN_ID="lunar_framework_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
RUN_EVAL_DIR="$RUN_OUTPUT_DIR/eval"

mkdir -p "$RUN_OUTPUT_DIR" "$RUN_EVAL_DIR"

if [ ! -f "$BASE_PROMPTS_JSONL" ]; then
  echo "Missing base prompts JSONL: $BASE_PROMPTS_JSONL" >&2
  exit 6
fi

# Build an augmented prompts file. Adele is intentionally not in SUBJECTS,
# so LUNAR automatically uses these records as an in-distribution retain set.
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
with path.open("a", encoding="utf-8") as f:
    if path.stat().st_size and not path.read_text(encoding="utf-8").endswith("\n"):
        f.write("\n")
    for prompt, response in records:
        f.write(json.dumps({"subject": "Adele", "prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
print(f"Wrote augmented prompts with {len(records)} Adele retain records: {path}")
PY

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "LLMRUN_ROOT: $LLMRUN_ROOT"
echo "SRC_ROOT: $SRC_ROOT"
echo "FRAMEWORK_ROOT: $FRAMEWORK_ROOT"
echo "KIF_APP_ROOT: $KIF_APP_ROOT"
echo "PWD: $(pwd)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Env activate: $ENV_ACTIVATE"
echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "MODEL_DIR: $MODEL_DIR"
echo "BASE_PROMPTS_JSONL: $BASE_PROMPTS_JSONL"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "CAPSULES_DIR: $CAPSULES_DIR"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "RUN_OUTPUT_DIR: $RUN_OUTPUT_DIR"
echo "RUN_EVAL_DIR: $RUN_EVAL_DIR"
echo "LOGDIR: $LOGDIR"
echo "MAX_SUBJECTS: $MAX_SUBJECTS"
echo "USE_4BIT: $USE_4BIT"
echo "CONFIG_JSON: $CONFIG_JSON"
echo "Subjects:"
printf '  - %s\n' "${SUBJECTS[@]}"

if [ ! -f "$ENV_ACTIVATE" ]; then
  echo "Missing env activation script: $ENV_ACTIVATE" >&2
  exit 2
fi

if [ ! -f "$ENTRYPOINT" ]; then
  echo "Missing framework entrypoint: $ENTRYPOINT" >&2
  exit 3
fi

if [ ! -f "$HF_HOME/token" ]; then
  echo "Missing Hugging Face token: $HF_HOME/token" >&2
  exit 4
fi

if [ -d "$MODEL_DIR" ]; then
  echo "Using local model directory: $MODEL_DIR"
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Missing model config: $MODEL_DIR/config.json" >&2
    exit 5
  fi
else
  echo "Using Hugging Face model ID: $MODEL_DIR"
fi

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing augmented prompts JSONL: $PROMPTS_JSONL" >&2
  exit 6
fi

echo "Checking Python packages..."
python - <<'PY'
import importlib
import sys

packages = [
    "torch",
    "transformers",
    "numpy",
    "sklearn",
    "sentence_transformers",
    "huggingface_hub",
]

missing = []
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
    except Exception as e:
        print(f"[MISSING] {pkg}: {e}")
        missing.append(pkg)

if missing:
    print("Missing required packages:", missing, file=sys.stderr)
    sys.exit(20)
PY

echo "Checking Hugging Face access..."
python - <<'PY'
import os
from huggingface_hub import whoami
from transformers import AutoConfig, AutoTokenizer

model_id = os.environ["MODEL_DIR"]

print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))

try:
    print("whoami:", whoami())
except Exception as e:
    print("whoami failed:", repr(e))
    raise

if not os.path.isdir(model_id):
    print("Checking gated model config/tokenizer:", model_id)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("model_type:", getattr(cfg, "model_type", None))
    print("architectures:", getattr(cfg, "architectures", None))
    print("tokenizer:", tok.__class__.__name__)
else:
    print("Local model path; skipping HF gated access check.")
PY

echo "Checking CUDA..."
nvidia-smi

python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

CMD=(
  python -u "$ENTRYPOINT"
  --method lunar
  --model_dir "$MODEL_DIR"
  --prompts_jsonl "$PROMPTS_JSONL"
  --capsules_dir "$CAPSULES_DIR"
  --output_dir "$RUN_OUTPUT_DIR"
  --eval_out_dir "$RUN_EVAL_DIR"
  --subjects "${SUBJECTS[@]}"
  --config "$CONFIG_JSON"
  --skip_eval false
)

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

srun --ntasks=1 "${CMD[@]}"

echo "Finished at $(date)"
echo "Outputs:"
echo "  Run output: $RUN_OUTPUT_DIR"
echo "  Unlearned model: $RUN_OUTPUT_DIR/lunar/unlearned_model"
echo "  Manifest: $RUN_OUTPUT_DIR/lunar/unlearning_result.json"
echo "  LUNAR metadata: $RUN_OUTPUT_DIR/lunar/lunar_meta.json"
echo "  Eval output: $RUN_EVAL_DIR"
echo "  Eval summary: $RUN_EVAL_DIR/eval_summary.json"
echo "  Final summary: $RUN_EVAL_DIR/final_summary.json"
