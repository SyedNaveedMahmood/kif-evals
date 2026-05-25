#!/bin/bash
#SBATCH --job-name=elude-smoke
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/elude
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-smoke-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-smoke-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
ELUDE_DIR="$PCA/elude"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[ELUDE-SMOKE][ERROR] Missing Python: $ENV_PY" >&2
  find "$LLMRUN" -maxdepth 4 -path '*/bin/python' -type f 2>/dev/null | sed 's#/bin/python##' >&2 || true
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ELUDE_DIR/src:$PCA:$SRC:$SRC/app/src:${PYTHONPATH:-}"
export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$PCA/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
cd "$ELUDE_DIR"

echo "[ELUDE-SMOKE] Host: $(hostname)"
echo "[ELUDE-SMOKE] CWD: $(pwd)"
echo "[ELUDE-SMOKE] ENV_PATH: $ENV_PATH"
echo "[ELUDE-SMOKE] Python: $ENV_PY"

# User-overridable smoke controls.
# SMOKE_EVAL_MODEL=1 checks model/tokenizer loading and tiny Module 8E eval.
# SMOKE_EVAL_MODEL=0 checks only imports + ELUDe HF data preparation.
export SMOKE_EVAL_MODEL="${SMOKE_EVAL_MODEL:-1}"
export SMOKE_TARGETS="${SMOKE_TARGETS:-}"
export SMOKE_MAX_TARGETS="${SMOKE_MAX_TARGETS:-1}"
export SMOKE_MAX_FORGET_TRAIN="${SMOKE_MAX_FORGET_TRAIN:-2}"
export SMOKE_MAX_RETAIN_TRAIN="${SMOKE_MAX_RETAIN_TRAIN:-2}"
export SMOKE_MAX_EVAL_ROWS="${SMOKE_MAX_EVAL_ROWS:-1}"
export ELUDE_SEED="${ELUDE_SEED:-17}"
export ELUDE_EXACT_PROTOCOL="${ELUDE_EXACT_PROTOCOL:-1}"
export ELUDE_RETAIN_EVAL_SPLIT="${ELUDE_RETAIN_EVAL_SPLIT:-test}"
export KIF_MODEL_DIR="${KIF_MODEL_DIR:-outputs/model}"
export KIF_USE_4BIT="${KIF_USE_4BIT:-1}"
export ELUDE_DO_GENERATION="${ELUDE_DO_GENERATION:-0}"
export ELUDE_EVAL_MAX_SEQ_LEN="${ELUDE_EVAL_MAX_SEQ_LEN:-384}"
export ELUDE_EVAL_MAX_NEW_TOKENS="${ELUDE_EVAL_MAX_NEW_TOKENS:-32}"

echo "[ELUDE-SMOKE] Controls: SMOKE_EVAL_MODEL=$SMOKE_EVAL_MODEL KIF_MODEL_DIR=$KIF_MODEL_DIR SMOKE_MAX_EVAL_ROWS=$SMOKE_MAX_EVAL_ROWS"

"$ENV_PY" - <<'PY'
import importlib, sys
print('[ELUDE-SMOKE] python', sys.version)
mods = [
    'torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes',
    'datasets', 'huggingface_hub', 'safetensors', 'sentence_transformers',
    'sklearn', 'rouge_score', 'evaluate', 'tqdm', 'pandas', 'numpy',
    'sentencepiece', 'einops', 'bs4', 'lxml', 'html5lib'
]
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f'[ELUDE-SMOKE] import OK {m:<24} {getattr(mod, "__version__", "unknown")}')
    except Exception as e:
        failed.append((m, repr(e)))
        print(f'[ELUDE-SMOKE] import FAIL {m:<24} {e!r}')
if failed:
    raise SystemExit('[ELUDE-SMOKE][ERROR] dependency import smoke test failed: ' + repr(failed))
from bs4 import BeautifulSoup
assert BeautifulSoup('<p>ok</p>', 'html.parser').find('p').text == 'ok'
import torch
print('[ELUDE-SMOKE] torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[ELUDE-SMOKE] gpu', torch.cuda.get_device_name(0))
PY

# Apply code-level stability patches and compile before touching data/model.
"$ENV_PY" -u apply_elude_kif_patches.py
"$ENV_PY" -m py_compile \
  run_elude_pipeline.py \
  run_elude20_benchmark.py \
  src/llama20/modules/module_elude.py \
  src/llama20/modules/module_b.py \
  src/llama20/modules/module_c.py \
  src/llama20/modules/module_d.py \
  src/llama20/modules/module_e.py \
  src/llama20/modules/module7.py \
  src/llama20/modules/module8e.py

"$ENV_PY" -u -X faulthandler - <<'PY'
import json, os, shutil
from pathlib import Path

from llama20.modules.module_elude import ELUDeConfig, build_elude_files

root = Path('outputs/elude_smoke')
datasets_dir = root / 'datasets'
elude_dir = root / 'elude'
eval_dir = root / 'eval_elude'
world_empty = root / 'empty_world_eval.json'
if root.exists():
    shutil.rmtree(root)
datasets_dir.mkdir(parents=True, exist_ok=True)
elude_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok=True)
world_empty.write_text('[]', encoding='utf-8')

targets_env = os.getenv('SMOKE_TARGETS', '').strip()
targets = [x.strip() for x in targets_env.split(',') if x.strip()] or None

print('[ELUDE-SMOKE] Building tiny ELUDe data files')
cfg = ELUDeConfig(
    output_dataset_dir=datasets_dir,
    output_elude_dir=elude_dir,
    seed=int(os.getenv('ELUDE_SEED', '17')),
    max_targets=int(os.getenv('SMOKE_MAX_TARGETS', '1')),
    max_forget_train_per_target=int(os.getenv('SMOKE_MAX_FORGET_TRAIN', '2')),
    max_retain_train_per_target=int(os.getenv('SMOKE_MAX_RETAIN_TRAIN', '2')),
    target_entities=targets,
    retain_eval_split=os.getenv('ELUDE_RETAIN_EVAL_SPLIT', 'test'),
    exact_elude_protocol=os.getenv('ELUDE_EXACT_PROTOCOL', '1') not in {'0', 'false', 'False'},
)
meta = build_elude_files(cfg)
print('[ELUDE-SMOKE] targets:', meta.get('targets'))
print('[ELUDE-SMOKE] stats:', json.dumps(meta.get('stats_by_target', {}), indent=2, ensure_ascii=False)[:2000])

required = [
    datasets_dir / 'prompts.jsonl',
    datasets_dir / 'triples.jsonl',
    elude_dir / 'elude_forget_train.jsonl',
    elude_dir / 'elude_retain_train.jsonl',
    elude_dir / 'elude_upu_train.jsonl',
    elude_dir / 'elude_forget_eval.jsonl',
    elude_dir / 'elude_retain_eval.jsonl',
    elude_dir / 'metadata.json',
]
missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
if missing:
    raise SystemExit('[ELUDE-SMOKE][ERROR] missing/empty ELUDe smoke files: ' + repr(missing))

for p in required:
    print(f'[ELUDE-SMOKE] file OK {p} size={p.stat().st_size}')

if os.getenv('SMOKE_EVAL_MODEL', '1') in {'0', 'false', 'False'}:
    print('[ELUDE-SMOKE] SMOKE_EVAL_MODEL=0: skipping model-load/Module8E smoke eval')
else:
    model_dir = Path(os.getenv('KIF_MODEL_DIR', 'outputs/model'))
    if not model_dir.exists():
        raise SystemExit(f'[ELUDE-SMOKE][ERROR] KIF_MODEL_DIR does not exist: {model_dir}. Run Module 0 first or set SMOKE_EVAL_MODEL=0 for data-only smoke.')

    print('[ELUDE-SMOKE] Running tiny Module 8E eval: 1 row/split, no generation')
    os.environ['ELUDE_DIR'] = str(elude_dir)
    os.environ['ELUDE_EVAL_OUT'] = str(eval_dir)
    os.environ['ELUDE_WORLD_EVAL_FILE'] = str(world_empty)
    os.environ['ELUDE_MAX_EVAL_ROWS'] = os.getenv('SMOKE_MAX_EVAL_ROWS', '1')
    os.environ['ELUDE_DO_GENERATION'] = os.getenv('ELUDE_DO_GENERATION', '0')

    from llama20.modules import module8e
    summary = module8e.run_module8_elude(adapter_path=None, out_dir=str(eval_dir))
    print('[ELUDE-SMOKE] Module8E summary keys:', sorted(summary.keys()))
    print('[ELUDE-SMOKE] Module8E core:', json.dumps({
        'forget/count': summary.get('forget/count'),
        'retain/count': summary.get('retain/count'),
        'FQ': summary.get('FQ'),
        'RQ': summary.get('RQ'),
    }, indent=2))
    for p in [eval_dir / 'elude_summary.json', eval_dir / 'elude_predictions.jsonl', eval_dir / 'elude_metrics.csv']:
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f'[ELUDE-SMOKE][ERROR] missing eval artifact: {p}')
        print(f'[ELUDE-SMOKE] eval artifact OK {p} size={p.stat().st_size}')

print('[ELUDE-SMOKE] PASS: ELUDe preflight smoke test completed successfully.')
PY
