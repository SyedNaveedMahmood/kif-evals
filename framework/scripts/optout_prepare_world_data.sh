#!/usr/bin/env bash
# Prepare upstream OPT-OUT Alpaca-GPT4 world/instruction data.
#
# Upstream OPT-OUT uses Alpaca-GPT4 data for the +wd component. This script
# downloads the upstream train split into the framework output directory without
# tracking it in git.

set -euo pipefail

LLMRUN_ROOT="${LLMRUN_ROOT:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
SRC_ROOT="${SRC_ROOT:-$LLMRUN_ROOT/src}"
FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$SRC_ROOT/framework}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FRAMEWORK_ROOT/outputs/optout}"
WORLD_DIR="$OUTPUT_ROOT/world"
WORLD_JSON="$WORLD_DIR/alpaca_gpt4_data_train.json"
WORLD_URL="https://raw.githubusercontent.com/brightjade/Opt-Out/main/data/alpaca_gpt4_data_train.json"

mkdir -p "$WORLD_DIR"

if [ -s "$WORLD_JSON" ]; then
  echo "World data already exists: $WORLD_JSON"
else
  echo "Downloading upstream OPT-OUT world data:"
  echo "  $WORLD_URL"
  echo "to:"
  echo "  $WORLD_JSON"
  tmp_path="$WORLD_JSON.tmp"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 5 "$WORLD_URL" -o "$tmp_path"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_path" "$WORLD_URL"
  else
    python - <<PY
import urllib.request
url = "$WORLD_URL"
out = "$tmp_path"
urllib.request.urlretrieve(url, out)
PY
  fi
  mv "$tmp_path" "$WORLD_JSON"
fi

python - <<PY
import json
from pathlib import Path
path = Path("$WORLD_JSON")
data = json.loads(path.read_text(encoding="utf-8"))
assert isinstance(data, list), type(data)
assert len(data) > 0
first = data[0]
prompt_keys = ("question", "instruction", "prompt", "input", "text")
answer_keys = ("answer", "output", "response", "completion", "target", "label")
has_prompt = any(k in first and str(first[k]).strip() for k in prompt_keys)
has_answer = any(k in first and str(first[k]).strip() for k in answer_keys)
assert has_prompt and has_answer, first.keys()
print(f"Validated world data: {path}")
print(f"Rows: {len(data)}")
print(f"First keys: {sorted(first.keys())}")
PY

echo "Use this path as OPTOUT_WORLD_JSON:"
echo "$WORLD_JSON"
