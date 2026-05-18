"""OPT-OUT world-data compatibility patch.

The upstream OPT-OUT method uses Alpaca-GPT4 world/instruction data for the
`+wd` component. In this framework, the dataset-check script prepares that file
at:

    framework/outputs/optout/world/alpaca_gpt4_data_train.json

This patch makes OptOutMethod use that prepared file automatically when the
config does not explicitly provide `world_json`.
"""

from __future__ import annotations

import os
from pathlib import Path

import methods.optout as optout

_original_build_rows = optout.OptOutMethod._build_rows


def _build_rows_with_default_world_json(self, prompts_jsonl: str, forget_subjects):
    if not getattr(self.cfg, "world_json", None):
        env_world = os.environ.get("OPTOUT_WORLD_JSON")
        default_world = Path(prompts_jsonl).resolve().parent / "world" / "alpaca_gpt4_data_train.json"
        if env_world and Path(env_world).exists():
            self.cfg.world_json = env_world
        elif default_world.exists():
            self.cfg.world_json = str(default_world)
    return _original_build_rows(self, prompts_jsonl, forget_subjects)


optout.OptOutMethod._build_rows = _build_rows_with_default_world_json
