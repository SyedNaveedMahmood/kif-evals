"""
methods/template.py
===================
Skeleton for a new unlearning method. Copy this file, rename it,
fill in run(), register, import in __init__.py. Done.

Steps
-----
1.  cp methods/template.py methods/my_method.py
2.  Replace "template" → "my_method" everywhere in the file.
3.  Implement run() below.
4.  In methods/__init__.py add:
        from methods.my_method import MyMethod   # noqa: F401
5.  Run:  python orchestrate.py --method my_method ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.base import METHOD_REGISTRY, UnlearningMethod, UnlearningResult

logger = logging.getLogger(__name__)


@METHOD_REGISTRY.register("template")          # ← change name
class TemplateMethod(UnlearningMethod):        # ← change class name
    name = "template"                          # ← match registry key

    def run(
        self,
        prompts_jsonl: str,
        output_dir: str,
        model_dir: str,
        subjects: Optional[List[str]],
    ) -> UnlearningResult:

        # ── 1. Subjects ──────────────────────────────────────────────────
        if subjects is None:
            subjects = self.load_subjects(prompts_jsonl)

        # ── 2. Forget QA pairs ───────────────────────────────────────────
        forget_qa = self.get_forget_qa(
            prompts_jsonl, subjects, max_per_subject=50
        )
        forget_questions = [r["question"] for r in forget_qa if r["question"]]

        # ── 3. Retain QA pairs (non-forget subjects in prompts.jsonl) ────
        all_subjects     = self.load_subjects(prompts_jsonl)
        forget_set       = set(subjects)
        retain_subjects  = [s for s in all_subjects if s not in forget_set]
        retain_qa        = self.get_forget_qa(
            prompts_jsonl, retain_subjects, max_per_subject=50
        )
        retain_questions = [r["question"] for r in retain_qa if r["question"]]

        logger.info(
            f"[{self.name}] Forget subjects: {subjects}\n"
            f"  forget_q={len(forget_questions)}  "
            f"retain_q={len(retain_questions)}"
        )

        # ── 4. TODO: implement your unlearning here ───────────────────────
        # Load model from model_dir, run your algorithm, save the result.
        raise NotImplementedError(
            f"[{self.name}] run() is not yet implemented."
        )

        # ── 5. Return result — choose ONE of the two modes ───────────────

        # Mode A: save a merged (full) model
        # merged_dir = str(Path(output_dir) / "unlearned_model")
        # model.save_pretrained(merged_dir)
        # tokenizer.save_pretrained(merged_dir)
        # return UnlearningResult(merged_model_dir=merged_dir,
        #                         subjects=subjects)

        # Mode B: save a LoRA adapter next to the original base model
        # adapter_dir = str(Path(output_dir) / "adapter")
        # peft_model.save_pretrained(adapter_dir)
        # return UnlearningResult(model_dir=model_dir,
        #                         adapter_path=adapter_dir,
        #                         subjects=subjects)
