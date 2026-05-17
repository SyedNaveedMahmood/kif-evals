"""
methods/template_new_method.py
==============================
Copy-paste this file to add any new unlearning method.

Checklist
---------
1.  Copy to  methods/my_method.py
2.  Replace  "template"  with your method name everywhere.
3.  Fill in  run()  with your actual unlearning logic.
4.  Add  from methods.my_method import MyMethod  to methods/__init__.py.
5.  Run:  python orchestrate.py --method my_method ...

Nothing else changes. The orchestrator, evaluator and Module 8 are
completely unaware of your method's internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.base import METHOD_REGISTRY, UnlearningMethod, UnlearningResult

logger = logging.getLogger(__name__)


@METHOD_REGISTRY.register("template")          # <-- change "template" to your method name
class TemplateMethod(UnlearningMethod):
    """
    Template / skeleton for a new unlearning method.

    Input interface  : outputs/datasets/prompts.jsonl  (KIF standard)
    Output interface : <output_dir>/template/unlearned_model/   OR
                       <output_dir>/template/adapter/
                         + <output_dir>/template/ for model_dir reference
    """

    name = "template"   # <-- match the registry key above

    def run(
        self,
        prompts_jsonl: str,
        output_dir: str,
        model_dir: str,
        subjects: Optional[List[str]],
    ) -> UnlearningResult:
        """
        Implement your unlearning procedure here.

        The three helpers below are inherited from UnlearningMethod and
        handle the KIF prompts.jsonl format for you:

            subjects = self.load_subjects_from_prompts(prompts_jsonl)
            by_subject = self.load_prompts_by_subject(prompts_jsonl)
            forget_records = self.subject_to_forget_questions(
                prompts_jsonl, subjects, max_per_subject=50
            )

        After unlearning, save the model and return an UnlearningResult.

        Option A — save a merged (full) model:
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            return UnlearningResult(merged_model_dir=merged_dir, subjects=subjects)

        Option B — save a LoRA adapter next to the base model:
            adapter.save_pretrained(adapter_dir)
            return UnlearningResult(
                model_dir=model_dir,        # point to the ORIGINAL base model
                adapter_path=adapter_dir,
                subjects=subjects,
            )
        """
        # ---- Resolve subjects ------------------------------------------------
        if subjects is None:
            subjects = self.load_subjects_from_prompts(prompts_jsonl)

        logger.info(f"[{self.name}] Subjects: {subjects}")

        # ---- Load prompts -------------------------------------------------------
        forget_records = self.subject_to_forget_questions(
            prompts_jsonl, subjects, max_per_subject=50
        )
        forget_questions = [r["question"] for r in forget_records if r["question"]]
        logger.info(f"[{self.name}] Forget set size: {len(forget_questions)}")

        # ----------------------------------------------------------------
        # TODO: implement your method here
        #   1. Load model from model_dir
        #   2. Run your unlearning procedure
        #   3. Save result to output_dir
        # ----------------------------------------------------------------

        raise NotImplementedError(
            f"[{self.name}] run() is not implemented. "
            "Fill in the TODO block in this template."
        )

        # ---- Example return for Option A ------------------------------------
        # merged_dir = str(Path(output_dir) / "unlearned_model")
        # return UnlearningResult(merged_model_dir=merged_dir, subjects=subjects)

        # ---- Example return for Option B (LoRA) -----------------------------
        # return UnlearningResult(
        #     model_dir=model_dir,
        #     adapter_path=str(Path(output_dir) / "adapter"),
        #     subjects=subjects,
        # )
