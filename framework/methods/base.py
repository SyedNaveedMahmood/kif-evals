"""
methods/base.py
===============
Abstract base class and registry for all unlearning methods.

Contract (every method must satisfy)
--------------------------------------
Input  : outputs/datasets/prompts.jsonl   (KIF format, unchanged)
Output : a saved model artefact + UnlearningResult manifest

UnlearningResult tells the evaluator how to load the model:
  Mode A — merged HF model dir  → merged_model_dir populated
  Mode B — base + LoRA adapter  → model_dir + adapter_path populated

The evaluator (module8_eval.py) reads this manifest and runs
SMR + EL10 + utility metrics regardless of which method produced the model.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class _MethodRegistry:
    def __init__(self):
        self._reg: Dict[str, type] = {}

    def register(self, name: str):
        """Decorator: @METHOD_REGISTRY.register('lunar')"""
        def decorator(cls):
            self._reg[name] = cls
            logger.info(f"[Registry] Registered: '{name}'")
            return cls
        return decorator

    def get(self, name: str) -> type:
        if name not in self._reg:
            raise KeyError(
                f"Unknown method '{name}'. Available: {self.list()}"
            )
        return self._reg[name]

    def list(self) -> List[str]:
        return sorted(self._reg.keys())


METHOD_REGISTRY = _MethodRegistry()


# ---------------------------------------------------------------------------
# Result dataclass — the handshake between method and evaluator
# ---------------------------------------------------------------------------

@dataclass
class UnlearningResult:
    """
    Produced by every unlearning method. Tells the evaluator exactly
    how to load the resulting model.

    Mode A (merged model — LUNAR, RMU, etc.):
        merged_model_dir = "path/to/full_model"
        model_dir = adapter_path = None

    Mode B (base + LoRA adapter — KIF, etc.):
        model_dir    = "path/to/base_model"
        adapter_path = "path/to/adapter"
        merged_model_dir = None
    """
    # Mode A
    merged_model_dir: Optional[str] = None
    # Mode B
    model_dir: Optional[str] = None
    adapter_path: Optional[str] = None
    # Shared
    method_name: str = ""
    subjects: List[str] = field(default_factory=list)
    output_dir: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "merged_model_dir": self.merged_model_dir,
            "model_dir": self.model_dir,
            "adapter_path": self.adapter_path,
            "subjects": self.subjects,
            "output_dir": self.output_dir,
            "extra": self.extra,
        }

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"[Result] Saved manifest → {path}")

    @staticmethod
    def load(path: str | Path) -> "UnlearningResult":
        d = json.loads(Path(path).read_text())
        r = UnlearningResult()
        for k, v in d.items():
            setattr(r, k, v)
        return r


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class UnlearningMethod(ABC):
    """
    Subclass this to implement a new unlearning method.

    Minimal implementation:
        @METHOD_REGISTRY.register("my_method")
        class MyMethod(UnlearningMethod):
            name = "my_method"
            def run(self, prompts_jsonl, output_dir, model_dir, subjects):
                ...
                return UnlearningResult(merged_model_dir=saved_path,
                                        subjects=subjects)
    """
    name: str = "unnamed"

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        self.config_overrides = config_overrides or {}

    # ------------------------------------------------------------------
    # Entry point called by orchestrator
    # ------------------------------------------------------------------

    def execute(
        self,
        prompts_jsonl: str,
        output_dir: str,
        model_dir: str,
        subjects: Optional[List[str]] = None,
    ) -> UnlearningResult:
        method_out = str(Path(output_dir) / self.name)
        Path(method_out).mkdir(parents=True, exist_ok=True)

        logger.info(f"[{self.name}] Starting  →  {method_out}")
        result = self.run(
            prompts_jsonl=prompts_jsonl,
            output_dir=method_out,
            model_dir=model_dir,
            subjects=subjects,
        )
        result.method_name = self.name
        result.output_dir  = method_out
        result.save(Path(method_out) / "unlearning_result.json")
        logger.info(f"[{self.name}] Done.  Result → {result.to_dict()}")
        return result

    @abstractmethod
    def run(
        self,
        prompts_jsonl: str,
        output_dir: str,
        model_dir: str,
        subjects: Optional[List[str]],
    ) -> UnlearningResult:
        """Implement unlearning here. Save model. Return UnlearningResult."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers — parse prompts.jsonl in the KIF format
    # ------------------------------------------------------------------

    @staticmethod
    def load_subjects(prompts_jsonl: str) -> List[str]:
        """All unique subjects in prompts.jsonl (insertion order preserved)."""
        subjects, seen = [], set()
        for line in Path(prompts_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                s = rec.get("subject") or rec.get("author")
                if s and s not in seen:
                    seen.add(s); subjects.append(s)
            except Exception:
                pass
        return subjects

    @staticmethod
    def load_prompts_by_subject(
        prompts_jsonl: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Returns {subject: [record, ...]} for every record in prompts.jsonl."""
        out: Dict[str, List] = {}
        for line in Path(prompts_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                s = rec.get("subject") or rec.get("author")
                if s:
                    out.setdefault(str(s), []).append(rec)
            except Exception:
                pass
        return out

    @staticmethod
    def get_forget_qa(
        prompts_jsonl: str,
        subjects: List[str],
        max_per_subject: int = 50,
    ) -> List[Dict[str, str]]:
        """
        Flatten prompts for the given subjects into:
            [{"question": ..., "answer": ..., "subject": ...}, ...]
        'answer' comes from the 'response' field if present, else empty string.
        """
        by_subj = UnlearningMethod.load_prompts_by_subject(prompts_jsonl)
        records = []
        for s in subjects:
            for r in by_subj.get(s, [])[:max_per_subject]:
                records.append({
                    "question": r.get("prompt", ""),
                    "answer":   r.get("response", ""),
                    "subject":  s,
                })
        return records
