#!/usr/bin/env python3
"""Patch Module C to use cross-subject real negatives before synthetic fallback.

This patch preserves the existing Module C behavior of using metadata controls
when available, but adds a stronger real-negative source:

  target subject positives vs positive/unknown prompts from other subjects

New config fields injected into SignatureMiningConfig:
  real_negative_source = "metadata_then_cross_subject"
    choices by convention:
      metadata_control          -> use only same-subject metadata controls
      cross_subject_positive    -> use only other-subject positive/unknown prompts
      metadata_then_cross_subject -> use metadata controls first, then cross-subject if insufficient
      mixed                     -> combine metadata controls and cross-subject positives
  max_cross_subject_negatives = 128

It also records negative_source in each layer result.
"""
from __future__ import annotations

from pathlib import Path

TARGETS = [
    Path("app/src/llama20/modules/module_c.py"),
    Path("elude/src/llama20/modules/module_c.py"),
]


def patch_one(path: Path) -> bool:
    if not path.exists():
        print(f"[SKIP] missing {path}")
        return False
    s = path.read_text(encoding="utf-8")
    original = s

    # 1. Add config fields after existing semantic-negative settings.
    old = '''    # Subject/Negative set settings
    use_semantic_negatives: bool = True
    min_controls_per_subject: int = 1
    allow_synthetic_fallback: bool = True
    positive_keys: List[str] = field(default_factory=lambda: ["leak", "direct", "context", "implicit", "reason", "reasoning"])
    control_keys: List[str] = field(default_factory=lambda: ["control"])
'''
    new = '''    # Subject/Negative set settings
    use_semantic_negatives: bool = True
    min_controls_per_subject: int = 1
    allow_synthetic_fallback: bool = True
    # Negative source policy:
    #   metadata_control: same-subject control prompts only
    #   cross_subject_positive: positive/unknown prompts from other subjects
    #   metadata_then_cross_subject: metadata controls first, cross-subject fallback if insufficient
    #   mixed: metadata controls + cross-subject positives
    real_negative_source: str = "metadata_then_cross_subject"
    max_cross_subject_negatives: int = 128
    positive_keys: List[str] = field(default_factory=lambda: ["leak", "direct", "context", "implicit", "reason", "reasoning"])
    control_keys: List[str] = field(default_factory=lambda: ["control"])
'''
    if old in s and "real_negative_source" not in s:
        s = s.replace(old, new)
    elif "real_negative_source" not in s:
        raise RuntimeError(f"Could not locate config insertion point in {path}")

    # 2. Add a subject_groups cache and helper methods inside CausalTracer.
    old = '''class CausalTracer:
    """Find signature directions contrasting leak (positive) vs control (negative) activations"""
    def __init__(self, config: SignatureMiningConfig, activation_manager: ActivationManager):
        self.config = config
        self.activation_manager = activation_manager
        self.memory_manager = MemoryManager(config)
        self.device = torch.device(config.device)

    def _select_paths_for_layer(self, prompt: Dict, layer: int) -> Optional[str]:
'''
    new = '''class CausalTracer:
    """Find signature directions contrasting leak (positive) vs real/synthetic negative activations"""
    def __init__(self, config: SignatureMiningConfig, activation_manager: ActivationManager):
        self.config = config
        self.activation_manager = activation_manager
        self.memory_manager = MemoryManager(config)
        self.device = torch.device(config.device)
        # Filled by SignatureExtractor.extract_all_signatures() to avoid regrouping.
        self.subject_groups: Optional[Dict[str, List[Dict]]] = None

    def _positive_like_prompts(self, prompts: List[Dict]) -> List[Dict]:
        return [p for p in prompts if p.get("class") in ("positive", "unknown")]

    def _sample_prompts(self, prompts: List[Dict], max_items: int, seed_offset: int = 0) -> List[Dict]:
        if max_items is None or max_items <= 0 or len(prompts) <= max_items:
            return list(prompts)
        rng = np.random.default_rng(self.config.random_state + seed_offset)
        idx = rng.choice(len(prompts), size=max_items, replace=False)
        return [prompts[int(i)] for i in idx]

    def get_cross_subject_negative_prompts(self, subject: str) -> List[Dict]:
        """Use other subjects' positive/unknown prompts as real semantic negatives."""
        groups = self.subject_groups
        if groups is None:
            groups = self.activation_manager.group_by_subject()
        candidates: List[Dict] = []
        for other_subject, other_prompts in groups.items():
            if other_subject == subject:
                continue
            candidates.extend(self._positive_like_prompts(other_prompts))
        # Stable per-subject sample to prevent large subjects from dominating.
        seed_offset = abs(hash(subject)) % 100000
        return self._sample_prompts(candidates, self.config.max_cross_subject_negatives, seed_offset=seed_offset)

    def _select_paths_for_layer(self, prompt: Dict, layer: int) -> Optional[str]:
'''
    if old in s and "get_cross_subject_negative_prompts" not in s:
        s = s.replace(old, new)
    elif "get_cross_subject_negative_prompts" not in s:
        raise RuntimeError(f"Could not locate CausalTracer insertion point in {path}")

    # 3. Replace the negative-selection block in analyze_subject.
    old = '''                # Preferred: semantic controls
                if self.config.use_semantic_negatives and len(controls_all) >= self.config.min_controls_per_subject:
                    neg_features, neg_fail = self.load_and_process_activations(controls_all, layer)
                    if len(neg_features) < self.config.min_controls_per_subject:
                        logger.info(f"Layer {layer}: found {len(neg_features)} controls < min {self.config.min_controls_per_subject}")
                        neg_features = []
                else:
                    neg_features = []

                # Fallback: synthesize negatives
                if not neg_features and self.config.allow_synthetic_fallback:
                    logger.info(f"Layer {layer}: using synthetic negatives fallback for subject '{subject}'")
                    neg_features = self.generate_synthetic_negatives(pos_features, num_negatives=len(pos_features))

                if len(neg_features) < 2:
                    logger.warning(f"Layer {layer}: insufficient negatives ({len(neg_features)})")
                    continue
'''
    new = '''                # Preferred: real semantic negatives. Module C supports both
                # same-subject metadata controls and cross-subject real negatives.
                neg_features = []
                negative_source = "none"
                policy = getattr(self.config, "real_negative_source", "metadata_then_cross_subject")

                if self.config.use_semantic_negatives and policy in ("metadata_control", "metadata_then_cross_subject", "mixed"):
                    if len(controls_all) >= self.config.min_controls_per_subject:
                        meta_neg_features, neg_fail = self.load_and_process_activations(controls_all, layer)
                        if len(meta_neg_features) >= self.config.min_controls_per_subject:
                            neg_features.extend(meta_neg_features)
                            negative_source = "metadata_control"
                        else:
                            logger.info(f"Layer {layer}: found {len(meta_neg_features)} metadata controls < min {self.config.min_controls_per_subject}")

                use_cross_subject = self.config.use_semantic_negatives and policy in ("cross_subject_positive", "metadata_then_cross_subject", "mixed")
                need_cross_subject = policy in ("cross_subject_positive", "mixed") or len(neg_features) < self.config.min_controls_per_subject
                if use_cross_subject and need_cross_subject:
                    cross_prompts = self.get_cross_subject_negative_prompts(subject)
                    cross_neg_features, cross_fail = self.load_and_process_activations(cross_prompts, layer)
                    if len(cross_neg_features) >= self.config.min_controls_per_subject:
                        if policy == "mixed" and neg_features:
                            neg_features.extend(cross_neg_features)
                            negative_source = "metadata_plus_cross_subject_positive"
                        else:
                            neg_features = cross_neg_features
                            negative_source = "cross_subject_positive"
                    else:
                        logger.info(f"Layer {layer}: found {len(cross_neg_features)} cross-subject negatives < min {self.config.min_controls_per_subject}")

                # Final fallback: synthesize negatives only if real semantic negatives are unavailable.
                if not neg_features and self.config.allow_synthetic_fallback:
                    logger.info(f"Layer {layer}: using synthetic negatives fallback for subject '{subject}'")
                    neg_features = self.generate_synthetic_negatives(pos_features, num_negatives=len(pos_features))
                    negative_source = "synthetic_fallback"

                if len(neg_features) < 2:
                    logger.warning(f"Layer {layer}: insufficient negatives ({len(neg_features)})")
                    continue
'''
    if old in s:
        s = s.replace(old, new)
    elif "negative_source = \"none\"" not in s:
        raise RuntimeError(f"Could not locate negative selection block in {path}")

    # 4. Add negative_source to saved layer result.
    old = '''                    "positive_count": len(pos_features),
                    "negative_count": len(neg_features),
                    "failed_activations": 0
'''
    new = '''                    "positive_count": len(pos_features),
                    "negative_count": len(neg_features),
                    "negative_source": negative_source,
                    "failed_activations": 0
'''
    if old in s and '"negative_source": negative_source' not in s:
        s = s.replace(old, new)

    # 5. Store subject_groups into the tracer once in SignatureExtractor.
    old = '''    def extract_all_signatures(self) -> Dict[str, Any]:
        subject_groups = self.activation_manager.group_by_subject()
        all_signatures = {}
'''
    new = '''    def extract_all_signatures(self) -> Dict[str, Any]:
        subject_groups = self.activation_manager.group_by_subject()
        self.causal_tracer.subject_groups = subject_groups
        all_signatures = {}
'''
    if old in s and "self.causal_tracer.subject_groups = subject_groups" not in s:
        s = s.replace(old, new)

    # 6. Add fields to signature index config summary if present.
    old = '''                "oversampling_enabled": self.config.enable_oversampling,
                "oversample_strategy": self.config.oversample_strategy
'''
    new = '''                "oversampling_enabled": self.config.enable_oversampling,
                "oversample_strategy": self.config.oversample_strategy,
                "real_negative_source": self.config.real_negative_source,
                "max_cross_subject_negatives": self.config.max_cross_subject_negatives
'''
    if old in s and '"real_negative_source": self.config.real_negative_source' not in s:
        s = s.replace(old, new)

    old = '''                "oversampling": {
                    "enabled": self.config.enable_oversampling,
                    "strategy": self.config.oversample_strategy,
                    "separate_classes": self.config.oversample_separately,
                    "preserve_ratio": self.config.preserve_original_ratio
                }
'''
    new = '''                "oversampling": {
                    "enabled": self.config.enable_oversampling,
                    "strategy": self.config.oversample_strategy,
                    "separate_classes": self.config.oversample_separately,
                    "preserve_ratio": self.config.preserve_original_ratio
                },
                "negative_mining": {
                    "use_semantic_negatives": self.config.use_semantic_negatives,
                    "real_negative_source": self.config.real_negative_source,
                    "max_cross_subject_negatives": self.config.max_cross_subject_negatives,
                    "allow_synthetic_fallback": self.config.allow_synthetic_fallback
                }
'''
    if old in s and '"negative_mining": {' not in s:
        s = s.replace(old, new)

    if s != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(s, encoding="utf-8")
        print(f"[PATCHED] {path}")
        return True
    print(f"[UNCHANGED] {path}")
    return False


def main() -> None:
    changed = 0
    for t in TARGETS:
        if patch_one(t):
            changed += 1
    print(f"Done. Files patched: {changed}")


if __name__ == "__main__":
    main()
