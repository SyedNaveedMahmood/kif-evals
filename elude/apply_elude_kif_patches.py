#!/usr/bin/env python3
"""Apply ELUDe-specific KIF stability patches.

This is intentionally conservative and idempotent. It edits the ELUDe copy of
KIF modules in-place before a cluster run:

1. Module C uses ELUDe retain-control activations as semantic negatives and
   disables synthetic Gaussian negatives by default.
2. Module C fixes a positive-class oversampling indexing bug.
3. Module D/E/7 use a softer default suppression configuration for ELUDe so the
   closely-related retain set is less likely to be suppressed.

Run from the repository root or from the elude/ directory:
    python elude/apply_elude_kif_patches.py
"""
from __future__ import annotations

from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve()
    return here.parent if here.parent.name == "elude" else Path.cwd()


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Pattern not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def replace_all(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_module_c(base: Path) -> None:
    p = base / "src/llama20/modules/module_c.py"
    changed = False

    changed |= replace_once(
        p,
        "    allow_synthetic_fallback: bool = True\n",
        "    # ELUDe default: use retain-control activations as semantic negatives.\n"
        "    # Synthetic Gaussian negatives are disabled unless explicitly enabled.\n"
        "    allow_synthetic_fallback: bool = field(default_factory=lambda: os.getenv(\"KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES\", \"0\") not in {\"0\", \"false\", \"False\", \"no\", \"NO\"})\n",
    )
    changed |= replace_once(
        p,
        "                    oversampled_positives = [prompts[i].copy() for i in indices]\n",
        "                    oversampled_positives = [positives[i].copy() for i in indices]\n",
    )
    changed |= replace_once(
        p,
        "        allow_synthetic_fallback=True,\n",
        "        allow_synthetic_fallback=os.getenv(\"KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES\", \"0\") not in {\"0\", \"false\", \"False\", \"no\", \"NO\"},\n",
    )
    changed |= replace_once(
        p,
        "                # Preferred: semantic controls\n                if self.config.use_semantic_negatives and len(controls_all) >= self.config.min_controls_per_subject:\n                    neg_features, neg_fail = self.load_and_process_activations(controls_all, layer)\n                    if len(neg_features) < self.config.min_controls_per_subject:\n                        logger.info(f\"Layer {layer}: found {len(neg_features)} controls < min {self.config.min_controls_per_subject}\")\n                        neg_features = []\n                else:\n                    neg_features = []\n\n                # Fallback: synthesize negatives\n                if not neg_features and self.config.allow_synthetic_fallback:\n                    logger.info(f\"Layer {layer}: using synthetic negatives fallback for subject '{subject}'\")\n                    neg_features = self.generate_synthetic_negatives(pos_features, num_negatives=len(pos_features))\n\n                if len(neg_features) < 2:\n                    logger.warning(f\"Layer {layer}: insufficient negatives ({len(neg_features)})\")\n                    continue\n",
        "                # Preferred for ELUDe: semantic negatives from the target's retain set.\n"
        "                negative_source = \"none\"\n"
        "                if self.config.use_semantic_negatives and len(controls_all) >= self.config.min_controls_per_subject:\n"
        "                    neg_features, neg_fail = self.load_and_process_activations(controls_all, layer)\n"
        "                    if len(neg_features) < self.config.min_controls_per_subject:\n"
        "                        logger.info(f\"Layer {layer}: found {len(neg_features)} retain controls < min {self.config.min_controls_per_subject}\")\n"
        "                        neg_features = []\n"
        "                    else:\n"
        "                        negative_source = \"elude_retain_controls\"\n"
        "                else:\n"
        "                    neg_features = []\n"
        "\n"
        "                # Optional fallback only when explicitly enabled. For ELUDe paper runs,\n"
        "                # keep KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES=0 so the signature\n"
        "                # contrast is forget-vs-retain rather than forget-vs-Gaussian.\n"
        "                if not neg_features and self.config.allow_synthetic_fallback:\n"
        "                    logger.info(f\"Layer {layer}: using synthetic negatives fallback for subject '{subject}'\")\n"
        "                    neg_features = self.generate_synthetic_negatives(pos_features, num_negatives=len(pos_features))\n"
        "                    negative_source = \"synthetic_fallback\"\n"
        "\n"
        "                if len(neg_features) < 2:\n"
        "                    logger.warning(f\"Layer {layer}: insufficient retain negatives ({len(neg_features)}); skipping instead of using Gaussian fallback\")\n"
        "                    continue\n",
    )
    changed |= replace_once(
        p,
        "                    \"negative_count\": len(neg_features),\n                    \"failed_activations\": 0\n",
        "                    \"negative_count\": len(neg_features),\n                    \"negative_source\": negative_source,\n                    \"failed_activations\": 0\n",
    )
    print(f"module_c.py patched: {changed}")


def patch_module_d(base: Path) -> None:
    p = base / "src/llama20/modules/module_d.py"
    changed = False
    changed |= replace_once(
        p,
        "    scaling_factor_init: float = -1.0  # Initial suppression strength\n    max_scaling_factor: float = -5.0   # Maximum suppression strength\n",
        "    # ELUDe uses closely related retain entities, so the default capsule\n"
        "    # suppression is softer than the original KIF setting. Override via env.\n"
        "    scaling_factor_init: float = float(os.getenv(\"KIF_CAPSULE_SCALING_INIT\", \"-0.55\"))\n"
        "    max_scaling_factor: float = float(os.getenv(\"KIF_CAPSULE_MAX_SCALING\", \"-2.0\"))\n",
    )
    changed |= replace_once(
        p,
        "            s = torch.clamp(self.suppression_strength, min=-5.0, max=0.0)  # safety clamp\n",
        "            s = torch.clamp(self.suppression_strength, min=float(os.getenv(\"KIF_CAPSULE_CLAMP_MIN\", \"-2.0\")), max=0.0)  # safety clamp\n",
    )
    print(f"module_d.py patched: {changed}")


def patch_module_e(base: Path) -> None:
    p = base / "src/llama20/modules/module_e.py"
    changed = False
    changed |= replace_once(
        p,
        "    z_tau: float = 3.0\n    soft_gate_k: float = 1.6\n    default_strength: float = -0.8\n",
        "    z_tau: float = float(os.getenv(\"KIF_SENTINEL_Z_TAU\", \"3.0\"))\n"
        "    soft_gate_k: float = float(os.getenv(\"KIF_SENTINEL_SOFT_GATE_K\", \"1.6\"))\n"
        "    default_strength: float = float(os.getenv(\"KIF_SENTINEL_DEFAULT_STRENGTH\", \"-0.55\"))\n",
    )
    changed |= replace_once(
        p,
        "    refusal_steer_strength: float = 15.0\n",
        "    refusal_steer_strength: float = float(os.getenv(\"KIF_REFUSAL_STEER_STRENGTH\", \"8.0\"))\n",
    )
    print(f"module_e.py patched: {changed}")


def patch_module7(base: Path) -> None:
    p = base / "src/llama20/modules/module7.py"
    changed = False
    changed |= replace_once(
        p,
        "    sig_suppression_strength: float = 1.0\n",
        "    # ELUDe default is softer than KIF's original 1.0 because retain\n"
        "    # examples are neighboring entities rather than generic benign text.\n"
        "    sig_suppression_strength: float = field(default_factory=lambda: float(os.getenv(\"KIF_SIG_SUPPRESSION_STRENGTH\", \"0.65\")))\n",
    )
    print(f"module7.py patched: {changed}")


def main() -> None:
    base = _root()
    if (base / "elude").exists() and (base / "elude/src").exists():
        base = base / "elude"
    if not (base / "src/llama20/modules").exists():
        raise SystemExit(f"Could not locate elude/src/llama20/modules from {Path.cwd()}")
    patch_module_c(base)
    patch_module_d(base)
    patch_module_e(base)
    patch_module7(base)
    print("ELUDe KIF patches applied successfully.")


if __name__ == "__main__":
    main()
