#!/usr/bin/env python3
"""Aggregate sweep metadata after Module E/7/8 runs.

This script is intentionally tolerant: if Module 8 has not been run yet, it
still aggregates Module E interaction counts and Module 7 adapter paths.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_state(smr: float | None, el10: float | None) -> str | None:
    if smr is None or el10 is None:
        return None
    if smr <= 0.05 and el10 < 1.0:
        return "Type I"
    if smr <= 0.05 and el10 >= 1.0:
        return "Type II"
    return "Type III"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", default="outputs/sweeps")
    parser.add_argument("--out", default="outputs/sweeps/sweep_summary.json")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    rows = []
    for cfg_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]):
        for seed_dir in sorted([p for p in cfg_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]):
            worker = load_json(seed_dir / "worker_result.json")
            worker_cfg = load_json(seed_dir / "worker_run_config.json")
            e_sum = load_json(seed_dir / "module_e" / "module_e_summary.json")
            m7 = load_json(seed_dir / "module7" / "last_adapter_path.json")
            ev = load_json(seed_dir / "module8" / "eval_summary.json")

            smr = None
            el10 = None
            util_delta_ppl = None
            if ev:
                smr = ev.get("robustness_post_lora", {}).get("avg_subject_mention_rate")
                el10 = ev.get("extraction_likelihood", {}).get("EL10_ratio")
                util_delta_ppl = ev.get("post_lora", {}).get("delta", {}).get("ppl")

            rows.append({
                "config_name": worker_cfg.get("config_name", cfg_dir.name),
                "seed": worker_cfg.get("seed", seed_dir.name.replace("seed", "")),
                "z_tau": worker_cfg.get("z_tau"),
                "soft_gate_k": worker_cfg.get("soft_gate_k"),
                "num_interactions": e_sum.get("num_interactions"),
                "num_firing_events": e_sum.get("num_firing_events"),
                "adapter_path": worker.get("adapter_path") or m7.get("adapter_path"),
                "smr": smr,
                "el10_ratio": el10,
                "utility_delta_ppl": util_delta_ppl,
                "state": classify_state(smr, el10),
            })

    grouped = {}
    for r in rows:
        grouped.setdefault(r["config_name"], []).append(r)

    config_summary = {}
    for cfg, rs in grouped.items():
        def stat(key: str):
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            return {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)} if vals else None
        config_summary[cfg] = {
            "z_tau": next((r.get("z_tau") for r in rs if r.get("z_tau") is not None), None),
            "soft_gate_k": next((r.get("soft_gate_k") for r in rs if r.get("soft_gate_k") is not None), None),
            "seeds": [r.get("seed") for r in rs],
            "num_interactions": stat("num_interactions"),
            "num_firing_events": stat("num_firing_events"),
            "smr": stat("smr"),
            "el10_ratio": stat("el10_ratio"),
            "utility_delta_ppl": stat("utility_delta_ppl"),
            "states": sorted(set(str(r.get("state")) for r in rs if r.get("state"))),
        }

    out = {"rows": rows, "config_summary": config_summary}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(config_summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
