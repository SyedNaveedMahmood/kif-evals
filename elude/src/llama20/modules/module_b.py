# Module B: Activation Probing (Hooks-in-Parallel + Batching)
# -----------------------------------------------------------
# One forward pass per batch captures ALL target layers via hooks, then saves
# per-prompt, per-layer activations to disk. This removes the O(num_layers)
# forward-pass bottleneck and fully utilizes the GPU with batching.
#
# 3090/ELUDe fast defaults:
#   KIF_PROBE_LAYERS=11,12,13,14
#   KIF_PROBE_BATCH_SIZE=16
#   KIF_PROBE_CAPTURE_SCOPE=last_token|full
#   KIF_PROBE_COMPRESSION_LEVEL=1
#   KIF_PROBE_SKIP_EXISTING=1

from __future__ import annotations

import os
import re
import gc
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False
from tqdm.auto import tqdm

try:
    import compress_pickle
except Exception:
    import pickle
    import gzip

    def _cp_dump(data, filename, compression="gzip", compresslevel=1, **kwargs):
        if compression != "gzip":
            raise ValueError("Only gzip compression supported in fallback.")
        with gzip.open(filename, "wb", compresslevel=compresslevel) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _cp_load(filename):
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

    compress_pickle = type("compress_pickle_fallback", (), {"dump": staticmethod(_cp_dump), "load": staticmethod(_cp_load)})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("kif_probe.log")]
)
logger = logging.getLogger("KIF-ModuleB")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _parse_layers(raw: str | None, default: List[int]) -> List[int]:
    if not raw:
        return list(default)
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out)) or list(default)


@dataclass
class ProbeConfig:
    model_dir: str = "outputs/model"
    prompts_file: str = "outputs/datasets/prompts.jsonl"
    output_dir: Path = Path("outputs/activations")

    # ELUDe/3090 default is deliberately restricted to KIF middle MLP layers.
    layers: List[int] = field(default_factory=lambda: _parse_layers(os.getenv("KIF_PROBE_LAYERS"), [11, 12, 13, 14]))
    targets: List[str] = field(default_factory=lambda: [x.strip() for x in os.getenv("KIF_PROBE_TARGETS", "mlp").split(",") if x.strip()])

    batch_size: int = field(default_factory=lambda: _env_int("KIF_PROBE_BATCH_SIZE", 16))
    max_length: int = field(default_factory=lambda: _env_int("KIF_PROBE_MAX_LENGTH", 128))
    use_half_precision: bool = field(default_factory=lambda: _env_bool("KIF_PROBE_FP16", True))
    use_4bit: bool = field(default_factory=lambda: _env_bool("KIF_USE_4BIT", True))
    save_dtype_fp16: bool = field(default_factory=lambda: _env_bool("KIF_PROBE_SAVE_FP16", True))

    device_map: str = field(default_factory=lambda: os.getenv("KIF_PROBE_DEVICE_MAP", "auto"))
    cleanup_every_batches: int = field(default_factory=lambda: _env_int("KIF_PROBE_CLEANUP_EVERY", 25))

    # gzip level 1 is much faster; activation files are already fp16.
    compression_level: int = field(default_factory=lambda: _env_int("KIF_PROBE_COMPRESSION_LEVEL", 1))

    # "last_token" is much smaller/faster on disk; "full" preserves old behavior.
    capture_scope: str = field(default_factory=lambda: os.getenv("KIF_PROBE_CAPTURE_SCOPE", "last_token"))
    skip_existing: bool = field(default_factory=lambda: _env_bool("KIF_PROBE_SKIP_EXISTING", True))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "mlp").mkdir(parents=True, exist_ok=True)
        if self.capture_scope not in {"full", "last_token"}:
            raise ValueError("KIF_PROBE_CAPTURE_SCOPE must be 'full' or 'last_token'")


def get_primary_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_numpy_for_saving(t: torch.Tensor, fp16: bool = True) -> np.ndarray:
    if t.is_cuda:
        t = t.detach().to("cpu")
    if fp16 and t.dtype in (torch.float32, torch.float64):
        t = t.half()
    t = t.contiguous()
    return t.numpy().astype(np.float16 if fp16 else np.float32, copy=False)


class ParallelActivationCollector:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ProbeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = get_primary_device(model)
        self.module_to_layer: Dict[str, int] = {}
        self.target_module_names: List[str] = []
        self.captured_activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self._discover_targets()
        self._register_all_hooks()

    def _discover_targets(self) -> None:
        names = []
        for name, module in self.model.named_modules():
            lname = name.lower()
            if any(t in lname for t in self.config.targets):
                m = re.search(r"layers\.(\d+)", name)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                if layer_idx in self.config.layers:
                    names.append((layer_idx, name))
        names.sort(key=lambda x: (x[0], x[1]))
        self.target_module_names = [n for _, n in names]
        self.module_to_layer = {n: i for i, n in names}
        if not self.target_module_names:
            raise RuntimeError("No target modules found. Check KIF_PROBE_LAYERS and KIF_PROBE_TARGETS.")
        logger.info("Target modules: %d across layers=%s", len(self.target_module_names), sorted(set(self.module_to_layer.values())))

    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured_activations[module_name] = out.detach()
        return hook_fn

    def _register_all_hooks(self) -> None:
        named = dict(self.model.named_modules())
        for module_name in self.target_module_names:
            mod = named.get(module_name, None)
            if mod is None:
                logger.warning("Module not found during hook registration: %s", module_name)
                continue
            self.hooks.append(mod.register_forward_hook(self._make_hook(module_name)))
        logger.info("Registered %d forward hooks", len(self.hooks))

    def remove_hooks(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        logger.info("Removed all hooks")

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _maybe_slice_last_token(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.capture_scope == "last_token":
            return t[:, -1:, :]
        return t

    def expected_paths_for_prompt(self, prompt_id: str) -> List[Path]:
        return [self.config.output_dir / "mlp" / f"{prompt_id}_layer{layer_idx}_mlp.pkl.gz" for layer_idx in sorted(set(self.module_to_layer.values()))]

    @torch.inference_mode()
    def collect_batch(self, prompts: List[str], prompt_ids: List[str]) -> Dict[str, List[Path]]:
        assert len(prompts) == len(prompt_ids)
        self.captured_activations.clear()

        if self.config.skip_existing:
            remaining = [(p, pid) for p, pid in zip(prompts, prompt_ids) if not all(x.exists() for x in self.expected_paths_for_prompt(pid))]
            if not remaining:
                return {pid: self.expected_paths_for_prompt(pid) for pid in prompt_ids}
            prompts, prompt_ids = zip(*remaining)
            prompts, prompt_ids = list(prompts), list(prompt_ids)

        inputs = self._tokenize_batch(prompts)
        try:
            _ = self.model(**inputs, use_cache=False, output_hidden_states=False, output_attentions=False)
        except TypeError:
            _ = self.model(**inputs)

        saved: Dict[str, List[Path]] = defaultdict(list)
        for module_name, tensor in self.captured_activations.items():
            layer_idx = self.module_to_layer.get(module_name)
            if layer_idx is None:
                continue
            tensor = self._maybe_slice_last_token(tensor)
            for b in range(tensor.shape[0]):
                pid = prompt_ids[b]
                path = self.config.output_dir / "mlp" / f"{pid}_layer{layer_idx}_mlp.pkl.gz"
                if self.config.skip_existing and path.exists():
                    saved[pid].append(path)
                    continue
                data_np = to_numpy_for_saving(tensor[b], fp16=self.config.save_dtype_fp16)
                compress_pickle.dump(data_np, path, compression="gzip", compresslevel=self.config.compression_level)
                saved[pid].append(path)
                del data_np
            del tensor

        self.captured_activations.clear()
        return saved


class OptimizedProbeRobot:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self._load_prompts()
        self._load_model()
        self.collector = ParallelActivationCollector(self.model, self.tokenizer, self.config)

    def _load_prompts(self) -> None:
        with open(self.config.prompts_file, "r", encoding="utf-8") as f:
            self.prompts: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]
        if not self.prompts:
            raise RuntimeError("No prompts found in prompts_file.")
        uniq = len({p.get("triple_id") for p in self.prompts})
        logger.info("Loaded %d prompts across %d triples", len(self.prompts), uniq)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Loading model from: %s", self.config.model_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        kwargs = dict(
            device_map=self.config.device_map,
            torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if self.config.use_4bit and _HAS_BNB and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_dir, **kwargs)
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if torch.cuda.is_available():
            logger.info("Model loaded on device: %s; CUDA device count: %d", get_primary_device(self.model), torch.cuda.device_count())
        else:
            logger.warning("CUDA not available; running on CPU.")

    def collect_all(self) -> Dict[str, List[Path]]:
        saved_all: Dict[str, List[Path]] = defaultdict(list)
        B = self.config.batch_size
        total = len(self.prompts)
        for batch_idx, start in enumerate(tqdm(range(0, total, B), desc="Collecting activations"), start=1):
            batch = self.prompts[start:start + B]
            texts = [p["prompt"] for p in batch]
            ids = [p["id"] for p in batch]
            try:
                saved = self.collector.collect_batch(texts, ids)
                for pid, paths in saved.items():
                    saved_all[pid].extend(paths)
            except torch.cuda.OutOfMemoryError as e:
                logger.error("OOM at batch %d-%d. Lower KIF_PROBE_BATCH_SIZE. Error: %s", start, start + len(batch), e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            except Exception as e:
                logger.error("Batch %d-%d failed: %s", start, start + len(batch), e)
                continue
            if torch.cuda.is_available() and (batch_idx % max(1, self.config.cleanup_every_batches) == 0):
                torch.cuda.empty_cache()
                gc.collect()
        return saved_all

    def _write_index(self, saved_paths: Dict[str, List[Path]]) -> None:
        prompt_by_id = {p.get("id"): p for p in self.prompts}
        index = {
            "config": {
                "layers": self.config.layers,
                "targets": self.config.targets,
                "model": self.config.model_dir,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "capture_scope": self.config.capture_scope,
                "fp16_model": self.config.use_half_precision,
                "use_4bit": self.config.use_4bit,
                "fp16_saves": self.config.save_dtype_fp16,
                "compression_level": self.config.compression_level,
                "skip_existing": self.config.skip_existing,
            },
            "prompts": {
                pid: {
                    "paths": [str(p) for p in paths],
                    "triple_id": prompt_by_id.get(pid, {}).get("triple_id"),
                    "activation_count": len(paths),
                }
                for pid, paths in saved_paths.items()
            },
            "summary": {
                "total_prompts": len(saved_paths),
                "total_files": sum(len(v) for v in saved_paths.values()),
            },
        }
        path = self.config.output_dir / "activation_index.json"
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        logger.info("Wrote activation index to %s", path)

    def _write_report(self, saved_paths: Dict[str, List[Path]]) -> None:
        total_size = 0
        layer_counts: Dict[int, int] = defaultdict(int)
        for plist in saved_paths.values():
            for p in plist:
                try:
                    total_size += os.path.getsize(p)
                    m = re.search(r"layer(\d+)", str(p))
                    if m:
                        layer_counts[int(m.group(1))] += 1
                except OSError:
                    pass
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.config.model_dir,
            "results": {
                "prompts_processed": len(saved_paths),
                "total_files": sum(len(v) for v in saved_paths.values()),
                "storage_gb": total_size / (1024 ** 3),
                "files_per_layer": dict(layer_counts),
            },
        }
        path = self.config.output_dir / "collection_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote collection report to %s", path)

    def run(self) -> None:
        logger.info("=" * 64)
        logger.info("Starting Module B: Hooks-in-Parallel + Batching")
        logger.info("Config: layers=%s batch_size=%s max_length=%s capture_scope=%s compression=%s skip_existing=%s", self.config.layers, self.config.batch_size, self.config.max_length, self.config.capture_scope, self.config.compression_level, self.config.skip_existing)
        logger.info("=" * 64)
        saved = self.collect_all()
        self._write_index(saved)
        self._write_report(saved)
        self.collector.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Module B completed successfully ✔")


def run_module_b():
    cfg = ProbeConfig()
    robot = OptimizedProbeRobot(cfg)
    robot.run()


if __name__ == "__main__":
    run_module_b()
