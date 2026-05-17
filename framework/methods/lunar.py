"""
methods/lunar.py
================
LUNAR unlearning method adapted to the KIF plug-and-play framework.

Key design decisions
---------------------
- Zero dependency on the LUNAR repo (no run_lunar.py, no Hydra, no their splits).
- All data comes exclusively from prompts.jsonl (KIF format).
- Forget set  : prompts for subjects to unlearn (from prompts.jsonl).
- Retain set  : prompts for subjects NOT in the forget list (from prompts.jsonl).
                If all subjects in prompts.jsonl are forget targets (e.g. the
                full musician dataset), retain_instr and Dref fall back to the
                built-in UNVERIFIABLE_PROMPTS — questions about fictional
                entities the model genuinely cannot answer.  This is the
                "unverifiable prompts" strategy from LUNAR §6.2, which is the
                correct choice when no real retain-subject data exists.
- Dref        : subset of retain_instr (retain subjects or unverifiable prompts).
                The UV direction points from forget activations toward how the
                model responds to content it has no knowledge of.
- Evaluation  : entirely via Module 8 (SMR + EL10). No LUNAR metrics used.

LUNAR mechanics (from the paper)
----------------------------------
1. Compute unlearning vector:
       r_UV = mean_act(Dref, layer l) − mean_act(Dforget, layer l)
2. Select layer l that maximises (s1 − s2) score (§3.2).
3. Solve closed-form weight update (Eq. 9):
       Wc = (H^T H + λI)^{-1} H^T A
   where H = pre-down-proj hidden states, A = redirected activations.
4. Patch down_proj weight at layer l. Save merged model.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from methods.base import METHOD_REGISTRY, UnlearningMethod, UnlearningResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unverifiable prompts — used as Dref + retain when all subjects are forgotten
# (LUNAR paper §6.2 "unverifiable prompts" strategy)
# These are questions about fictional entities the model genuinely cannot
# answer, so its natural response is uncertainty / "I don't know".
# The UV direction will redirect forget activations toward this uncertainty
# region, which is the correct target when there are no real retain subjects.
# ---------------------------------------------------------------------------

UNVERIFIABLE_PROMPTS: List[str] = [
    "What is the population of the mythical country of Zarbonia?",
    "Describe the rules of the imaginary sport ftszeqohwq.",
    "What is the lifespan of a blorpfish?",
    "Who won the 2087 Galactic Championship?",
    "Explain the monetary system of the country Quelthar.",
    "What are the main exports of the fictional nation Dravencia?",
    "Describe the constitution of Peloria written in 3042.",
    "What is the official language of Grestovia?",
    "Who is the current prime minister of Flondria?",
    "What is the speed of a standard Xenocraft spaceship?",
    "Explain the history of the imaginary city Vorthex.",
    "What is the climate like in the made-up region of Blornheim?",
    "Describe the culture of the fictional tribe Zeptari.",
    "Who founded the mythical organisation QWERZYX?",
    "What is the capital of the non-existent country Murdalon?",
    "Describe the diet of the imaginary creature the Snorflax.",
    "What year did the fictional war of Krenthia end?",
    "Who wrote the legendary text of the Dorbitian Scrolls?",
    "What is the average temperature on the planet Frobulon?",
    "Describe the economic policy of the invented nation Trezvia.",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LUNARConfig:
    # Model family — controls which chat template is applied.
    # Supported: llama3-8b | plain | llama3-8b-instruct | llama2-7b-chat |
    #            Qwen2-7B-Instruct | Qwen2.5-7B-Instruct | gemma-7b-it |
    #            mistral-7b-instruct
    model_family: str = "llama3-8b"

    # Layer selection
    auto_select_layer: bool = True   # True  → pick via cosine-sim score (paper §3.2)
                                     # False → use layer_modified below
    layer_modified: int = 22         # used only when auto_select_layer=False

    # UV computation
    positions: int = -1              # token position for mean-diff (−1 = last)

    # Optimisation (closed-form solve is default; SGD as fallback)
    use_closed_form: bool = True
    lambda_reg: float = 1e-3         # ridge regularisation for closed-form
    num_epochs: int = 20             # only used for SGD fallback
    lr: float = 1e-2                 # only used for SGD fallback

    # Data limits
    n_forget: int = 128              # max forget prompts fed to LUNAR
    n_retain: int = 128              # max retain prompts
    n_ref: int = 64                  # max Dref prompts (subset of retain)
    max_per_subject: int = 50        # prompts pulled per subject from prompts.jsonl

    # Subjects
    retain_subjects: Optional[List[str]] = None   # None → all non-forget subjects
    max_forget_subjects: Optional[int] = None     # None → all subjects in list

    # Misc
    batch_size: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Chat templates (matching LUNAR's data_loader.py where chat models are used)
# ---------------------------------------------------------------------------

_CHAT_TEMPLATES: Dict[str, str] = {
    "llama3-8b":
        "{instruction}",   # base model, no chat template
    "plain":
        "{instruction}",
    "llama3-8b-instruct":
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "llama2-7b-chat":
        "[INST] {instruction} [/INST]",
    "Qwen2-7B-Instruct":
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    "Qwen2.5-7B-Instruct":
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    "gemma-7b-it":
        "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
    "mistral-7b-instruct":
        "[INST] {instruction} [/INST]",
}

def _apply_chat_template(text: str, model_family: str) -> str:
    tmpl = _CHAT_TEMPLATES.get(model_family, "{instruction}")
    return tmpl.format(instruction=text)


# ---------------------------------------------------------------------------
# Step 1 — Mean activations (pre-block hook, mirrors LUNAR's generate_directions.py)
# ---------------------------------------------------------------------------

def _mean_activations_at_layer(
    model,
    tokenizer,
    instructions: List[str],
    model_family: str,
    layer_idx: int,
    position: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Returns mean residual-stream activation at position `position` of
    `layer_idx`'s input (pre-block hook), shape [d_model], float32.
    """
    n = len(instructions)
    d = model.config.hidden_size
    acc = torch.zeros(d, dtype=torch.float64, device="cpu")

    block = model.model.layers[layer_idx]

    def _hook(module, inputs):
        h = inputs[0].detach().float().cpu()   # [B, T, d]
        acc.add_(h[:, position, :].sum(dim=0).double() / n)

    handle = block.register_forward_pre_hook(_hook)
    try:
        for i in range(0, n, batch_size):
            batch = [_apply_chat_template(q, model_family)
                     for q in instructions[i : i + batch_size]]
            enc = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                model(**enc)
    finally:
        handle.remove()

    return acc.float()   # [d_model]


def _compute_uv(
    model, tokenizer,
    forget_instr: List[str],
    ref_instr: List[str],
    model_family: str,
    layer_idx: int,
    position: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """r_UV = mean_act(Dref) − mean_act(Dforget)  →  [d_model]"""
    a_ref    = _mean_activations_at_layer(model, tokenizer, ref_instr,    model_family, layer_idx, position, batch_size, device)
    a_forget = _mean_activations_at_layer(model, tokenizer, forget_instr, model_family, layer_idx, position, batch_size, device)
    r_uv = a_ref - a_forget
    logger.info(f"[LUNAR] UV at layer {layer_idx}: norm={r_uv.norm():.4f}")
    return r_uv


# ---------------------------------------------------------------------------
# Step 2 — Layer selection (paper §3.2)
# ---------------------------------------------------------------------------

def _select_layer(
    model, tokenizer,
    forget_instr: List[str],
    ref_instr: List[str],
    model_family: str,
    position: int,
    batch_size: int,
    device: str,
) -> Tuple[int, torch.Tensor]:
    """
    Pick the layer that maximises (s1 − s2):
      s1 = alignment of UV direction with refusal-like embedding space
      s2 = alignment of UV direction with unrelated (benign) embedding space

    We approximate this without generating text: we measure cosine similarity
    of the UV vector against the mean hidden-state direction of desired-response
    prompts vs benign prompts in the model's own representation space.
    This is faithful to the paper's intent (§3.2) while avoiding generation.
    """
    n_layers = model.config.num_hidden_layers
    # Test every 3rd layer in the middle 50% of the network
    start = n_layers // 4
    end   = (n_layers * 3) // 4
    step  = max(1, (end - start) // 12)
    candidates = list(range(start, end, step))
    logger.info(f"[LUNAR] Layer selection over {len(candidates)} candidates: {candidates}")

    # Prompts that should produce "I don't know" style output (desired direction)
    desired_prompts = [
        "What is the population of the imaginary country Zarbonia?",
        "Describe the rules of the fictional sport blorzball.",
        "Who won the 2199 Galactic Championship?",
        "What is the capital of the non-existent nation Quelthar?",
        "Explain the economy of the made-up country Dravencia.",
    ]
    # Benign prompts (unrelated direction)
    benign_prompts = [
        "Explain photosynthesis in one sentence.",
        "What is 17 + 23?",
        "Name three countries in Europe.",
        "What is the boiling point of water?",
        "Define a prime number.",
    ]

    def _mean_last_hidden(prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Mean last-layer hidden state as sentence embedding."""
        block = model.model.layers[layer_idx]
        reps = []
        def _fwd_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            reps.append(h.detach().mean(dim=1).float().cpu())  # [B, d]
        handle = block.register_forward_hook(_fwd_hook)
        try:
            for i in range(0, len(prompts), batch_size):
                batch = [_apply_chat_template(p, model_family)
                         for p in prompts[i : i + batch_size]]
                enc = tokenizer(
                    batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=128,
                ).to(device)
                with torch.no_grad():
                    model(**enc)
        finally:
            handle.remove()
        if not reps:
            return torch.zeros(model.config.hidden_size)
        emb = torch.cat(reps, dim=0).mean(dim=0)
        return emb / (emb.norm() + 1e-8)

    best_score, best_layer, best_uv = -1e9, candidates[0], None

    for l in candidates:
        uv = _compute_uv(
            model, tokenizer,
            forget_instr[:32], ref_instr[:32],
            model_family, l, position, batch_size, device,
        )
        uv_norm = uv / (uv.norm() + 1e-8)

        desired_dir = _mean_last_hidden(desired_prompts, l)
        benign_dir  = _mean_last_hidden(benign_prompts,  l)

        # Resize UV to match hidden dim if needed (shouldn't differ but guard)
        d = desired_dir.numel()
        if uv_norm.numel() != d:
            uv_norm = uv_norm[:d] if uv_norm.numel() > d else torch.nn.functional.pad(uv_norm, (0, d - uv_norm.numel()))
            uv_norm = uv_norm / (uv_norm.norm() + 1e-8)

        s1 = float(torch.dot(uv_norm, desired_dir).abs())
        s2 = float(torch.dot(uv_norm, benign_dir).abs())
        score = s1 - s2
        logger.info(f"[LUNAR] Layer {l:2d}  s1={s1:.4f}  s2={s2:.4f}  score={score:.4f}")

        if score > best_score:
            best_score = score
            best_layer = l
            best_uv    = uv

    logger.info(f"[LUNAR] → Selected layer {best_layer}  (score={best_score:.4f})")
    return best_layer, best_uv


# ---------------------------------------------------------------------------
# Step 3 — Collect hidden states for weight solve
# ---------------------------------------------------------------------------

def _collect_pre_down_proj(
    model, tokenizer,
    instructions: List[str],
    model_family: str,
    layer_idx: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Collect inputs to down_proj (= gate * up_proj output) for all tokens.
    Returns [N_tokens, d_inner], float32 on CPU.
    """
    states = []
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        lambda m, inp: states.append(inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1]))
    )
    try:
        for i in range(0, len(instructions), batch_size):
            batch = [_apply_chat_template(q, model_family)
                     for q in instructions[i : i + batch_size]]
            enc = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                model(**enc)
    finally:
        handle.remove()
    return torch.cat(states, dim=0) if states else torch.zeros(0)


def _collect_mlp_outputs(
    model, tokenizer,
    instructions: List[str],
    model_family: str,
    layer_idx: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Collect MLP block outputs for all tokens.
    Returns [N_tokens, d_model], float32 on CPU.
    """
    outs = []
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(
        lambda m, inp, out: outs.append(out.detach().float().cpu().reshape(-1, out.shape[-1]))
    )
    try:
        for i in range(0, len(instructions), batch_size):
            batch = [_apply_chat_template(q, model_family)
                     for q in instructions[i : i + batch_size]]
            enc = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                model(**enc)
    finally:
        handle.remove()
    return torch.cat(outs, dim=0) if outs else torch.zeros(0)


# ---------------------------------------------------------------------------
# Step 4 — Closed-form weight solve (Eq. 9 from the paper)
# ---------------------------------------------------------------------------

def _solve_closed_form(
    H_forget: torch.Tensor,   # [Nf, d_inner]
    H_retain: torch.Tensor,   # [Nr, d_inner]
    A_forget: torch.Tensor,   # [Nf, d_model]  redirected targets
    A_retain: torch.Tensor,   # [Nr, d_model]  unchanged (original)
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """
    Wc = ([Hf; Hr]^T [Hf; Hr] + λI)^{-1} [Hf; Hr]^T [Af; Ar]
    Returns Wc of shape [d_inner, d_model].
    HF stores down_proj as [d_model, d_inner], so caller transposes before patching.
    """
    H = torch.cat([H_forget, H_retain], dim=0).double()
    A = torch.cat([A_forget, A_retain], dim=0).double()
    logger.info(f"[LUNAR] Closed-form solve: H={list(H.shape)}  A={list(A.shape)}  λ={lambda_reg}")
    G = H.T @ H + lambda_reg * torch.eye(H.shape[1], dtype=torch.float64)
    try:
        G_inv = torch.linalg.inv(G)
    except Exception:
        logger.warning("[LUNAR] linalg.inv failed, falling back to pinv")
        G_inv = torch.linalg.pinv(G)
    Wc = G_inv @ (H.T @ A)   # [d_inner, d_model]
    logger.info(f"[LUNAR] Solved Wc: shape={list(Wc.shape)}  norm={Wc.norm():.4f}")
    return Wc.float()


# ---------------------------------------------------------------------------
# Method class
# ---------------------------------------------------------------------------

@METHOD_REGISTRY.register("lunar")
class LUNARMethod(UnlearningMethod):
    """
    LUNAR: LLM Unlearning via Neural Activation Redirection (NeurIPS 2025).

    Input  : prompts.jsonl (KIF format)
    Output : merged HF model saved to <output_dir>/unlearned_model/
    Eval   : Module 8 (SMR + EL10) — same as KIF evaluation
    """

    name = "lunar"

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        super().__init__(config_overrides)
        self.cfg = LUNARConfig()
        for k, v in (config_overrides or {}).items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                logger.warning(f"[LUNAR] Unknown config key ignored: '{k}'")

    def run(
        self,
        prompts_jsonl: str,
        output_dir: str,
        model_dir: str,
        subjects: Optional[List[str]],
    ) -> UnlearningResult:
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # ── 1. Subjects ────────────────────────────────────────────────
        all_subjects = self.load_subjects(prompts_jsonl)
        forget_subjects = subjects if subjects is not None else all_subjects

        if cfg.max_forget_subjects:
            forget_subjects = forget_subjects[: cfg.max_forget_subjects]

        logger.info(f"[LUNAR] Forget subjects ({len(forget_subjects)}): {forget_subjects}")

        # ── 2. Build instruction lists from prompts.jsonl ───────────────
        forget_qa    = self.get_forget_qa(prompts_jsonl, forget_subjects,
                                          max_per_subject=cfg.max_per_subject)
        forget_instr = [r["question"] for r in forget_qa if r["question"]]
        forget_instr = forget_instr[: cfg.n_forget]

        if not forget_instr:
            raise ValueError("[LUNAR] No forget instructions found in prompts.jsonl!")

        # ── Retain set + Dref — two paths ──────────────────────────────
        #
        # Path A (has retain subjects in prompts.jsonl):
        #   retain_instr = prompts for non-forget subjects
        #   ref_instr    = subset of retain_instr
        #   → UV points forget activations toward how the model talks about
        #     subjects it should keep.
        #
        # Path B (all subjects are forgotten — your musician dataset):
        #   retain_instr = UNVERIFIABLE_PROMPTS (repeated to fill n_retain)
        #   ref_instr    = same unverifiable prompts
        #   → UV points forget activations toward the model's natural
        #     "I don't know / cannot find this information" region.
        #     This is exactly LUNAR §6.2's unverifiable-prompts strategy.
        #
        if cfg.retain_subjects is not None:
            retain_subjects = cfg.retain_subjects
        else:
            forget_set      = set(forget_subjects)
            retain_subjects = [s for s in all_subjects if s not in forget_set]

        if retain_subjects:
            # Path A
            logger.info(
                f"[LUNAR] Path A — retain subjects ({len(retain_subjects)}): "
                f"{retain_subjects}"
            )
            retain_qa    = self.get_forget_qa(prompts_jsonl, retain_subjects,
                                              max_per_subject=cfg.max_per_subject)
            retain_instr = [r["question"] for r in retain_qa if r["question"]]
            retain_instr = retain_instr[: cfg.n_retain]
            dref_source  = "retain_subjects_from_prompts_jsonl"
        else:
            # Path B — all subjects are being forgotten
            logger.info(
                "[LUNAR] Path B — all subjects are forget targets. "
                "No retain subjects in prompts.jsonl. "
                "Using UNVERIFIABLE_PROMPTS as retain set and Dref "
                "(LUNAR §6.2 unverifiable-prompts strategy)."
            )
            # Repeat to fill n_retain slots
            reps         = (cfg.n_retain // len(UNVERIFIABLE_PROMPTS)) + 1
            retain_instr = (UNVERIFIABLE_PROMPTS * reps)[: cfg.n_retain]
            dref_source  = "unverifiable_prompts_fallback"

        ref_instr = retain_instr[: cfg.n_ref]   # Dref ⊆ retain_instr

        logger.info(
            f"[LUNAR] Forget: {len(forget_instr)}  "
            f"Retain: {len(retain_instr)}  "
            f"Dref: {len(ref_instr)}  "
            f"Source: {dref_source}"
        )

        # ── 3. Load model ───────────────────────────────────────────────
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"[LUNAR] Loading model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

        # ── 4. Layer selection + UV ─────────────────────────────────────
        if cfg.auto_select_layer:
            layer_idx, r_uv = _select_layer(
                model, tokenizer,
                forget_instr, ref_instr,
                cfg.model_family, cfg.positions,
                cfg.batch_size, cfg.device,
            )
        else:
            layer_idx = cfg.layer_modified
            r_uv = _compute_uv(
                model, tokenizer,
                forget_instr, ref_instr,
                cfg.model_family, layer_idx,
                cfg.positions, cfg.batch_size, cfg.device,
            )

        # ── 5. Collect hidden states ────────────────────────────────────
        logger.info("[LUNAR] Collecting pre-down-proj states (forget)...")
        H_forget = _collect_pre_down_proj(
            model, tokenizer, forget_instr,
            cfg.model_family, layer_idx, cfg.device, cfg.batch_size,
        )
        logger.info("[LUNAR] Collecting pre-down-proj states (retain)...")
        H_retain = _collect_pre_down_proj(
            model, tokenizer, retain_instr,
            cfg.model_family, layer_idx, cfg.device, cfg.batch_size,
        )

        # ── 6. Build target activation matrices ─────────────────────────
        logger.info("[LUNAR] Collecting MLP outputs (forget)...")
        A_forget_origin = _collect_mlp_outputs(
            model, tokenizer, forget_instr,
            cfg.model_family, layer_idx, cfg.device, cfg.batch_size,
        )
        logger.info("[LUNAR] Collecting MLP outputs (retain)...")
        A_retain_origin = _collect_mlp_outputs(
            model, tokenizer, retain_instr,
            cfg.model_family, layer_idx, cfg.device, cfg.batch_size,
        )

        # Forget target = original + raw UV  (the redirect, no scaling)
        # Retain target = original           (unchanged)
        r_uv_cpu = r_uv.float().cpu()
        A_forget_target = A_forget_origin + r_uv_cpu.unsqueeze(0)
        A_retain_target = A_retain_origin

        # ── 7. Solve for new down_proj weight ───────────────────────────
        if cfg.use_closed_form:
            Wc = _solve_closed_form(
                H_forget, H_retain,
                A_forget_target, A_retain_target,
                lambda_reg=cfg.lambda_reg,
            )
        else:
            raise NotImplementedError(
                "SGD optimisation path not implemented in this version. "
                "Set use_closed_form=True (default)."
            )

        # ── 8. Patch the model ──────────────────────────────────────────
        updated_model = copy.deepcopy(model)
        orig_dtype = updated_model.model.layers[layer_idx].mlp.down_proj.weight.dtype
        # HF stores weight as [d_out, d_in] = [d_model, d_inner]
        # Wc is [d_inner, d_model], so we transpose
        with torch.no_grad():
            updated_model.model.layers[layer_idx].mlp.down_proj.weight.copy_(
                Wc.T.to(orig_dtype)
            )
        logger.info(f"[LUNAR] Patched down_proj at layer {layer_idx}.")

        # ── 9. Save merged model ────────────────────────────────────────
        save_path = str(Path(output_dir) / "unlearned_model")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"[LUNAR] Saving merged model → {save_path}")
        updated_model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

        # ── 10. Save run metadata ───────────────────────────────────────
        meta = {
            "method": "lunar",
            "layer_idx": layer_idx,
            "auto_select_layer": cfg.auto_select_layer,
            "uv_norm": float(r_uv.norm()),
            "lambda_reg": cfg.lambda_reg,
            "model_family": cfg.model_family,
            "forget_subjects": forget_subjects,
            "retain_subjects": retain_subjects if retain_subjects else [],
            "n_forget_instr": len(forget_instr),
            "n_retain_instr": len(retain_instr),
            "model_dir": model_dir,
            "dref_source": dref_source,
        }
        (Path(output_dir) / "lunar_meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        logger.info(f"[LUNAR] Metadata saved → {output_dir}/lunar_meta.json")

        return UnlearningResult(
            merged_model_dir=save_path,
            subjects=forget_subjects,
        )
