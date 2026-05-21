#!/usr/bin/env python3
"""Patch real_negative_signature_separability.py for the current KIF capsule schema.

Fixes:
1. Use target_module_name as the hook module field.
2. Ignore giant adapter_state_dict arrays.
3. Hook the pre-activation input of gate_proj/up_proj/down_proj so 4096-dim
   signature vectors align with hidden states.
"""
from pathlib import Path

p = Path("analysis/real_negative_signature_separability.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    'MODULE_KEYS = {"target_module", "module", "module_name", "layer_name", "hook_module", "activation_module", "mlp_module"}',
    'MODULE_KEYS = {"target_module", "target_module_name", "module", "module_name", "layer_name", "hook_module", "activation_module", "mlp_module"}'
)

s = s.replace(
    'if arr.size < 64:\n        return None\n    if not np.all(np.isfinite(arr)):',
    'if arr.size < 64:\n        return None\n    if arr.size > 20000:\n        return None\n    if not np.all(np.isfinite(arr)):'
)

start = s.index('@torch.inference_mode()\ndef collect_pooled_activations(')
end = s.index('\ndef cohen_d', start)
new_func = '''@torch.inference_mode()
def collect_pooled_activations(model, tok, prompts: Sequence[str], module_name: str, device: str, batch_size: int, max_length: int) -> np.ndarray:
    resolved_name, module = resolve_module(model, module_name)
    collected: List[torch.Tensor] = []

    def pre_hook_fn(_module, inputs):
        y = inputs[0]
        if not torch.is_tensor(y):
            raise RuntimeError(f"Pre-hook input at {resolved_name} is not a tensor: {type(y)}")
        if y.ndim != 3:
            raise RuntimeError(f"Pre-hook input at {resolved_name} has shape {tuple(y.shape)}, expected [B,T,D]")
        collected.append(y.detach().float().cpu())

    handle = module.register_forward_pre_hook(pre_hook_fn)
    pooled: List[np.ndarray] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = list(prompts[i:i + batch_size])
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            attention = enc["attention_mask"].detach().cpu().float()
            enc = {k: v.to(device) for k, v in enc.items()}
            before = len(collected)
            _ = model(**enc)
            if len(collected) != before + 1:
                raise RuntimeError(f"Expected one pre-hook capture, got {len(collected) - before}")
            acts = collected[-1]
            if acts.shape[:2] != attention.shape:
                min_t = min(acts.shape[1], attention.shape[1])
                acts = acts[:, -min_t:, :]
                attention = attention[:, -min_t:]
            denom = attention.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled_batch = (acts * attention.unsqueeze(-1)).sum(dim=1) / denom
            pooled.extend([x.numpy().astype(np.float32) for x in pooled_batch])
    finally:
        handle.remove()
    if not pooled:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(pooled, axis=0)

'''
s = s[:start] + new_func + s[end+1:]

p.write_text(s, encoding="utf-8")
print(f"Patched {p}")
