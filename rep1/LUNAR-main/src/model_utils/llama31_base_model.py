# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for KIF apple-to-apple evaluation on meta-llama/Llama-3.1-8B base.

from __future__ import annotations

import functools
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_utils.model_base import ModelBase
from src.utils.utils import get_orthogonalized_matrix


def format_instruction_llama31_base(
    instruction: str,
    output: str | None = None,
    include_trailing_whitespace: bool = False,
) -> str:
    """Plain base-model prompt formatter.

    For an apple-to-apple KIF comparison we deliberately do NOT wrap prompts in an
    instruction/chat template. KIF Module 8 evaluates raw entity prompts, so LUNAR
    activation direction mining and down-projection fitting should see the same
    prompt distribution.
    """
    formatted = str(instruction).strip()
    if include_trailing_whitespace:
        formatted = formatted + "\n"
    if output is not None:
        formatted = formatted + str(output)
    return formatted


def tokenize_instructions_llama31_base(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] | None = None,
    include_trailing_whitespace: bool = False,
):
    if outputs is not None:
        prompts = [
            format_instruction_llama31_base(
                instruction=instruction,
                output=output,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama31_base(
                instruction=instruction,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def orthogonalize_llama31_base_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_llama31_base_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Llama31BaseModel(ModelBase):
    """LUNAR wrapper for meta-llama/Llama-3.1-8B base, not Instruct.

    This wrapper keeps prompts raw rather than applying a chat template. That is
    the important change for fair KIF dual-metric evaluation.
    """

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_llama31_base,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=False,
        )

    def _get_eoi_toks(self):
        # LUNAR uses len(eoi_toks) only to choose final-token positions.
        # For raw base prompts, a single final-token position is desired.
        return [self.tokenizer.eos_token_id]

    def _get_refusal_toks(self):
        ids = self.tokenizer.encode("I", add_special_tokens=False)
        return ids[:1] if ids else [40]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama31_base_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_llama31_base_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
        # If the model is loaded with device_map="auto", forcing .to("cuda") can
        # break accelerate's placement. Keep this method conservative.
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        return self

    def _eval(self):
        self.model.eval()
        return self

    def _forward(self, batch):
        return self.model(**batch)

    def _device(self):
        return next(self.model.parameters()).device.type

    def _generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        max_new_tokens,
        do_sample,
        num_beams,
        num_return_sequences,
        use_cache,
        pad_token_id,
        output_scores,
        return_dict_in_generate,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
        )

    def _save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory, safe_serialization=True)
        self.tokenizer.save_pretrained(save_directory)
