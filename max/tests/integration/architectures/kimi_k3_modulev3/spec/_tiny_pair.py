# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""The tiny random-weight Kimi K3 + DSpark pair, as a buildable fixture.

The checkpoints ``test_spec_parity.py`` and the capture test both build
here, over the same seeds and shapes, differing only in the target's MLA
per-head geometry (``LatentGeometry``): the parity fixture serves a narrow
16-wide latent that keeps the target's decompressed cache path under test,
while the capture test serves the production geometry, whose 576-wide
latent runs the same fused latent kernels the real K3 does and reads the
prefill-plan extent it guards. The two targets' streams are not
comparable, and no test compares them.

The export-contract fixture (``mach/tools/gen-mef``) builds here too, at
the production latent widths and the production KDA widths
(``PRODUCTION_KDA``, 96 heads of width 128) with the published drafter's
depth, because the frozen Inferact export contract pins those widths while
a reduced layer count still keeps its checkpoint tiny. The default widths
(``KDA``, two layers) stay what the numerics and parity tests build.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HIDDEN = 64
VOCAB = 256
MASK_TOKEN_ID = VOCAB - 2
EOS_TOKEN_ID = VOCAB - 1
LAYERS = 4
KDA_LAYERS = [1, 2, 3]
"""1-indexed, the checkpoint's own convention; KDA layers 0-2 0-based."""
FULL_LAYERS = [4]
TAPS = [1, 2]
"""The drafter's ``target_layer_ids``, 0-based post-layer taps, held to
the tiny target's depth."""
DRAFT_LAYERS = 2
DRAFT_INTERMEDIATE = 96
INTERMEDIATE = 48
EXPERTS = 64
EXPERT_LATENT = 24
EXPERT_INTERMEDIATE = 16
KDA_HEADS = 2
KDA_HEAD_DIM = 32
KDA_CONV_KERNEL = 4
MLA_HEADS = 4
Q_LORA = 16
MARKOV_RANK = 8
MAX_LENGTH = 512
DRAFT_WIDTH = 7

# The drafter's head geometry is the published one: the fused MLA kernels
# its latent path runs on fix the latent width at 576 (kv_lora 512 + rope
# 64), so whatever the target's latent width is, the drafter runs the same
# fused latent path the production K3 does.
DRAFT_KV_LORA = 512
DRAFT_QK_NOPE = 128
DRAFT_QK_ROPE = 64
DRAFT_V_DIM = 128


@dataclass(frozen=True)
class LatentGeometry:
    """The target's per-head MLA geometry, the fixture's one dial."""

    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_dim: int


@dataclass(frozen=True)
class KdaWidths:
    """The KDA state's head geometry, the fixture's second dial."""

    heads: int
    head_dim: int


FUSED_LATENT = LatentGeometry(kv_lora=512, qk_nope=128, qk_rope=64, v_dim=128)
"""The production per-head geometry: the target's verify instantiates the
same fused latent kernels the real K3 does."""


NARROW_LATENT = LatentGeometry(kv_lora=12, qk_nope=8, qk_rope=4, v_dim=8)
"""A 16-wide latent, which keeps the target's decompressed cache path
under test instead of the fused one."""


KDA = KdaWidths(heads=KDA_HEADS, head_dim=KDA_HEAD_DIM)
"""The stock reduced KDA widths, which keep the state cheap for the
numerics and parity tests."""

PRODUCTION_KDA = KdaWidths(heads=96, head_dim=128)
"""The production KDA widths, 96 heads of width 128: the shared K3 shape
the Inferact export contract pins
(mach/docs/models/kimi-k3-inferact-dspark.md), whose validator refuses a
sidecar recorded from any other state geometry. Heads and conv width both
divide at TP2 (48 heads, 18,432 conv channels per device)."""


def _text_config(geometry: LatentGeometry, kda: KdaWidths) -> dict[str, object]:
    return {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "dtype": "bfloat16",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": MLA_HEADS,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": geometry.kv_lora,
        "qk_nope_head_dim": geometry.qk_nope,
        "qk_rope_head_dim": geometry.qk_rope,
        "v_head_dim": geometry.v_dim,
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "intermediate_size": INTERMEDIATE,
        "routed_expert_hidden_size": EXPERT_LATENT,
        "moe_intermediate_size": EXPERT_INTERMEDIATE,
        "num_experts": EXPERTS,
        "num_experts_per_token": 4,
        # The real K3 ships two, whose shared experts run at twice the MoE
        # width -- the shapes below are written for that.
        "num_shared_experts": 2,
        "rms_norm_eps": 1e-05,
        "activation_situ_beta": 1.5,
        "activation_situ_linear_beta": 1.5,
        "hidden_act": "situ",
        "latent_moe_use_norm": True,
        "moe_renormalize": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": MAX_LENGTH,
        "bos_token_id": 0,
        "eos_token_id": EOS_TOKEN_ID,
        "linear_attn_config": {
            "kda_layers": KDA_LAYERS,
            "full_attn_layers": FULL_LAYERS,
            "num_heads": kda.heads,
            "head_dim": kda.head_dim,
            "short_conv_kernel_size": KDA_CONV_KERNEL,
            "gate_lower_bound": -5.0,
        },
    }


_DRAFT_CONFIG = {
    "architectures": ["K3DSparkModel"],
    "model_type": "k3_dspark",
    "hidden_size": HIDDEN,
    "intermediate_size": DRAFT_INTERMEDIATE,
    "num_hidden_layers": DRAFT_LAYERS,
    "num_attention_heads": MLA_HEADS,
    "num_key_value_heads": MLA_HEADS,
    "q_lora_rank": Q_LORA,
    "kv_lora_rank": DRAFT_KV_LORA,
    "qk_nope_head_dim": DRAFT_QK_NOPE,
    "qk_rope_head_dim": DRAFT_QK_ROPE,
    "v_head_dim": DRAFT_V_DIM,
    "mla_use_nope": False,
    "mla_use_output_gate": False,
    "vocab_size": VOCAB,
    "draft_vocab_size": VOCAB,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": MAX_LENGTH,
    "rope_theta": 50000.0,
    # The tap count and the target's depth, as the real drafter's config
    # names them; MAX's pairing reads the target's depth off the draft
    # config, so both must be present and distinct.
    "num_target_layers": len(TAPS),
    "target_hidden_size": HIDDEN,
    "target_num_hidden_layers": LAYERS,
    "target_layer_ids": TAPS,
    "mask_token_id": MASK_TOKEN_ID,
    "bos_token_id": 0,
    "eos_token_id": EOS_TOKEN_ID,
    "pad_token_id": VOCAB - 3,
    "markov_rank": MARKOV_RANK,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "tie_word_embeddings": False,
    "_torchspec_version": "0.1.0",
    "torch_dtype": "bfloat16",
    # The real drafter's yarn rope, scaled to the tiny context.
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 128,
        "rope_theta": 50000.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}

# The K3 target's remote-code AutoConfig class, as the real checkpoint
# ships; transformers does not know either model_type. The real drafter
# snapshot ships no remote code and no ``auto_map``, so the fixture's
# does not either.
_K3_CONFIGURATION = """
from transformers import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(self, text_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = KimiLinearConfig(**(text_config or {}))
"""


@dataclass(frozen=True)
class TinyPair:
    """A tiny random K3 target, DSpark drafter, and prompts file."""

    target: Path
    drafter: Path
    prompts: Path
    prompt_rows: tuple[list[int], ...]
    """The token-id rows ``prompts`` holds, in file order."""


def _write_tokenizer(path: Path) -> None:
    """A word-level vocabulary of ``t0``..., which no prompt encodes with:
    the harness feeds token ids directly."""
    vocab = {f"t{i}": i for i in range(MASK_TOKEN_ID)}
    vocab.update({"<mask>": MASK_TOKEN_ID, "<eos>": EOS_TOKEN_ID})
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [
                    {
                        "id": token_id,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                    for token, token_id in (
                        ("<mask>", MASK_TOKEN_ID),
                        ("<eos>", EOS_TOKEN_ID),
                    )
                ],
                "normalizer": None,
                "pre_tokenizer": {"type": "WhitespaceSplit"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": "<eos>",
                },
            }
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "pad_token": "<eos>",
                "model_max_length": MAX_LENGTH,
            }
        )
    )


def _weights(
    rng: np.random.Generator, shapes: dict[str, tuple[int, ...]], f32: set[str]
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        ).to(torch.float32 if name in f32 else torch.bfloat16)
        for name, shape in shapes.items()
    }


def _target_shapes(
    geometry: LatentGeometry,
    kda: KdaWidths,
) -> dict[str, tuple[int, ...]]:
    """The checkpoint's tensors, under the HF names the adapter reads."""
    kda_inner = kda.heads * kda.head_dim
    q_b = MLA_HEADS * (geometry.qk_nope + geometry.qk_rope)
    kv_a = geometry.kv_lora + geometry.qk_rope
    kv_b = MLA_HEADS * (geometry.qk_nope + geometry.v_dim)
    o_inner = MLA_HEADS * geometry.v_dim
    shared_intermediate = 2 * EXPERT_INTERMEDIATE
    shapes: dict[str, tuple[int, ...]] = {
        "language_model.model.embed_tokens.weight": (VOCAB, HIDDEN),
        "language_model.lm_head.weight": (VOCAB, HIDDEN),
        "language_model.model.norm.weight": (HIDDEN,),
        "language_model.model.output_attn_res_proj.weight": (1, HIDDEN),
        "language_model.model.output_attn_res_norm.weight": (HIDDEN,),
    }
    for i in range(LAYERS):
        src = f"language_model.model.layers.{i}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attention_res_norm",
            "mlp_res_norm",
        ):
            shapes[f"{src}.{name}.weight"] = (HIDDEN,)
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            shapes[f"{src}.{name}.weight"] = (1, HIDDEN)
        attn = f"{src}.self_attn"
        if i in [layer - 1 for layer in KDA_LAYERS]:
            shapes.update(
                {
                    f"{attn}.q_proj.weight": (kda_inner, HIDDEN),
                    f"{attn}.k_proj.weight": (kda_inner, HIDDEN),
                    f"{attn}.v_proj.weight": (kda_inner, HIDDEN),
                    f"{attn}.q_conv1d.weight": (kda_inner, 1, KDA_CONV_KERNEL),
                    f"{attn}.k_conv1d.weight": (kda_inner, 1, KDA_CONV_KERNEL),
                    f"{attn}.v_conv1d.weight": (kda_inner, 1, KDA_CONV_KERNEL),
                    f"{attn}.f_a_proj.weight": (kda.head_dim, HIDDEN),
                    f"{attn}.f_b_proj.weight": (kda_inner, kda.head_dim),
                    f"{attn}.b_proj.weight": (kda.heads, HIDDEN),
                    f"{attn}.g_proj.weight": (kda_inner, HIDDEN),
                    f"{attn}.o_norm.weight": (kda.head_dim,),
                    f"{attn}.o_proj.weight": (HIDDEN, kda_inner),
                    f"{attn}.A_log": (kda.heads,),
                    f"{attn}.dt_bias": (kda_inner,),
                }
            )
        else:
            shapes.update(
                {
                    f"{attn}.q_a_proj.weight": (Q_LORA, HIDDEN),
                    f"{attn}.q_a_layernorm.weight": (Q_LORA,),
                    f"{attn}.q_b_proj.weight": (q_b, Q_LORA),
                    f"{attn}.kv_a_proj_with_mqa.weight": (kv_a, HIDDEN),
                    f"{attn}.kv_a_layernorm.weight": (geometry.kv_lora,),
                    f"{attn}.kv_b_proj.weight": (kv_b, geometry.kv_lora),
                    f"{attn}.g_proj.weight": (o_inner, HIDDEN),
                    f"{attn}.o_proj.weight": (HIDDEN, o_inner),
                }
            )
        if i < 1:
            shapes.update(
                {
                    f"{src}.mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
                    f"{src}.mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
                    f"{src}.mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
                }
            )
        else:
            moe = f"{src}.block_sparse_moe"
            shapes.update(
                {
                    f"{moe}.gate.weight": (EXPERTS, HIDDEN),
                    f"{moe}.gate.e_score_correction_bias": (EXPERTS,),
                    f"{moe}.routed_expert_down_proj.weight": (
                        EXPERT_LATENT,
                        HIDDEN,
                    ),
                    f"{moe}.routed_expert_up_proj.weight": (
                        HIDDEN,
                        EXPERT_LATENT,
                    ),
                    f"{moe}.routed_expert_norm.weight": (EXPERT_LATENT,),
                    f"{moe}.shared_experts.gate_proj.weight": (
                        shared_intermediate,
                        HIDDEN,
                    ),
                    f"{moe}.shared_experts.up_proj.weight": (
                        shared_intermediate,
                        HIDDEN,
                    ),
                    f"{moe}.shared_experts.down_proj.weight": (
                        HIDDEN,
                        shared_intermediate,
                    ),
                }
            )
            for e in range(EXPERTS):
                for w, shape in (
                    ("w1", (EXPERT_INTERMEDIATE, EXPERT_LATENT)),
                    ("w2", (EXPERT_LATENT, EXPERT_INTERMEDIATE)),
                    ("w3", (EXPERT_INTERMEDIATE, EXPERT_LATENT)),
                ):
                    shapes[f"{moe}.experts.{e}.{w}.weight"] = shape
    return shapes


def _draft_shapes(draft_layers: int) -> dict[str, tuple[int, ...]]:
    q_b = MLA_HEADS * (DRAFT_QK_NOPE + DRAFT_QK_ROPE)
    kv_a = DRAFT_KV_LORA + DRAFT_QK_ROPE
    kv_b = MLA_HEADS * (DRAFT_QK_NOPE + DRAFT_V_DIM)
    o_inner = MLA_HEADS * DRAFT_V_DIM
    shapes: dict[str, tuple[int, ...]] = {
        "context_proj.weight": (HIDDEN, len(TAPS) * HIDDEN),
        "context_norm.weight": (HIDDEN,),
        "embed_tokens.weight": (VOCAB, HIDDEN),
        "final_norm.weight": (HIDDEN,),
        "markov_head.markov_w1.weight": (VOCAB, MARKOV_RANK),
        "markov_head.markov_w2.weight": (VOCAB, MARKOV_RANK),
        "confidence_head.proj.weight": (1, HIDDEN + MARKOV_RANK),
        "confidence_head.proj.bias": (1,),
    }
    for i in range(draft_layers):
        src = f"layers.{i}"
        shapes.update(
            {
                f"{src}.input_layernorm.weight": (HIDDEN,),
                f"{src}.post_attention_layernorm.weight": (HIDDEN,),
                f"{src}.mlp.gate_proj.weight": (DRAFT_INTERMEDIATE, HIDDEN),
                f"{src}.mlp.up_proj.weight": (DRAFT_INTERMEDIATE, HIDDEN),
                f"{src}.mlp.down_proj.weight": (HIDDEN, DRAFT_INTERMEDIATE),
                f"{src}.self_attn.q_a_proj.weight": (Q_LORA, HIDDEN),
                f"{src}.self_attn.q_a_layernorm.weight": (Q_LORA,),
                f"{src}.self_attn.q_b_proj.weight": (q_b, Q_LORA),
                f"{src}.self_attn.kv_a_proj_with_mqa.weight": (kv_a, HIDDEN),
                f"{src}.self_attn.kv_a_layernorm.weight": (DRAFT_KV_LORA,),
                f"{src}.self_attn.kv_b_proj.weight": (kv_b, DRAFT_KV_LORA),
                f"{src}.self_attn.o_proj.weight": (HIDDEN, o_inner),
            }
        )
    return shapes


def _checkpoint(
    path: Path,
    config: dict[str, object],
    shapes: dict[str, tuple[int, ...]],
    f32: set[str],
    rng: np.random.Generator,
    *,
    configuration_module: str | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    if configuration_module is not None:
        (path / "configuration.py").write_text(configuration_module)
    _write_tokenizer(path)
    save_file(_weights(rng, shapes, f32), str(path / "model.safetensors"))
    return path


def _prompt_ids(length: int, seed: int) -> list[int]:
    """One token-id prompt: ordinary ids only, never a special token."""
    rng = np.random.default_rng(seed)
    return [int(t) for t in rng.integers(0, MASK_TOKEN_ID, length)]


def random_prompt_rows(
    prompt_lengths: Sequence[int],
) -> list[list[int]]:
    """One seeded token-id prompt row per requested length."""
    return [_prompt_ids(length, seed=length) for length in prompt_lengths]


def build_tiny_pair(
    base: Path,
    prompt_rows: Sequence[Sequence[int]],
    geometry: LatentGeometry = FUSED_LATENT,
    kda: KdaWidths = KDA,
    draft_layers: int = DRAFT_LAYERS,
) -> TinyPair:
    """Builds the tiny pair with the given token-id prompt rows.

    The weights are drawn from one seeded generator in a fixed order, so
    the same geometry, call path and rows always build byte-identical
    checkpoints. ``kda`` and ``draft_layers`` widen the KDA state and deepen
    the drafter for the export-contract fixture, whose validator pins the
    production KDA widths and the published drafter's five layers.
    """
    rng = np.random.default_rng(0)
    target = _checkpoint(
        base / "target",
        {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "auto_map": {"AutoConfig": "configuration.KimiK3Config"},
            "model_type": "kimi_k3",
            "dtype": "bfloat16",
            "tie_word_embeddings": False,
            "text_config": _text_config(geometry, kda),
        },
        _target_shapes(geometry, kda),
        {
            f"language_model.model.layers.{layer - 1}.self_attn.{name}"
            for layer in KDA_LAYERS
            for name in ("A_log", "dt_bias")
        }
        | {
            f"language_model.model.layers.{index}.block_sparse_moe."
            "gate.e_score_correction_bias"
            for index in range(1, LAYERS)
        },
        rng,
        configuration_module=_K3_CONFIGURATION,
    )
    drafter = _checkpoint(
        base / "drafter",
        {**_DRAFT_CONFIG, "num_hidden_layers": draft_layers},
        _draft_shapes(draft_layers),
        set(),
        rng,
    )
    rows = [list(row) for row in prompt_rows]
    prompts = base / "prompts.txt"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return TinyPair(
        target=target, drafter=drafter, prompts=prompts, prompt_rows=tuple(rows)
    )
