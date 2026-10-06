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
"""A tiny random-weight MiMo-V2 checkpoint in the NVFP4 export's format.

Four decoder layers: layer 0 is full attention with the dense MLP, layers 1
and 2 are sliding-window attention with sinks and the MoE, and layer 3 is
full attention with the MoE. Each head has the real model's shape (Q/K 192,
V 128, RoPE over the first 64 dims, a 128-key window); the widths are small.
Dense weights are F32 values that FP8 with 128x128 block scales represents
exactly, and the experts are NVFP4 codes whose scale pairs are MXFP4 scales,
so the weight adapter's repacks run as they do on the real export.

The weights are scaled so that each sublayer adds a fraction of the residual
stream: random weights at unit gain amplify rounding until BF16 alone flips
a tenth of the top-1 tokens. A window off by one changes one attention input
in 128, which that noise would hide, so the long prompt carries a beacon:
token :data:`BEACON`, at row :data:`BEACON_ROW`, has a key that every query
of a sliding layer attends to almost wholly, in dims RoPE leaves alone. Row
``BEACON_ROW + 128`` is the first a 128-key window hides it from, and a
129-key window would not.

The weights come from a seeded generator and depend on nothing but numpy,
so the serving numerics test and the golden generator build the same
checkpoint; :func:`digest` pins it.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

VOCAB, HIDDEN, HEADS = 256, 256, 16
FULL_KV_HEADS, SLIDING_KV_HEADS = 4, 8
HEAD_DIM, V_HEAD_DIM, ROTARY_DIM = 192, 128, 64
DENSE_DIM, MOE_DIM, EXPERTS, TOP_K = 512, 512, 32, 8
LAYERS = [(0, 0), (1, 1), (1, 1), (0, 1)]
"""(sliding, MoE) per decoder layer."""
SLIDING_WINDOW = 128

PROMPTS = {"long": 200, "short": 48}
"""Prompt lengths. From row 128 of the long one on, a sliding layer drops
the oldest key at each row."""

BEACON, BEACON_ROW = 3, 20
BEACON_DIM, CONSTANT_DIM = 7, 11
"""The hidden dim only the beacon's embedding has, and the one every other
token's embedding holds at 1."""

Tensors = dict[str, tuple[str, np.ndarray]]
"""Checkpoint tensors: safetensors dtype and the raw array."""


def config() -> dict[str, Any]:
    """Returns the checkpoint's ``config.json``."""
    return {
        "architectures": ["MiMoV2ForCausalLM"],
        "model_type": "mimo_v2",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE_DIM,
        "num_hidden_layers": len(LAYERS),
        "hybrid_layer_pattern": [sliding for sliding, _ in LAYERS],
        "moe_layer_freq": [moe for _, moe in LAYERS],
        "num_attention_heads": HEADS,
        "swa_num_attention_heads": HEADS,
        "num_key_value_heads": FULL_KV_HEADS,
        "swa_num_key_value_heads": SLIDING_KV_HEADS,
        "head_dim": HEAD_DIM,
        "swa_head_dim": HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "swa_v_head_dim": V_HEAD_DIM,
        "partial_rotary_factor": 0.334,
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "rope_parameters": {
            "partial_rotary_factor": 0.334,
            "rope_theta": 10000000.0,
            "rope_type": "default",
            "type": "default",
        },
        "max_position_embeddings": 4096,
        "sliding_window": SLIDING_WINDOW,
        "attention_value_scale": 0.707,
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True,
        "attention_bias": False,
        "attention_projection_layout": "fused_qkv",
        "layernorm_epsilon": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": MOE_DIM,
        "n_shared_experts": None,
        "norm_topk_prob": True,
        "routed_scaling_factor": None,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "torch_dtype": "bfloat16",
        "conversion_metadata": {"qkv_layout": "global_q_k_v"},
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "quantized_layers": {
                f"model.layers.{i}.mlp.experts.{e}.{proj}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for i, (_, moe) in enumerate(LAYERS)
                if moe
                for e in range(EXPERTS)
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
        },
    }


def _e4m3_values() -> np.ndarray:
    """E4M3FN byte -> value; the NaN bytes are NaN."""
    codes = np.arange(256)
    exp, man = (codes >> 3) & 0xF, codes & 7
    magnitude = np.where(
        exp == 0, man * 2.0**-9, (1 + man / 8) * 2.0 ** (exp - 7)
    )
    magnitude[(codes & 0x7F) == 0x7F] = np.nan
    return (np.where(codes & 0x80, -1.0, 1.0) * magnitude).astype(np.float32)


_E4M3 = _e4m3_values()


def _fp8_exact(weight: np.ndarray) -> np.ndarray:
    """Rounds ``[N, K]`` to values FP8 E4M3 with 128x128 block scales holds.

    Each block's scale is a power of two and its largest element becomes
    448 times it, so the block's ``amax / 448`` is its scale exactly.
    """
    rows, cols = weight.shape
    tiles = weight.reshape(rows // 128, 128, cols // 128, 128)
    amax = np.abs(tiles).max(axis=(1, 3), keepdims=True)
    scale = np.exp2(np.round(np.log2(amax / 448))).astype(np.float32)
    x = (tiles / amax * 448).astype(np.float32)
    positive = _E4M3[:127]
    above = np.clip(np.searchsorted(positive, np.abs(x)), 1, 126)
    below = above - 1
    nearest = np.where(
        positive[above] - np.abs(x) < np.abs(x) - positive[below], above, below
    )
    # No negative zeros: the adapter's exactness check compares bits.
    sign = np.where((x < 0) & (nearest > 0), 0x80, 0)
    codes = (nearest | sign).astype(np.uint8)
    return (_E4M3[codes] * scale).reshape(rows, cols)


def _bf16(x: np.ndarray) -> tuple[str, np.ndarray]:
    """``x`` rounded to BF16, nearest even, as raw ``uint16``."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return "BF16", rounded.astype(np.uint16)


def _qkv_proj(rng: np.random.Generator, sliding: bool) -> np.ndarray:
    """An F32 ``qkv_proj`` in the export's global ``[Q; K; V]`` order whose
    chunk-order blocks, pad rows included, are FP8-exact.

    A sliding layer's queries add a constant from :data:`CONSTANT_DIM`, and
    its keys the beacon's :data:`BEACON_DIM`, along the same direction in
    each head's unrotated dims, which gives the beacon's key an attention
    logit of about 8 from every query.
    """
    chunks = FULL_KV_HEADS
    kv_heads = SLIDING_KV_HEADS if sliding else FULL_KV_HEADS
    q = HEADS * HEAD_DIM // chunks
    k = kv_heads * HEAD_DIM // chunks
    v = kv_heads * V_HEAD_DIM // chunks
    rows = q + k + v
    padded = -(-rows // 128) * 128
    weight = np.zeros((chunks, padded, HIDDEN), np.float32)
    weight[:, :rows] = rng.normal(0, 0.0625, (chunks, rows, HIDDEN))
    if sliding:
        queries = weight[:, :q].reshape(chunks, -1, HEAD_DIM, HIDDEN)
        queries[:, :, ROTARY_DIM:, CONSTANT_DIM] = 0.25
        keys = weight[:, q : q + k].reshape(chunks, -1, HEAD_DIM, HIDDEN)
        keys[:, :, ROTARY_DIM:, BEACON_DIM] = 0.22
    exact = _fp8_exact(weight.reshape(-1, HIDDEN)).reshape(weight.shape)
    return np.concatenate(
        [
            exact[:, :q].reshape(-1, HIDDEN),
            exact[:, q : q + k].reshape(-1, HIDDEN),
            exact[:, q + k : rows].reshape(-1, HIDDEN),
        ]
    )


def _nvfp4_expert(
    rng: np.random.Generator, rows: int, cols: int
) -> dict[str, tuple[str, np.ndarray]]:
    """Random E2M1 codes with MXFP4 scales of 2^-7 or 2^-6, stored as the
    export stores them: each E8M0 scale as two E4M3 scales under a global
    ``weight_scale_2`` of 2^-8."""
    codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    exponents = rng.integers(-7, -5, (rows, cols // 32))
    e4m3 = ((exponents + 8 + 7) << 3).astype(np.uint8)
    return {
        "weight": ("U8", codes),
        "weight_scale": ("F8_E4M3", np.repeat(e4m3, 2, axis=-1)),
        "weight_scale_2": ("F32", np.array([2.0**-8], np.float32)),
    }


def tensors() -> Tensors:
    """Returns the checkpoint's tensors, the same on every call."""
    rng = np.random.default_rng(20260929)
    embed = rng.normal(0, 1, (VOCAB, HIDDEN))
    embed[:, BEACON_DIM] = 0
    embed[:, CONSTANT_DIM] = 1
    embed[BEACON] = 0
    embed[BEACON, BEACON_DIM] = 16
    out: Tensors = {
        "model.embed_tokens.weight": _bf16(embed),
        "model.norm.weight": _bf16(rng.normal(1, 0.1, HIDDEN)),
        "lm_head.weight": _bf16(rng.normal(0, 0.1, (VOCAB, HIDDEN))),
    }
    for i, (sliding, moe) in enumerate(LAYERS):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = _bf16(rng.normal(1, 0.1, HIDDEN))
        out[p + "post_attention_layernorm.weight"] = _bf16(
            rng.normal(1, 0.1, HIDDEN)
        )
        out[p + "self_attn.qkv_proj.weight"] = (
            "F32",
            _qkv_proj(rng, bool(sliding)),
        )
        out[p + "self_attn.o_proj.weight"] = _bf16(
            rng.normal(0, 0.03, (HIDDEN, HEADS * V_HEAD_DIM))
        )
        if sliding:
            out[p + "self_attn.attention_sink_bias"] = _bf16(
                rng.normal(1, 1, HEADS)
            )
        if not moe:
            for proj, shape, std in (
                ("gate_proj", (DENSE_DIM, HIDDEN), 0.0625),
                ("up_proj", (DENSE_DIM, HIDDEN), 0.0625),
                ("down_proj", (HIDDEN, DENSE_DIM), 0.02),
            ):
                out[p + f"mlp.{proj}.weight"] = (
                    "F32",
                    _fp8_exact(rng.normal(0, std, shape).astype(np.float32)),
                )
            continue
        out[p + "mlp.gate.weight"] = _bf16(
            rng.normal(0, 0.0625, (EXPERTS, HIDDEN))
        )
        out[p + "mlp.gate.e_score_correction_bias"] = (
            "F32",
            rng.normal(0, 0.2, EXPERTS).astype(np.float32),
        )
        for e in range(EXPERTS):
            for proj, (n, k) in (
                ("gate_proj", (MOE_DIM, HIDDEN)),
                ("up_proj", (MOE_DIM, HIDDEN)),
                ("down_proj", (HIDDEN, MOE_DIM)),
            ):
                base = f"{p}mlp.experts.{e}.{proj}."
                for field, value in _nvfp4_expert(rng, n, k).items():
                    out[base + field] = value
    return out


def prompts() -> dict[str, list[int]]:
    """Returns the token ids of each prompt.

    Only a prompt long enough to reach the beacon's window edge holds the
    beacon, once, at :data:`BEACON_ROW`.
    """
    rng = np.random.default_rng(1)
    out = {}
    for name, length in PROMPTS.items():
        ids = rng.integers(0, VOCAB, length)
        ids[ids == BEACON] = BEACON + 1
        if length > BEACON_ROW + SLIDING_WINDOW:
            ids[BEACON_ROW] = BEACON
        out[name] = ids.tolist()
    return out


def digest(checkpoint: Tensors) -> str:
    """Returns a SHA-256 over every tensor's name, dtype, shape and bytes."""
    h = hashlib.sha256()
    for name in sorted(checkpoint):
        dtype, array = checkpoint[name]
        h.update(f"{name}:{dtype}:{list(array.shape)}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def write(directory: Path, checkpoint: Tensors) -> None:
    """Writes ``config.json`` and ``model.safetensors`` to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config(), indent=1))
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, array) in checkpoint.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    with open(directory / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for _, array in checkpoint.values():
            f.write(np.ascontiguousarray(array).tobytes())
