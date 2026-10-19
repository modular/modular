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
"""A tiny random-weight MiMo-V2 checkpoint in Xiaomi's format.

Four decoder layers: layer 0 is full attention with the dense MLP, layers 1
and 2 are sliding-window attention with sinks and the MoE, and layer 3 is
full attention with the MoE. Each head has the real model's shape (Q/K 192,
V 128, RoPE over the first 64 dims, a 128-key window); the widths are small.
Dense projections are FP8 E4M3 with float32 128x128 block scales, and
``qkv_proj`` is in Xiaomi's chunk order with each full-attention chunk's
scales padded to whole blocks, as in the real checkpoint. The experts are
MXFP4: E2M1 codes with one E8M0 scale per 32 elements.

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

import ml_dtypes
import numpy as np

VOCAB, HIDDEN, HEADS = 256, 256, 16
FULL_KV_HEADS, SLIDING_KV_HEADS = 4, 8
HEAD_DIM, V_HEAD_DIM, ROTARY_DIM = 192, 128, 64
DENSE_DIM, MOE_DIM, EXPERTS, TOP_K = 512, 512, 32, 8
LAYERS = [(0, 0), (1, 1), (1, 1), (0, 1)]
"""(sliding, MoE) per decoder layer."""
SLIDING_WINDOW = 128
CHUNKS = FULL_KV_HEADS
"""``qkv_proj`` chunks, one per full-attention KV head, as in Xiaomi's."""

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
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
            "ignored_layers": [
                f"model.layers.{i}.self_attn.o_proj" for i in range(len(LAYERS))
            ],
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


def fp8_blocks(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes ``[N, K]`` to E4M3 codes with float32 128x128 block scales.

    Each block's scale is its ``amax / 448``, as in the real checkpoint.
    """
    rows, cols = weight.shape
    tiles = weight.reshape(rows // 128, 128, cols // 128, 128)
    scale = (np.abs(tiles).max(axis=(1, 3), keepdims=True) / 448).astype(
        np.float32
    )
    x = np.abs(tiles / scale)
    positive = _E4M3[:127]
    above = np.clip(np.searchsorted(positive, x), 1, 126)
    below = above - 1
    nearest = np.where(positive[above] - x < x - positive[below], above, below)
    sign = np.where((tiles < 0) & (nearest > 0), 0x80, 0)
    codes = (nearest | sign).astype(np.uint8).reshape(rows, cols)
    return codes, scale.reshape(rows // 128, cols // 128)


def fp8_decode(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Returns the float32 values of ``[N, K]`` codes under block scales."""
    grid = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
    return _E4M3[codes] * grid[: codes.shape[0], : codes.shape[1]]


def chunked_fp8_blocks(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes a ``[chunks, rows, K]`` ``qkv_proj`` as Xiaomi stores it.

    Each chunk's scales cover it padded to whole 128-row blocks; the codes
    hold only its real rows, ``[chunks * rows, K]``.
    """
    chunks, rows, cols = weight.shape
    padded = -(-rows // 128) * 128
    full = np.zeros((chunks, padded, cols), np.float32)
    full[:, :rows] = weight
    codes, scales = fp8_blocks(full.reshape(-1, cols))
    codes = codes.reshape(chunks, padded, cols)[:, :rows]
    return codes.reshape(-1, cols), scales


def chunked_fp8_decode(
    codes: np.ndarray, scales: np.ndarray, chunks: int
) -> np.ndarray:
    """Returns the ``[chunks, rows, K]`` values of a stored ``qkv_proj``."""
    rows, cols = codes.shape[0] // chunks, codes.shape[1]
    padded = -(-rows // 128) * 128
    full = np.zeros((chunks, padded, cols), np.uint8)
    full[:, :rows] = codes.reshape(chunks, rows, cols)
    values = fp8_decode(full.reshape(-1, cols), scales)
    return values.reshape(chunks, padded, cols)[:, :rows]


def _bf16(x: np.ndarray) -> tuple[str, np.ndarray]:
    """``x`` rounded to BF16, nearest even, as raw ``uint16``."""
    rounded = (
        np.asarray(x, dtype=np.float32)
        .astype(ml_dtypes.bfloat16)
        .view(np.uint16)
    )
    return "BF16", rounded


def qkv_rows(sliding: bool) -> tuple[int, int, int]:
    """Returns the ``(q, k, v)`` rows of one ``qkv_proj`` chunk."""
    kv_heads = SLIDING_KV_HEADS if sliding else FULL_KV_HEADS
    return (
        HEADS * HEAD_DIM // CHUNKS,
        kv_heads * HEAD_DIM // CHUNKS,
        kv_heads * V_HEAD_DIM // CHUNKS,
    )


def _qkv_proj(
    rng: np.random.Generator, sliding: bool
) -> tuple[np.ndarray, np.ndarray]:
    """A ``qkv_proj`` in chunk order: codes and scales as Xiaomi stores them.

    A sliding layer's queries add a constant from :data:`CONSTANT_DIM`, and
    its keys the beacon's :data:`BEACON_DIM`, along the same direction in
    each head's unrotated dims, which gives the beacon's key an attention
    logit of about 8 from every query.
    """
    q, k, v = qkv_rows(sliding)
    weight = rng.normal(0, 0.0625, (CHUNKS, q + k + v, HIDDEN))
    if sliding:
        queries = weight[:, :q].reshape(CHUNKS, -1, HEAD_DIM, HIDDEN)
        queries[:, :, ROTARY_DIM:, CONSTANT_DIM] = 0.25
        keys = weight[:, q : q + k].reshape(CHUNKS, -1, HEAD_DIM, HIDDEN)
        keys[:, :, ROTARY_DIM:, BEACON_DIM] = 0.22
    return chunked_fp8_blocks(weight.astype(np.float32))


def _mxfp4_expert(
    rng: np.random.Generator, rows: int, cols: int
) -> dict[str, tuple[str, np.ndarray]]:
    """Random E2M1 codes with E8M0 scales of 2^-7 or 2^-6."""
    codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    exponents = rng.integers(-7, -5, (rows, cols // 32))
    return {
        "weight": ("U8", codes),
        "weight_scale": ("U8", (exponents + 127).astype(np.uint8)),
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
        codes, scales = _qkv_proj(rng, bool(sliding))
        out[p + "self_attn.qkv_proj.weight"] = ("F8_E4M3", codes)
        out[p + "self_attn.qkv_proj.weight_scale_inv"] = ("F32", scales)
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
                codes, scales = fp8_blocks(
                    rng.normal(0, std, shape).astype(np.float32)
                )
                out[p + f"mlp.{proj}.weight"] = ("F8_E4M3", codes)
                out[p + f"mlp.{proj}.weight_scale_inv"] = ("F32", scales)
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
                for field, value in _mxfp4_expert(rng, n, k).items():
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
