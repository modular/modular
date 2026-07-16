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
"""Weight adaptation utilities for FLUX2 transformer models.

Handles NVFP4 checkpoint conversion and stacked QKV splitting so that
both the ComponentModel and PipelineExecutor paths share the same logic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from max._core.safetensors import safe_open
from max.driver import Buffer
from max.dtype import DType
from max.graph.shape import Shape
from max.graph.weights import WeightData
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig

from .nvfp4_weight_adapter import convert_nvfp4_state_dict

# Linear weight-key suffixes that carry a quantized (int8 W8A8) weight. Must
# stay in lockstep with which Linears get the int8 ``quant_config`` in
# ``flux2.py`` (``Flux2BlockQuant.resolve`` for int8 quantizes every block
# Linear; the single-block fused proj + out; head/tail Linears stay bf16).
_INT8_QUANTIZED_WEIGHT_SUFFIXES = (
    ".attn.to_q.weight",
    ".attn.to_k.weight",
    ".attn.to_v.weight",
    ".attn.to_out.0.weight",
    ".attn.add_q_proj.weight",
    ".attn.add_k_proj.weight",
    ".attn.add_v_proj.weight",
    ".attn.to_add_out.weight",
    ".ff.linear_in.weight",
    ".ff.linear_out.weight",
    ".ff_context.linear_in.weight",
    ".ff_context.linear_out.weight",
    ".attn.to_qkv_mlp_proj.weight",
    ".attn.to_out.weight",
)

# Mapping from stacked QKV key infixes to the split (Q, K, V) infixes.
_STACKED_QKV_INFIXES = {
    ".attn.qkv_proj.": (".attn.to_q.", ".attn.to_k.", ".attn.to_v."),
    ".attn.add_qkv_proj.": (
        ".attn.add_q_proj.",
        ".attn.add_k_proj.",
        ".attn.add_v_proj.",
    ),
}


def _split_stacked_qkv(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Split fused QKV weights into separate Q, K, V entries."""
    out: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        matched = False
        for stacked, (q, k, v) in _STACKED_QKV_INFIXES.items():
            if stacked not in key:
                continue
            matched = True
            if key.endswith((".weight", ".weight_scale")):
                buf = value.to_buffer()
                chunk = buf.shape[0] // 3
                for infix, i in zip([q, k, v], range(3), strict=False):
                    split_name = key.replace(stacked, infix)
                    split_buf = buf[i * chunk : (i + 1) * chunk, :]
                    out[split_name] = WeightData(
                        split_buf,
                        split_name,
                        value.dtype,
                        Shape(split_buf.shape),
                    )
            elif key.endswith((".weight_scale_2", ".input_scale")):
                # Per-tensor scales are shared across Q/K/V.
                for infix in (q, k, v):
                    out[key.replace(stacked, infix)] = value
            break
        if not matched:
            out[key] = value
    return out


def parse_nvfp4_quantization_metadata(
    paths: Sequence[Path],
) -> frozenset[str]:
    """Return BFL-named layers tagged ``nvfp4`` in the checkpoint metadata.

    modelopt/BFL NVFP4 single-file exports embed a ``_quantization_metadata``
    JSON blob in the safetensors header listing each Linear's format. Entries
    with ``"format": "nvfp4"`` are block-scaled FP4; layers absent from the
    list stay in BF16. Returns an empty set when no path carries the metadata,
    in which case the caller falls back to the legacy uniform-NVFP4 assumption.
    """
    out: set[str] = set()
    for path in paths:
        with safe_open(path) as f:
            md = f.metadata()
        raw = md.get("_quantization_metadata")
        if not raw:
            continue
        for name, spec in json.loads(raw).get("layers", {}).items():
            if isinstance(spec, dict) and spec.get("format") == "nvfp4":
                out.add(name)
    return frozenset(out)


def _bf16_weightdata_to_fp32(value: WeightData) -> np.ndarray:
    """Return a WeightData's values as an fp32 numpy array.

    numpy has no bfloat16, so bf16 buffers are viewed as uint16 and widened
    to fp32 via a bit shift into the fp32 mantissa (bf16 is the fp32 high 16
    bits). Non-bf16 float buffers go through numpy directly.
    """
    buf = value.to_buffer()
    if value.dtype == DType.bfloat16:
        u16 = buf.view(DType.uint16).to_numpy().astype(np.uint32)
        return (u16 << 16).view(np.float32)
    return buf.to_numpy().astype(np.float32)


def _rtn_quantize_int8(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """RTN-quantize the targeted bf16 Linear weights to symmetric int8.

    For each weight matching :data:`_INT8_QUANTIZED_WEIGHT_SUFFIXES`, computes
    a per-output-channel (per-row of the ``[N, K]`` weight) symmetric
    absmax/127 int8 quantization and emits both the int8 weight (same key) and
    a new ``<name>_scale`` fp32 ``[N, 1]`` scale (``.weight`` -> ``.weight_scale``).
    Non-matching weights (norms, embedders, head/tail Linears) pass through
    unchanged in bf16.
    """
    out: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        if (
            not key.endswith(_INT8_QUANTIZED_WEIGHT_SUFFIXES)
            or len(value.shape) != 2
        ):
            out[key] = value
            continue

        w = _bf16_weightdata_to_fp32(value)  # [N, K] fp32
        absmax = np.abs(w).max(axis=1, keepdims=True)  # [N, 1]
        scale = np.where(absmax != 0.0, absmax / 127.0, np.float32(1.0)).astype(
            np.float32
        )
        q = np.rint(w / scale).astype(np.int32)
        q = np.clip(q, -127, 127).astype(np.int8)  # [N, K] int8

        q_buf = Buffer.from_numpy(np.ascontiguousarray(q))
        out[key] = WeightData(q_buf, key, DType.int8, Shape(q.shape))

        scale_key = key[: -len(".weight")] + ".weight_scale"
        scale_c = np.ascontiguousarray(scale)  # [N, 1]
        scale_buf = Buffer.from_numpy(scale_c)
        out[scale_key] = WeightData(
            scale_buf, scale_key, DType.float32, Shape(scale_c.shape)
        )
    return out


def verify_int8_quantization_consistency(
    model: Module,
    state_dict: dict[str, WeightData],
) -> None:
    """Cross-check the int8 W8A8 weights against the model's int8 Linears.

    :data:`_INT8_QUANTIZED_WEIGHT_SUFFIXES` (which weights RTN-quantizes to
    int8) has to stay in lockstep with which Linears :meth:`Flux2BlockQuant.
    resolve` configures as int8. When they drift, the failure surfaces far
    away as ``weight b must be int8, got bf16`` from ``_apple_int8_w8a8_matmul``
    at graph build. This reconciles the two sets at load time and raises a
    clear error naming the offending layer(s), in either direction:

    - a Linear configured int8 whose weight stayed bf16 (whitelist too narrow);
    - a weight quantized to int8 whose Linear is bf16 (whitelist too broad).

    Args:
        model: The constructed transformer module (its ``_iter_named_weights``
            FQNs match the ``state_dict`` keys passed to ``load_state_dict``).
        state_dict: The adapted state dict about to be loaded.

    Raises:
        ValueError: If the int8-configured Linears and the int8 weights in
            ``state_dict`` do not match.
    """
    # Linears the model configured for int8 W8A8, by their weight FQN.
    int8_module_weights = {
        name
        for name, weight, layer in model._iter_named_weights()
        if isinstance(layer, Linear)
        and weight is layer.weight
        and layer.quant_config is not None
        and layer.quant_config.is_int8_w8a8
    }
    # Weights RTN-quantized to int8 in the state dict.
    int8_state_weights = {
        key
        for key, value in state_dict.items()
        if key.endswith(".weight") and value.dtype == DType.int8
    }

    configured_but_bf16 = sorted(int8_module_weights - int8_state_weights)
    quantized_but_unused = sorted(int8_state_weights - int8_module_weights)
    if configured_but_bf16 or quantized_but_unused:
        raise ValueError(
            "int8 W8A8 weight/Linear mismatch: `_INT8_QUANTIZED_WEIGHT_"
            "SUFFIXES` is out of sync with `Flux2BlockQuant.resolve`.\n"
            f"  configured int8 but weight left bf16: {configured_but_bf16}\n"
            f"  quantized to int8 but Linear is bf16: {quantized_but_unused}"
        )


def adapt_weights(
    state_dict: dict[str, WeightData],
    quant_config: QuantConfig | None = None,
) -> dict[str, WeightData]:
    """Apply NVFP4 conversion and QKV splitting to a raw state dict.

    Args:
        state_dict: Raw checkpoint weights keyed by parameter name.
        quant_config: If not None, apply NVFP4 weight conversion.

    Returns:
        Adapted state dict with BFL naming converted and stacked QKV
        weights split into separate Q, K, V entries.
    """
    if quant_config is not None and quant_config.is_int8_w8a8:
        # int8 W8A8 rides on the bf16 (diffusers-named) checkpoint: RTN-quantize
        # the targeted Linear weights to int8 + synthesize their scales. No
        # BFL conversion / QKV split (klein bf16 already has split diffusers
        # naming).
        return _rtn_quantize_int8(state_dict)

    if quant_config is not None:
        state_dict = convert_nvfp4_state_dict(state_dict)

    stacked_qkv = any(
        ".attn.qkv_proj." in k or ".attn.add_qkv_proj." in k for k in state_dict
    )
    if stacked_qkv:
        state_dict = _split_stacked_qkv(state_dict)

    return state_dict
