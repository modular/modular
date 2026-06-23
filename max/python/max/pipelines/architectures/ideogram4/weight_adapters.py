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
"""Weight adapter for the Ideogram 4 transformer.

The diffusers ``transformer/`` checkpoint uses module names identical to our
Graph-API model, so no key remapping is needed. The only transformation is
weight-only FP8 dequantization: quantized Linear layers store a
``float8_e4m3fn`` ``<name>.weight`` plus a per-output-channel float32
``<name>.weight_scale`` such that ``weight ~= weight_fp8 * scale[:, None]``.
We dequantize to float32 here; the component loader then casts to the compute
dtype (bfloat16).
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph.weights import WeightData

FP8_SCALE_SUFFIX = ".weight_scale"

# Per-block Linear projections kept as native FP8 (weight + rowwise scale)
# instead of being dequantized to bf16. These are the per-step GEMM hot spots
# (run 34 layers x 2 branches x N steps); the remaining quantized linears
# (adaLN, input_proj, llm_cond_proj, t_embedding, final_layer) run once or are
# tiny, so they keep the simpler dequantized bf16 path.
FP8_RUNTIME_LINEAR_SUFFIXES = (
    ".attention.qkv",
    ".attention.o",
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3",
)


def _is_fp8_runtime_linear(base: str) -> bool:
    """Whether ``base`` (a weight key minus ``.weight``) is a hot FP8 Linear."""
    return base.startswith("layers.") and base.endswith(
        FP8_RUNTIME_LINEAR_SUFFIXES
    )


def _rowwise_scale(scale: WeightData, name: str) -> WeightData:
    """Reshape a per-output-channel scale ``[N]`` to rowwise ``[N, 1]`` f32.

    ``dynamic_scaled_matmul`` expects a rank-2 rowwise weight scale whose
    trailing dimension is 1.
    """
    s = np.from_dlpack(scale.astype(DType.float32).data).astype(np.float32)
    return WeightData.from_numpy(np.ascontiguousarray(s.reshape(-1, 1)), name)


@functools.lru_cache(maxsize=1)
def _e4m3fn_lut() -> npt.NDArray[np.float32]:
    """256-entry lookup decoding every ``float8_e4m3fn`` byte to float32.

    OCP E4M3FN: 1 sign / 4 exponent (bias 7) / 3 mantissa bits, finite-only
    (the sole NaN encoding is ``S.1111.111``; max finite magnitude is 448).
    Verified bit-exact against ``torch.float8_e4m3fn``.
    """
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if (byte & 0x80) else 1.0
        exp = (byte >> 3) & 0x0F
        mant = byte & 0x07
        if exp == 0:
            value = (mant / 8.0) * 2.0 ** (1 - 7)
        elif exp == 15 and mant == 7:
            value = np.nan
        else:
            value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        table[byte] = sign * value
    return table


def _fp8_to_float32(weight: WeightData) -> npt.NDArray[np.float32]:
    """Decode a ``float8_e4m3fn`` weight to float32 via raw-byte lookup.

    The engine's host cast model can't lower ``float8_e4m3fn -> float32``, so
    we reinterpret the underlying buffer as ``uint8`` and index a precomputed
    decode table instead.
    """
    raw = np.from_dlpack(weight.to_buffer().view(DType.uint8))
    return _e4m3fn_lut()[raw]


def _dequantize_fp8(
    weight: WeightData, scale: WeightData, name: str
) -> WeightData:
    """Dequantize a per-row FP8 weight to float32: ``w_fp8 * scale[:, None]``."""
    w_f32 = _fp8_to_float32(weight)
    scale_f32 = np.from_dlpack(scale.astype(DType.float32).data).astype(
        np.float32
    )
    if w_f32.ndim != 2:
        raise ValueError(
            f"FP8 weight '{name}' expected 2-D, got shape {w_f32.shape}"
        )
    if scale_f32.shape[0] != w_f32.shape[0]:
        raise ValueError(
            f"FP8 scale for '{name}' has length {scale_f32.shape[0]} but "
            f"weight has {w_f32.shape[0]} output channels"
        )
    deq = np.ascontiguousarray(w_f32 * scale_f32[:, None])
    return WeightData.from_numpy(deq, name)


def dequantize_fp8_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Apply FP8 weight-only dequantization; pass other tensors through.

    Quantized Linear weights are stored as a ``float8_e4m3fn`` ``<name>.weight``
    paired with a per-output-channel float32 ``<name>.weight_scale``. Each such
    pair is dequantized to float32; the ``.weight_scale`` entries are dropped.
    All other tensors pass through unchanged.
    """
    scale_keys = [k for k in state_dict if k.endswith(FP8_SCALE_SUFFIX)]
    quantized_weight_keys = {
        k[: -len(FP8_SCALE_SUFFIX)] + ".weight" for k in scale_keys
    }

    converted: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            continue
        if key in quantized_weight_keys:
            scale_key = key[: -len(".weight")] + FP8_SCALE_SUFFIX
            converted[key] = _dequantize_fp8(value, state_dict[scale_key], key)
        else:
            converted[key] = value
    return converted


def convert_ideogram4_transformer_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Adapt the DiT checkpoint, keeping the hot Linears as native FP8.

    The per-block ``qkv``/``o``/``w1``/``w2``/``w3`` projections (see
    :data:`FP8_RUNTIME_LINEAR_SUFFIXES`) pass their ``float8_e4m3fn`` weight
    through unchanged and keep a rowwise ``[N, 1]`` float32 ``weight_scale``,
    so they can run as native FP8 GEMMs. Every other quantized Linear is
    dequantized to float32 and its scale dropped (as before). Non-quantized
    tensors pass through unchanged.
    """
    scale_keys = [k for k in state_dict if k.endswith(FP8_SCALE_SUFFIX)]
    quantized_bases = {k[: -len(FP8_SCALE_SUFFIX)] for k in scale_keys}

    converted: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            base = key[: -len(FP8_SCALE_SUFFIX)]
            # Keep the scale only for the native-FP8 linears; for the rest the
            # scale is folded into the dequantized weight below and dropped.
            if _is_fp8_runtime_linear(base):
                converted[key] = _rowwise_scale(value, key)
            continue

        if key.endswith(".weight"):
            weight_base = key[: -len(".weight")]
            if weight_base in quantized_bases:
                if _is_fp8_runtime_linear(weight_base):
                    converted[key] = value  # native FP8: keep packed weight
                else:
                    converted[key] = _dequantize_fp8(
                        value, state_dict[weight_base + FP8_SCALE_SUFFIX], key
                    )
                continue

        converted[key] = value

    required_prefixes = (
        "input_proj.",
        "llm_cond_norm.",
        "llm_cond_proj.",
        "t_embedding.",
        "adaln_proj.",
        "embed_image_indicator.",
        "layers.0.",
        "final_layer.",
    )
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in converted):
            raise ValueError(
                f"Missing required Ideogram 4 transformer weights with prefix "
                f"'{prefix}'"
            )
    return converted
