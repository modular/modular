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

"""FP8 activation quantize/dequantize, as DeepSeek-V4 trained with it.

``inference/model.py`` calls ``act_quant(kv[..., :-rd], 64, scale_fmt,
scale_dtype, True)`` on the non-RoPE dims of the KV latent. The trailing
``True`` is ``inplace``, which in ``inference/kernel.py`` means *fused quantize
then dequantize back to bf16* -- the tensor stays bf16-shaped but carries only
the information an FP8 e4m3 value can hold.

This is not an optimization to skip. The model was trained against it (QAT), and
vLLM refuses to serve V4 with anything but an FP8 KV cache
(``fp8_ds_mla layout only supports fp8 kv-cache``). Dropping it changes results.

V4's ``quantization_config`` sets ``scale_fmt: "ue8m0"``, which selects
``round_scale=True``: the per-block scale is rounded *up* to a power of two.
The reference does that with float bit manipulation::

    fast_log2_ceil(x): (bits >> 23 & 0xFF) - 127 + (mantissa != 0)
    fast_pow2(k):      reinterpret_float((k + 127) << 23)
    fast_round_scale(amax, 1/448) = fast_pow2(fast_log2_ceil(amax / 448))

MAX exposes neither a bitcast nor ``log2``/``exp2``, so ``_ceil_log2`` below
computes it from ``log`` and corrects the result to be exact. Approximating with
``log(x)/log(2)`` alone is not safe here: an error of one ULP at an exact power
of two flips the ceiling and changes the scale by a factor of two.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import TensorValue, ops

# From ``inference/kernel.py::act_quant_kernel``.
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
KV_QUANT_BLOCK = 64

# From ``inference/kernel.py::fp4_quant_kernel``. The FP4 floor is the smallest
# normal float32 times ``fp4_max``, not the FP8 path's 1e-4.
FP4_MAX = 6.0
FP4_AMAX_FLOOR = 6.0 * 2.0**-126
FP4_QUANT_BLOCK = 32

_LN2 = math.log(2.0)


def _pow2(k: TensorValue) -> TensorValue:
    """``2 ** k`` for a tensor holding integer-valued floats."""
    return ops.pow(ops.constant(2.0, DType.float32, k.device), k)


def _ceil_log2(x: TensorValue) -> TensorValue:
    """Exact ``ceil(log2(x))`` for positive ``x``.

    Starts from the float estimate and corrects it. ``ceil(log2(x))`` is the
    smallest integer ``k`` with ``2**k >= x``, so at most one step in each
    direction is needed once the estimate is within one.
    """
    k = ops.floor(ops.log(x) / _LN2)
    # Estimate too small: 2**k still below x.
    k = k + ops.cast(_pow2(k) < x, DType.float32)
    # Estimate too large: 2**(k-1) already reaches x.
    k = k - ops.cast(_pow2(k - 1) >= x, DType.float32)
    return k


def block_scale(
    x: TensorValue,
    block_size: int = KV_QUANT_BLOCK,
    *,
    value_max: float = FP8_MAX,
    amax_floor: float = AMAX_FLOOR,
) -> TensorValue:
    """Per-block power-of-two scale, ``[..., n_blocks, 1]``.

    ``amax`` over each group, floored, then rounded up to a power of two --
    ``fast_round_scale(amax, 1/value_max)`` in the reference. Separated from the
    quantize step so it can be checked on a device without FP8/FP4 support.
    """
    grouped = _group(x, block_size)
    amax = ops.max(ops.abs(grouped), axis=-1)
    amax = ops.max(amax, ops.constant(amax_floor, DType.float32, amax.device))
    return _pow2(_ceil_log2(amax * (1.0 / value_max)))


def _group(x: TensorValue, block_size: int) -> TensorValue:
    """``[..., width]`` -> ``[..., width // block_size, block_size]``, float32."""
    width = int(x.shape[-1])
    if width % block_size != 0:
        raise ValueError(
            f"last axis {width} is not a multiple of the quant block "
            f"{block_size}"
        )
    x32 = ops.cast(x, DType.float32)
    return ops.reshape(
        x32, list(x32.shape[:-1]) + [width // block_size, block_size]
    )


def fp8_qat_quantize(
    x: TensorValue, block_size: int = KV_QUANT_BLOCK
) -> TensorValue:
    """Fused FP8 e4m3 quantize/dequantize over groups of the last axis.

    Args:
        x: Tensor whose last axis is a multiple of ``block_size``.
        block_size: Elements sharing one scale. The reference passes 64 for the
            KV latent (its own default is 128).

    Returns:
        ``x``'s dtype and shape, values snapped to the FP8 grid.

    Requires a device with FP8 support; ``float8_e4m3fn`` does not compile for
    CPU ("The f8e4m3fn data type is not supported on device 'cpu:0'").
    """
    return _roundtrip(x, block_size, DType.float8_e4m3fn, FP8_MAX, AMAX_FLOOR)


def fp4_qat_quantize(
    x: TensorValue, block_size: int = FP4_QUANT_BLOCK
) -> TensorValue:
    """Fused FP4 e2m1 quantize/dequantize, the indexer's variant.

    ``fp4_act_quant(x, 32, True)`` in the reference. Same shape as the FP8 path
    but over the *whole* vector rather than just the non-RoPE dims, since the
    indexer Hadamard-rotates first and the RoPE dims stop being separable.

    The e2m1 grid is only {0, .5, 1, 1.5, 2, 3, 4, 6} in magnitude, so this
    discards far more than the FP8 path; that is what the model was trained
    against for indexer scoring, which only has to rank entries.
    """
    grouped = _group(x, block_size)
    scale = block_scale(
        x, block_size, value_max=FP4_MAX, amax_floor=FP4_AMAX_FLOOR
    )
    lo = ops.constant(-FP4_MAX, DType.float32, grouped.device)
    hi = ops.constant(FP4_MAX, DType.float32, grouped.device)
    clamped = ops.min(ops.max(grouped / scale, lo), hi)
    roundtripped = _round_e2m1(clamped) * scale
    return ops.cast(ops.reshape(roundtripped, x.shape), x.dtype)


# Magnitudes representable in e2m1: 0, .5, 1, 1.5, 2, 3, 4, 6. Each threshold
# below is a midpoint between neighbours, and the comparison operator encodes
# round-half-to-even: ``>`` rounds the tie down, ``>=`` rounds it up. The even-
# mantissa values are 0, 1, 2 and 4, so ties alternate direction, which is why
# these are not all the same operator.
_E2M1_STEPS = (
    (0.25, 0.5, False),
    (0.75, 0.5, True),
    (1.25, 0.5, False),
    (1.75, 0.5, True),
    (2.5, 1.0, False),
    (3.5, 1.0, True),
    (5.0, 2.0, False),
)


def _round_e2m1(x: TensorValue) -> TensorValue:
    """Round to the nearest e2m1 magnitude, ties to even, keeping the sign.

    MAX has ``DType.float4_e2m1fn`` but not the conversion into it -- the graph
    compiler reports "conversion from 'f32' to 'f4e2m1fn' is not implemented" --
    so this is done arithmetically. With only eight magnitudes, a sum of step
    functions is exact and branch-free.
    """
    a = ops.abs(x)
    acc = ops.constant(0.0, DType.float32, x.device)
    for threshold, step, tie_up in _E2M1_STEPS:
        t = ops.constant(threshold, DType.float32, x.device)
        crossed = a >= t if tie_up else a > t
        acc = acc + ops.cast(crossed, DType.float32) * step
    sign = ops.where(
        x < ops.constant(0.0, DType.float32, x.device),
        ops.constant(-1.0, DType.float32, x.device),
        ops.constant(1.0, DType.float32, x.device),
    )
    return sign * acc


def _roundtrip(
    x: TensorValue,
    block_size: int,
    dtype: DType,
    value_max: float,
    amax_floor: float,
) -> TensorValue:
    grouped = _group(x, block_size)
    scale = block_scale(
        x, block_size, value_max=value_max, amax_floor=amax_floor
    )

    lo = ops.constant(-value_max, DType.float32, grouped.device)
    hi = ops.constant(value_max, DType.float32, grouped.device)
    clamped = ops.min(ops.max(grouped / scale, lo), hi)

    roundtripped = ops.cast(ops.cast(clamped, dtype), DType.float32) * scale
    return ops.cast(ops.reshape(roundtripped, x.shape), x.dtype)
