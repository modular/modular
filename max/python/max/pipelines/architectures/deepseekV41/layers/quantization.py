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
"""Rounds KV-cache values onto the NVFP4 grid, keeping the input dtype.

Attention then sees exactly the precision an NVFP4 cache would store. NVFP4
pairs E2M1 values with an E4M3 block scale, so unlike power-of-two scales
the scale itself must be rounded through E4M3.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops

from ...deepseekV4.layers.quantization import FP4_MAX, FP8_MAX

# One E4M3 scale per 16 elements.
NVFP4_QUANT_BLOCK = 16
# Lower clamp on the block scale, not on the block amax.
NVFP4_SCALE_MIN = 2.0**-9


def _group(x: TensorValue, block_size: int) -> TensorValue:
    """Splits the last axis into float32 blocks of ``block_size`` elements."""
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


# e2m1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6. Each threshold is a midpoint;
# the flag picks ``>=`` (tie rounds up) or ``>`` (tie rounds down) so ties go
# to the even-mantissa neighbour (0, 1, 2 or 4).
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
    """Rounds to the nearest e2m1 magnitude, ties to even, keeping the sign.

    Done arithmetically as an exact sum of step functions, since the graph
    compiler does not implement the f32 to ``float4_e2m1fn`` cast.
    """
    a = ops.abs(x)
    acc = ops.constant(0.0, DType.float32, x.device)
    for threshold, step, tie_up in _E2M1_STEPS:
        crossed = a >= threshold if tie_up else a > threshold
        acc = acc + ops.cast(crossed, DType.float32) * step
    sign = ops.where(
        x < 0.0,
        ops.constant(-1.0, DType.float32, x.device),
        ops.constant(1.0, DType.float32, x.device),
    )
    return sign * acc


def nvfp4_qat_quantize(
    x: TensorValue, block_size: int = NVFP4_QUANT_BLOCK
) -> TensorValue:
    """Fused NVFP4 e2m1 quantize/dequantize with an E4M3 block scale.

    Computes ``scale = clamp(amax / 6, 2**-9, 448)``, rounds the scale
    through e4m3, then snaps ``x / scale`` to the e2m1 grid. Requires a GPU:
    ``float8_e4m3fn`` does not compile for CPU.

    Args:
        x: Tensor whose last axis is a multiple of ``block_size``.
        block_size: Elements sharing one scale.

    Returns:
        ``x``'s dtype and shape, values snapped to the NVFP4 grid.

    Raises:
        ValueError: If the last axis is not a multiple of ``block_size``.
    """
    grouped = _group(x, block_size)
    amax = ops.max(ops.abs(grouped), axis=-1)
    scale_f32 = ops.min(
        ops.max(amax * (1.0 / FP4_MAX), NVFP4_SCALE_MIN), FP8_MAX
    )
    scale = ops.cast(ops.cast(scale_f32, DType.float8_e4m3fn), DType.float32)
    clamped = ops.min(ops.max(grouped / scale, -FP4_MAX), FP4_MAX)
    roundtripped = _round_e2m1(clamped) * scale
    return ops.cast(ops.reshape(roundtripped, x.shape), x.dtype)
