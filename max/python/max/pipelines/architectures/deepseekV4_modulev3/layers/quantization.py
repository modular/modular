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
"""QAT fake-quant ops and the blockwise fp8 linear, on ModuleV3.

Op-for-op ports of the ModuleV2 implementation
(``deepseekV4/layers/quantization.py`` at dfc8109fc0c); the reasoning (ue8m0
scales, exact ``ceil(log2)``, arithmetic e2m1 rounding) is there. The
fake-quant ops are model semantics: the checkpoint was trained with them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from ..model_config import DeepseekV4Config

__all__ = [
    "DeepseekV4Fp8Linear",
    "block_scale",
    "dequantize_fp8_blocks",
    "fp4_qat_quantize",
    "fp8_block_quant_config",
    "fp8_qat_quantize",
    "linear_for",
    "shard_linear",
]

# Extracted from deepseekV4/layers/quantization.py at dfc8109fc0c.
# From ``inference/kernel.py::act_quant_kernel``.
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
KV_QUANT_BLOCK = 64
# ``inference/model.py`` ``block_size``: the activation group and the weight
# block edge of every fp8 linear (``linear()`` -> ``act_quant(x, 128, ...)``).
LINEAR_QUANT_BLOCK = 128

# From ``inference/kernel.py::fp4_quant_kernel``. The FP4 floor is the smallest
# normal float32 times ``fp4_max``, not the FP8 path's 1e-4.
FP4_MAX = 6.0
FP4_AMAX_FLOOR = 6.0 * 2.0**-126
FP4_QUANT_BLOCK = 32

_LN2 = math.log(2.0)


_WEIGHT_BLOCK_SIZE = [LINEAR_QUANT_BLOCK, LINEAR_QUANT_BLOCK]


def fp8_block_quant_config(
    hf_quantization_config: Mapping[str, object] | None, num_layers: int
) -> QuantConfig | None:
    """The blockwise-fp8 ``QuantConfig`` a DeepSeek-V4 checkpoint declares.

    ``None`` when the checkpoint declares no quantization (a dequantized copy).
    The reference's ``linear()`` quantizes every activation with 128-element
    groups and a power-of-two (``ue8m0``) scale and stores 128x128 weight
    blocks with e8m0 scales, so anything else is refused rather than misread.
    The scale dtype here is float32: the adapter widens the checkpoint's e8m0
    scales (exactly, ``2 ** (e - 127)``) because the blockwise fp8 matmul only
    accepts float32 scales. No repo code reads ``scale_fmt``; the rounding it
    asks for is guaranteed by :class:`DeepseekV4Fp8Linear` instead.
    """
    if not hf_quantization_config:
        return None
    method = hf_quantization_config.get("quant_method")
    if method != "fp8":
        raise ValueError(
            f"DeepSeek-V4 expects quant_method 'fp8', got {method!r}"
        )
    fmt = hf_quantization_config.get("fmt")
    if fmt != "e4m3":
        raise ValueError(f"DeepSeek-V4 expects fmt 'e4m3', got {fmt!r}")
    scheme = hf_quantization_config.get("activation_scheme")
    if scheme != "dynamic":
        raise ValueError(
            f"DeepSeek-V4 expects a dynamic activation scheme, got {scheme!r}"
        )
    scale_fmt = hf_quantization_config.get("scale_fmt")
    if scale_fmt != "ue8m0":
        raise ValueError(
            "DeepSeek-V4 native fp8 implements only power-of-two activation "
            f"scales (scale_fmt 'ue8m0'), got {scale_fmt!r}"
        )
    block_size = hf_quantization_config.get("weight_block_size")
    if (
        not isinstance(block_size, (list, tuple))
        or list(block_size) != _WEIGHT_BLOCK_SIZE
    ):
        raise ValueError(
            f"DeepSeek-V4 expects weight_block_size {_WEIGHT_BLOCK_SIZE}, got "
            f"{block_size!r}"
        )
    layers = range(num_layers)
    return QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=layers,
        attn_quantized_layers=layers,
        block_size=LINEAR_QUANT_BLOCK,
    )


def _pow2(k: Tensor) -> Tensor:
    return 2.0**k


def _ceil_log2(x: Tensor) -> Tensor:
    """Exact ``ceil(log2(x))``: the ``log`` estimate, corrected by one step."""
    k = F.floor(F.log(x) / _LN2)
    k = k + (_pow2(k) < x).cast(DType.float32)
    k = k - (_pow2(k - 1) >= x).cast(DType.float32)
    return k


def _group(x: Tensor, block_size: int) -> Tensor:
    """``[..., width]`` -> float32 ``[..., width // block_size, block_size]``."""
    width = int(x.shape[-1])
    if width % block_size != 0:
        raise ValueError(
            f"last axis {width} is not a multiple of the quant block "
            f"{block_size}"
        )
    x32 = x.cast(DType.float32)
    return x32.reshape([*x32.shape[:-1], width // block_size, block_size])


def block_scale(
    x: Tensor,
    block_size: int = KV_QUANT_BLOCK,
    *,
    value_max: float = FP8_MAX,
    amax_floor: float = AMAX_FLOOR,
) -> Tensor:
    """Per-block power-of-two scale, ``[..., n_blocks, 1]``."""
    amax = F.abs(_group(x, block_size)).max(axis=-1)
    amax = F.elementwise_max(amax, amax_floor)
    return _pow2(_ceil_log2(amax * (1.0 / value_max)))


def _roundtrip(
    x: Tensor,
    block_size: int,
    dtype: DType,
    value_max: float,
    amax_floor: float,
) -> Tensor:
    grouped = _group(x, block_size)
    scale = block_scale(
        x, block_size, value_max=value_max, amax_floor=amax_floor
    )
    clamped = F.elementwise_min(
        F.elementwise_max(grouped / scale, -value_max), value_max
    )
    roundtripped = clamped.cast(dtype).cast(DType.float32) * scale
    return roundtripped.reshape(x.shape).cast(x.dtype)


def fp8_qat_quantize(x: Tensor, block_size: int = KV_QUANT_BLOCK) -> Tensor:
    """Fused FP8 e4m3 quantize/dequantize over groups of the last axis."""
    return _roundtrip(x, block_size, DType.float8_e4m3fn, FP8_MAX, AMAX_FLOOR)


# ``(threshold, step, tie_up)``: midpoints between e2m1 magnitudes; the
# operator alternates to round half to even.
_E2M1_STEPS = (
    (0.25, 0.5, False),
    (0.75, 0.5, True),
    (1.25, 0.5, False),
    (1.75, 0.5, True),
    (2.5, 1.0, False),
    (3.5, 1.0, True),
    (5.0, 2.0, False),
)


def _round_e2m1(x: Tensor) -> Tensor:
    """Nearest e2m1 magnitude, ties to even, sign kept; arithmetic, exact."""
    a = F.abs(x)
    acc: Tensor | None = None
    for threshold, step, tie_up in _E2M1_STEPS:
        crossed = a >= threshold if tie_up else a > threshold
        term = crossed.cast(DType.float32) * step
        acc = term if acc is None else acc + term
    assert acc is not None
    sign = 1.0 - 2.0 * (x < 0.0).cast(DType.float32)
    return sign * acc


def fp4_qat_quantize(x: Tensor, block_size: int = FP4_QUANT_BLOCK) -> Tensor:
    """Fused FP4 e2m1 quantize/dequantize, the indexer's variant."""
    grouped = _group(x, block_size)
    scale = block_scale(
        x, block_size, value_max=FP4_MAX, amax_floor=FP4_AMAX_FLOOR
    )
    clamped = F.elementwise_min(
        F.elementwise_max(grouped / scale, -FP4_MAX), FP4_MAX
    )
    roundtripped = _round_e2m1(clamped) * scale
    return roundtripped.reshape(x.shape).cast(x.dtype)


class DeepseekV4Fp8Linear(Module[[Tensor], Tensor]):
    """``inference/model.py::linear`` for an fp8 weight with 128x128 blocks.

    Not ``QuantizedLinear``: that quantizes activations with ``amax / 448``
    float32 scales, where the checkpoint was trained on power-of-two (e8m0)
    ones. So the activation is quantized with e8m0 scales, which the quantizer
    writes widened to float32 for ``mo.matmul_dynamic_scaled_fp8``: the V2
    layer's values, without its separate cast.
    """

    weight: Tensor
    weight_scale: Tensor

    def __init__(
        self, in_dim: int, out_dim: int, quant_config: QuantConfig
    ) -> None:
        super().__init__()
        if in_dim % LINEAR_QUANT_BLOCK or out_dim % LINEAR_QUANT_BLOCK:
            raise ValueError(
                f"fp8 linear {out_dim}x{in_dim} is not a multiple of "
                f"{LINEAR_QUANT_BLOCK} on both axes"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quant_config = quant_config
        self.weight = Tensor.zeros([out_dim, in_dim], dtype=DType.float8_e4m3fn)
        self.weight_scale = Tensor.zeros(
            [out_dim // LINEAR_QUANT_BLOCK, in_dim // LINEAR_QUANT_BLOCK],
            dtype=DType.float32,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Quantizes ``x`` to fp8 (e8m0 scales) and applies the weight."""
        qc = self.quant_config
        leading = list(x.shape[:-1])
        x2 = x.reshape([-1, x.shape[-1]]) if x.rank != 2 else x
        quantized, scales = quant_ops.quantize_dynamic_scaled_float8(
            x2.cast(DType.bfloat16),
            qc.input_scale,
            qc.weight_scale,
            group_size_or_per_token=LINEAR_QUANT_BLOCK,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float32,
            pow2_scales=True,
        )
        out = quant_ops.dynamic_scaled_matmul(
            quantized,
            self.weight,
            scales,
            self.weight_scale,
            qc.input_scale,
            qc.weight_scale,
            out_type=DType.bfloat16,
        ).cast(x.dtype)
        return out if x.rank == 2 else out.reshape([*leading, self.out_dim])


def dequantize_fp8_blocks(
    weight: Tensor, scale: Tensor, dtype: DType
) -> Tensor:
    """An fp8 weight times its ``[N / 128, K / 128]`` block scale, in ``dtype``.

    Exact in bf16 for power-of-two scales (3 mantissa bits into 7).
    """
    rows, cols = (int(d) for d in weight.shape)
    block = LINEAR_QUANT_BLOCK
    values = weight.cast(DType.float32).reshape(
        [rows // block, block, cols // block, block]
    )
    scales = scale.reshape([rows // block, 1, cols // block, 1])
    return (values * scales).reshape([rows, cols]).cast(dtype)


def linear_for(
    config: DeepseekV4Config, in_dim: int, out_dim: int
) -> Linear | DeepseekV4Fp8Linear:
    """The fp8 linear on a quantized checkpoint, else a plain bias-free one.

    Only for projections the checkpoint stores in fp8.
    """
    if config.quant_config is None:
        return Linear(in_dim, out_dim, bias=False)
    return DeepseekV4Fp8Linear(in_dim, out_dim, config.quant_config)


def shard_linear(linear: Linear | DeepseekV4Fp8Linear, axis: int) -> None:
    """Splits ``linear``'s ``[out, in]`` weight on ``axis`` over the ``tp`` axis.

    Axis 0 is column-parallel (each device a run of outputs), axis 1
    row-parallel (a run of the contraction, so a partial sum out). An fp8
    weight's ``[out / 128, in / 128]`` block scale is split the same way.
    """
    mesh = linear.weight.mesh
    spec = (TP, None) if axis == 0 else (None, TP)
    linear.weight = linear.weight.to(NamedMapping(mesh, spec))
    if isinstance(linear, DeepseekV4Fp8Linear):
        blocks = int(linear.weight_scale.shape[axis])
        if blocks % mesh.num_devices:
            raise ValueError(
                f"{blocks} fp8 scale blocks on axis {axis} do not split "
                f"across {mesh.num_devices} devices"
            )
        linear.weight_scale = linear.weight_scale.to(NamedMapping(mesh, spec))
