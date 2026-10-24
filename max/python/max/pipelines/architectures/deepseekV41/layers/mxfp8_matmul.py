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
"""MXFP8 projections on the cluster split-K matmul.

At decode sizes one CTA per 128-row weight tile leaves most SMs idle, so
``mo.matmul.mxfp8.cluster_splitk`` splits each tile's K over a cluster of
CTAs where that pays. For the weight shapes and M its measured tables name
it runs a pair-CTA GEMM (the swap-AB decode GEMM or the prefill GEMM), and
the production block-scaled matmul elsewhere.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import Dim, TensorType, TensorValue, ops
from max.nn.kernels import (
    _is_sm10x_gpu,
    block_scales_interleave,
    quantize_dynamic_block_scaled,
)
from max.nn.linear import Linear
from max.nn.quant_config import QuantFormat
from max.nn.quant_ops import _matmul_float8_mxfp8

from ...deepseekV4.layers.rope import apply_rope_tail
from ..model_config import mxfp8_splitk_enabled

# One E8M0 scale per 32 elements.
_MXFP8_BLOCK_SIZE = 32


def _splitk_supported(layer: Linear) -> bool:
    if (
        not mxfp8_splitk_enabled()
        or layer.quant_config is None
        or layer.quant_config.format != QuantFormat.MXFP8
        or layer.weight_scale is None
        or layer.bias is not None
    ):
        return False
    return _is_sm10x_gpu()


def mxfp8_linear(
    layer: Linear, x: TensorValue, x_scales: TensorValue | None = None
) -> TensorValue:
    """Applies an MXFP8 ``layer`` through the cluster split-K matmul.

    Any other layer, or any device but SM100, runs ``layer(x)``.

    Args:
        layer: The projection.
        x: The bf16 input, ``[..., in_dim]``, or the ``float8_e4m3fn``
            ``[rows, in_dim]`` activation when ``x_scales`` is given.
        x_scales: The activation's E8M0 block scales when ``x`` arrives
            already MXFP8 from a fused activation and quantize, in the layout
            :func:`~max.nn.kernels.quantize_dynamic_block_scaled` produces.

    Returns:
        ``[..., out_dim]`` bf16.
    """
    if not _splitk_supported(layer):
        if x_scales is None:
            return layer(x)
        assert layer.weight_scale is not None
        return _matmul_float8_mxfp8(
            x,
            layer.weight.to(x.device),
            layer.weight_scale,
            x_scales=x_scales,
        )
    assert layer.weight_scale is not None
    leading = list(x.shape[:-1])
    rows = Dim(1)
    for d in leading:
        rows = rows * d
    x_fp8 = ops.reshape(x, [rows, x.shape[-1]])
    if x_scales is None:
        x_fp8, x_scales = quantize_dynamic_block_scaled(
            x_fp8,
            sf_vector_size=_MXFP8_BLOCK_SIZE,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    weight = layer.weight.to(x.device)
    w_scales = block_scales_interleave(
        layer.weight_scale.to(x.device), sf_vector_size=_MXFP8_BLOCK_SIZE
    )
    out = ops.custom(
        "mo.matmul.mxfp8.cluster_splitk",
        device=x.device,
        values=[x_fp8, weight, x_scales, w_scales],
        out_types=[
            TensorType(DType.bfloat16, [rows, weight.shape[0]], x.device)
        ],
    )[0].tensor
    return ops.reshape(out, [*leading, weight.shape[0]])


def mxfp8_linear_rope_tail(
    layer: Linear,
    x: TensorValue,
    freqs_cis: TensorValue,
    head_dim: int,
    x_scales: TensorValue | None = None,
) -> TensorValue:
    """Applies ``layer``, then rotates the RoPE tail of each output head.

    On SM100 an MXFP8 ``layer`` rotates in its matmul's epilogue, so the
    output is written once. Otherwise this is :func:`mxfp8_linear`, then
    :func:`apply_rope_tail` on the heads.

    Args:
        layer: The projection; its output is whole ``head_dim`` heads.
        x: The bf16 input, ``[..., in_dim]``, one row per token.
        freqs_cis: ``[T, rope_dim // 2, 2]`` float32 cos/sin at each row's
            position.
        head_dim: Channels per output head.
        x_scales: The E8M0 block scales of ``x`` when it arrives already
            ``float8_e4m3fn`` from a fused quantize. Only the SM100 MXFP8
            path takes it.

    Returns:
        ``[..., out_dim]`` bf16.
    """
    rope_dim = int(freqs_cis.shape[-2]) * 2
    out_dim = int(layer.weight.shape[0])
    leading = list(x.shape[:-1])
    if (
        layer.quant_config is None
        or layer.quant_config.format != QuantFormat.MXFP8
        or layer.weight_scale is None
        or layer.bias is not None
        or not _is_sm10x_gpu()
    ):
        assert x_scales is None, "pre-quantized input needs the SM100 path"
        out = mxfp8_linear(layer, x)
        heads = ops.reshape(out, [1, -1, out_dim // head_dim, head_dim])
        return ops.reshape(
            apply_rope_tail(heads, freqs_cis, rope_dim), [*leading, out_dim]
        )
    rows = Dim(1)
    for d in leading:
        rows = rows * d
    x_fp8 = ops.reshape(x, [rows, x.shape[-1]])
    if x_scales is None:
        x_fp8, x_scales = quantize_dynamic_block_scaled(
            x_fp8,
            sf_vector_size=_MXFP8_BLOCK_SIZE,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    w_scales = block_scales_interleave(
        layer.weight_scale.to(x.device), sf_vector_size=_MXFP8_BLOCK_SIZE
    )
    out = ops.custom(
        "mo.matmul.mxfp8.rope_tail",
        device=x.device,
        values=[
            x_fp8,
            layer.weight.to(x.device),
            x_scales,
            w_scales,
            ops.reshape(freqs_cis, [-1, rope_dim]),
        ],
        out_types=[TensorType(DType.bfloat16, [rows, out_dim], x.device)],
        parameters={"head_dim": head_dim, "rope_dim": rope_dim},
    )[0].tensor
    return ops.reshape(out, [*leading, out_dim])
