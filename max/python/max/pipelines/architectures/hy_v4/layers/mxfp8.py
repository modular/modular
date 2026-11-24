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
"""Hy4's dense MXFP8 projections on the cluster split-K matmul.

At decode a projection has 4 to 24 rows (MTP verifies 4 a request), so the
production block-scaled matmul runs one CTA per 128 weight rows and uses
few SMs. ``matmul.mxfp8.cluster_splitk.bf16`` splits K over a cluster of
CTAs where that leaves SMs idle, and at decode M quantizes the bf16
activation inside the matmul, so no quantize kernel runs before it. Past
decode M one quantize serves every matmul of the activation, which then run
on E4M3 tokens as before, so prefill keeps its kernels.
"""

from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn._graph_ops import (
    _block_scaled_quantize_out_types,
    _is_sm10x_gpu,
    swiglu_quantize_dynamic_block_scaled,
)
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import MXFP8Tensor
from max.nn.kernels import block_scales_interleave
from max.tensor import Tensor
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.placements import Sharded
from max.tensor.sharding.types import TensorLayout

_SF_VECTOR_SIZE = 32


def _cluster_splitk(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
) -> TensorValue:
    return ops.custom(
        "matmul.mxfp8.cluster_splitk",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(DType.bfloat16, [a.shape[0], b.shape[0]], a.device)
        ],
    )[0].tensor


_FUSED_ROWS = 32
"""Mirrors the kernels' ``FUSED_ROWS``: the most rows the matmul quantizes
itself; the op's scratch for a shape it declines holds that many."""


def _quant_out_types(
    rows: Dim, cols: Dim, device: DeviceRef
) -> list[TensorType]:
    return _block_scaled_quantize_out_types(
        rows,
        cols,
        "x.shape[1]",
        _SF_VECTOR_SIZE,
        MXFP8Tensor.scale_dtype,
        MXFP8Tensor.storage_dtype,
        device,
    )


def _quantize_unfused(x: TensorValue) -> tuple[TensorValue, TensorValue]:
    codes, scales = ops.custom(
        "matmul.mxfp8.cluster_splitk.quantize",
        device=x.device,
        values=[x],
        out_types=_quant_out_types(x.shape[0], x.shape[1], x.device),
    )
    return codes.tensor, scales.tensor


def _quantize_unfused_rule(x: TensorLayout) -> list[AxisAssignment]:
    return [
        AxisAssignment((R,), (R, R)),
        AxisAssignment((Sharded(0),), (Sharded(0), Sharded(0))),
        AxisAssignment((Sharded(1),), (Sharded(1), Sharded(1))),
    ]


_quantize_unfused_op = F.functional(
    _quantize_unfused, rule=_quantize_unfused_rule
)


def _cluster_splitk_bf16(
    x: TensorValue,
    x_codes: TensorValue,
    x_scales: TensorValue,
    b: TensorValue,
    b_scales: TensorValue,
) -> TensorValue:
    return ops.custom(
        "matmul.mxfp8.cluster_splitk.bf16",
        device=x.device,
        values=[x, x_codes, x_scales, b, b_scales],
        out_types=[
            TensorType(DType.bfloat16, [x.shape[0], b.shape[0]], x.device),
            *_quant_out_types(Dim(_FUSED_ROWS), x.shape[1], x.device),
        ],
    )[0].tensor


def _cluster_splitk_bf16_rule(
    x: TensorLayout,
    x_codes: TensorLayout,
    x_scales: TensorLayout,
    b: TensorLayout,
    b_scales: TensorLayout,
) -> list[AxisAssignment]:
    # The block-scaled matmul's placements: column- and row-parallel weights,
    # and row-sharded activations.
    return [
        AxisAssignment((R, R, R, R, R), (R,)),
        AxisAssignment((R, R, R, Sharded(0), Sharded(0)), (Sharded(1),)),
        AxisAssignment((Sharded(1),) * 5, (P,)),
        AxisAssignment((Sharded(0),) * 3 + (R, R), (Sharded(0),)),
    ]


_cluster_splitk_bf16_op = F.functional(
    _cluster_splitk_bf16, rule=_cluster_splitk_bf16_rule
)


def _swiglu_quantize(gate_up: TensorValue) -> tuple[TensorValue, TensorValue]:
    return swiglu_quantize_dynamic_block_scaled(
        gate_up,
        GatedActivation("silu"),
        sf_vector_size=_SF_VECTOR_SIZE,
        scales_type=MXFP8Tensor.scale_dtype,
        out_type=MXFP8Tensor.storage_dtype,
    )


def _swiglu_quantize_rule(gate_up: TensorLayout) -> list[AxisAssignment]:
    # Column-sharded only as a rank's own [gate_r | up_r] columns.
    return [
        AxisAssignment((R,), (R, R)),
        AxisAssignment((Sharded(1),), (Sharded(1), Sharded(1))),
    ]


_swiglu_quantize_op = F.functional(_swiglu_quantize, rule=_swiglu_quantize_rule)


def _cluster_splitk_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
) -> list[AxisAssignment]:
    # Row-parallel only: each rank's K slice gives a partial sum.
    return [
        AxisAssignment((R, R, R, R), (R,)),
        AxisAssignment((Sharded(1),) * 4, (P,)),
    ]


_cluster_splitk_op = F.functional(_cluster_splitk, rule=_cluster_splitk_rule)


def matmul(x: Tensor, weight: quant_ops.QuantAwareTensor) -> Tensor:
    """Returns ``x @ weight.T``, MXFP8 weights on the cluster split-K matmul.

    Other weights, and MXFP8 off SM100, take :func:`quant_ops.matmul`.
    """
    if not isinstance(weight, MXFP8Tensor) or not _is_sm10x_gpu():
        return quant_ops.matmul(x, weight)
    weight_scales = block_scales_interleave(
        weight.weight_scale, sf_vector_size=_SF_VECTOR_SIZE
    )
    # Graph CSE shares the quantize among the matmuls of one activation;
    # at decode M it launches nothing.
    codes, scales = _quantize_unfused_op(x)
    return _cluster_splitk_bf16_op(
        x, codes, scales, weight.weight, weight_scales
    )


def swiglu_matmul(gate_up: Tensor, weight: MXFP8Tensor) -> Tensor:
    """Returns ``silu(gate) * up @ weight.T`` for ``gate_up = [gate | up]``.

    One kernel computes the SwiGLU, rounds it to bfloat16 and quantizes it
    to MXFP8, so the activation is never written. With ``weight`` split
    along K, each rank's ``gate_up`` must hold its own ``[gate_r | up_r]``
    columns and the result is that rank's unreduced partial sum.

    Raises:
        NotImplementedError: Off SM10x GPUs, where the cluster split-K MXFP8
            matmul has no kernel.
    """
    if not _is_sm10x_gpu():
        raise NotImplementedError(
            "Hy4's MLP needs an SM10x GPU (B200 or B300): its cluster"
            " split-K MXFP8 matmul has no kernel for other targets"
        )
    weight_scales = block_scales_interleave(
        weight.weight_scale, sf_vector_size=_SF_VECTOR_SIZE
    )
    codes, scales = _swiglu_quantize_op(gate_up)
    return _cluster_splitk_op(codes, weight.weight, scales, weight_scales)


def concat_rows(parts: Sequence[MXFP8Tensor]) -> MXFP8Tensor:
    """Stacks ``parts`` by row, each rank its own rows once sharded.

    The checkpoint stores the parts of a fused projection apart; the graph
    stacks them, a weight-only op that runs once in ``$init``.
    """
    return MXFP8Tensor(
        weight=F.concat([p.weight for p in parts]),
        weight_scale=F.concat([p.weight_scale for p in parts]),
    )


def dequantize(weight: MXFP8Tensor) -> Tensor:
    """Returns ``weight`` times its block scales as bfloat16.

    Exact: an E4M3 value times a power of two fits bfloat16.
    """
    rows, cols = (int(d) for d in weight.weight.shape)
    blocks = cols // _SF_VECTOR_SIZE
    values = F.cast(weight.weight, DType.float32).reshape(
        [rows, blocks, _SF_VECTOR_SIZE]
    )
    scales = F.cast(weight.weight_scale, DType.float32).reshape(
        [rows, blocks, 1]
    )
    return F.cast((values * scales).reshape([rows, cols]), DType.bfloat16)
