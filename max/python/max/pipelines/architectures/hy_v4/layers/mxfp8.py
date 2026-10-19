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
"""MXFP8 weight tensors, linear layer and grouped expert matmul (NVIDIA SM100).

The ModuleV3 front end of the kernels :func:`max.nn.quant_ops._matmul_float8_mxfp8`
and the MXFP8 arm of :class:`max.nn.moe.MoEQuantized` call, with the same
operands in the same order, so both front ends produce identical bits.

Weight contract with the weight adapter: ``data`` is ``float8_e4m3fn``
``[N, K]`` (``[E, N, K]`` for stacked experts, K contiguous) and
``weight_scale`` is the ``float8_e8m0fnu`` block scales already in the
tcgen05 SF-atom layout, ``[ceil(N / 128), ceil(K / 128), 32, 4, 4]`` (with a
leading ``E`` for stacked experts). The adapter prepares them on the host, so
weight init allocates no scale buffers beyond the weights themselves. An
:class:`MXFP8Linear` named ``proj`` loads them as ``proj.weight.data`` and
``proj.weight.weight_scale``, a stacked :class:`MXFP8Tensor` named
``experts`` as ``experts.data`` and ``experts.weight_scale``.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.quant_ops import (
    dynamic_block_scaled_matmul,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled,
)
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import TensorValue

MXFP8_BLOCK = 32
# One SM100 scale granule spans 128 rows and 128 elements of K.
_SF_GRANULE = 128


class MXFP8Tensor(Module[[], None]):
    """MXFP8 weight: E4M3 data plus one E8M0 scale per 32 elements along K."""

    data: Tensor
    weight_scale: Tensor

    def __init__(self, *, data: Tensor, weight_scale: Tensor) -> None:
        super().__init__()
        self.data = data
        self.weight_scale = weight_scale

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> MXFP8Tensor:
        """Builds a zero-filled tensor of logical ``shape`` (``[.., N, K]``)."""
        *lead, n, k = shape
        return cls(
            data=Tensor.zeros(shape, dtype=DType.float8_e4m3fn),
            weight_scale=Tensor.zeros(
                (*lead, -(-n // _SF_GRANULE), -(-k // _SF_GRANULE), 32, 4, 4),
                dtype=DType.float8_e8m0fnu,
            ),
        )

    def shard(self, spec: tuple[str | None, ...]) -> None:
        """Distributes ``data`` and ``weight_scale`` over their mesh by ``spec``.

        The scales' leading axes index the same experts, row granules and
        column granules as ``data``'s experts, N and K, so ``spec`` places
        them too; the three atom axes stay whole. A split must land on
        granule boundaries (128 rows, 128 elements of K).
        """
        self.data = self.data.to(NamedMapping(self.data.mesh, spec))
        self.weight_scale = self.weight_scale.to(
            NamedMapping(self.weight_scale.mesh, (*spec, None, None, None))
        )

    def forward(self) -> None:
        raise NotImplementedError("MXFP8Tensor is not a callable layer")


def _quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    return quantize_dynamic_block_scaled(
        x,
        sf_vector_size=MXFP8_BLOCK,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )


def matmul(x: Tensor, weight: MXFP8Tensor) -> Tensor:
    """Computes ``x @ weight.T`` in bf16 from bf16 ``x`` of shape ``[M, K]``."""
    x_fp8, x_scales = _quantize(x)
    return dynamic_block_scaled_matmul(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale,
        sf_vector_size=MXFP8_BLOCK,
        out_type=DType.bfloat16,
    )


def grouped_matmul(
    x: Tensor,
    weight: MXFP8Tensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    scales_offset: Tensor,
) -> Tensor:
    """Grouped expert matmul over stacked ``[E, N, K]`` MXFP8 weights.

    Args:
        x: bf16 tokens already permuted into expert order, ``[M, K]``.
        weight: Stacked expert weights.
        expert_start_indices: Row offsets per active expert, from
            ``moe_create_indices``.
        expert_ids: Expert id per active group, from ``moe_create_indices``.
        scales_offset: Per-expert offsets into the padded activation scales,
            from ``moe_create_indices(needs_scales_offset=True)``.

    Returns:
        The bf16 output, ``[M, N]``, in the same row order as ``x``.
    """
    unit_scales = F.broadcast_to(
        F.constant(1.0, DType.float32, device=x.device), expert_ids.shape
    )
    x_fp8, x_scales = grouped_quantize_dynamic_block_scaled(
        x,
        row_offsets=expert_start_indices,
        scales_offsets=scales_offset,
        expert_ids=expert_ids,
        sf_tensor=unit_scales,
        sf_vector_size=MXFP8_BLOCK,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    # The kernel reads the active-expert count from host stats; 8192 is the
    # max-tokens-per-expert bound the graph-API path passes too.
    usage_stats_host = F.constant(
        [8192, int(expert_ids.shape[0])], dtype=DType.uint32, device=CPU()
    )
    estimated_total_m = F.shape_to_tensor(x.shape)[0].cast(DType.uint32)
    return grouped_matmul_block_scaled(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale,
        expert_start_indices,
        scales_offset,
        expert_ids,
        unit_scales,
        usage_stats_host,
        out_type=DType.bfloat16,
        estimated_total_m=TensorValue(estimated_total_m),
    )


class MXFP8Linear(Module[[Tensor], Tensor]):
    """Bias-free linear layer ``x @ weight.T`` with an MXFP8 weight."""

    weight: MXFP8Tensor

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = MXFP8Tensor.zeros((out_dim, in_dim))

    def forward(self, x: Tensor) -> Tensor:
        return matmul(x, self.weight)
