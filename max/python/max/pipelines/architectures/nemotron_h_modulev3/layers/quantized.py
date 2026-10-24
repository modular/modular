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
"""Linear layers that run the checkpoint's quantized weights as stored."""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.linear import _float8_matmul_op, _matmul_mapping
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.nn import kernels
from max.nn.quant_config import QuantConfig
from max.support.math import ceildiv

from ..model_config import NemotronHConfig
from ..quantization import (
    FP8_STATIC_TENSOR_QUANT,
    NVFP4_GROUP_SIZE,
    ModuleFormat,
    Parallelism,
    linear_parallelism,
)
from .sharding import shard_segments

# The largest E2M1 value times the largest E4M3 value: a row scaled by
# amax / _NVFP4_RANGE has its largest block scale at the E4M3 maximum.
_NVFP4_RANGE = 6.0 * 448.0

# Whether to quantize activations with the fused row-amax kernel
# (`kernels.grouped_quantize_nvfp4_row_scaled`), whose output is
# bit-identical to the op chain in `nvfp4_grouped_matmul`. The fused kernel
# makes the B=64 decode step about 8% faster (SERVOPT-1732); the chain stays
# as the reference path.
_FUSED_NVFP4_ROW_QUANT = True


def nvfp4_grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    block_scales: TensorValue,
    global_scales: TensorValue,
    expert_start_indices: TensorValue,
    scales_offsets: TensorValue,
    expert_ids: TensorValue,
    rows: TensorValue | None = None,
    relu2: bool = False,
) -> TensorValue:
    """Quantizes ``x`` to NVFP4 per row and runs the W4A4 grouped matmul.

    The checkpoint has no activation scale, so each row gets its own global
    scale from its amax, which the matmul epilogue multiplies back in. With
    ``_FUSED_NVFP4_ROW_QUANT``, one kernel takes the amax, rounds the scales,
    quantizes and applies relu2; otherwise a chain of ops does the same, with
    the same bits out.

    Args:
        x: BF16 activations, ``[tokens, K]``.
        weight: Packed E2M1 expert weights, ``[experts, N, K / 2]``.
        block_scales: The weights' interleaved E4M3 scales.
        global_scales: The weights' float32 ``weight_scale_2``, ``[experts]``.
        expert_start_indices: Each expert's first routed row.
        scales_offsets: Each expert's scale-tile offset.
        expert_ids: The expert of each group.
        rows: The ``x`` row of each routed row, when ``x`` is not already in
            routed order.
        relu2: Whether to multiply with ``relu(x) ** 2`` in place of ``x``.

    Returns:
        The BF16 output, ``[routed rows, N]``.
    """
    num_experts = int(weight.shape[0])
    if _FUSED_NVFP4_ROW_QUANT:
        x_fp4, x_block_scales, row_scales = (
            kernels.grouped_quantize_nvfp4_row_scaled(
                x,
                row_offsets=expert_start_indices,
                scales_offsets=scales_offsets,
                indices=rows,
                relu2=relu2,
            )
        )
    else:
        if relu2:
            r = ops.relu(x)
            x = r * r
        amax = ops.max(ops.abs(ops.cast(x, DType.float32)), axis=-1)
        # A row of zeros would otherwise divide by zero.
        row_scales = ops.cast(
            ops.max(
                amax / _NVFP4_RANGE,
                ops.constant(1e-30, DType.float32, x.device),
            ),
            DType.bfloat16,
        )
        # Float32 division is slow when the numerator is zero, and relu2
        # makes about half of these zero, so multiply by the reciprocal.
        scaled = ops.cast(
            ops.cast(x, DType.float32)
            * (1.0 / ops.cast(row_scales, DType.float32)),
            DType.bfloat16,
        )
        row_scales = ops.reshape(row_scales, [-1])
        if rows is not None:
            row_scales = ops.gather(row_scales, rows, axis=0)
        x_fp4, x_block_scales = kernels.grouped_quantize_dynamic_block_scaled(
            scaled,
            row_offsets=expert_start_indices,
            scales_offsets=scales_offsets,
            expert_ids=expert_ids,
            sf_tensor=ops.broadcast_to(
                ops.constant(1.0, DType.float32, x.device), [num_experts]
            ),
            sf_vector_size=NVFP4_GROUP_SIZE,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
            indices=rows,
        )
    # The kernel picks its tiling from the average rows per expert.
    routed_rows = ops.cast(
        ops.shape_to_tensor([x_fp4.shape[0]])[0], DType.uint32
    )
    return kernels.grouped_matmul_block_scaled(
        x_fp4,
        weight,
        x_block_scales,
        block_scales,
        expert_start_indices,
        scales_offsets,
        expert_ids,
        global_scales,
        ops.constant(
            [8192, num_experts], dtype=DType.uint32, device=DeviceRef.CPU()
        ),
        estimated_total_m=routed_rows,
        a_row_scales=row_scales,
    )


def nvfp4_matmul(
    x: TensorValue,
    weight: TensorValue,
    block_scales: TensorValue,
    global_scale: TensorValue,
) -> TensorValue:
    """Runs ``x @ weight.T`` as a one-group W4A4 grouped matmul.

    The dense block-scaled matmul takes one activation scale for the whole
    batch, which would make each request's output depend on the others in
    its batch. The grouped matmul scales each row on its own.

    Args:
        x: BF16 activations, ``[tokens, K]``.
        weight: Packed E2M1 weight, ``[N, K / 2]``.
        block_scales: The weight's interleaved E4M3 scales, ``[1,
            ceil(N / 128), K / 64, 32, 4, 4]``.
        global_scale: The weight's float32 ``weight_scale_2``, ``[1]``.

    Returns:
        The BF16 output, ``[tokens, N]``.
    """
    device = x.device
    cpu = DeviceRef.CPU()
    tokens = ops.cast(ops.shape_to_tensor([x.shape[0]])[0], DType.uint32)
    # The group's row offsets, [0, tokens], are generated on the device: a
    # host-to-device copy would invalidate device graph capture. A batch
    # always has a token, and the floor of 1 only keeps the step nonzero.
    step = ops.max(tokens, ops.constant(1, DType.uint32, cpu))
    return nvfp4_grouped_matmul(
        x,
        ops.unsqueeze(weight, 0),
        block_scales,
        global_scale,
        expert_start_indices=ops.range(
            ops.constant(0, DType.uint32, cpu),
            step * 2,
            step,
            out_dim=2,
            dtype=DType.uint32,
            device=device,
        ),
        scales_offsets=ops.constant([0], DType.uint32, device),
        expert_ids=ops.constant([0], DType.int32, device),
    )


def _place(weight: Tensor, parallelism: Parallelism) -> Tensor:
    """Places a ``[N, K]`` weight's rows or columns across the TP axis."""
    match parallelism:
        case Parallelism.COLUMN:
            return weight.to(NamedMapping(weight.mesh, (TP, None)))
        case Parallelism.ROW:
            return weight.to(NamedMapping(weight.mesh, (None, TP)))
    return weight


def nvfp4_block_scale_shape(
    out_dim: int, in_dim: int, parallelism: Parallelism, num_devices: int
) -> list[int]:
    """Returns the shape of an NVFP4 linear's interleaved block scales on
    the devices.

    Each device's share is interleaved on its own, since the interleaved
    layout pads the output rows to whole 128-row granules. The first axis
    holds the shares.
    """
    parts = 1 if parallelism is Parallelism.REPLICATED else num_devices
    rows = (
        out_dim // num_devices if parallelism is Parallelism.COLUMN else out_dim
    )
    cols = in_dim // num_devices if parallelism is Parallelism.ROW else in_dim
    return [
        parts,
        ceildiv(rows, 128),
        cols // (4 * NVFP4_GROUP_SIZE),
        32,
        4,
        4,
    ]


class NVFP4Linear(Module[[Tensor], Tensor]):
    """A linear layer with NVFP4 weights, run W4A4.

    The checkpoint quantizes the weight only. Each activation row is
    quantized to NVFP4 with its own scale, as the routed experts are.

    The block scales are placed like the weight, as the checkpoint stores
    them, and the graph interleaves each device's share for the
    block-scaled matmul. The quantized kernels have no sharding rules, so
    each device runs its own share: a column-parallel layer splits the
    output's columns, and a row-parallel one leaves each device its share
    of the sum, which the residual add reduces.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        parallelism: Parallelism,
        num_devices: int,
        row_segments: Sequence[int] = (),
    ) -> None:
        """Initializes the layer.

        Args:
            in_dim: The input dimension.
            out_dim: The output dimension.
            parallelism: How the weight splits across devices.
            num_devices: The number of devices.
            row_segments: The sizes of the parts the weight's rows stack,
                each split across devices, or none for one part. Only a
                column-parallel layer has parts.
        """
        self.parallelism = parallelism
        self.row_segments = tuple(row_segments) if num_devices > 1 else ()
        shapes = (
            ([out_dim, in_dim // 2], DType.uint8),
            ([out_dim, in_dim // NVFP4_GROUP_SIZE], DType.float8_e4m3fn),
        )
        if self.row_segments:
            # A plain split would give a device whole parts, so both stay on
            # the host for the graph to give each device its share of each.
            self.weight, self.weight_scale = (
                Tensor.zeros(shape, dtype=dtype, device=CPU())
                for shape, dtype in shapes
            )
        else:
            self.weight, self.weight_scale = (
                _place(Tensor.zeros(shape, dtype=dtype), parallelism)
                for shape, dtype in shapes
            )
        self.weight_scale_2 = Tensor.zeros([], dtype=DType.float32)

    def placed(self, mesh: DeviceMesh) -> tuple[Tensor, Tensor]:
        """Returns the weight and its block scales, each device holding its
        share."""
        if not self.row_segments:
            return self.weight, self.weight_scale
        return (
            shard_segments(self.weight, self.row_segments, mesh),
            shard_segments(self.weight_scale, self.row_segments, mesh),
        )

    def device_params(
        self, mesh: DeviceMesh
    ) -> list[tuple[TensorValue, TensorValue, TensorValue]]:
        """Returns each device's weight, interleaved block scales and global
        scale, as the matmul reads them."""
        weight, scales = self.placed(mesh)
        with ensure_context():
            return [
                (
                    TensorValue(weight.local_shards[d]),
                    ops.unsqueeze(
                        kernels.block_scales_interleave(
                            TensorValue(scales.local_shards[d]),
                            NVFP4_GROUP_SIZE,
                        ),
                        0,
                    ),
                    ops.reshape(
                        TensorValue(self.weight_scale_2.local_shards[d]), [1]
                    ),
                )
                for d in range(mesh.num_devices)
            ]

    def matmul_on(self, x: TensorValue, device: int) -> TensorValue:
        """Runs ``x @ weight.T`` on one device with its share of the weight."""
        params = self.device_params(self.weight_scale_2.mesh)[device]
        return nvfp4_matmul(x, *params)

    def forward(self, x: Tensor) -> Tensor:
        with ensure_context():
            shards = [
                nvfp4_matmul(TensorValue(shard), *params)
                for shard, params in zip(
                    x.local_shards, self.device_params(x.mesh), strict=True
                )
            ]
        if (
            x.mesh.num_devices == 1
            or self.parallelism is Parallelism.REPLICATED
        ):
            mapping = x.mapping
        elif self.parallelism is Parallelism.COLUMN:
            mapping = DeviceMapping(x.mesh, (Sharded(x.rank - 1),))
        else:
            mapping = DeviceMapping(x.mesh, (Partial(),))
        return Tensor.from_shard_values(shards, mapping)


class RowGroupedLinear(Linear):
    """A column-parallel linear whose weight rows stack parts, each split
    across devices.

    A plain split of the rows would give a device whole parts. The weight
    stays whole on the host as stored, and the graph gives each device its
    share of every part.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        row_segments: Sequence[int],
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__(in_dim, out_dim, bias=False, quant_config=quant_config)
        self.weight = Tensor.zeros(
            [out_dim, in_dim], dtype=self.weight.dtype, device=CPU()
        )
        self.row_segments = tuple(row_segments)

    def placed_weight(self, mesh: DeviceMesh) -> Tensor:
        """Returns the weight, each device holding its rows."""
        return shard_segments(self.weight, self.row_segments, mesh)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.placed_weight(x.mesh)
        if self.quant_config is None:
            return x @ weight.T
        y = _float8_matmul_op(x, weight, self.weight_scale, self.input_scale)
        return y.rebind_mapping(_matmul_mapping(x, weight))


def quantized_linear(
    config: NemotronHConfig, module: str, row_segments: Sequence[int] = ()
) -> Linear | NVFP4Linear:
    """Builds a dense linear module that reads its weight as stored.

    Args:
        config: The model config.
        module: The module's checkpoint path, such as
            ``backbone.layers.0.mixer.in_proj``.
        row_segments: The sizes of the parts a column-parallel weight's rows
            stack, each split across devices, or none for one part.
    """
    in_dim, out_dim = config.linear_shape(module)
    parallelism = linear_parallelism(module)
    fmt = config.quant_scheme.format_of(module)
    num_devices = len(config.devices)
    if fmt is ModuleFormat.NVFP4_WEIGHT_ONLY:
        return NVFP4Linear(
            in_dim, out_dim, parallelism, num_devices, row_segments
        )
    quant_config = (
        FP8_STATIC_TENSOR_QUANT
        if fmt is ModuleFormat.FP8_STATIC_TENSOR
        else None
    )
    if row_segments and num_devices > 1:
        return RowGroupedLinear(in_dim, out_dim, row_segments, quant_config)
    linear = Linear(in_dim, out_dim, bias=False, quant_config=quant_config)
    match parallelism:
        case Parallelism.COLUMN:
            return col_parallel(linear)
        case Parallelism.ROW:
            return row_parallel(linear)
    return linear
