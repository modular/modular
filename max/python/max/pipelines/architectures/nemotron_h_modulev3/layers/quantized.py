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

from max.dtype import DType
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.linear import _matmul_mapping
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.nn import kernels
from max.support.math import ceildiv

from ..model_config import NemotronHConfig
from ..quantization import (
    FP8_STATIC_TENSOR_QUANT,
    NVFP4_GROUP_SIZE,
    ModuleFormat,
    Parallelism,
    linear_parallelism,
)

# The largest E2M1 value times the largest E4M3 value: a row scaled by
# amax / _NVFP4_RANGE has its largest block scale at the E4M3 maximum.
_NVFP4_RANGE = 6.0 * 448.0

# The SM100 block-scaled matmul reads its scales in granules of 128 rows by 4
# scale columns.
_SF_GRANULE_ROWS = 128
_SF_ATOM_ROWS = 32
_SF_ATOM_COLS = 4


def interleave_expert_scales(scales: TensorValue) -> TensorValue:
    """Permutes stacked E4M3 block scales into the kernel layout.

    Each expert's element ``(r, c)`` lands at ``[r // 128, c // 4, r % 32,
    (r % 128) // 32, c % 4]``, where ``set_scale_factor`` in
    ``max/kernels/src/linalg/fp4_utils.mojo`` stores it. Rows are
    zero-padded to a whole granule. The scales are weights, so the graph
    compiler runs this once, at model init.

    Args:
        scales: The ``[experts, N, K / 16]`` scales. ``K / 16`` must be a
            multiple of 4.

    Returns:
        The ``[experts, ceil(N / 128), K / 64, 32, 4, 4]`` scales.
    """
    experts, rows, cols = (int(d) for d in scales.shape)
    if cols % _SF_ATOM_COLS:
        raise ValueError(f"cannot interleave NVFP4 scales {scales.shape}")
    granules = ceildiv(rows, _SF_GRANULE_ROWS)
    padding = granules * _SF_GRANULE_ROWS - rows
    if padding:
        scales = ops.pad(scales, [0, 0, 0, padding, 0, 0])
    atoms = ops.reshape(
        scales,
        [
            experts,
            granules,
            _SF_GRANULE_ROWS // _SF_ATOM_ROWS,
            _SF_ATOM_ROWS,
            cols // _SF_ATOM_COLS,
            _SF_ATOM_COLS,
        ],
    )
    return ops.permute(atoms, [0, 1, 4, 3, 2, 5])


def nvfp4_grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    block_scales: TensorValue,
    global_scales: TensorValue,
    expert_start_indices: TensorValue,
    scales_offsets: TensorValue,
    expert_ids: TensorValue,
    rows: TensorValue | None = None,
) -> TensorValue:
    """Quantizes ``x`` to NVFP4 per row and runs the W4A4 grouped matmul.

    The checkpoint has no activation scale, so each row gets its own global
    scale from its amax, which the matmul epilogue multiplies back in.

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

    Returns:
        The BF16 output, ``[routed rows, N]``.
    """
    amax = ops.max(ops.abs(ops.cast(x, DType.float32)), axis=-1)
    # A row of zeros would otherwise divide by zero.
    row_scales = ops.cast(
        ops.max(
            amax / _NVFP4_RANGE, ops.constant(1e-30, DType.float32, x.device)
        ),
        DType.bfloat16,
    )
    # Float32 division is slow when the numerator is zero, and relu2 makes
    # about half of these zero, so multiply by the reciprocal instead.
    scaled = ops.cast(
        ops.cast(x, DType.float32)
        * (1.0 / ops.cast(row_scales, DType.float32)),
        DType.bfloat16,
    )
    row_scales = ops.reshape(row_scales, [-1])
    if rows is not None:
        row_scales = ops.gather(row_scales, rows, axis=0)
    num_experts = int(weight.shape[0])
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


class _PerDeviceLinear(Module[[Tensor], Tensor]):
    """Runs a quantized matmul on each device's shard of the input.

    The quantized kernels have no sharding rules, so the output takes the
    placement the framework's FP8 :class:`Linear` derives from its weight's.
    A row-parallel layer leaves each device its share of the sum, which the
    residual add reduces.
    """

    weight: Tensor

    def _local(self, x: TensorValue, device: int) -> TensorValue:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        with ensure_context():
            shards = [
                self._local(TensorValue(shard), d)
                for d, shard in enumerate(x.local_shards)
            ]
        mapping = (
            _matmul_mapping(x, self.weight)
            if x.mesh.num_devices > 1
            else x.mapping
        )
        return Tensor.from_shard_values(shards, mapping)


def nvfp4_block_scale_shape(
    out_dim: int, in_dim: int, parallelism: Parallelism, num_devices: int
) -> list[int]:
    """Returns the shape of an NVFP4 linear's interleaved block scales.

    Each device's share is interleaved on its own, since the interleaved
    layout pads the output rows to whole 128-row granules. The first axis
    holds the shares, so a sharded layer splits it across devices.
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


class NVFP4Linear(_PerDeviceLinear):
    """A linear layer with NVFP4 weights, run W4A4.

    The checkpoint quantizes the weight only. Each activation row is
    quantized to NVFP4 with its own scale, as the routed experts are.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        parallelism: Parallelism,
        num_devices: int,
    ) -> None:
        self.weight = _place(
            Tensor.zeros([out_dim, in_dim // 2], dtype=DType.uint8),
            parallelism,
        )
        block_scale = Tensor.zeros(
            nvfp4_block_scale_shape(out_dim, in_dim, parallelism, num_devices),
            dtype=DType.float8_e4m3fn,
        )
        if parallelism is not Parallelism.REPLICATED:
            block_scale = block_scale.to(
                NamedMapping(block_scale.mesh, (TP,) + (None,) * 5)
            )
        self.weight_scale = block_scale
        self.weight_scale_2 = Tensor.zeros([1], dtype=DType.float32)

    def _local(self, x: TensorValue, device: int) -> TensorValue:
        return nvfp4_matmul(
            x,
            TensorValue(self.weight.local_shards[device]),
            TensorValue(self.weight_scale.local_shards[device]),
            TensorValue(self.weight_scale_2.local_shards[device]),
        )


def quantized_linear(
    config: NemotronHConfig, module: str
) -> Linear | NVFP4Linear:
    """Builds a dense linear module that reads its weight as stored.

    Args:
        config: The model config.
        module: The module's checkpoint path, such as
            ``backbone.layers.0.mixer.in_proj``.
    """
    in_dim, out_dim = config.linear_shape(module)
    parallelism = linear_parallelism(module)
    fmt = config.quant_scheme.format_of(module)
    if fmt is ModuleFormat.NVFP4_WEIGHT_ONLY:
        return NVFP4Linear(in_dim, out_dim, parallelism, len(config.devices))
    quant_config = (
        FP8_STATIC_TENSOR_QUANT
        if fmt is ModuleFormat.FP8_STATIC_TENSOR
        else None
    )
    linear = Linear(in_dim, out_dim, bias=False, quant_config=quant_config)
    match parallelism:
        case Parallelism.COLUMN:
            return col_parallel(linear)
        case Parallelism.ROW:
            return row_parallel(linear)
    return linear
