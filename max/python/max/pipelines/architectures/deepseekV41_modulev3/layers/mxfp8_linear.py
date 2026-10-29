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
"""Dense V4.1 FP8 projections with compact checkpoint scales.

The basic production path only: each site quantizes its bf16 activations
into per-32-block E4M3 codes with a UE8M0 block scale (the QAT recipe,
amax floored at ``1e-4``), interleaves both scale sets into the
block-scaled matmul's atom layout, and runs
``dynamic_block_scaled_matmul``. The checkpoint holds the weights as
compact E4M3 codes with one E8M0 scale per 32x32 block, never
dequantized.

The one grouped site -- the attention output's ``wo_a`` -- runs the same
quantize once, token-major, and one block-diagonal block-scaled matmul
(main #105486's mechanism, :func:`grouped_dense_op`); CPU and Apple
traces, and a shard too narrow to tile, keep the grouped reference
launch.
"""

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
)
from max.nn.legacy.kernels import (
    _is_apple_gpu,
    _is_sm10x_gpu,
    block_diagonal_block_scaled_matmul,
    block_scales_interleave,
    dynamic_block_scaled_matmul,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.types import TensorLayout


def _act_quant(
    x: TensorValue, amax_floor: float = 1e-4
) -> tuple[TensorValue, TensorValue]:
    """The reference activation quantize: E4M3 codes and UE8M0 block scales.

    Per 32-wide block of the last axis, in fp32: ``amax = max(|x|, floor)``,
    ``scale = 2 ** ceil(log2(amax / 448))`` (the two ``where`` corrections
    keep the ceil exact at power-of-two boundaries), and the codes clamp to
    ``+-448`` before the E4M3 cast. The caller interleaves the scales.
    """
    k = int(x.shape[-1])
    if k % 32:
        raise ValueError("MXFP8 activation K must be divisible by 32")
    grouped = ops.reshape(ops.cast(x, DType.float32), [-1, k // 32, 32])
    amax = ops.max(ops.abs(grouped), axis=-1)
    unrounded = ops.max(amax, amax_floor) * (1.0 / 448.0)
    # Correct the logarithm's last-bit error at exact power-of-two boundaries.
    candidate = ops.pow(
        2.0, ops.floor(ops.log(unrounded) * (1.0 / 0.6931471805599453))
    )
    scale = ops.where(candidate < unrounded, candidate * 2.0, candidate)
    scale = ops.where(scale / 2.0 >= unrounded, scale / 2.0, scale)
    codes = ops.cast(
        ops.min(ops.max(grouped / scale, -448.0), 448.0), DType.float8_e4m3fn
    )
    return ops.reshape(codes, x.shape), ops.cast(
        ops.squeeze(scale, axis=-1), DType.float8_e8m0fnu
    )


act_quant = F.functional(_act_quant)


def _dense(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    amax_floor: float,
) -> TensorValue:
    """``x @ weight.T`` on the block-scaled MXFP8 matmul, bf16 out.

    The weight arrives as compact checkpoint leaves (E4M3 codes plus one
    E8M0 scale per 32x32 block); its scale is expanded per output row and
    interleaved, the activation quantize feeds the matching interleave, and
    the production ``dynamic_block_scaled_matmul`` computes the product.
    """
    if (
        weight.dtype != DType.float8_e4m3fn
        or scale.dtype != DType.float8_e8m0fnu
    ):
        raise ValueError(
            "V4.1 dense weights require E4M3 and compact E8M0 scales"
        )
    rows = int(weight.shape[0])
    padding = (-rows) % 32
    if padding:
        # SM100's output TMA requires an aligned row stride, including when
        # the checkpoint's final 32-row scale block is partial.
        weight = ops.pad(weight, [0, padding, 0, 0])
    codes, scales = _act_quant(x, amax_floor)
    a_scales = block_scales_interleave(scales, sf_vector_size=32)
    # This function runs per shard: repeat only the already sliced
    # compact leaf. The expansion is two plain elementwise launches.
    expanded = ops.repeat_interleave(scale, 32, axis=0)[:rows]
    if padding:
        expanded = ops.pad(expanded, [0, padding, 0, 0], value=1.0)
    b_scales = block_scales_interleave(expanded, sf_vector_size=32)
    return dynamic_block_scaled_matmul(
        codes,
        weight,
        a_scales,
        b_scales,
        sf_vector_size=32,
        out_type=DType.bfloat16,
    )[:, :rows]


def _dense_rule(
    x: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
    amax_floor: float,
) -> list[AxisAssignment]:
    return [
        AxisAssignment((R, R, R), (R,)),
        AxisAssignment((R, Sharded(0), Sharded(0)), (Sharded(1),)),
        AxisAssignment((Sharded(1), Sharded(1), Sharded(1)), (P,)),
        AxisAssignment((Sharded(0), R, R), (Sharded(0),)),
    ]


_dense_op = F.functional(_dense, rule=_dense_rule)


def _trace_device_is_gpu(device: object) -> bool:
    """Whether a trace-time device names a GPU, in either of its two forms.

    A guard runs on devices in two flavors: the raw driver ``Accelerator``
    of a tensor the op built itself, and the graph ``DeviceRef`` of every
    value ``F.functional`` converts for a wrapped op's parameters and
    every ``ops.*`` result. No single method or isinstance answers both;
    this call does.
    """
    if isinstance(device, DeviceRef):
        return device.is_gpu()
    return isinstance(device, Accelerator)


def _use_block_diagonal_woa(x: TensorValue) -> bool:
    """Whether the block-diagonal grouped matmul can take these operands.

    The op is SM100-only; CPU and Apple traces keep the grouped
    reference launch below.
    """
    return (
        _trace_device_is_gpu(x.device)
        and not _is_apple_gpu()
        and _is_sm10x_gpu()
    )


def _grouped_dense(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
    amax_floor: float,
) -> TensorValue:
    """``[T, g * d] -> [T, g * group_rows]``: group ``j`` uses its own rows.

    Runs per shard, where ``g`` is the shard's group count. Each group is a
    dense projection, so the weight is read as FP8 and never dequantized.
    With two or more groups on SM100 the groups run as ONE
    block-diagonal block-scaled matmul (main #105486's mechanism): one
    token-major quantize over the whole width, one launch. The per-32
    amax blocks the quantize computes are independent of the group
    slicing, so the one launch's codes are the bytes the per-group
    chains produce, and each output column keeps its separate-launch
    value. Every other trace -- a non-SM100 device, or a site whose K
    or group rank is not whole 128-block tiles -- runs the grouped
    reference launch (:func:`_grouped_dense_reference`); a rank below a
    whole 32-row scale block is refused below, before any arm runs.
    """
    if (
        weight.dtype != DType.float8_e4m3fn
        or scale.dtype != DType.float8_e8m0fnu
    ):
        raise ValueError(
            "V4.1 dense weights require E4M3 and compact E8M0 scales"
        )
    rows = int(weight.shape[0])
    if rows % group_rows:
        raise ValueError(
            f"grouped weight holds {rows} rows, not a whole multiple of the"
            f" {group_rows}-row group"
        )
    if group_rows % 32:
        # A rank below a whole 32-row scale block has no scale rows for
        # its own group: the reference arm would slice its compact scales
        # an empty range, and the single-group dense path would read a
        # degenerate zero-row scale leaf as fabricated unit scales.
        raise ValueError(
            f"group rank {group_rows} must hold whole 32-row scale blocks"
        )
    groups = rows // group_rows
    if groups == 1:
        return _dense(x, weight, scale, amax_floor)
    k = int(weight.shape[1])
    # The block-diagonal launch also needs whole 128-column scale tiles
    # along K, and its B-scales are one interleave over the whole shard
    # split per group by a view, so a group must cover whole 128-row
    # interleave blocks. A site that misses either -- rank 64 included --
    # keeps the grouped reference launch, which needs only the whole
    # 32-row blocks above.
    if _use_block_diagonal_woa(x) and k % 128 == 0 and group_rows % 128 == 0:
        return _grouped_dense_block_diagonal(
            x, weight, scale, group_rows, amax_floor
        )
    return _grouped_dense_reference(x, weight, scale, group_rows, amax_floor)


def _grouped_dense_block_diagonal(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
    amax_floor: float,
) -> TensorValue:
    """The production launch: one token-major quantize, one matmul.

    The whole-width quantize's codes and interleaved scales are
    byte-identical to the per-group chains the reference arm runs (the
    per-32 amax blocks never straddle a group boundary), so the matmul
    reads the same values the grouped launch would -- without the
    per-group chains, their interleaves and the per-group concat of the
    products.
    """
    groups = int(weight.shape[0]) // group_rows
    codes, scales = _act_quant(x, amax_floor)
    a_scales = block_scales_interleave(scales, sf_vector_size=32)
    # The weight-side interleave is the same bytes the reference arm
    # builds per group, unsplit: [rows / 128, k / 128, 32, 4, 4].
    expanded = ops.repeat_interleave(scale, 32, axis=0)
    b_scales = block_scales_interleave(expanded, sf_vector_size=32)
    return block_diagonal_block_scaled_matmul(
        codes, weight, a_scales, b_scales, groups
    )


def _grouped_dense_reference(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
    amax_floor: float,
) -> TensorValue:
    """The reference arm: one quantize chain and one dense launch per group.

    The values every arm must reproduce: each group quantizes its own
    K-slice of ``x`` through the reference chain and multiplies it by
    its own rows of the leaf -- the per-group dense launches the
    block-diagonal matmul replaces with one batched kernel.
    """
    groups = int(weight.shape[0]) // group_rows
    width = int(x.shape[-1]) // groups
    outputs: list[TensorValue] = []
    for j in range(groups):
        codes, group_scales = _act_quant(
            x[:, j * width : (j + 1) * width], amax_floor
        )
        a_scales = block_scales_interleave(group_scales, sf_vector_size=32)
        # Group j's compact scale rows are its own weight rows' blocks:
        # the leaf's rows are group-ordered.
        blocks = scale[j * group_rows // 32 : (j + 1) * group_rows // 32]
        b_scales = block_scales_interleave(
            ops.repeat_interleave(blocks, 32, axis=0), sf_vector_size=32
        )
        outputs.append(
            dynamic_block_scaled_matmul(
                codes,
                weight[j * group_rows : (j + 1) * group_rows],
                a_scales,
                b_scales,
                sf_vector_size=32,
                out_type=DType.bfloat16,
            )
        )
    return ops.concat(outputs, axis=1)


def _grouped_dense_rule(
    x: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
    group_rows: int,
    amax_floor: float,
) -> list[AxisAssignment]:
    return [
        AxisAssignment((R, R, R), (R,)),
        AxisAssignment((Sharded(0), R, R), (Sharded(0),)),
        AxisAssignment((Sharded(1), Sharded(0), Sharded(0)), (Sharded(1),)),
    ]


grouped_dense_op = F.functional(_grouped_dense, rule=_grouped_dense_rule)


def _gate_up(x: TensorValue) -> tuple[TensorValue, TensorValue]:
    """Splits a stacked ``[gate | up]`` projection output down the middle.

    The stacked ``w13`` leaf interleaves its parts per TP rank
    (:func:`~..weight_adapters.stack_projections`), so each rank's slice
    is ``[gate_r | up_r]`` of equal halves -- a per-rank half split, never
    a global one. Positive bounds only: the upper bound is the merged
    width, never a negative index.
    """
    width = int(x.shape[-1]) // 2
    return x[:, :width], x[:, width : int(x.shape[-1])]


def _gate_up_rule(x: TensorLayout) -> list[AxisAssignment]:
    return [
        AxisAssignment((R,), (R, R)),
        AxisAssignment((Sharded(0),), (Sharded(0), Sharded(0))),
        AxisAssignment((Sharded(1),), (Sharded(1), Sharded(1))),
    ]


gate_up_op = F.functional(_gate_up, rule=_gate_up_rule)


class MXFP8Linear(Module[[Tensor], Tensor]):
    """A bias-free E4M3 projection with compact 32x32 E8M0 scales.

    Args:
        in_dim: Input width, divisible by 32.
        out_dim: Output width; a partial final scale block is trimmed.
        amax_floor: Minimum activation block amax before ceil-pow2 scaling.
    """

    weight: Tensor
    scale: Tensor

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        amax_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if in_dim % 32:
            raise ValueError("MXFP8 input width must be divisible by 32")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.amax_floor = amax_floor
        self.weight = Tensor.zeros((out_dim, in_dim), dtype=DType.float8_e4m3fn)
        self.scale = Tensor.zeros(
            ((out_dim + 31) // 32, in_dim // 32), dtype=DType.float8_e8m0fnu
        )

    def shard(self, axis: int, mesh: DeviceMesh) -> "MXFP8Linear":
        """Shards compact leaves together, on whole 32-element blocks."""
        if axis not in (0, 1) or int(self.weight.shape[axis]) % (
            32 * mesh.num_devices
        ):
            raise ValueError(
                "TP slices must contain whole 32-element scale blocks"
            )
        mapping = DeviceMapping(mesh, (Sharded(axis),))
        self.weight = self.weight.to(mapping)
        self.scale = self.scale.to(mapping)
        if mesh.num_devices == 1:
            # Single-device realization canonicalizes placements to R;
            # retain the checkpoint's S0/S1 intent without moving data.
            self.weight = self.weight.rebind_mapping(
                DeviceMapping(self.weight.mesh, (Sharded(axis),))
            )
            self.scale = self.scale.rebind_mapping(
                DeviceMapping(self.scale.mesh, (Sharded(axis),))
            )
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Quantizes 1x32 activation groups and applies the projection."""
        if isinstance(self.weight, FP8BlockTensor):
            raise ValueError(
                "V4.1 requires compact E8M0 scales, not FP8BlockTensor"
            )
        return _dense_op(x, self.weight, self.scale, self.amax_floor)
