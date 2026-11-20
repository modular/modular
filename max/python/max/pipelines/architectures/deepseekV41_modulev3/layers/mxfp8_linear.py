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

Each site quantizes its bf16 activations through main's fused
``quantize_dynamic_block_scaled`` (per-32-block E4M3 codes with UE8M0
block scales, already in the matmul's atom layout) and runs the
cluster split-K MXFP8 matmul V2 serves with. The checkpoint holds the
weights as compact E4M3 codes with one E8M0 scale per 32x32 block, never
dequantized.

The one grouped site -- the attention output's ``wo_a`` -- runs the same
quantize once, token-major, and one block-diagonal block-scaled matmul
(main #105486's mechanism, :func:`grouped_dense_op`); CPU and Apple
traces, and a shard too narrow to tile, keep the grouped reference
launch. ``wq_b`` rotates its RoPE tail in the matmul epilogue
(:func:`dense_rope_tail_op`).
"""

from dataclasses import dataclass

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn import Module
from max.nn._graph_ops import (
    _is_apple_gpu,
    _is_sm10x_gpu,
    block_scales_interleave,
    dynamic_block_scaled_matmul,
    quantize_dynamic_block_scaled,
    rope_quantize_dynamic_block_scaled,
    swiglu_quantize_dynamic_block_scaled,
)
from max.nn.activation import GatedActivation
from max.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.types import TensorLayout

from ...deepseekV4.layers.rope import apply_rope_tail as apply_rope_tail_values
from ...deepseekV41.layers.mxfp8_matmul import mxfp8_block_diagonal
from ...deepseekV41.model_config import mxfp8_splitk_enabled


def _quantize(x: TensorValue) -> tuple[TensorValue, TensorValue]:
    """E4M3 codes and UE8M0 block scales of ``x``, scales in the atom layout.

    Main's fused quantize emits the interleaved layout on SM100; a trace
    for any other device gets rank-2 scales, which are interleaved here so
    the reference arms keep tracing there.
    """
    if int(x.shape[-1]) % 32:
        raise ValueError("MXFP8 activation K must be divisible by 32")
    codes, scales = quantize_dynamic_block_scaled(
        ops.cast(x, DType.bfloat16),
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    if scales.rank == 2:
        scales = block_scales_interleave(scales, sf_vector_size=32)
    return codes, scales


@dataclass(frozen=True)
class Mxfp8Rows:
    """Activation codes and scales in the matmul's interleaved layout."""

    codes: Tensor
    scales: Tensor


def _check_compact(weight: TensorValue, scale: TensorValue) -> None:
    """Accept ceil-div scales; matmul pads N while load stacking needs full blocks."""
    if weight.rank != 2 or scale.rank != 2:
        raise ValueError("V4.1 dense weights and scales must be rank-2")
    rows, cols = (int(d) for d in weight.shape)
    expected = ((rows + 31) // 32, cols // 32)
    if (
        cols % 32
        or tuple(int(d) for d in scale.shape) != expected
        or weight.dtype != DType.float8_e4m3fn
        or scale.dtype != DType.float8_e8m0fnu
    ):
        raise ValueError(
            f"V4.1 dense weights require E4M3 and compact E8M0 scales "
            f"{expected}, got {weight.shape} and {scale.shape}"
        )


def _matmul(
    codes: TensorValue,
    a_scales: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
) -> TensorValue:
    """Quantized ``codes @ weight.T`` using compact MXFP8 weights.

    Padding to 32 lets ceil-div scales cover rows; load stacking needs full blocks.
    Expansion is about weight bytes / 32 (V2 pays once at load, this graph
    per site per step).
    """
    _check_compact(weight, scale)
    rows = int(weight.shape[0])
    padded_rows = (rows + 31) // 32 * 32
    if padded_rows != rows:
        weight = ops.pad(weight, [0, padded_rows - rows, 0, 0])
    expanded = ops.repeat_interleave(scale, 32, axis=0)
    b_scales = block_scales_interleave(expanded, sf_vector_size=32)
    if mxfp8_splitk_enabled() and _is_sm10x_gpu():
        out = ops.custom(
            "matmul.mxfp8.cluster_splitk",
            device=codes.device,
            values=[codes, weight, a_scales, b_scales],
            out_types=[
                TensorType(
                    DType.bfloat16,
                    [codes.shape[0], padded_rows],
                    codes.device,
                )
            ],
        )[0].tensor
    else:
        out = dynamic_block_scaled_matmul(
            codes,
            weight,
            a_scales,
            b_scales,
            sf_vector_size=32,
            out_type=DType.bfloat16,
        )
    return out[:, :rows]


def _dense(
    x: TensorValue, weight: TensorValue, scale: TensorValue
) -> TensorValue:
    """``x @ weight.T`` with ``x`` quantized on the way in; see :func:`_matmul`."""
    _check_compact(weight, scale)
    return _matmul(*_quantize(x), weight, scale)


def _dense_rule(
    x: TensorLayout, weight: TensorLayout, scale: TensorLayout
) -> list[AxisAssignment]:
    return [
        AxisAssignment((R, R, R), (R,)),
        AxisAssignment((R, Sharded(0), Sharded(0)), (Sharded(1),)),
        AxisAssignment((Sharded(1), Sharded(1), Sharded(1)), (P,)),
        AxisAssignment((Sharded(0), R, R), (Sharded(0),)),
    ]


_dense_op = F.functional(_dense, rule=_dense_rule)


def _dense_rule_quantized(
    codes: TensorLayout,
    a_scales: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
) -> list[AxisAssignment]:
    """Keep activation scale blocks intact by avoiding a K split."""
    return [
        AxisAssignment((R, R, R, R), (R,)),
        AxisAssignment((R, R, Sharded(0), Sharded(0)), (Sharded(1),)),
        AxisAssignment((Sharded(0), Sharded(0), R, R), (Sharded(0),)),
    ]


_dense_quantized_op = F.functional(_matmul, rule=_dense_rule_quantized)


def _swiglu_dense(
    gate_up: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    limit: float,
) -> TensorValue:
    """Computes the clamped SwiGLU of a ``[gate | up]`` row.

    The activation is fused into the quantize that feeds ``weight`` (V2's
    shared-expert ``w2``).

    ``limit`` clamps ``gate`` above and ``up`` on both sides; ``0`` leaves
    the activation unclamped. Off SM100 the kernel is unavailable, so the
    activation runs as graph ops and :func:`_dense` quantizes it.
    """
    _check_compact(weight, scale)
    if _is_sm10x_gpu():
        codes, a_scales = swiglu_quantize_dynamic_block_scaled(
            gate_up,
            GatedActivation("silu", limit=limit),
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        return _matmul(codes, a_scales, weight, scale)
    half = int(gate_up.shape[1]) // 2
    gate = ops.cast(gate_up[:, :half], DType.float32)
    up = ops.cast(gate_up[:, half:], DType.float32)
    if limit > 0:
        gate = ops.min(gate, limit)
        up = ops.min(ops.max(up, -limit), limit)
    return _dense(ops.cast(ops.silu(gate) * up, DType.bfloat16), weight, scale)


def _swiglu_dense_rule(
    gate_up: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
    limit: float,
) -> list[AxisAssignment]:
    return _dense_rule(gate_up, weight, scale)


swiglu_dense_op = F.functional(_swiglu_dense, rule=_swiglu_dense_rule)


def _dense_rope_tail(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    cos_sin: TensorValue,
    head_dim: int,
) -> TensorValue:
    """:func:`_dense`, then each ``head_dim`` output head's RoPE tail rotated.

    On SM100 the rotation runs in the matmul's epilogue, so the output is
    written once; elsewhere it is the matmul followed by the rotation.
    ``cos_sin`` is ``[T, rope_dim // 2, 2]`` float32, one row per token.
    """
    return _rope_tail_matmul(*_quantize(x), weight, scale, cos_sin, head_dim)


def _rope_tail_matmul(
    codes: TensorValue,
    a_scales: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    cos_sin: TensorValue,
    head_dim: int,
) -> TensorValue:
    _check_compact(weight, scale)
    rope_dim = int(cos_sin.shape[-2]) * 2
    if not (mxfp8_splitk_enabled() and _is_sm10x_gpu()):
        out = _matmul(codes, a_scales, weight, scale)
        heads = ops.reshape(
            out, [1, -1, int(out.shape[1]) // head_dim, head_dim]
        )
        return ops.reshape(
            apply_rope_tail_values(heads, cos_sin, rope_dim), out.shape
        )
    b_scales = block_scales_interleave(
        ops.repeat_interleave(scale, 32, axis=0), sf_vector_size=32
    )
    return ops.custom(
        "matmul.mxfp8.rope_tail",
        device=codes.device,
        values=[
            codes,
            weight,
            a_scales,
            b_scales,
            ops.reshape(cos_sin, [-1, rope_dim]),
        ],
        out_types=[
            TensorType(
                DType.bfloat16, [codes.shape[0], weight.shape[0]], codes.device
            )
        ],
        parameters={"head_dim": head_dim, "rope_dim": rope_dim},
    )[0].tensor


def _dense_rope_tail_rule(
    x: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
    cos_sin: TensorLayout,
    head_dim: int,
) -> list[AxisAssignment]:
    """Returns ``wq_b``'s placements, which shard whole heads by rows.

    ``cos_sin`` stays replicated, since it shares ``x``'s row axis.
    """
    return [
        AxisAssignment((R, R, R, R), (R,)),
        AxisAssignment((R, Sharded(0), Sharded(0), R), (Sharded(1),)),
        AxisAssignment((Sharded(0), R, R, Sharded(0)), (Sharded(0),)),
    ]


dense_rope_tail_op = F.functional(_dense_rope_tail, rule=_dense_rope_tail_rule)


def _rope_tail_matmul_rule(
    codes: TensorLayout,
    a_scales: TensorLayout,
    weight: TensorLayout,
    scale: TensorLayout,
    cos_sin: TensorLayout,
    head_dim: int,
) -> list[AxisAssignment]:
    """Keep activation scale blocks intact by avoiding a K split."""
    return [
        AxisAssignment((R, R, R, R, R), (R,)),
        AxisAssignment((R, R, Sharded(0), Sharded(0), R), (Sharded(1),)),
        AxisAssignment(
            (Sharded(0), Sharded(0), R, R, Sharded(0)), (Sharded(0),)
        ),
    ]


rope_tail_matmul_op = F.functional(
    _rope_tail_matmul, rule=_rope_tail_matmul_rule
)


def trace_device_is_gpu(device: object) -> bool:
    """Recognize GPU devices from either graph or driver values."""
    if isinstance(device, DeviceRef):
        return device.is_gpu()
    return isinstance(device, Accelerator)


def block_diagonal_woa_supported(
    device: object, k: int, group_rows: int
) -> bool:
    """Whether a grouped site can run as one block-diagonal launch.

    The op is SM100-only. It also needs whole 128-column scale tiles along
    K, and its B-scales are one interleave over the whole shard split per
    group by a view, so a group must cover whole 128-row interleave
    blocks. A site that misses any of these -- rank 64 included --
    keeps the grouped reference launch, which needs only whole 32-row
    blocks.
    """
    return (
        trace_device_is_gpu(device)
        and not _is_apple_gpu()
        and _is_sm10x_gpu()
        and k % 128 == 0
        and group_rows % 128 == 0
    )


def norm_quantize_store_supported(
    device: object, q_cols: int, kv_cols: int
) -> bool:
    """Whether the device supports fused q/kv norm, quantization, and storage."""
    return (
        trace_device_is_gpu(device)
        and not _is_apple_gpu()
        and _is_sm10x_gpu()
        and q_cols % 256 == 0
        and kv_cols % 256 == 0
    )


def _grouped_dense(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
    rope: tuple[TensorValue, int] | None = None,
) -> TensorValue:
    """Run grouped MXFP8 projections, fusing supported groups into one launch."""
    _check_compact(weight, scale)
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
    block_diagonal = groups > 1 and block_diagonal_woa_supported(
        x.device, int(weight.shape[1]), group_rows
    )
    if rope is not None and not block_diagonal:
        # The other arms quantize ``x`` as given, so an un-rotated ``x``
        # would silently drop the rotation.
        raise ValueError(
            "the fused inverse rope needs the block-diagonal wo_a launch"
        )
    if groups == 1:
        return _dense(x, weight, scale)
    if block_diagonal:
        return _grouped_dense_block_diagonal(x, weight, scale, group_rows, rope)
    return _grouped_dense_reference(x, weight, scale, group_rows)


def _grouped_dense_block_diagonal(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
    rope: tuple[TensorValue, int] | None = None,
) -> TensorValue:
    """The production launch: one token-major quantize, one matmul.

    The whole-width quantize's codes and interleaved scales are
    byte-identical to the per-group chains the reference arm runs (the
    per-32 amax blocks never straddle a group boundary), so the matmul
    reads the same values the grouped launch would -- without the
    per-group chains, their interleaves and the per-group concat of the
    products.

    With ``rope`` the quantize also rotates each head's tail back, through
    main's rope-quantize op.
    """
    if rope is None:
        codes, a_scales = _quantize(x)
    else:
        cos_sin, head_dim = rope
        codes, a_scales = rope_quantize_dynamic_block_scaled(
            x,
            cos_sin,
            head_dim,
            inverse=True,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    return _block_diagonal_matmul(codes, a_scales, weight, scale, group_rows)


def _block_diagonal_matmul(
    codes: TensorValue,
    a_scales: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
) -> TensorValue:
    """The block-diagonal launch over quantized activations."""
    # The weight-side interleave is the same bytes the reference arm
    # builds per group, unsplit: [rows / 128, k / 128, 32, 4, 4].
    expanded = ops.repeat_interleave(scale, 32, axis=0)
    b_scales = block_scales_interleave(expanded, sf_vector_size=32)
    return mxfp8_block_diagonal(
        codes, weight, a_scales, b_scales, int(weight.shape[0]) // group_rows
    )


def _grouped_dense_quantized(
    codes: TensorValue,
    a_scales: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
) -> TensorValue:
    """Run grouped MXFP8 matmul on pre-quantized rows."""
    _check_compact(weight, scale)
    return _block_diagonal_matmul(codes, a_scales, weight, scale, group_rows)


def _grouped_dense_reference(
    x: TensorValue,
    weight: TensorValue,
    scale: TensorValue,
    group_rows: int,
) -> TensorValue:
    """Reference grouped MXFP8 projection with one launch per group."""
    groups = int(weight.shape[0]) // group_rows
    width = int(x.shape[-1]) // groups
    outputs: list[TensorValue] = []
    for j in range(groups):
        codes, a_scales = _quantize(x[:, j * width : (j + 1) * width])
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
    rope: tuple[TensorLayout, int] | None = None,
) -> list[AxisAssignment]:
    """Returns the grouped projection's placements.

    The rope table (``cos_sin``) shares ``x``'s row axis and stays replicated
    under a TP column split.
    """
    if rope is None:
        return [
            AxisAssignment((R, R, R), (R,)),
            AxisAssignment((Sharded(0), R, R), (Sharded(0),)),
            AxisAssignment((Sharded(1), Sharded(0), Sharded(0)), (Sharded(1),)),
        ]
    return [
        AxisAssignment((R, R, R, R), (R,)),
        AxisAssignment((Sharded(0), R, R, Sharded(0)), (Sharded(0),)),
        AxisAssignment((Sharded(1), Sharded(0), Sharded(0), R), (Sharded(1),)),
    ]


grouped_dense_op = F.functional(_grouped_dense, rule=_grouped_dense_rule)
# No sharding rule: only the single-device-per-rank layouts run it, where
# the K slices that a tensor-parallel split would pair with weight shards do
# not exist.
grouped_dense_quantized_op = F.functional(_grouped_dense_quantized)


class MXFP8Linear(Module[[Tensor], Tensor]):
    """A bias-free E4M3 projection with compact 32x32 E8M0 scales.

    Args:
        in_dim: Input width, divisible by 32.
        out_dim: Output width; a partial final scale block is trimmed.
    """

    weight: Tensor
    scale: Tensor

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim % 32:
            raise ValueError("MXFP8 input width must be divisible by 32")
        self.in_dim = in_dim
        self.out_dim = out_dim
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
        return _dense_op(x, self.weight, self.scale)

    def forward_quantized(self, x: Mxfp8Rows) -> Tensor:
        """:meth:`forward` on an activation that arrives already quantized."""
        if isinstance(self.weight, FP8BlockTensor):
            raise ValueError(
                "V4.1 requires compact E8M0 scales, not FP8BlockTensor"
            )
        return _dense_quantized_op(x.codes, x.scales, self.weight, self.scale)

    def forward_swiglu(self, gate_up: Tensor, limit: float | None) -> Tensor:
        """Applies the clamped SwiGLU of ``gate_up``, then this projection.

        ``gate_up`` is ``[gate | up]`` per row. ``limit=None`` leaves the
        activation unclamped.
        """
        if isinstance(self.weight, FP8BlockTensor):
            raise ValueError(
                "V4.1 requires compact E8M0 scales, not FP8BlockTensor"
            )
        return swiglu_dense_op(gate_up, self.weight, self.scale, limit or 0.0)

    def forward_rope_tail(
        self, x: Tensor, cos_sin: Tensor, head_dim: int
    ) -> Tensor:
        """:meth:`forward`, with each output head's RoPE tail rotated."""
        if isinstance(self.weight, FP8BlockTensor):
            raise ValueError(
                "V4.1 requires compact E8M0 scales, not FP8BlockTensor"
            )
        return dense_rope_tail_op(x, self.weight, self.scale, cos_sin, head_dim)

    def forward_quantized_rope_tail(
        self, x: Mxfp8Rows, cos_sin: Tensor, head_dim: int
    ) -> Tensor:
        """:meth:`forward_rope_tail` on an activation that arrives quantized."""
        if isinstance(self.weight, FP8BlockTensor):
            raise ValueError(
                "V4.1 requires compact E8M0 scales, not FP8BlockTensor"
            )
        return rope_tail_matmul_op(
            x.codes, x.scales, self.weight, self.scale, cos_sin, head_dim
        )
