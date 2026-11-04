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
"""Op implementation for fused all-gather + RMSNorm."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import BoolAttr, IntegerAttr, IntegerType
from max.dtype import DType

from ..dim import Dim, StaticDim
from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorType,
    TensorValue,
    TensorValueLike,
)
from .rms_norm import _float32_attr
from .utils import _buffer_values, _tensor_values

# Mirrors the kernels' `MXFP8_SF_VECTOR_SIZE`: elements per MX block scale.
_FP6_FORMAT_CODE = {"e2m3": 0, "e3m2": 1}
"""OCP FP6 encodings, in the order the kernel's `fp6_format` parameter takes."""

_MX_SF_VECTOR_SIZE = 32


def _validate_ag_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    group_size: int | None,
) -> tuple[
    list[TensorValue],
    list[BufferValue],
    list[TensorValue],
    list[DeviceRef],
    int,
    int,
]:
    """Shared operand validation for the fused all-gather + RMSNorm ops.

    Returns:
        ``(inputs, signal_buffers, gammas, devices, num_devices,
        group_size)`` with ``group_size`` defaulted to the device count.

    Raises:
        ValueError: On any operand condition the fused kernel cannot honor.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    gammas = _tensor_values(gammas)

    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "allgather_rms_norm requires at least two inputs (one per device); "
            f"the all-gather is a no-op otherwise. Got: {num_devices}"
        )
    if len(signal_buffers) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of signal "
            f"buffers ({len(signal_buffers)}) to match"
        )
    if len(gammas) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of gammas "
            f"({len(gammas)}) to match"
        )

    input_dtype = inputs[0].dtype
    if input_dtype != DType.bfloat16:
        raise ValueError(
            "allgather_rms_norm is bfloat16-only (the kernel and fuse threshold "
            f"assume it). Got: {input_dtype}"
        )
    if not all(t.dtype == input_dtype for t in inputs[1:]):
        raise ValueError(
            "allgather_rms_norm requires the same dtype across all input "
            f"tensors. Got: {inputs=}"
        )
    if not all(t.shape.rank == inputs[0].shape.rank for t in inputs[1:]):
        raise ValueError(
            "allgather_rms_norm requires the same rank across all input "
            f"tensors. Got: {inputs=}"
        )
    # Fused kernel indexes rows/cols directly (2D only); fail fast on higher rank.
    if inputs[0].shape.rank != 2:
        raise ValueError(
            "allgather_rms_norm is 2D-only ([rows, cols]); the fused kernel "
            f"indexes rows and cols directly. Got rank: {inputs[0].shape.rank}"
        )
    group_size = group_size or num_devices
    if group_size < 2:
        raise ValueError(
            "allgather_rms_norm requires group_size to be at least 2 (the "
            f"all-gather is a no-op otherwise). Got: {group_size=}"
        )
    if num_devices % group_size != 0:
        raise ValueError(
            "allgather_rms_norm requires group_size to evenly divide the "
            f"number of input tensors. Got: {group_size=} and {num_devices=}"
        )
    # Only axis 0 (gathered) may differ across shards. Axis 0 MUST stay
    # exempt: the shards are a reduce-scatter residual, whose ragged binning
    # gives each group-local rank a structurally different symbolic dim
    # (`(S + (g-1-lr)) // g`), so they never compare equal. The other dims
    # must match across the whole world, not just within a group: the kernel
    # sizes every group's windows and fuse gate from device 0's columns.
    for t in inputs[1:]:
        for i in range(1, inputs[0].shape.rank):
            if t.shape[i] != inputs[0].shape[i]:
                raise ValueError(
                    "allgather_rms_norm requires the same shape in all "
                    "dimensions except axis 0 (rows) across every input "
                    f"shard, including across groups. Got: {inputs=}"
                )
    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allgather_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )
    return inputs, signal_buffers, gammas, devices, num_devices, group_size


def allgather_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    group_size: int | None = None,
    axis: int = 0,
    round_offset_gamma: bool = False,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused all-gather + RMSNorm across devices (bf16 in/out).

    All-gathers ``inputs`` (per-device ``[shard_i, cols]`` row shards) along axis
    0 so every device holds its group's ``[sum(shard_i), cols]`` tensor, then
    RMSNorms every gathered row in the same launch, consuming it from registers
    (no global-memory round-trip). The norm is ``multiply_before_cast=True``.
    Without ``group_size`` the group is every device, so the result is the
    world-wide gather.

    A gathered row is a verbatim copy, so the residual is a drop-in for
    ``allgather`` along axis 0.

    The fuse-vs-fallback threshold is calibrated on AMD (gfx950): at/below it the
    fused kernel is bit-identical to ``allgather`` + ``rms_norm``, above it the
    two-launch fallback wins. On other backends the ``block.sum`` geometry may
    differ, so the paths agree only to RMSNorm ULP tolerance.

    Args:
        inputs: The input row shards to gather, one per device. Only axis 0
            may differ between shards, within a group (see ``group_size``) or
            across groups; the fused kernel sizes every group from device 0's
            column count.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (input dtype, length
            ``cols``).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Constant offset added to gamma at runtime (folded in
            float32). ``1.0`` for Gemma-style norms, ``0.0`` otherwise.
        group_size: Optional number of contiguous devices per independent
            all-gather group. Defaults to all devices (a full-world collective).
            Under TP-within-DP this is the TP degree, so each replica gathers
            only within its own group.
        axis: The gathered axis: ``0`` for row shards, ``1`` for column
            shards of equal shape, concatenated in rank order. Column shards
            always run the fused kernel.
        round_offset_gamma: Whether ``gamma + weight_offset`` is rounded to
            the input dtype before the float32 multiply, as a norm that
            scales by a stored ``1 + w`` does. Always runs the fused kernel.

    Returns:
        A tuple ``(normed, residual)`` of two lists, each with one tensor per
        device: ``normed[i]`` is the RMSNorm of the gathered tensor and
        ``residual[i]`` is the raw gathered tensor itself (the residual stream).
        Both are the gathered shape of the device's own group, replicated across
        that group.

    Raises:
        ValueError: If ``axis`` is not 0 or 1, if column shards differ in
            rows, or on any condition the fused kernel cannot honor.
    """
    inputs, signal_buffers, gammas, devices, num_devices, group_size = (
        _validate_ag_rms_norm(inputs, signal_buffers, gammas, group_size)
    )
    if axis not in (0, 1):
        raise ValueError(
            f"allgather_rms_norm gathers axis 0 or 1. Got: {axis=}"
        )
    if axis == 1 and any(t.shape[0] != inputs[0].shape[0] for t in inputs):
        raise ValueError(
            "allgather_rms_norm over columns requires the same rows in every "
            f"shard. Got: {inputs=}"
        )
    input_dtype = inputs[0].dtype
    graph = Graph.current

    # Gathered shape, replicated within each group: axis 0 = sum of the GROUP's
    # shard rows (Dim arithmetic), other dims from the group's first shard. No
    # ragged binning.
    normed_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        group_start = (dev_idx // group_size) * group_size
        group_inputs = inputs[group_start : group_start + group_size]
        gathered_dim = group_inputs[0].shape[axis]
        for t in group_inputs[1:]:
            gathered_dim = gathered_dim + t.shape[axis]
        full_shape = list(group_inputs[0].shape)
        full_shape[axis] = gathered_dim
        normed_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllgatherRmsNormOp,
        normed_types,
        residual_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        in_chain,
        _float32_attr(epsilon),
        _float32_attr(weight_offset),
        IntegerAttr(IntegerType(64), group_size),
        IntegerAttr(IntegerType(64), axis),
        BoolAttr(round_offset_gamma),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[:num_devices]]
    residual = [res.tensor for res in results[num_devices:]]
    return normed, residual


def allgather_rms_norm_quant_mxfp8(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    group_size: int | None = None,
    gather_quant_only: bool = False,
) -> tuple[
    list[TensorValue], list[TensorValue], list[TensorValue], list[TensorValue]
]:
    """:obj:`allgather_rms_norm` that also emits an MXFP8 copy of the norm.

    The quantize rides the collective's epilogue on the same bfloat16 written to
    the normed output, so the pair is byte-identical to a standalone quantize.
    Scales are the plain ``[rows, cols / 32]`` row-major layout
    ``block_scaled_matmul_amd`` takes as ``a_scales`` -- not the SM100 SF-atom
    interleave, and NOT what ``block_scaled_matmul_amd_preb`` needs: same dtype
    and element count, permuted bytes, so it dequants wrongly without erroring.

    Args:
        inputs: The input row shards to gather, one per device. Only axis 0
            may differ between shards, within or across groups.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device.
        epsilon: Numerical stability epsilon for RMSNorm.
        weight_offset: Constant offset added to ``gammas``.
        group_size: Devices per independent gather group; defaults to all.
        gather_quant_only: When every group gathers at least ~1024 rows,
            norms and quantizes only each device's own row shard and gathers
            just ``quantized`` and ``scales``, about half the bytes of the
            bfloat16 residual; below that the default runs. Each shard is
            normed with its owner's gamma, so it requires the same ``gammas``
            on every device of a group. ``normed`` and ``residual`` then hold
            valid values only in each device's own rows, so set it only when
            every consumer reads nothing else of them.

    Returns:
        ``(normed, quantized, scales, residual)``, each a per-device list.
        ``quantized`` is ``float8_e4m3fn`` with the normed shape and ``scales``
        is ``float8_e8m0fnu`` with its last dim divided by 32.

    Raises:
        ValueError: If the hidden dim is not a static multiple of 32, or on any
            condition :obj:`allgather_rms_norm` rejects.
    """
    inputs, signal_buffers, gammas, devices, num_devices, group_size = (
        _validate_ag_rms_norm(inputs, signal_buffers, gammas, group_size)
    )
    input_dtype = inputs[0].dtype

    graph = Graph.current

    normed_types: list[TensorType] = []
    quant_types: list[TensorType] = []
    scale_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        group_start = (dev_idx // group_size) * group_size
        group_inputs = inputs[group_start : group_start + group_size]
        gathered_dim = group_inputs[0].shape[0]
        for t in group_inputs[1:]:
            gathered_dim = gathered_dim + t.shape[0]
        full_shape = list(group_inputs[0].shape)
        full_shape[0] = gathered_dim

        cols = full_shape[-1]
        if (
            not isinstance(cols, StaticDim)
            or int(cols) % _MX_SF_VECTOR_SIZE != 0
        ):
            raise ValueError(
                "allgather_rms_norm_quant_mxfp8 requires a static hidden dim "
                f"that is a multiple of {_MX_SF_VECTOR_SIZE} (one block "
                f"scale's worth); got {cols}."
            )
        scale_shape = list(full_shape)
        scale_shape[-1] = Dim(int(cols) // _MX_SF_VECTOR_SIZE)

        normed_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )
        quant_types.append(
            TensorType(
                dtype=DType.float8_e4m3fn, shape=full_shape, device=device
            )
        )
        scale_types.append(
            TensorType(
                dtype=DType.float8_e8m0fnu, shape=scale_shape, device=device
            )
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllgatherRmsNormQuantMxfp8Op,
        normed_types,
        quant_types,
        scale_types,
        residual_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        in_chain,
        _float32_attr(epsilon),
        _float32_attr(weight_offset),
        IntegerAttr(IntegerType(64), group_size),
        BoolAttr(gather_quant_only),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[0 * num_devices : 1 * num_devices]]
    quant = [res.tensor for res in results[1 * num_devices : 2 * num_devices]]
    scales = [res.tensor for res in results[2 * num_devices : 3 * num_devices]]
    residual = [res.tensor for res in results[3 * num_devices :]]
    return normed, quant, scales, residual


def allgather_rms_norm_quant_mxfp6(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    group_size: int | None = None,
    fp6_format: str = "e2m3",
) -> tuple[
    list[TensorValue], list[TensorValue], list[TensorValue], list[TensorValue]
]:
    """:obj:`allgather_rms_norm` that also emits a packed MXFP6 copy of the norm.

    The quantize rides the collective's epilogue on the same bfloat16 written to
    the normed output, so the pair is byte-identical to a standalone quantize.
    Scales are the plain ``[rows, cols / 32]`` row-major layout
    ``block_scaled_matmul_amd`` takes as ``a_scales`` -- not the SM100 SF-atom
    interleave, and NOT what ``block_scaled_matmul_amd_preb`` needs: same dtype
    and element count, permuted bytes, so it dequants wrongly without erroring.

    Args:
        inputs: The input row shards to gather, one per device. Only axis 0
            may differ between shards, within or across groups.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device.
        epsilon: Numerical stability epsilon for RMSNorm.
        weight_offset: Constant offset added to ``gammas``.
        group_size: Devices per independent gather group; defaults to all.
        fp6_format: OCP FP6 encoding to produce, ``e2m3`` or ``e3m2``. Carried
            explicitly because both are six bits and arrive packed as
            ``uint8``, so the result type cannot record which one it holds.

    Returns:
        ``(normed, quantized, scales, residual)``, each a per-device list.
        ``quantized`` is ``uint8`` holding four packed FP6 codes per three
        bytes, so its last dim is three quarters the normed one, and ``scales``
        is ``float8_e8m0fnu`` with its last dim divided by 32.

    Raises:
        ValueError: If ``fp6_format`` is not a known encoding, if the hidden
            dim is not a static multiple of 32 (which also makes it a multiple
            of the four-code packing group), or on any condition
            :obj:`allgather_rms_norm` rejects.
    """
    inputs, signal_buffers, gammas, devices, num_devices, group_size = (
        _validate_ag_rms_norm(inputs, signal_buffers, gammas, group_size)
    )
    input_dtype = inputs[0].dtype

    graph = Graph.current

    normed_types: list[TensorType] = []
    quant_types: list[TensorType] = []
    scale_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        group_start = (dev_idx // group_size) * group_size
        group_inputs = inputs[group_start : group_start + group_size]
        gathered_dim = group_inputs[0].shape[0]
        for t in group_inputs[1:]:
            gathered_dim = gathered_dim + t.shape[0]
        full_shape = list(group_inputs[0].shape)
        full_shape[0] = gathered_dim

        cols = full_shape[-1]
        if (
            not isinstance(cols, StaticDim)
            or int(cols) % _MX_SF_VECTOR_SIZE != 0
        ):
            raise ValueError(
                "allgather_rms_norm_quant_mxfp6 requires a static hidden dim "
                f"that is a multiple of {_MX_SF_VECTOR_SIZE} (one block "
                f"scale's worth); got {cols}."
            )
        scale_shape = list(full_shape)
        scale_shape[-1] = Dim(int(cols) // _MX_SF_VECTOR_SIZE)

        normed_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )
        packed_shape = list(full_shape)
        packed_shape[-1] = Dim(int(cols) * 3 // 4)
        quant_types.append(
            TensorType(dtype=DType.uint8, shape=packed_shape, device=device)
        )
        scale_types.append(
            TensorType(
                dtype=DType.float8_e8m0fnu, shape=scale_shape, device=device
            )
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )

    if fp6_format not in _FP6_FORMAT_CODE:
        raise ValueError(
            f"unknown FP6 encoding {fp6_format!r}; expected one of "
            f"{sorted(_FP6_FORMAT_CODE)}"
        )

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllgatherRmsNormQuantMxfp6Op,
        normed_types,
        quant_types,
        scale_types,
        residual_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        in_chain,
        _float32_attr(epsilon),
        _float32_attr(weight_offset),
        IntegerAttr(IntegerType(64), group_size),
        IntegerAttr(IntegerType(64), _FP6_FORMAT_CODE[fp6_format]),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[0 * num_devices : 1 * num_devices]]
    quant = [res.tensor for res in results[1 * num_devices : 2 * num_devices]]
    scales = [res.tensor for res in results[2 * num_devices : 3 * num_devices]]
    residual = [res.tensor for res in results[3 * num_devices :]]
    return normed, quant, scales, residual


def allgather_gated_residual_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    residuals: Iterable[TensorValueLike],
    post_weights: Iterable[TensorValueLike],
    gates: Iterable[TensorValueLike],
    norm_weights: Iterable[TensorValueLike],
    post_eps: float,
    norm_eps: float,
    temperature: float,
    gate_eps: float,
    scale_branch: bool,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Gathers a branch's column shards and joins it to the residual stream.

    Each device holds ``[rows, cols / N]``, a contiguous block of a branch's
    columns in rank order. One launch per device gathers the whole branch
    ``b`` and computes, as a learned residual gate joins a branch:

    .. code-block:: text

        n = rms_norm(b * (1 + post_weight))       # weightless; scale_branch
        stream = alpha * residual + beta * n      # float32, rounded once
        normed = rms_norm(stream) * (1 + norm_weight)

    with ``beta = sigmoid(gate / temperature)`` and
    ``alpha = sqrt(max(sigmoid(-gate / temperature) * (1 + beta), gate_eps))``.
    Each step rounds to bfloat16 where that graph does.

    Args:
        inputs: The branch's column shards, one per device.
        signal_buffers: Device buffer values used for synchronization.
        residuals: Each device's incoming residual stream ``[rows, cols]``.
        post_weights: Each device's branch scale weight ``[cols]``; not read
            unless ``scale_branch``.
        gates: Each device's residual gate logits ``[cols]``.
        norm_weights: Each device's next pre-norm weight ``[cols]``.
        post_eps: The branch norm's epsilon.
        norm_eps: The next pre-norm's epsilon.
        temperature: The residual gate's temperature.
        gate_eps: The floor under the residual coefficient's root.
        scale_branch: Whether the branch is scaled by ``1 + post_weight``
            before its norm.

    Returns:
        ``(stream, normed)``, each one ``[rows, cols]`` tensor per device.

    Raises:
        ValueError: If the shards differ in shape, or on any condition
            :obj:`allgather_rms_norm` rejects.
    """
    inputs, signal_buffers, _, devices, num_devices, _ = _validate_ag_rms_norm(
        inputs, signal_buffers, post_weights, None
    )
    if any(t.shape != inputs[0].shape for t in inputs):
        raise ValueError(
            "allgather_gated_residual_rms_norm requires equal column shards. "
            f"Got: {inputs=}"
        )
    per_device = [
        _tensor_values(v)
        for v in (residuals, post_weights, gates, norm_weights)
    ]
    if any(len(v) != num_devices for v in per_device):
        raise ValueError(
            "allgather_gated_residual_rms_norm takes one residual and one of "
            "each weight per device."
        )
    rows, cols = inputs[0].shape
    out_types = [
        TensorType(
            dtype=inputs[0].dtype, shape=[rows, cols * num_devices], device=d
        )
        for d in devices
    ]
    graph = Graph.current
    in_chain = graph.device_chains.merge_for(devices)
    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllgatherGatedResidualRmsNormOp,
        out_types,
        out_types,
        _ChainType(),
        inputs,
        signal_buffers,
        *per_device,
        in_chain,
        _float32_attr(post_eps),
        _float32_attr(norm_eps),
        _float32_attr(temperature),
        _float32_attr(gate_eps),
        BoolAttr(scale_branch),
    )
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain
    return (
        [res.tensor for res in results[:num_devices]],
        [res.tensor for res in results[num_devices:]],
    )
