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
from max._core.dialects.builtin import IntegerAttr, IntegerType
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .constant import constant
from .utils import _buffer_values, _tensor_values


def allgather_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    group_size: int | None = None,
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
        inputs: The input row shards to gather, one per device. Within a group
            (see ``group_size``) only axis 0 may differ; different groups are
            independent collectives and may differ in any dim.
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

    Returns:
        A tuple ``(normed, residual)`` of two lists, each with one tensor per
        device: ``normed[i]`` is the RMSNorm of the gathered tensor and
        ``residual[i]`` is the raw gathered tensor itself (the residual stream).
        Both are the gathered shape of the device's own group, replicated across
        that group.
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
    # Only axis 0 (gathered) may differ across shards; other dims must match
    # within a group. Axis 0 MUST stay exempt: the shards are a reduce-scatter
    # residual, whose ragged binning gives each group-local rank a structurally
    # different symbolic dim (`(S + (g-1-lr)) // g`), so they never compare
    # equal. Groups are independent collectives (DP replicas) and may differ
    # from each other in any dim.
    for group_start in range(0, num_devices, group_size):
        group_inputs = inputs[group_start : group_start + group_size]
        for t in group_inputs[1:]:
            for i in range(1, group_inputs[0].shape.rank):
                if t.shape[i] != group_inputs[0].shape[i]:
                    raise ValueError(
                        "allgather_rms_norm requires the same shape in all "
                        "dimensions except axis 0 (rows) across the input "
                        f"shards of each group. Got: {inputs=}"
                    )
    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allgather_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    # Gathered shape, replicated within each group: axis 0 = sum of the GROUP's
    # shard rows (Dim arithmetic), other dims from the group's first shard. No
    # ragged binning.
    normed_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        group_start = (dev_idx // group_size) * group_size
        group_inputs = inputs[group_start : group_start + group_size]
        gathered_dim = group_inputs[0].shape[0]
        for t in group_inputs[1:]:
            gathered_dim = gathered_dim + t.shape[0]
        full_shape = list(group_inputs[0].shape)
        full_shape[0] = gathered_dim
        normed_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=full_shape, device=device)
        )

    # epsilon/weight_offset are CPU scalars (kernel reads them host-side); one
    # slot per device (SameVariadicOperandSize), same constant reused.
    cpu = DeviceRef.CPU()
    eps_const = constant(epsilon, DType.float32, cpu)
    weight_offset_const = constant(weight_offset, input_dtype, cpu)
    epsilons = [eps_const] * num_devices
    weight_offsets = [weight_offset_const] * num_devices

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllgatherRmsNormOp,
        normed_types,
        residual_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        epsilons,
        weight_offsets,
        in_chain,
        IntegerAttr(IntegerType(64), group_size),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[:num_devices]]
    residual = [res.tensor for res in results[num_devices:]]
    return normed, residual
