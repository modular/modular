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
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused all-gather + RMSNorm across devices (bf16 in/out).

    All-gathers ``inputs`` (per-device ``[shard_i, cols]`` row shards) along axis
    0 so every device holds the full ``[sum(shard_i), cols]`` tensor, then
    RMSNorms every gathered row in the same launch, consuming it from registers
    (no global-memory round-trip). The norm is ``multiply_before_cast=True``.

    Full-world (no device grouping): outputs are replicated on every device. A
    gathered row is a verbatim copy, so the residual is a drop-in for
    ``allgather`` along axis 0.

    The fuse-vs-fallback threshold is calibrated on AMD (gfx950): at/below it the
    fused kernel is bit-identical to ``allgather`` + ``rms_norm``, above it the
    two-launch fallback wins. On other backends the ``block.sum`` geometry may
    differ, so the paths agree only to RMSNorm ULP tolerance.

    Args:
        inputs: The input row shards to gather, one per device.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (input dtype, length
            ``cols``).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Constant offset added to gamma at runtime (folded in
            float32). ``1.0`` for Gemma-style norms, ``0.0`` otherwise.

    Returns:
        A tuple ``(normed, residual)`` of two lists, each with one tensor per
        device: ``normed[i]`` is the RMSNorm of the full gathered tensor and
        ``residual[i]`` is the raw gathered tensor itself (the residual stream).
        Both are the full replicated shape on every device.
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
    # Only axis 0 (gathered) may differ across shards; other dims must match.
    for t in inputs[1:]:
        for i in range(1, inputs[0].shape.rank):
            if t.shape[i] != inputs[0].shape[i]:
                raise ValueError(
                    "allgather_rms_norm requires the same shape in all "
                    "dimensions except axis 0 (rows) across input shards. "
                    f"Got: {inputs=}"
                )
    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allgather_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    # Full replicated shape: axis 0 = sum of shard rows (Dim arithmetic), other
    # dims from input[0]. No ragged binning.
    gathered_dim = inputs[0].shape[0]
    for t in inputs[1:]:
        gathered_dim = gathered_dim + t.shape[0]
    full_shape = list(inputs[0].shape)
    full_shape[0] = gathered_dim
    normed_types: list[TensorType] = [
        TensorType(dtype=input_dtype, shape=full_shape, device=device)
        for device in devices
    ]
    residual_types: list[TensorType] = [
        TensorType(dtype=input_dtype, shape=full_shape, device=device)
        for device in devices
    ]

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
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[:num_devices]]
    residual = [res.tensor for res in results[num_devices:]]
    return normed, residual
