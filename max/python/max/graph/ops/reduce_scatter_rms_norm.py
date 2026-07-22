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
"""Op implementation for fused reduce-scatter + RMSNorm."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .constant import constant
from .utils import _buffer_values, _tensor_values


def reduce_scatter_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused reduce-scatter sum + RMSNorm across devices (bf16 in/out).

    Reduce-scatters ``inputs`` (one ``[rows, cols]`` tensor per device) along
    axis 0 across all devices, then RMSNorm-normalizes each device's owned row
    shard in the same collective launch, keeping the reduced sum in float32
    registers so there is no global-memory round-trip between the reduce-scatter
    and the norm. The norm is ``multiply_before_cast=True`` (gamma folded in
    float32, single cast to the input dtype last).

    This is a full-world reduce-scatter (no device grouping): every input
    participates in one reduction. Rows are partitioned with the same ragged
    binning as :func:`reducescatter.sum` (remainder rows go to low ranks), so
    the sum output is a drop-in for ``reducescatter.sum`` along axis 0.

    Args:
        inputs: The input tensors to reduce and scatter, one per device.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (input dtype, length
            ``cols``).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Constant offset added to gamma at runtime (folded in
            float32). ``1.0`` for Gemma-style norms, ``0.0`` otherwise.

    Returns:
        A tuple ``(normed, residual)`` of two lists, each with one tensor per
        device: ``normed[i]`` is the RMSNorm of device ``i``'s reduce-scatter
        shard and ``residual[i]`` is the reduce-scatter sum shard itself (the
        residual stream). Both have the input shape with axis 0 divided across
        devices.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    gammas = _tensor_values(gammas)

    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "reduce_scatter_rms_norm requires at least two inputs (one per "
            f"device); the reduce-scatter is a no-op otherwise. Got: {num_devices}"
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
            "reduce_scatter_rms_norm is bfloat16-only (the kernel and fuse "
            f"threshold assume it). Got: {input_dtype}"
        )
    if not all(t.dtype == input_dtype for t in inputs[1:]):
        raise ValueError(
            "reduce_scatter_rms_norm requires the same dtype across all input "
            f"tensors. Got: {inputs=}"
        )
    if not all(t.shape == inputs[0].shape for t in inputs[1:]):
        raise ValueError(
            "reduce_scatter_rms_norm requires the same shape across all input "
            f"tensors. Got: {inputs=}"
        )
    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "reduce_scatter_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    # Per-device output types: axis-0 ragged binning (matches reducescatter.sum
    # so the residual is a drop-in). Normed and residual share the shard shape.
    scatter_dim = inputs[0].shape[0]
    normed_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        shard_shape = list(inputs[dev_idx].shape)
        shard_shape[0] = (
            scatter_dim + (num_devices - dev_idx - 1)
        ) // num_devices
        normed_types.append(
            TensorType(dtype=input_dtype, shape=shard_shape, device=device)
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=shard_shape, device=device)
        )

    # epsilon/weight_offset as CPU scalars: the kernel reads them host-side for
    # launch params. One operand slot per device (SameVariadicOperandSize),
    # reusing the same host constant.
    cpu = DeviceRef.CPU()
    eps_const = constant(epsilon, DType.float32, cpu)
    weight_offset_const = constant(weight_offset, input_dtype, cpu)
    epsilons = [eps_const] * num_devices
    weight_offsets = [weight_offset_const] * num_devices

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedReduceScatterRmsNormOp,
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
