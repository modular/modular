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
"""Op implementation for fused allreduce + RMSNorm."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import BoolAttr
from max.dtype import DType

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .rms_norm import _float32_attr
from .utils import _buffer_values, _tensor_values


def allreduce_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused allreduce sum + RMSNorm across devices (bf16 in/out).

    Sum-reduces ``inputs`` (same-shaped per-device tensors) so every device
    holds the sum, then RMSNorm-normalizes the reduced rows in the same
    collective launch, consuming the peer sum from registers (no
    global-memory round-trip between the allreduce and the norm). The norm
    runs on the unrounded float32 sum, so it is not bit-identical to a
    standalone ``allreduce.sum`` -- which rounds the sum to the input dtype
    before the norm reads it.

    Args:
        inputs: The input tensors to reduce, one per device. Shapes must all
            match, since every device contributes the same-sized shard.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (input dtype, length
            ``cols``). Every device must pass the same replicated weight:
            the dispatch's 2-stage arm normalizes each row shard on its
            owner rank, so a gamma that differed per device would leak the
            owner's into the other devices' outputs.
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Constant offset added to gamma at runtime (folded in
            float32). ``1.0`` for Gemma-style norms, ``0.0`` otherwise.
        multiply_before_cast: The norm's gamma cast order, matching
            :func:`rms_norm`. ``False`` casts the normalized value back to the
            input dtype before multiplying by gamma.

    Returns:
        A tuple ``(normed, summed)`` of two lists, each with one tensor per
        device: ``normed[i]`` holds the RMSNorm of the reduced sum.
        ``summed[i]`` holds the raw reduced sum, and is written only when
        the op takes its two-launch arm (large payloads, non-AMD targets,
        and hosts without P2P access); the fused arm's peer sum never
        materializes, so callers that need the sum must use
        ``allreduce.sum`` instead.

    Raises:
        ValueError: On any operand condition the fused kernel cannot honor.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    gammas = _tensor_values(gammas)

    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "allreduce_rms_norm requires at least two inputs (one per "
            f"device); the fused kernel asserts ngpus >= 2. Got: {num_devices}"
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
            "allreduce_rms_norm is bfloat16-only (the fused kernel assumes "
            f"it). Got: {input_dtype}"
        )
    if not all(t.dtype == input_dtype for t in inputs[1:]):
        raise ValueError(
            "allreduce_rms_norm requires the same dtype across all input "
            f"tensors. Got: {inputs=}"
        )
    if not all(t.shape == inputs[0].shape for t in inputs[1:]):
        raise ValueError(
            "allreduce_rms_norm requires the same shape across all input "
            f"tensors (each device contributes the same-sized shard). Got: "
            f"{inputs=}"
        )
    if inputs[0].shape.rank != 2:
        raise ValueError(
            "allreduce_rms_norm is 2D-only ([rows, cols]); the fused kernel "
            f"indexes rows and cols directly. Got rank: {inputs[0].shape.rank}"
        )
    for dev_idx, (gamma, inp) in enumerate(zip(gammas, inputs, strict=True)):
        if gamma.shape != inp.shape[-1:]:
            raise ValueError(
                "allreduce_rms_norm requires each gamma to match its input's "
                f"last dimension. On device {dev_idx}: gamma {gamma.shape} "
                f"vs cols {inp.shape[-1]}"
            )
        # The Mojo handler reads gammas[i] on device i in the input dtype;
        # anything else fails deep in elaboration, so reject it here.
        if gamma.dtype != inp.dtype:
            raise ValueError(
                "allreduce_rms_norm requires each gamma to match its input's "
                f"dtype. On device {dev_idx}: gamma {gamma.dtype} vs input "
                f"{inp.dtype}"
            )
        if gamma.device != inp.device:
            raise ValueError(
                "allreduce_rms_norm requires each gamma to sit on its "
                "input's device, where the kernel reads it. On device "
                f"{dev_idx}: gamma on {gamma.device} vs input on "
                f"{inp.device}"
            )

    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allreduce_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    in_chain = graph.device_chains.merge_for(devices)

    output_types = [inp.type for inp in inputs]
    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllreduceRmsNormOp,
        output_types,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        in_chain,
        _float32_attr(epsilon),
        _float32_attr(weight_offset),
        BoolAttr(multiply_before_cast),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[:num_devices]]
    summed = [res.tensor for res in results[num_devices:]]
    return normed, summed
