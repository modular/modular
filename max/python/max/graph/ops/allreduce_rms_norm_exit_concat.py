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
"""Op implementation for the fused allreduce + RMSNorm + exit activation."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max.dtype import DType

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .allreduce_rms_norm import gamma_dtype_supported
from .rms_norm import _float32_attr
from .utils import _buffer_values, _tensor_values


def allreduce_rms_norm_exit_concat(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    gate_ups: Iterable[TensorValueLike],
    epsilon: float,
    beta: float,
    linear_beta: float,
) -> list[TensorValue]:
    """Fused allreduce sum + RMSNorm + column-shard cast + SiTU-and-mul concat.

    Sum-reduces ``inputs`` (one float32 or bfloat16 ``[rows, cols]`` partial
    per device) across the devices, RMSNorm-normalizes every reduced row over
    all ``cols`` columns in the same collective launch, keeps device ``r``'s
    column shard of the normed row (columns
    ``[r * cols / N, (r + 1) * cols / N)``) rounded to ``gate_ups``' dtype,
    and writes it beside the SiTU-and-mul of that device's own gated shared
    projection. The result is one ``[rows, cols / N + half]`` bf16 tensor
    per device -- a tensor-parallel latent MoE's exit activation.

    The norm reads the peer sum rounded to the input dtype, as a standalone
    ``allreduce.sum`` writes it, normalized the way
    :func:`~max.graph.ops.rms_norm` does with its defaults (``weight_offset``
    0, ``multiply_before_cast`` False).

    Args:
        inputs: The float32 or bfloat16 partials to reduce, one per device;
            every device contributes the same-shaped shard.
        signal_buffers: Device buffer values used for synchronization. The
            bytes behind each ``Signal`` header hold the collective's staging
            scratch, so the buffer must be at least ``sizeof(Signal)`` plus
            ``ceil(rows / N) * cols * sizeof(dtype)`` bytes: the plain
            allreduce's partition. The chain arm's full-width sum is a
            workspace of the kernel's own, not part of this payload.
        gammas: The norm weights, one per device (length ``cols``), in the
            inputs' dtype or, on float32 inputs, bfloat16/float16. Every
            device must pass the same replicated weight: one device
            normalizes the columns it holds for every device's rows.
        gate_ups: The gated shared projection, one per device (bf16,
            ``[rows, 2 * half]``, gate half first).
        epsilon: RMSNorm epsilon for numerical stability.
        beta: The SiTU activation's gate beta.
        linear_beta: The SiTU activation's up-half saturation beta; ``0.0``
            passes the up half through unchanged.

    Returns:
        One ``[rows, cols / N + half]`` bfloat16 tensor per device: that
        device's normed column shard, then the activated shared half.

    Raises:
        ValueError: On any operand condition the fused kernel cannot honor.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    gammas = _tensor_values(gammas)
    gate_ups = _tensor_values(gate_ups)

    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "allreduce_rms_norm_exit_concat requires at least two inputs (one"
            f" per device); the fused kernel asserts ngpus >= 2. Got: "
            f"{num_devices}"
        )
    for name, per_device in (
        ("signal buffers", signal_buffers),
        ("gammas", gammas),
        ("gated projections", gate_ups),
    ):
        if len(per_device) != num_devices:
            raise ValueError(
                f"expected number of inputs ({num_devices}) and number of "
                f"{name} ({len(per_device)}) to match"
            )

    for dev_idx, (gamma, inp) in enumerate(zip(gammas, inputs, strict=True)):
        if gamma.shape != inp.shape[-1:]:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires each gamma to match "
                f"its input's last dimension. On device {dev_idx}: gamma "
                f"{gamma.shape} vs cols {inp.shape[-1]}"
            )
        if not gamma_dtype_supported(gamma.dtype, inp.dtype):
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires each gamma to match "
                "its input's dtype, or to be bfloat16/float16 on a float32 "
                f"input. On device {dev_idx}: gamma {gamma.dtype} vs input "
                f"{inp.dtype}"
            )
        if gamma.device != inp.device:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires each gamma to sit on "
                f"its input's device, where the kernel reads it. On device "
                f"{dev_idx}: gamma on {gamma.device} vs input on {inp.device}"
            )

    input_dtype = inputs[0].dtype
    if input_dtype not in (DType.float32, DType.bfloat16):
        raise ValueError(
            "allreduce_rms_norm_exit_concat takes float32 or bfloat16 partials"
            f" (the only dtypes the fused kernel accepts). Got: {input_dtype}"
        )
    out_dtype = gate_ups[0].dtype
    if out_dtype != DType.bfloat16:
        raise ValueError(
            "allreduce_rms_norm_exit_concat writes bfloat16 (the fused kernel"
            f" assumes it). Got: {out_dtype}"
        )
    for dev_idx, (gate_up, inp) in enumerate(
        zip(gate_ups, inputs, strict=True)
    ):
        if gate_up.rank != 2 or inp.rank != 2:
            raise ValueError(
                "allreduce_rms_norm_exit_concat is 2D-only ([rows, cols]); "
                f"the fused kernel indexes rows and cols directly. On device "
                f"{dev_idx}: input rank {inp.rank}, gate_up rank "
                f"{gate_up.rank}"
            )
        if gate_up.dtype != out_dtype:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires the same gated-"
                f"projection dtype on every device. On device {dev_idx}: "
                f"{gate_up.dtype} vs {out_dtype}"
            )
        if gate_up.device != inp.device:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires each gated projection "
                f"to sit on its input's device. On device {dev_idx}: gate_up "
                f"on {gate_up.device} vs input on {inp.device}"
            )
        if gate_up.shape[0] != inp.shape[0]:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires the gated projection "
                f"to have one row per input row. On device {dev_idx}: "
                f"{gate_up.shape[0]} vs {inp.shape[0]}"
            )
        if int(gate_up.shape[1]) % 2 != 0:
            raise ValueError(
                "allreduce_rms_norm_exit_concat requires an even gated "
                f"projection width (gate and up halves). On device {dev_idx}: "
                f"{gate_up.shape[1]}"
            )

    if not all(t.dtype == input_dtype for t in inputs[1:]):
        raise ValueError(
            "allreduce_rms_norm_exit_concat requires the same dtype across all "
            f"input tensors. Got: {inputs=}"
        )
    if not all(t.shape == inputs[0].shape for t in inputs[1:]):
        raise ValueError(
            "allreduce_rms_norm_exit_concat requires the same shape across "
            f"all input tensors (each device contributes a same-sized "
            f"partial). Got: {inputs=}"
        )
    cols = int(inputs[0].shape[1])
    if cols % num_devices != 0:
        raise ValueError(
            "allreduce_rms_norm_exit_concat splits every row into "
            f"{num_devices} equal column shards, so cols ({cols}) must be a "
            f"multiple of the device count ({num_devices})"
        )

    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allreduce_rms_norm_exit_concat requires unique devices across "
            f"its input tensors. Got: {devices=}"
        )

    half_cols = int(gate_ups[0].shape[1]) // 2
    output_types = [
        TensorType(
            out_dtype,
            [inputs[0].shape[0], cols // num_devices + half_cols],
            device=inp.device,
        )
        for inp in inputs
    ]

    graph = Graph.current

    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllreduceRmsNormExitConcatOp,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        gate_ups,
        in_chain,
        _float32_attr(epsilon),
        _float32_attr(beta),
        _float32_attr(linear_beta),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
