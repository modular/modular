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

from ..dim import StaticDim
from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .rms_norm import _float32_attr
from .utils import _buffer_values, _tensor_values

# The post-norm residual kernel runs a row in one block of at most 1024
# threads, each holding one 128-bit pack of 8 bfloat16 values.
_POST_NORM_RESIDUAL_PACK_COLS = 8
_POST_NORM_RESIDUAL_MAX_COLS = 1024 * _POST_NORM_RESIDUAL_PACK_COLS


def gamma_dtype_supported(gamma_dtype: DType, input_dtype: DType) -> bool:
    """Returns whether a fused allreduce + norm kernel reads this gamma dtype.

    Gamma is the input dtype, or a bfloat16/float16 weight on a float32
    input, which the kernels widen exactly in registers (Kimi K3's
    routed-expert norm keeps a bfloat16 weight on a float32 latent).
    """
    return gamma_dtype == input_dtype or (
        input_dtype == DType.float32
        and gamma_dtype in (DType.bfloat16, DType.float16)
    )


def allreduce_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused allreduce sum + RMSNorm across devices (bf16 or float32 in/out).

    Sum-reduces ``inputs`` (same-shaped per-device tensors) so every device
    holds the sum, then RMSNorm-normalizes the reduced rows in the same
    collective launch, consuming the peer sum from registers (no
    global-memory round-trip between the allreduce and the norm). The norm
    runs on the unrounded float32 sum, so it is not bit-identical to a
    standalone ``allreduce.sum`` -- which rounds the sum to the input dtype
    before the norm reads it. On a float32 input there is no such rounding,
    but the peer sum and the mean square may accumulate in a different
    order, so the two still differ by float32 rounding.

    Args:
        inputs: The input tensors to reduce, one per device. Shapes must all
            match, since every device contributes the same-sized shard.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (length ``cols``), in
            the input dtype or, on a float32 input, bfloat16/float16 (the
            kernels widen it exactly in registers). Every device must pass
            the same replicated weight: the dispatch's 2-stage arm
            normalizes each row shard on its owner rank, so a gamma that
            differed per device would leak the owner's into the other
            devices' outputs.
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
    if input_dtype not in (DType.bfloat16, DType.float32):
        raise ValueError(
            "allreduce_rms_norm takes bfloat16 or float32 inputs (the only "
            f"dtypes the fused kernel dispatch accepts). Got: {input_dtype}"
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
        # The Mojo handler reads gammas[i] on device i in the input dtype,
        # or widens a narrower float gamma on a float32 input; anything else
        # fails deep in elaboration, so reject it here.
        if not gamma_dtype_supported(gamma.dtype, inp.dtype):
            raise ValueError(
                "allreduce_rms_norm requires each gamma to match its input's "
                "dtype, or to be bfloat16/float16 on a float32 input. On "
                f"device {dev_idx}: gamma {gamma.dtype} vs input {inp.dtype}"
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


def allreduce_post_norm_residual_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    residuals: Iterable[TensorValueLike],
    gammas: Iterable[TensorValueLike],
    gamma2s: Iterable[TensorValueLike],
    mixes: Iterable[TensorValueLike],
    epsilon: float,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused allreduce sum + a post-norm residual join (bf16 in/out).

    For a transformer block's branch output, one launch per device runs the
    branch's post-norm, its scaled join to the residual stream and the next
    pre-norm:

    .. code-block:: text

        joined = mix[0] * rms_norm(sum(inputs), gamma) + mix[1] * residual
        normed = rms_norm(joined, gamma2)

    Each product and the sum round to bfloat16 on their own, as the unfused
    ops do, and both norms multiply by their weight in float32 before the
    cast. A Lamport kernel runs it, in row chunks that each fit its 1 MiB
    workspace, with one thread block per row: ``cols`` must be a multiple
    of 8 and at most 8192. A static ``cols`` outside that range raises here;
    a symbolic one raises when the graph runs. The kernel stores straight
    into the peers' signal buffers, so it raises at run time on a host
    without P2P access between the GPUs.

    Args:
        inputs: The ``[rows, cols]`` partial sums to reduce, one per device.
        signal_buffers: Device buffer values used for synchronization.
        residuals: Each device's residual stream ``[rows, cols]``.
        gammas: Each device's post-norm weight ``[cols]``.
        gamma2s: Each device's next pre-norm weight ``[cols]``.
        mixes: Each device's ``[2]`` float32 weights of the normed sum and
            of the residual.
        epsilon: Both norms' epsilon.

    Returns:
        ``(joined, normed)``, each one ``[rows, cols]`` tensor per device.

    Raises:
        ValueError: If the operand lists differ in length, the inputs are
            not same-shaped 2-D bfloat16 tensors on distinct devices, a
            static ``cols`` does not fit the kernel, a weight does not match
            its input, or a mix is not a ``[2]`` float32 tensor.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    per_device = [
        _tensor_values(v) for v in (residuals, gammas, gamma2s, mixes)
    ]
    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm requires at least two "
            f"inputs, one per device. Got: {num_devices}"
        )
    if len(signal_buffers) != num_devices or any(
        len(v) != num_devices for v in per_device
    ):
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm takes one signal buffer, "
            "residual, pair of norm weights and mix per input."
        )
    first = inputs[0]
    if first.dtype != DType.bfloat16 or first.shape.rank != 2:
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm takes 2-D bfloat16 inputs. "
            f"Got: {first.type}"
        )
    if any(
        t.type.dtype != first.dtype or t.shape != first.shape for t in inputs
    ):
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm requires same-shaped "
            f"inputs. Got: {inputs=}"
        )
    cols = first.shape[-1]
    if isinstance(cols, StaticDim) and (
        int(cols) % _POST_NORM_RESIDUAL_PACK_COLS != 0
        or int(cols) > _POST_NORM_RESIDUAL_MAX_COLS
    ):
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm runs a row in one thread "
            "block, so cols must be a multiple of "
            f"{_POST_NORM_RESIDUAL_PACK_COLS} and at most "
            f"{_POST_NORM_RESIDUAL_MAX_COLS}. Got: {cols}"
        )
    residual_list, gamma_list, gamma2_list, mix_list = per_device
    for i, t in enumerate(inputs):
        if residual_list[i].type != t.type:
            raise ValueError(
                f"residual on device {i} must match its input: "
                f"{residual_list[i].type} vs {t.type}"
            )
        for name, weight in (
            ("gamma", gamma_list[i]),
            ("gamma2", gamma2_list[i]),
        ):
            if (
                weight.dtype != t.dtype
                or weight.shape != t.shape[-1:]
                or weight.device != t.device
            ):
                raise ValueError(
                    f"{name} on device {i} must be [cols] {t.dtype} on "
                    f"{t.device}. Got: {weight.type}"
                )
        mix = mix_list[i]
        # The kernel reads both weights, so a shorter mix reads past its end.
        if (
            mix.dtype != DType.float32
            or mix.shape != [2]
            or mix.device != t.device
        ):
            raise ValueError(
                f"mix on device {i} must be [2] float32 on {t.device}. "
                f"Got: {mix.type}"
            )
    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm requires unique devices "
            f"across its input tensors. Got: {devices=}"
        )

    graph = Graph.current
    in_chain = graph.device_chains.merge_for(devices)
    output_types = [t.type for t in inputs]
    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedAllreducePostNormResidualRmsNormOp,
        output_types,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        *per_device,
        in_chain,
        _float32_attr(epsilon),
    )
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain
    return (
        [res.tensor for res in results[:num_devices]],
        [res.tensor for res in results[num_devices:]],
    )
