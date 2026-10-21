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
"""Python wrapper for the varlen causal depthwise conv1d kernel.

:func:`causal_conv1d_varlen_fwd`
  Slot-indexed depthwise causal conv1d over a ragged batch. Reads/writes a
  per-layer conv-state pool in place at slot ``cache_indices[batch_item]``;
  handles prefill and decode in one op.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops


def causal_conv1d_varlen_fwd(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue,
    conv_states: BufferValue,
    query_start_loc: TensorValue,
    cache_indices: TensorValue,
    has_initial_state: TensorValue,
    activation: str = "silu",
    channels_last: bool = False,
    use_residual: bool = False,
    update_state: TensorValue | None = None,
) -> TensorValue:
    """Applies a slot-indexed varlen causal depthwise conv1d for prefill
    and decode.

    Mutates the conv-state pool ``conv_states`` in place at slot
    ``cache_indices[batch_item]`` — the Qwen3.5 GatedDeltaNet conv
    pattern. The builtin registers ``conv_states`` as a
    ``MutableInputTensor`` at operand position 4 (after ``output, x,
    weight, bias``). ``update_state`` is the last operand.

    Args:
        x: The ``[dim, total_seqlen]`` input (channels-first, model
            dtype), or ``[total_seqlen, dim]`` when ``channels_last`` is
            true.
        weight: The ``[dim, width]`` depthwise conv weights.
        bias: The ``[dim]`` per-channel bias (empty to disable).
        conv_states: The ``[max_slots, dim, width - 1]`` mutable
            conv-state pool.
        query_start_loc: The ``[batch + 1]`` int32 cumulative sequence
            lengths.
        cache_indices: The ``[batch]`` uint32 slot indices into
            ``conv_states``, or one ``[1, batch]`` row of a per-layer table.
        has_initial_state: The ``[batch]`` bool, whether to use the
            stored state.
        activation: ``"silu"`` or ``"none"``.
        channels_last: If true, ``x`` and the output are tokens-major
            ``[total_seqlen, dim]``. The kernel indexes through runtime
            strides, so this only relabels the axes — it avoids the
            materialized transposes the ``[dim, total_seqlen]`` contract
            forces on a tokens-major caller.
        use_residual: If true, adds ``x`` to the convolution sum at each
            output position, before ``activation``.
        update_state: The ``[batch]`` bool, whether each sequence writes its
            new window back to its ``conv_states`` slot. ``None`` writes every
            sequence's, as before the flag existed. A sequence with the flag
            false leaves its slot untouched, so the op reads the pool without
            advancing it. The output does not depend on the flag.

    Returns:
        The conv output with the same shape/layout as ``x``.
        ``conv_states`` is mutated in place.

    The following example convolves a ragged batch of two sequences, of
    lengths 3 and 2, with a width-4 kernel. The first sequence continues
    from the history stored in slot 2. The second starts fresh in slot 0:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU, Accelerator, Buffer, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import BufferType, DeviceRef, Graph, TensorType
        from max.nn.state_space import causal_conv1d_varlen_fwd

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)
        dim, total_seqlen, width, max_slots = 8, 5, 4, 3

        with Graph(
            "causal_conv1d_varlen_fwd_example",
            input_types=(
                TensorType(
                    DType.float32, [dim, total_seqlen], device=device_ref
                ),
                TensorType(DType.float32, [dim, width], device=device_ref),
                TensorType(DType.float32, [dim], device=device_ref),
                BufferType(
                    DType.float32,
                    [max_slots, dim, width - 1],
                    device=device_ref,
                ),
                TensorType(DType.int32, [3], device=device_ref),
                TensorType(DType.uint32, [2], device=device_ref),
                TensorType(DType.bool, [2], device=device_ref),
            ),
        ) as graph:
            (
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
            ) = graph.inputs
            output = causal_conv1d_varlen_fwd(
                x.tensor,
                weight.tensor,
                bias.tensor,
                conv_states.buffer,
                query_start_loc.tensor,
                cache_indices.tensor,
                has_initial_state.tensor,
            )
            graph.output(output)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((dim, total_seqlen), dtype=np.float32)
        weight_np = rng.standard_normal((dim, width), dtype=np.float32)
        bias_np = rng.standard_normal(dim, dtype=np.float32)
        states_np = rng.standard_normal(
            (max_slots, dim, width - 1), dtype=np.float32
        )
        starts = np.array([0, 3, 5], dtype=np.int32)
        slots = np.array([2, 0], dtype=np.uint32)
        use_history = np.array([True, False])

        inputs = [
            Buffer.from_numpy(array).to(device)
            for array in (
                x_np,
                weight_np,
                bias_np,
                states_np.copy(),
                starts,
                slots,
                use_history,
            )
        ]
        model = InferenceSession(devices=[device]).load(graph)
        result = model.execute(*inputs)[0].to_numpy()
        # The op updated the state pool, inputs[3], in place.
        states = inputs[3]
        # result.shape is (8, 5), the same layout as x.

    .. invisible-code-block: python

        assert result.shape == (dim, total_seqlen)
        assert result.dtype == np.float32

        # out[:, t] = silu(bias + sum_j weight[:, j] * window[:, t + j]),
        # where window prepends the slot's history (or zeros for a fresh
        # sequence). The slot then keeps the last width - 1 inputs.
        expected = np.empty((dim, total_seqlen))
        expected_states = states_np.astype(np.float64)
        sequences = zip(slots, use_history, starts[:-1], starts[1:])
        for slot, use_state, start, end in sequences:
            history = states_np[slot] if use_state else 0 * states_np[slot]
            window = np.concatenate([history, x_np[:, start:end]], axis=1)
            for t in range(end - start):
                taps = weight_np * window[:, t : t + width]
                pre = bias_np + taps.sum(axis=1)
                expected[:, start + t] = pre / (1 + np.exp(-pre))
            expected_states[slot] = window[:, -(width - 1) :]
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            states.to_numpy(), expected_states, rtol=1e-6, atol=1e-6
        )
    """
    device = x.device

    out_type = TensorType(x.dtype, x.shape, device)
    if update_state is None:
        # An empty flag tensor means every sequence writes.
        update_state = ops.constant(
            np.zeros(0, dtype=np.bool_), DType.bool, device=device
        )

    # Operand order matches the builtin registration
    # ``CausalConv1DVarlenFwd.execute``: after the ``output`` operand,
    # ``x, weight, bias, conv_states (MutableInput), query_start_loc,
    # cache_indices, has_initial_state, update_state``.
    results = ops.inplace_custom(
        "causal_conv1d_varlen_fwd",
        device,
        [
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            update_state,
        ],
        [out_type],
        parameters={
            "activation": activation,
            "channels_last": channels_last,
            "use_residual": use_residual,
        },
    )
    return cast(TensorValue, results[0])
