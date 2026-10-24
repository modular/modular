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
"""Python wrappers for the Gated DeltaNet two-pass kernels.

Provides two graph-level wrappers that call the Mojo ops registered in the
``state_space`` package:

  ``gated_delta_conv1d_fwd()``
    Pass 1: causal depthwise conv1d over a ragged batch of sequences.
    Reads/writes a single mutable conv-state pool of shape
    ``[max_slots, conv_dim, K-1]`` at slot ``slot_idx[batch_item]`` —
    no gather/scatter, no working buffers.

  ``gated_delta_recurrence_fwd()``
    Pass 2: gated delta rule recurrence over the conv1d outputs. Reads/
    writes a single mutable recurrent-state pool of shape
    ``[max_slots, nv, KD, VD]`` at slot ``slot_idx[batch_item]``.

Both ops mutate their pool inputs in place (``ops.inplace_custom``); the
graph output is just the per-token tensor (conv output / recurrence
output respectively). This matches vLLM's ``selective_state_update``
design — kernel does pointer arithmetic ``state_ptr += slot * stride``
into a long-lived pool, no per-step pool allocation.


**Usage:**

.. skip: next

.. code-block:: python

    # Illustrative fragment: the pools and projected tensors come from a
    # live gated-deltanet layer, so it isn't runnable standalone.
    from max.nn.state_space import (
        gated_delta_conv1d_fwd,
        gated_delta_recurrence_fwd,
    )

    # Pass 1: conv_pool is a BufferValue mutated in place at slot_idx[b].
    conv_output = gated_delta_conv1d_fwd(
        qkv_input_ragged=qkv_f32,         # [total_N, conv_dim]
        conv_weight=conv_weight_flat,     # [conv_dim, K]
        conv_state=conv_pool,             # [max_slots, conv_dim, K-1] (mut)
        slot_idx=slot_idx_uint32,         # [B]
        input_row_offsets=offsets_uint32, # [B+1]
    )

    # Pass 2: recurrent_pool is mutated in place too; there is no
    # state-out graph output.
    recurrence_output = gated_delta_recurrence_fwd(
        qkv_conv_output=conv_output,      # [total_N, conv_dim]
        decay_per_token=decay,            # [total_N, nv]
        beta_per_token=beta,              # [total_N, nv]
        recurrent_state=recurrent_pool,   # [max_slots, nv, kd, vd] (mut)
        slot_idx=slot_idx_uint32,         # [B]
        input_row_offsets=offsets_uint32, # [B+1]
    )
"""

from __future__ import annotations

from typing import NamedTuple, cast

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
    ops,
)


class GatedDeltaReplayInputs(NamedTuple):
    """One layer's per-token inputs to the two state kernels.

    Speculative decoding runs the verify pass over K+1 positions and then has
    to land the pools on the accepted prefix instead. Every op feeding these
    tensors is causal or pointwise, so re-running the two kernels over the
    accepted rows from the pre-verify state reproduces the state the verify
    pass held at that length. Captured only when the caller asks for it; see
    ``mach/docs/qwen38_27b/mtp-serving.md``.
    """

    qkv: TensorValue
    """``[total_seq_len, conv_dim]`` float32 conv input."""
    conv_weight: TensorValue
    """``[conv_dim, kernel_size]`` float32 depthwise weights."""
    decay: TensorValue
    """``[total_seq_len, num_value_heads]`` float32 decays."""
    beta: TensorValue
    """``[total_seq_len, num_value_heads]`` float32 beta gates."""
    conv_output: TensorValue
    """``[total_seq_len, conv_dim]`` float32 activated conv output.

    The conv is causal and a verify reads the window as it was before it
    ran, so a replay of the recurrence reads its rows here instead of running
    the conv again.
    """
    qkv_window: TensorValue | None = None
    """``[batch_size * (1 + K), conv_dim]`` float32 copy of ``qkv`` that a
    verify's conv kernel writes when there are drafts, or ``None`` outside a
    verify. A conv replay that reads this instead of ``qkv`` lets the next
    layer reuse ``qkv``'s memory."""


def _as_uint32(value: TensorValue) -> TensorValue:
    return (
        value if value.type.dtype == DType.uint32 else value.cast(DType.uint32)
    )


def verify_width_operand(num_draft_tokens: DimLike) -> TensorValue:
    """Returns the operand the speculative state ops read the verify width from.

    The ops read only its length, which is host metadata, so choosing between
    landing the state on the forward and deferring it to the rollback costs
    no device synchronization. At a width of zero there is no draft to
    reject.

    Args:
        num_draft_tokens: The graph's draft width ``K``, as a dim.

    Returns:
        A ``[K]`` int64 tensor on the host.
    """
    return ops.range(
        start=0,
        stop=Dim(num_draft_tokens),
        out_dim=num_draft_tokens,
        device=DeviceRef.CPU(),
        dtype=DType.int64,
    )


def gated_delta_conv1d_fwd(
    qkv_input_ragged: TensorValue,
    conv_weight: TensorValue,
    conv_state: BufferValue,
    slot_idx: TensorValue,
    input_row_offsets: TensorValue,
    *,
    write_state: bool = True,
) -> TensorValue:
    """Applies the causal conv1d pass, mutating a slot-indexed conv-state
    pool in place.

    ``conv_state`` is a mutable pool of shape ``[max_slots, conv_dim,
    kernel_size - 1]`` and the kernel reads/writes slot
    ``slot_idx[batch_item]`` directly. There is no ``conv_state_out``
    graph output: the pool is mutated in place.

    Args:
        qkv_input_ragged: The ``[total_seq_len, conv_dim]`` projected QKV
            input.
        conv_weight: The ``[conv_dim, kernel_size]`` depthwise conv
            weights.
        conv_state: The ``[max_slots, conv_dim, kernel_size - 1]``
            mutable pool.
        slot_idx: The ``[batch_size]`` uint32 slot indices into the pool.
        input_row_offsets: The ``[batch_size + 1]`` uint32 ragged offsets.
        write_state: Whether to write the updated window back to
            ``conv_state``. ``False`` only reads it for the look-back.

    Returns:
        The conv output, ``[total_seq_len, conv_dim]``.

    The following example convolves a ragged batch of two sequences, of
    lengths 3 and 2, on a GPU. The sequences read and update conv-state
    slots 2 and 0:

    .. Skipped without an NVIDIA or AMD GPU: the kernel has no CPU
       implementation, and Apple GPUs are untested. The ``start``/``end``
       pair also covers the invisible check, which reads this block's
       results.
    .. skip: start if(__import__("max.driver").driver.accelerator_count() == 0 or __import__("sys").platform == "darwin", reason="requires an NVIDIA or AMD GPU")

    .. code-block:: python

        import numpy as np
        from max.driver import Accelerator, Buffer
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import BufferType, DeviceRef, Graph, TensorType
        from max.nn.state_space import gated_delta_conv1d_fwd

        total_seq_len, conv_dim, kernel_size, max_slots = 5, 8, 4, 3
        gpu = DeviceRef.GPU()

        with Graph(
            "gated_delta_conv1d_fwd_example",
            input_types=(
                TensorType(
                    DType.float32, [total_seq_len, conv_dim], device=gpu
                ),
                TensorType(DType.float32, [conv_dim, kernel_size], device=gpu),
                BufferType(
                    DType.float32,
                    [max_slots, conv_dim, kernel_size - 1],
                    device=gpu,
                ),
                TensorType(DType.uint32, [2], device=gpu),
                TensorType(DType.uint32, [3], device=gpu),
            ),
        ) as graph:
            x, conv_weight, conv_state, slot_idx, input_row_offsets = (
                graph.inputs
            )
            output = gated_delta_conv1d_fwd(
                qkv_input_ragged=x.tensor,
                conv_weight=conv_weight.tensor,
                conv_state=conv_state.buffer,
                slot_idx=slot_idx.tensor,
                input_row_offsets=input_row_offsets.tensor,
            )
            graph.output(output)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((total_seq_len, conv_dim), dtype=np.float32)
        weight_np = rng.standard_normal(
            (conv_dim, kernel_size), dtype=np.float32
        )
        state_np = rng.standard_normal(
            (max_slots, conv_dim, kernel_size - 1), dtype=np.float32
        )
        slots = np.array([2, 0], dtype=np.uint32)
        offsets = np.array([0, 3, 5], dtype=np.uint32)

        gpu = Accelerator()
        inputs = [
            Buffer.from_numpy(array).to(gpu)
            for array in (x_np, weight_np, state_np.copy(), slots, offsets)
        ]
        model = InferenceSession(devices=[gpu]).load(graph)
        result = model.execute(*inputs)[0].to_numpy()
        # The op updated the state pool, inputs[2], in place.
        state = inputs[2]
        # result.shape is (5, 8). Slots 2 and 0 of state now hold the last
        # three inputs of each sequence, and slot 1 is unchanged.

    .. invisible-code-block: python

        assert result.shape == (total_seq_len, conv_dim)
        assert result.dtype == np.float32

        # Each sequence continues from the history in its slot (oldest
        # first): out[t, c] = sum_j weight[c, j] * window[t + j, c].
        expected = np.empty((total_seq_len, conv_dim))
        expected_state = state_np.astype(np.float64)
        for slot, start, end in zip(slots, offsets[:-1], offsets[1:]):
            window = np.concatenate([state_np[slot].T, x_np[start:end]])
            for t in range(end - start):
                expected[start + t] = (
                    weight_np.T * window[t : t + kernel_size]
                ).sum(axis=0)
            expected_state[slot] = window[-(kernel_size - 1) :].T
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            state.to_numpy(), expected_state, rtol=1e-6, atol=1e-6
        )

    .. skip: end
    """
    device = qkv_input_ragged.device
    total_seq_len = qkv_input_ragged.shape[0]
    conv_dim = qkv_input_ragged.shape[1]

    conv_output_ragged_type = TensorType(
        DType.float32, [total_seq_len, conv_dim], device
    )

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )
    slot_idx_uint32 = (
        slot_idx
        if slot_idx.type.dtype == DType.uint32
        else slot_idx.cast(DType.uint32)
    )

    results = ops.inplace_custom(
        "gated_delta_conv1d_fwd",
        device,
        [
            qkv_input_ragged,
            conv_weight,
            conv_state,
            slot_idx_uint32,
            offsets_uint32,
        ],
        [conv_output_ragged_type],
        parameters={"write_state": write_state},
    )
    return cast(TensorValue, results[0])


def gated_delta_conv1d_verify_fwd(
    qkv_input_ragged: TensorValue,
    conv_weight: TensorValue,
    conv_state: BufferValue,
    slot_idx: TensorValue,
    input_row_offsets: TensorValue,
    verify_width: TensorValue,
    *,
    rollback: bool,
) -> tuple[TensorValue, TensorValue]:
    """Runs :func:`gated_delta_conv1d_fwd` in a speculative verify graph.

    At a verify width of zero the forward launch writes the window and the
    rollback launch does nothing, leaving its output unwritten. Otherwise the
    forward launch only reads the window and the rollback launch writes it.

    The forward launch with drafts also copies its input for the rollback to
    replay. Every row is then one token and its drafts, so the copy is the
    whole input. Every other launch leaves the copy unwritten.

    Args:
        qkv_input_ragged: The ``[total_seq_len, conv_dim]`` conv input.
        conv_weight: The ``[conv_dim, kernel_size]`` depthwise weights.
        conv_state: The ``[max_slots, conv_dim, kernel_size - 1]`` pool.
        slot_idx: The ``[batch_size]`` uint32 pool rows.
        input_row_offsets: The ``[batch_size + 1]`` uint32 ragged offsets.
        verify_width: The operand :func:`verify_width_operand` returns.
        rollback: Whether this is the rollback's launch.

    Returns:
        The conv output, ``[total_seq_len, conv_dim]``, and the copy of the
        input, ``[batch_size * (1 + K), conv_dim]``.
    """
    device = qkv_input_ragged.device
    conv_dim = qkv_input_ragged.shape[1]
    results = ops.inplace_custom(
        "gated_delta_conv1d_verify_fwd",
        device,
        [
            qkv_input_ragged,
            conv_weight,
            conv_state,
            _as_uint32(slot_idx),
            _as_uint32(input_row_offsets),
            verify_width,
        ],
        [
            TensorType(
                DType.float32, [qkv_input_ragged.shape[0], conv_dim], device
            ),
            TensorType(
                qkv_input_ragged.dtype,
                [
                    Dim(slot_idx.shape[0]) * (Dim(verify_width.shape[0]) + 1),
                    conv_dim,
                ],
                device,
            ),
        ],
        parameters={"rollback": rollback},
    )
    return cast(TensorValue, results[0]), cast(TensorValue, results[1])


def gated_delta_recurrence_fwd(
    qkv_conv_output: TensorValue,
    decay_per_token: TensorValue,
    beta_per_token: TensorValue,
    recurrent_state: BufferValue,
    slot_idx: TensorValue,
    input_row_offsets: TensorValue,
) -> TensorValue:
    """Applies the gated delta recurrence pass, mutating a slot-indexed
    state pool in place.

    ``recurrent_state`` is a mutable pool of shape ``[max_slots, nv, KD,
    VD]`` and the kernel reads/writes slot ``slot_idx[batch_item]``
    directly. There is no ``recurrent_state_out`` graph output: the pool
    is mutated in place.

    Args:
        qkv_conv_output: The ``[total_seq_len, conv_dim]`` output of
            :func:`gated_delta_conv1d_fwd`.
        decay_per_token: The ``[total_seq_len, num_value_heads]`` decays.
        beta_per_token: The ``[total_seq_len, num_value_heads]`` beta
            gates.
        recurrent_state: The ``[max_slots, nv, KD, VD]`` mutable pool.
        slot_idx: The ``[batch_size]`` uint32 slot indices into the pool.
        input_row_offsets: The ``[batch_size + 1]`` uint32 ragged offsets.

    Returns:
        The recurrence output, ``[total_seq_len, value_dim]``.
    """
    device = qkv_conv_output.device
    total_seq_len = qkv_conv_output.shape[0]
    num_value_heads = decay_per_token.shape[1]
    # recurrent_state.shape is [max_slots, nv, KD, VD]; index 3 is value_head_dim.
    value_head_dim = recurrent_state.shape[3]
    value_dim = num_value_heads * value_head_dim

    recurrence_output_type = TensorType(
        DType.float32, [total_seq_len, value_dim], device
    )

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )
    slot_idx_uint32 = (
        slot_idx
        if slot_idx.type.dtype == DType.uint32
        else slot_idx.cast(DType.uint32)
    )

    results = ops.inplace_custom(
        "gated_delta_recurrence_fwd",
        device,
        [
            qkv_conv_output,
            decay_per_token,
            beta_per_token,
            recurrent_state,
            slot_idx_uint32,
            offsets_uint32,
        ],
        [recurrence_output_type],
    )
    return cast(TensorValue, results[0])


def gated_delta_recurrence_verify_ring_fwd(
    qkv_conv_output: TensorValue,
    decay_per_token: TensorValue,
    beta_per_token: TensorValue,
    recurrent_state: BufferValue,
    ring: BufferValue,
    slot_idx: TensorValue,
    ring_slot_idx: TensorValue,
    input_row_offsets: TensorValue,
    verify_width: TensorValue,
) -> TensorValue:
    """Runs the recurrence over a speculative verify window into a ring.

    Produces the same readout as :func:`gated_delta_recurrence_fwd` without
    writing ``recurrent_state``. Each token writes its decay, raw key and
    delta factor to its request's ``ring`` row, which
    :func:`gated_delta_state_fold` applies once acceptance is known.

    At a verify width of zero this is :func:`gated_delta_recurrence_fwd` on
    ``recurrent_state``, and no ring is written.

    Args:
        qkv_conv_output: The ``[total_seq_len, conv_dim]`` conv output.
        decay_per_token: The ``[total_seq_len, nv]`` decays.
        beta_per_token: The ``[total_seq_len, nv]`` beta gates.
        recurrent_state: The ``[max_slots, nv, KD, VD]`` live pool, written
            only at a verify width of zero.
        ring: The ``[ring_rows, nk, RING_LEN, record_stride]`` ring pool.
        slot_idx: The ``[batch_size]`` live pool rows.
        ring_slot_idx: The ``[batch_size]`` ring rows.
        input_row_offsets: The ``[batch_size + 1]`` uint32 ragged offsets.
        verify_width: The operand :func:`verify_width_operand` returns.

    Returns:
        The recurrence output, ``[total_seq_len, value_dim]``.
    """
    device = qkv_conv_output.device
    value_dim = decay_per_token.shape[1] * recurrent_state.shape[3]
    results = ops.inplace_custom(
        "gated_delta_recurrence_verify_ring_fwd",
        device,
        [
            qkv_conv_output,
            decay_per_token,
            beta_per_token,
            recurrent_state,
            ring,
            _as_uint32(slot_idx),
            _as_uint32(ring_slot_idx),
            _as_uint32(input_row_offsets),
            verify_width,
        ],
        [
            TensorType(
                DType.float32, [qkv_conv_output.shape[0], value_dim], device
            )
        ],
    )
    return cast(TensorValue, results[0])


def gated_delta_state_fold(
    recurrent_state: BufferValue,
    ring: BufferValue,
    row_ids: TensorValue,
    ring_row_ids: TensorValue,
    num_accepted: TensorValue,
    verify_width: TensorValue,
) -> None:
    """Applies a verify's accepted ring records to the state pool.

    Advances ``recurrent_state`` over the first ``num_accepted[b]`` records
    written by :func:`gated_delta_recurrence_verify_ring_fwd`, for every
    layer in one launch. A row with ``num_accepted[b] == 0`` is left
    unchanged. Nothing runs at a verify width of zero, where the verify wrote
    no records.

    Args:
        recurrent_state: The ``[max_slots, nv, KD, VD]`` live pool, folded in
            place.
        ring: The ``[ring_rows, nk, RING_LEN, record_stride]`` ring pool.
        row_ids: The ``[num_layers, batch_size]`` live pool rows.
        ring_row_ids: The ``[num_layers, batch_size]`` ring rows.
        num_accepted: The ``[batch_size]`` records to fold. A count past
            ``RING_LEN`` names a row the verify was too long to record and
            committed itself, and is skipped.
        verify_width: The operand :func:`verify_width_operand` returns.
    """
    ops.inplace_custom(
        "gated_delta_state_fold",
        recurrent_state.device,
        [
            recurrent_state,
            ring,
            _as_uint32(row_ids),
            _as_uint32(ring_row_ids),
            _as_uint32(num_accepted),
            verify_width,
        ],
    )
