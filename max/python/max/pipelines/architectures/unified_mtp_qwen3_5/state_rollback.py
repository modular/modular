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
"""Recurrent-state rollback for Qwen3.5 speculative decoding.

Rejecting a draft token has to un-advance 48 Gated DeltaNet recurrences, and
neither pool has a length pointer to move. The two leaves roll back by
different mechanisms:

- **Recurrent.** The verify reads the live pool without writing it and
  records each token's update in a ring, a scratch leaf of the state cache.
  Once acceptance is known, :func:`fold_state_pools` applies the accepted
  records to the live pool in one launch per device.
- **Conv.** The window is the last ``kernel_size - 1`` raw inputs, so the
  verify leaves it unwritten and :func:`replay_conv_pools` re-runs the conv
  kernel over the accepted rows of the verify's own input. Every op feeding it
  is causal or pointwise, so those rows are bit-identical to what a forward
  over the accepted prefix alone would have produced.

A window with no drafts has nothing to roll back, so at a verify width of zero
the verify's forward lands both leaves on the live pools and every rollback op
launches nothing. A window with drafts is ``K + 1`` tokens, which the ring is
sized to hold.

Every pool is addressed as one buffer per leaf plus a
``[num_layers, batch_size]`` tensor of the rows each layer occupies, and the
engine supplies both for the live leaves and the ring alike.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.state_space import (
    gated_delta_conv1d_verify_fwd,
    gated_delta_state_fold,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs

__all__ = [
    "accepted_lengths",
    "accepted_row_plan",
    "fold_state_pools",
    "replay_conv_pools",
]


def accepted_lengths(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
) -> TensorValue:
    """Returns the ``[batch]`` rows of the verify window each request keeps.

    This is the prompt length on a prefill and ``1 + num_accepted`` on a
    decode.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K``, zero on a prefill.

    Returns:
        ``[batch]`` int64 accepted row counts.
    """
    offsets = merged_offsets.cast(DType.int64)
    return (
        ops.rebind(offsets[1:], ["batch_size"])
        - ops.rebind(offsets[:-1], ["batch_size"])
        - num_draft_tokens
        + num_accepted.cast(DType.int64).rebind(["batch_size"])
    )


def accepted_row_plan(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    total_rows: Dim,
    device: DeviceRef,
    *,
    plan_rows: Dim | None = None,
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay's gather indices and ragged offsets.

    A row's accepted length is its merged length minus the drafts it did not
    accept, which is the prompt length during prefill (where there are no
    drafts) and ``1 + num_accepted`` during decode -- one expression, no phase
    branch.

    The gather is deliberately *not* trimmed to the accepted total: that
    length is only known on the device, and materializing it as a shape would
    cost a device-to-host sync every step. Instead the index vector is
    ``plan_rows`` long and the trailing entries repeat the last accepted row.
    ``replay_offsets`` stops at the accepted total, so the conv kernel never
    reaches those trailing entries.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K`` on ``device``.
        total_rows: Row count of the verify pass's per-token tensors.
        device: Device the plan is built on.
        plan_rows: Length of the index vector, at least the accepted total.
            Defaults to ``total_rows``.

    Returns:
        ``(row_indices, replay_offsets)``: which row of the verify's per-token
        tensors feeds each replay row, and the ragged offsets over them.
    """
    starts = ops.rebind(merged_offsets.cast(DType.int64)[:-1], ["batch_size"])
    accepted = accepted_lengths(merged_offsets, num_accepted, num_draft_tokens)

    replay_offsets = ops.concat(
        [
            ops.constant(0, DType.int64, device=device).reshape([1]),
            ops.cumsum(accepted, axis=0),
        ],
        axis=0,
    )

    if plan_rows is None:
        plan_rows = total_rows
    out_pos = ops.range(
        start=0,
        stop=plan_rows,
        out_dim=plan_rows,
        device=device,
        dtype=DType.int64,
    )
    last_row = _shape_to_scalar(Dim("batch_size"), device) - 1
    row_of_pos = ops.min(
        ops.sum(
            (
                ops.unsqueeze(out_pos, -1)
                >= ops.unsqueeze(replay_offsets[1:], 0)
            ).cast(DType.int64),
            axis=-1,
        ).reshape([-1]),
        last_row,
    )
    row_indices = ops.gather(starts, row_of_pos, axis=0) + (
        out_pos - ops.gather(replay_offsets[:-1], row_of_pos, axis=0)
    )
    # Rows past the accepted total are never read; clamp them so the gather
    # itself stays in bounds.
    return (
        ops.min(row_indices, _shape_to_scalar(total_rows, device) - 1),
        replay_offsets,
    )


def replay_conv_pools(
    captures: Sequence[Sequence[GatedDeltaReplayInputs]],
    live_conv_pools: Sequence[BufferValue],
    conv_row_ids: Sequence[TensorValue],
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Writes each conv window at the accepted length.

    Re-runs the conv kernel over the accepted rows of the verify pass's own
    input, carrying forward from the pre-verify window still in the pool.

    Args:
        captures: Per-device, per-layer inputs captured by the verify pass.
        live_conv_pools: Per-device conv pool, still pre-verify.
        conv_row_ids: Per-device ``[num_layers, batch_size]`` conv rows.
        row_indices: Rows of the verify tensors the replay consumes.
        replay_offsets: ``[batch + 1]`` ragged offsets over those rows.
        signal_buffers: Used only to place the plan on each device.
        verify_width: The verify's width operand, which skips every launch
            at width zero.
    """
    offsets_per_dev = (
        ops.distributed_broadcast(replay_offsets, list(signal_buffers))
        if len(captures) > 1
        else [replay_offsets]
    )
    rows_per_dev = (
        ops.distributed_broadcast(row_indices, list(signal_buffers))
        if len(captures) > 1
        else [row_indices]
    )

    for device_idx, device_captures in enumerate(captures):
        rows = rows_per_dev[device_idx]
        offsets = offsets_per_dev[device_idx].cast(DType.uint32)
        conv_pool = live_conv_pools[device_idx]
        conv_row_id = conv_row_ids[device_idx].cast(DType.uint32)
        for layer_idx, capture in enumerate(device_captures):
            # No resume row: the live rows still hold the pre-verify window.
            gated_delta_conv1d_verify_fwd(
                qkv_input_ragged=ops.gather(capture.qkv, rows, axis=0),
                conv_weight=capture.conv_weight,
                conv_state=conv_pool,
                slot_idx=conv_row_id[layer_idx],
                input_row_offsets=offsets,
                verify_width=verify_width,
                rollback=True,
            )


def fold_state_pools(
    live_recurrent_pools: Sequence[BufferValue],
    recurrent_row_ids: Sequence[TensorValue],
    ring_pools: Sequence[BufferValue],
    ring_row_ids: Sequence[TensorValue],
    fold_accepted: TensorValue,
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Applies the verify's accepted ring records to the live recurrent pool.

    One launch per device covers every layer, and none at a verify width of
    zero.

    Args:
        live_recurrent_pools: Per-device recurrent pool, still pre-verify.
        recurrent_row_ids: Per-device ``[num_layers, batch_size]`` state rows.
        ring_pools: Per-device ring pool.
        ring_row_ids: Per-device ``[num_layers, batch_size]`` ring rows.
        fold_accepted: ``[batch]`` records to fold.
        signal_buffers: Used only to place ``fold_accepted`` on each device.
        verify_width: The verify's width operand.
    """
    n_devs = len(live_recurrent_pools)
    fold_accepted = fold_accepted.cast(DType.uint32)
    accepted_per_dev = (
        ops.distributed_broadcast(fold_accepted, list(signal_buffers))
        if n_devs > 1
        else [fold_accepted]
    )
    for device_idx in range(n_devs):
        gated_delta_state_fold(
            recurrent_state=live_recurrent_pools[device_idx],
            ring=ring_pools[device_idx],
            row_ids=recurrent_row_ids[device_idx],
            ring_row_ids=ring_row_ids[device_idx],
            num_accepted=accepted_per_dev[device_idx],
            verify_width=verify_width,
        )
