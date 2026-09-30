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
neither pool has a length pointer to move. This module implements the
snapshot-and-replay design of ``mach/docs/qwen38_27b/mtp-spec-design.md``,
in the cheap form the measurements allow:

- **Snapshot.** Before the verify, each request's live recurrent slot is
  copied into a batch-indexed shadow pool. The verify's recurrence then runs
  on the shadow, so the live pool still holds the pre-verify state when
  acceptance is known. The conv leaf takes no copy. With drafts to reject,
  the verify leaves the conv window unwritten and the replay writes it.
- **Replay.** Every op feeding the two state kernels is causal or pointwise,
  so rows ``[0, j)`` of the verify's own conv input, decay and beta are
  bit-identical to what a forward over the accepted prefix alone would have
  produced. Re-running just the two state kernels over those rows, from the
  untouched live pool, lands the live pool exactly where the verify pass was
  at length ``j`` — without a second pass over the model's weights.

The replay is therefore ~2 kernels per linear layer instead of a whole extra
target forward, which is what makes the TPOT win survive the rollback. The
recurrence replay is unconditional: prefill takes the same path with an
accepted length equal to the whole prompt, so no branch depends on the phase.
A window with no drafts lands its conv window on the verify, so there the conv
replay launches nothing.

With ``speculative_config.recurrent_state_rollback == "ring"``, the verify
reads the live recurrent pool and records each token's update in a ring, and
:func:`fold_state_pools` replaces that leaf's snapshot and replay. A window
with no drafts has nothing to roll back, so there the verify runs the plain
recurrence on the live pool and the fold launches nothing. A window with
drafts is ``K + 1`` tokens, which the ring is sized to hold.

Every pool is addressed as one buffer per leaf plus a
``[num_layers, batch_size]`` tensor of the rows each layer occupies. The engine
supplies the live and ring rows, and a shadow is this graph's own scratch, so
:func:`shadow_row_ids` picks that layout here.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.state_space import (
    gated_delta_conv1d_verify_fwd,
    gated_delta_recurrence_fwd,
    gated_delta_state_fold,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs

__all__ = [
    "accepted_lengths",
    "accepted_row_plan",
    "fold_state_pools",
    "replay_state_pools",
    "shadow_row_ids",
    "snapshot_state_pools",
]


_SHADOW_SPAN = "shadow_span"
"""Name of the shadow slice a snapshot fills: ``batch_size * num_layers``."""


def shadow_row_ids(num_layers: int, device: DeviceRef) -> TensorValue:
    """Returns the ``[num_layers, batch_size]`` uint32 shadow-pool rows.

    Request ``r``'s layer ``l`` sits at row ``l * batch_size + r``, matching
    where the snapshot's layer-major flatten of the live rows lands it.
    """
    span = num_layers * Dim("batch_size")
    rows = ops.range(
        start=0, stop=span, out_dim=span, device=device, dtype=DType.uint32
    )
    return rows.reshape([num_layers, "batch_size"])


def snapshot_state_pools(
    live_pools: Sequence[BufferValue],
    shadow_pools: Sequence[BufferValue],
    live_row_ids: Sequence[TensorValue],
    shadow_span: TensorValue,
) -> None:
    """Copies each request's live rows into the shadow pool.

    Args:
        live_pools: One persistent pool per device, for a single leaf.
        shadow_pools: One scratch pool per device, at least
            ``max_batch_size * num_layers`` rows deep.
        live_row_ids: Per-device ``[num_layers, batch_size]`` live rows.
        shadow_span: Scalar ``batch_size * num_layers``, the slice filled.
    """
    for live, shadow, rows in zip(
        live_pools, shadow_pools, live_row_ids, strict=True
    ):
        gathered = ops.gather(
            ops.buffer_load(live), ops.reshape(rows, [-1]), axis=0
        )
        ops.buffer_store_slice(
            shadow,
            ops.rebind(gathered, [_SHADOW_SPAN, *gathered.shape[1:]]),
            [(slice(0, shadow_span), _SHADOW_SPAN)],
        )


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
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay's gather indices and ragged offsets.

    A row's accepted length is its merged length minus the drafts it did not
    accept, which is the prompt length during prefill (where there are no
    drafts) and ``1 + num_accepted`` during decode -- one expression, no phase
    branch.

    The gather is deliberately *not* trimmed to the accepted total: that
    length is only known on the device, and materializing it as a shape would
    cost a device-to-host sync every step. Instead the index vector keeps the
    verified window's row count and the trailing entries repeat the last
    accepted row. ``replay_offsets`` stops at the accepted total, so the
    state kernels never reach those trailing entries.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K`` on ``device``.
        total_rows: Row count of the verify pass's per-token tensors.
        device: Device the plan is built on.

    Returns:
        ``(row_indices, replay_offsets)``: which row of the verify's per-token
        tensors feeds each replay row, and the ragged offsets over them.
    """
    starts = ops.rebind(merged_offsets.cast(DType.int64)[:-1], ["batch_size"])
    accepted = accepted_lengths(merged_offsets, num_accepted, num_draft_tokens)

    # ``ops.cumsum`` is CPU-only, and a device-to-host hop here would stall
    # the stream every step; the batch is small enough that a triangular sum
    # is cheaper than the sync.
    batch_pos = ops.range(
        start=0,
        stop=Dim("batch_size"),
        out_dim=Dim("batch_size"),
        device=device,
        dtype=DType.int64,
    )
    lower = (ops.unsqueeze(batch_pos, -1) >= ops.unsqueeze(batch_pos, 0)).cast(
        DType.int64
    )
    replay_offsets = ops.concat(
        [
            ops.constant(0, DType.int64, device=device).reshape([1]),
            ops.sum(lower * ops.unsqueeze(accepted, 0), axis=-1).reshape([-1]),
        ],
        axis=0,
    )

    out_pos = ops.range(
        start=0,
        stop=total_rows,
        out_dim=total_rows,
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


def replay_state_pools(
    captures: Sequence[Sequence[GatedDeltaReplayInputs]],
    live_conv_pools: Sequence[BufferValue],
    live_recurrent_pools: Sequence[BufferValue] | None,
    conv_row_ids: Sequence[TensorValue],
    recurrent_row_ids: Sequence[TensorValue] | None,
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Re-runs the state kernels over the accepted prefix.

    Reuses the verify pass's own per-token inputs rather than recomputing the
    projections, so the replayed arithmetic is the same arithmetic — not
    merely a close approximation of it.

    The conv replay writes the conv window at the accepted length, carrying
    forward from the pre-verify window still in the pool. The recurrence
    replays over the verify's own activated conv output.

    Args:
        captures: Per-device, per-layer inputs captured by the verify pass.
        live_conv_pools: Per-device conv pool, still pre-verify.
        live_recurrent_pools: Per-device recurrent pool, or ``None`` on the
            ring rollback, which folds the recurrence instead of replaying
            it.
        conv_row_ids: Per-device ``[num_layers, batch_size]`` conv rows.
        recurrent_row_ids: Per-device ``[num_layers, batch_size]`` state
            rows, ``None`` with ``live_recurrent_pools``.
        row_indices: Rows of the verify tensors the replay consumes.
        replay_offsets: ``[batch + 1]`` ragged offsets over those rows.
        signal_buffers: Used only to place the plan on each device.
        verify_width: The verify's width operand, which skips the conv
            launch at width zero.
    """
    assert (live_recurrent_pools is None) == (recurrent_row_ids is None)
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
        recurrent_row_id = (
            None
            if recurrent_row_ids is None
            else recurrent_row_ids[device_idx].cast(DType.uint32)
        )
        for layer_idx, capture in enumerate(device_captures):
            # No resume row: the live rows still hold the pre-verify state.
            gated_delta_conv1d_verify_fwd(
                qkv_input_ragged=ops.gather(capture.qkv, rows, axis=0),
                conv_weight=capture.conv_weight,
                conv_state=conv_pool,
                slot_idx=conv_row_id[layer_idx],
                input_row_offsets=offsets,
                verify_width=verify_width,
                rollback=True,
            )
            if live_recurrent_pools is None or recurrent_row_id is None:
                continue
            gated_delta_recurrence_fwd(
                qkv_conv_output=ops.gather(capture.conv_output, rows, axis=0),
                decay_per_token=ops.gather(capture.decay, rows, axis=0),
                beta_per_token=ops.gather(capture.beta, rows, axis=0),
                recurrent_state=live_recurrent_pools[device_idx],
                slot_idx=recurrent_row_id[layer_idx],
                input_row_offsets=offsets,
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
