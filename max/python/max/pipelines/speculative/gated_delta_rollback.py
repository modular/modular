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
"""Recurrent-state rollback for a Gated DeltaNet target's spec decoding.

Rejecting a draft token has to un-advance every Gated DeltaNet recurrence in
the target, and neither pool has a length pointer to move. The two leaves
roll back by different mechanisms:

- **Recurrent.** The verify reads the live pool without writing it and
  records each token's update in a ring, a scratch leaf of the state cache.
  Once acceptance is known, :func:`fold_state_pools` applies the accepted
  records to the live pool in one launch per device.
- **Conv.** The window is the last ``kernel_size - 1`` raw inputs, so the
  verify leaves it unwritten and :func:`replay_conv_pools` re-runs the conv
  kernel over the accepted rows of the copy the verify keeps of its input,
  one verify-window (``K + 1``) slot per row.
  Every op feeding it is causal or pointwise, so those rows are bit-identical
  to what a forward over the accepted prefix alone would have produced.

A window with no drafts has nothing to roll back, so at a verify width of zero
the verify's forward lands both leaves on the live pools and every rollback op
launches nothing. A decode's window is ``K + 1`` tokens; a prompt mixed
into the batch runs longer than the verify window, and the verify commits
both leaves of such a row itself at its last real token, so it neither
replays nor folds anything and takes no slot of the copy.

Every pool is addressed as one buffer per leaf plus a
``[num_layers, batch_size]`` tensor of the rows each layer occupies, and the
engine supplies both for the live leaves and the ring alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, DimLike, TensorValue, ops
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_verify_fwd,
    gated_delta_state_fold,
)

from .ragged_token_merger import _shape_to_scalar

__all__ = [
    "ConvReplayInputs",
    "ReplicaAcceptInputs",
    "accepted_lengths",
    "accepted_row_plan",
    "fold_state_pools",
    "replay_conv_pools",
    "replica_accept_inputs",
    "ring_accepted_lengths",
    "window_row_plan",
]


class ConvReplayInputs(Protocol):
    """What the conv replay reads from one layer's captured verify inputs.

    Any linear-attention layer that runs ``gated_delta_conv1d_fwd`` captures
    these, whatever its recurrence.
    """

    @property
    def qkv_window(self) -> TensorValue | None:
        """The verify's copy of its conv input, or ``None`` outside a verify."""
        ...

    @property
    def conv_weight(self) -> TensorValue:
        """``[conv_dim, kernel_size]`` float32 depthwise weights."""
        ...


def _window_lengths(
    merged_offsets: TensorValue, batch_dim: DimLike = "batch_size"
) -> TensorValue:
    """Returns the ``[batch]`` row lengths of the verified sequence."""
    offsets = merged_offsets.cast(DType.int64)
    return ops.rebind(offsets[1:], [batch_dim]) - ops.rebind(
        offsets[:-1], [batch_dim]
    )


def accepted_lengths(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    *,
    batch_dim: DimLike = "batch_size",
) -> TensorValue:
    """Returns the ``[batch]`` rows of the verify window each request keeps.

    This is the prompt length on a prefill and ``1 + num_accepted`` on a
    decode.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K``, zero on a prefill.
        batch_dim: The batch's dimension: ``batch_size`` for the whole
            batch, a replica's own under data parallelism
            (:func:`replica_accept_inputs`).

    Returns:
        ``[batch]`` int64 accepted row counts.
    """
    return (
        _window_lengths(merged_offsets, batch_dim)
        - num_draft_tokens
        + num_accepted.cast(DType.int64).rebind([batch_dim])
    )


def ring_accepted_lengths(
    window_lengths: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
) -> TensorValue:
    """Returns :func:`accepted_lengths`, zero for a row the verify committed.

    A row longer than the verify window was committed by the verify itself
    at its last real token, so the rollback neither replays nor folds any
    of it.

    Args:
        window_lengths: ``[batch]`` row lengths of the verified sequence,
            from :func:`_window_lengths`.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K``, zero on a prefill.

    Returns:
        ``[batch]`` int64 row counts the rollback applies.
    """
    # ``window_lengths - 1 > K`` is ``window_lengths > K + 1`` without a
    # ``[1]``-sized add, which would launch its own kernel.
    return ops.where(
        window_lengths - 1 > num_draft_tokens,
        ops.constant(0, DType.int64, device=window_lengths.device).broadcast_to(
            ["batch_size"]
        ),
        window_lengths
        - num_draft_tokens
        + num_accepted.cast(DType.int64).rebind(["batch_size"]),
    )


def accepted_row_plan(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    total_rows: Dim,
    device: DeviceRef,
    *,
    plan_rows: Dim | None = None,
    batch_dim: DimLike = "batch_size",
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay's gather indices and ragged offsets.

    A row's accepted length is its merged length minus the drafts it did not
    accept, which is the prompt length during prefill (where there are no
    drafts) and ``1 + num_accepted`` during decode. One expression, no phase
    branch. Rows are captured row for row; a row longer than the verify
    window replays through :func:`window_row_plan`.

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
        total_rows: Row count of the captured tensors the replay gathers
            from.
        device: Device the plan is built on.
        plan_rows: Length of the index vector, at least the accepted total.
            Defaults to ``total_rows``.
        batch_dim: The batch's dimension, as :func:`accepted_lengths`.

    Returns:
        ``(row_indices, replay_offsets)``: which row of the verify's per-token
        tensors feeds each replay row, and the ragged offsets over them.
    """
    starts = ops.rebind(merged_offsets.cast(DType.int64)[:-1], [batch_dim])
    accepted = accepted_lengths(
        merged_offsets, num_accepted, num_draft_tokens, batch_dim=batch_dim
    )
    return _row_plan(
        starts,
        None,
        accepted,
        total_rows,
        plan_rows if plan_rows is not None else total_rows,
        device,
        batch_dim,
    )


def window_row_plan(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    total_rows: Dim,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay over the verify's per-row window copy.

    The verify copies each row of at most ``K + 1`` tokens into one
    ``K + 1`` slot, so request ``r``'s rows start at ``r * (K + 1)`` and
    a row longer than the window, which the verify committed itself,
    replays nothing (:func:`ring_accepted_lengths`).

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K`` on ``device``.
        total_rows: Row count of the window copy the replay gathers from.
        device: Device the plan is built on.

    Returns:
        ``(row_indices, replay_offsets)``: which row of the window copy
        feeds each replay row, and the ragged offsets over them.
    """
    accepted = ring_accepted_lengths(
        _window_lengths(merged_offsets),
        num_accepted,
        num_draft_tokens,
    )
    return _row_plan(
        None, num_draft_tokens, accepted, total_rows, total_rows, device
    )


def _row_plan(
    starts: TensorValue | None,
    stride: TensorValue | None,
    accepted: TensorValue,
    total_rows: Dim,
    out_rows: Dim,
    device: DeviceRef,
    batch_dim: DimLike = "batch_size",
) -> tuple[TensorValue, TensorValue]:
    """Returns the replay rows of ``accepted`` and their ragged offsets."""
    replay_offsets = ops.concat(
        [
            ops.constant(0, DType.int64, device=device).reshape([1]),
            ops.cumsum(accepted, axis=0),
        ],
        axis=0,
    )
    return (
        _ragged_source_rows(
            starts,
            stride,
            replay_offsets,
            total_rows,
            out_rows,
            device,
            batch_dim,
        ),
        replay_offsets,
    )


def _ragged_source_rows(
    starts: TensorValue | None,
    stride: TensorValue | None,
    offsets: TensorValue,
    total_rows: Dim,
    out_rows: Dim,
    device: DeviceRef,
    batch_dim: DimLike = "batch_size",
) -> TensorValue:
    """Returns the source row of each of ``out_rows`` ragged output rows.

    Request ``r`` owns output rows ``[offsets[r], offsets[r + 1])`` and reads
    them from source rows ``starts[r]`` on. Output rows past
    ``offsets[-1]`` repeat a clamped row; callers stop at that total.

    Args:
        starts: ``[batch_size]`` int64 first source row of each request.
        offsets: ``[batch_size + 1]`` int64 ragged offsets over the output.
        total_rows: Source row bound of the gather.
        out_rows: Output row count.
        device: Device the indices are built on.
        stride: When given, each request's rows are one ``stride + 1``
            slot apart, so request ``r`` starts at ``r * (stride + 1)``
            and ``starts`` must be ``None``.
        batch_dim: The batch's dimension, as :func:`accepted_lengths`.

    Returns:
        ``[out_rows]`` int64 source rows.
    """
    assert (starts is None) != (stride is None), (
        "exactly one of starts and stride must be given"
    )
    # An output row's request is the number of request ends at or before it.
    # Ends at or past ``out_rows`` land in an extra sink position.
    pos = ops.range(
        start=0,
        stop=Dim(out_rows) + 1,
        out_dim=Dim(out_rows) + 1,
        device=device,
        dtype=DType.int64,
    )
    out_pos = ops.rebind(pos[:-1], [out_rows])
    ends = ops.min(offsets[1:], pos[-1])
    ones = ops.broadcast_to(
        ops.constant(1, DType.int64, device=device), ends.shape
    )
    ends_per_pos = ops.scatter_add(pos * 0, ones, ends, axis=0)
    last_row = _shape_to_scalar(Dim(batch_dim), device) - 1
    row_of_pos = ops.min(
        ops.rebind(ops.cumsum(ends_per_pos, axis=0)[:-1], [out_rows]),
        last_row,
    )
    if stride is not None:
        # ``row_of_pos * K + row_of_pos`` is ``row_of_pos * (K + 1)`` with
        # the + 1 folded in, so no ``[1]``-sized add breaks the fusion.
        row_base = row_of_pos * stride + row_of_pos
    else:
        assert starts is not None
        row_base = ops.gather(starts, row_of_pos, axis=0)
    row_indices = row_base + (
        out_pos - ops.gather(offsets[:-1], row_of_pos, axis=0)
    )
    # Rows past the total are never read; clamp them so the gather itself
    # stays in bounds.
    return ops.min(row_indices, _shape_to_scalar(total_rows, device) - 1)


def replay_conv_pools(
    captures: Sequence[Sequence[ConvReplayInputs]],
    live_conv_pools: Sequence[BufferValue],
    conv_row_ids: Sequence[TensorValue],
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Writes each conv window at the accepted length.

    Re-runs the conv kernel over the accepted rows of the verify pass's copy
    of its input, carrying forward from the pre-verify window still in the
    pool.

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
            assert capture.qkv_window is not None, "a replay needs a verify"
            # No resume row: the live rows still hold the pre-verify window.
            gated_delta_conv1d_verify_fwd(
                qkv_input_ragged=ops.gather(capture.qkv_window, rows, axis=0),
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
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    signal_buffers: Sequence[BufferValue],
    verify_width: TensorValue,
) -> None:
    """Applies the verify's accepted ring records to the live recurrent pool.

    One launch per device covers every layer, and none at a verify width of
    zero.

    A row longer than the verify window was committed by the verify itself,
    so its records were never written and it folds nothing
    (:func:`ring_accepted_lengths`).

    Args:
        live_recurrent_pools: Per-device recurrent pool, still pre-verify.
        recurrent_row_ids: Per-device ``[num_layers, batch_size]`` state rows.
        ring_pools: Per-device ring pool.
        ring_row_ids: Per-device ``[num_layers, batch_size]`` ring rows.
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K`` on the batch's device.
        signal_buffers: Used only to place the fold counts on each device.
        verify_width: The verify's width operand.
    """
    n_devs = len(live_recurrent_pools)
    fold_accepted = ring_accepted_lengths(
        _window_lengths(merged_offsets),
        num_accepted,
        num_draft_tokens,
    ).cast(DType.uint32)
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


@dataclass(frozen=True)
class ReplicaAcceptInputs:
    """One data-parallel replica's share of a verify's accept.

    Under data parallelism each replica verifies only its own requests, so
    each rolls its own state back over its own rows. These are the batch-wide
    accept inputs cut to the replica's requests, rebased and placed on its
    device, ready for :func:`accepted_lengths` and :func:`accepted_row_plan`
    with ``batch_dim``.
    """

    merged_offsets: TensorValue
    """``[batch_dim + 1]`` offsets of the replica's verified rows, from 0."""
    num_accepted: TensorValue
    """``[batch_dim]`` accepted draft tokens of the replica's requests."""
    num_draft_tokens: TensorValue
    """Scalar ``K`` on :attr:`device`."""
    batch_dim: Dim
    """The replica's request count, named per replica."""
    device: DeviceRef
    """The replica's device, which every value lives on."""


def replica_accept_inputs(
    merged_offsets_per_dev: Sequence[TensorValue],
    num_accepted_per_dev: Sequence[TensorValue],
    num_draft_tokens: Dim,
    data_parallel_splits: TensorValue,
    prefix: str = "rollback",
) -> list[ReplicaAcceptInputs]:
    """Cuts a verify's batch-wide accept into each replica's.

    Device ``i`` is replica ``i``: one attention replica per device, as every
    data-parallel rollback runs today. The bounds are host values and each
    device cuts its own copies, so the cut costs no device sync and moves no
    data between devices: a peer copy would merge the devices' capture
    sequences under device graph capture.

    Args:
        merged_offsets_per_dev: The batch's ``[batch + 1]`` verify offsets on
            each device.
        num_accepted_per_dev: ``[batch]`` accepted draft tokens, in batch
            order, on each device (broadcast, as the driver broadcasts its
            per-device inputs).
        num_draft_tokens: The verify's draft width ``K``.
        data_parallel_splits: ``[replicas + 1]`` int64 first request of each
            replica, then the batch size, on the host.
        prefix: Prepended to the per-replica dimensions' names. Every rollback
            in one graph that cuts the batch needs its own.

    Raises:
        ValueError: If the offsets and counts don't both come one per device.
    """
    if len(merged_offsets_per_dev) != len(num_accepted_per_dev):
        raise ValueError(
            f"{len(merged_offsets_per_dev)} offset tensors and"
            f" {len(num_accepted_per_dev)} accepted counts: a replica rollback"
            " takes both on every device"
        )
    per_replica = []
    for replica, (offsets, num_accepted) in enumerate(
        zip(merged_offsets_per_dev, num_accepted_per_dev, strict=True)
    ):
        device = offsets.device
        first = data_parallel_splits[replica]
        last = data_parallel_splits[replica + 1]
        batch_dim = Dim(f"{prefix}_batch_{replica}")
        offsets = ops.slice_tensor(
            offsets.cast(DType.int64),
            [(slice(first, last + 1), f"{prefix}_offsets_{replica}")],
        )
        accepted = ops.slice_tensor(
            num_accepted, [(slice(first, last), batch_dim)]
        )
        per_replica.append(
            ReplicaAcceptInputs(
                merged_offsets=offsets - offsets[0],
                num_accepted=accepted,
                num_draft_tokens=_shape_to_scalar(num_draft_tokens, device),
                batch_dim=batch_dim,
                device=device,
            )
        )
    return per_replica
