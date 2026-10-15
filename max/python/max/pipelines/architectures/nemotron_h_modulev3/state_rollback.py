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
"""Accepted-prefix replay for Nemotron's Mamba state.

Verification runs on the live state pools without advancing them. The conv and
SSD kernels take an ``update_state`` flag per request, and a verify with drafts
clears it, so the pools keep their pre-verify state until acceptance is known.
Replay then consumes only the accepted rows of the verification's own conv and
SSD inputs and writes the result to the same pools. No full state is recorded
per token, none is copied, and no target projection is repeated.

A step either verifies drafts or it does not, and the two commit differently:

- With drafts (``K > 0``), every request's rows are the anchor plus its ``K``
  drafts. The verify captures exactly those ``K + 1`` rows of each Mamba
  layer's replay inputs, and replay applies the accepted ``1 + num_accepted``
  of them to the live rows.
- Without drafts (``K == 0``, a prefill or any batch advancing draft-less),
  every row of every request is committed, so the verify sets ``update_state``
  and advances the live rows itself. Replay launches nothing: its offsets are
  all zero.

Both ops are in every graph. Each takes its extent from ``K``, and the one that
does not apply has no rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

from ..unified_mtp_qwen3_5.state_rollback import accepted_row_plan
from .layers.mamba2 import MambaReplayInputs

__all__ = [
    "replay_plan",
    "replay_state_pools",
    "verify_update_state",
    "verify_window",
]


def verify_update_state(
    num_draft_tokens: Dim, batch_size: Dim, device: DeviceRef
) -> TensorValue:
    """Returns the verify's ``update_state`` flags: set only without drafts.

    With drafts the verify must leave the live state alone for the replay to
    start from. Without them nothing can be rejected, so the verify advances
    the live rows itself and nothing replays.

    Args:
        num_draft_tokens: ``K``.
        batch_size: The batch's request count.
        device: The device the flags are built on.

    Returns:
        The ``[batch_size]`` bool flags, all ``K == 0``.
    """
    no_drafts = _shape_to_scalar(num_draft_tokens, device) == 0
    return ops.broadcast_to(no_drafts, [batch_size])


def replay_state_pools(
    captures: Sequence[MambaReplayInputs[TensorValue]],
    live_conv_pool: BufferValue,
    live_ssm_pool: BufferValue,
    conv_rows: TensorValue,
    ssm_rows: TensorValue,
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    has_initial_state: TensorValue,
) -> None:
    """Commits the accepted prefix of one device's captured Mamba layers.

    Uses the same conv and SSD kernels as verification. The SSD inputs were
    produced causally, so their accepted rows already include exactly the
    accepted conv history. The conv replay updates its persistent window;
    recomputing its outputs is unnecessary for the SSD replay.

    Args:
        captures: One capture per Mamba layer, in row-table order.
        live_conv_pool: Convolution state, as the verify left it.
        live_ssm_pool: SSM state, as the verify left it.
        conv_rows: Live ``[num_layers, batch_size]`` convolution row ids.
        ssm_rows: Live ``[num_layers, batch_size]`` SSM row ids.
        row_indices: Gather plan returned by :func:`accepted_row_plan`.
        replay_offsets: Ragged offsets of the accepted rows.
        has_initial_state: The verification's per-request initialization flags.
    """
    offsets = replay_offsets.cast(DType.int32)
    for layer, capture in enumerate(captures):
        causal_conv1d_varlen_fwd(
            x=ops.gather(TensorValue(capture.conv_input), row_indices, axis=0),
            weight=TensorValue(capture.conv_weight),
            bias=TensorValue(capture.conv_bias),
            conv_states=live_conv_pool,
            query_start_loc=offsets,
            cache_indices=conv_rows[layer].cast(DType.uint32),
            has_initial_state=has_initial_state,
            activation="silu",
            channels_last=True,
        )
        mamba2_ssd_chunk_scan_varlen_fwd_inplace(
            x=ops.gather(TensorValue(capture.x), row_indices, axis=0),
            dt=ops.gather(TensorValue(capture.dt), row_indices, axis=0),
            A=TensorValue(capture.A),
            B=ops.gather(TensorValue(capture.B), row_indices, axis=0),
            C=ops.gather(TensorValue(capture.C), row_indices, axis=0),
            D=TensorValue(capture.D),
            dt_bias=TensorValue(capture.dt_bias),
            ssm_pool=live_ssm_pool,
            query_start_loc=offsets,
            has_initial_state=has_initial_state,
            cache_indices=ssm_rows[layer].cast(DType.uint32),
        )


def verify_window(
    merged_offsets: TensorValue,
    num_draft_tokens: Dim,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Returns the token rows a verify captures, and their ragged offsets.

    Each request's window is its last ``K + 1`` rows. With drafts that is the
    whole request, the anchor and its ``K`` drafts. Without, it is the last
    row, which keeps the capture to one row per request.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequences.
        num_draft_tokens: ``K``.
        device: The device of the offsets.

    Returns:
        ``(rows, offsets)``: the ``[batch * (K + 1)]`` int64 token rows, and
        the ``[batch + 1]`` int64 offsets of each request's window in them.
    """
    width = Dim(num_draft_tokens) + 1
    width_value = _shape_to_scalar(width, device)
    ends = merged_offsets.cast(DType.int64)[1:]
    steps = ops.range(
        0, width, 1, out_dim=width, dtype=DType.int64, device=device
    )
    # Named, since the rows' extent crosses the Mamba subgraph boundary,
    # whose signature cannot hold an expression of the batch and ``K``.
    rows = ops.rebind(
        (
            ops.unsqueeze(ends, -1) - width_value + ops.unsqueeze(steps, 0)
        ).reshape((-1,)),
        ["verify_window_rows"],
    )
    offsets = (
        ops.range(
            0,
            merged_offsets.shape[0],
            1,
            out_dim=merged_offsets.shape[0],
            dtype=DType.int64,
            device=device,
        )
        * width_value
    )
    return rows, offsets


def replay_plan(
    window_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: Dim,
    total_rows: Dim,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay over the captured windows.

    A request replays its ``1 + num_accepted`` accepted rows when there are
    drafts, and none when there are not (see the module docstring).

    Args:
        window_offsets: The ``[batch + 1]`` offsets from :func:`verify_window`.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: ``K``.
        total_rows: The row count of the captured windows.
        device: The device the plan is built on.

    Returns:
        ``(row_indices, replay_offsets)`` as :func:`accepted_row_plan` does.
    """
    k = _shape_to_scalar(num_draft_tokens, device)
    one = ops.constant(1, DType.int64, device=device)
    # A window is ``K + 1`` rows and a request keeps ``rows - n + accepted``,
    # so ``n = K`` keeps ``1 + accepted`` and ``n = 1`` keeps none of the
    # window's single row when ``K`` is zero and nothing was accepted.
    drafts = k + ops.max(
        one - ops.min(k, one), ops.constant(0, DType.int64, device=device)
    )
    return accepted_row_plan(
        window_offsets, num_accepted, drafts, total_rows, device
    )
