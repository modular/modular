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
"""Snapshot-once and accepted-prefix replay for Nemotron's Mamba state.

Verification advances Jenga-owned scratch leaves. The live leaves retain the
pre-verify state until acceptance is known, then replay consumes only the
accepted rows of the verification's own conv and SSD inputs. No full state is
recorded per token and no target projection is repeated.

A step either verifies drafts or it does not, and the two commit differently:

- With drafts (``K > 0``), every request's rows are the anchor plus its ``K``
  drafts. The verify captures exactly those ``K + 1`` rows of each Mamba
  layer's replay inputs, and replay applies the accepted ``1 + num_accepted``
  of them to the live rows.
- Without drafts (``K == 0``, a prefill or any batch advancing draft-less),
  every row of every request is committed, so the scratch state after the
  verify is the live state after the step, and the commit is a row copy.
  Replay launches nothing: its offsets are all zero.

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
    "commit_state_rows",
    "replay_plan",
    "replay_state_pools",
    "snapshot_state_pools",
    "verify_window",
]


def snapshot_state_pools(
    live_pool: BufferValue,
    shadow_pool: BufferValue,
    live_rows: TensorValue,
    shadow_rows: TensorValue,
) -> None:
    """Copies active rows once into a Jenga-owned verification scratch leaf.

    Args:
        live_pool: The untouched pre-verification pool.
        shadow_pool: The scratch pool used by verification.
        live_rows: Live ``[num_layers, batch_size]`` row ids.
        shadow_rows: Independently assigned scratch row ids of the same shape.
    """
    if live_pool.dtype != shadow_pool.dtype:
        raise ValueError("The live and verification state dtypes must match")
    if live_pool.shape[1:] != shadow_pool.shape[1:]:
        raise ValueError(
            "The live and verification state row shapes must match"
        )
    ops.inplace_custom(
        "state_space_copy_rows",
        live_pool.device,
        [
            shadow_pool,
            live_pool,
            shadow_rows.cast(DType.uint32),
            live_rows.cast(DType.uint32),
        ],
    )


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
        live_conv_pool: Pre-verify convolution state.
        live_ssm_pool: Pre-verify SSM state.
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


def commit_state_rows(
    live_pool: BufferValue,
    scratch_pool: BufferValue,
    live_rows: TensorValue,
    scratch_rows: TensorValue,
    num_draft_tokens: Dim,
    batch_size: Dim,
) -> None:
    """Copies the verified scratch rows over the live rows when there are no drafts.

    Launches nothing when there are drafts, since the replay commits those
    steps.

    Args:
        live_pool: The pool the live rows address.
        scratch_pool: The pool the verify advanced.
        live_rows: Live ``[num_layers, batch_size]`` row ids.
        scratch_rows: Scratch ``[num_layers, batch_size]`` row ids.
        num_draft_tokens: ``K``.
        batch_size: The batch's request count.
    """
    cpu = DeviceRef.CPU()
    k = ops.shape_to_tensor([num_draft_tokens])[0]
    one = ops.constant(1, DType.int64, device=cpu)
    no_drafts = ops.max(
        one - ops.min(k, one), ops.constant(0, DType.int64, device=cpu)
    )
    count = ops.shape_to_tensor([batch_size])[0] * no_drafts
    out_dim = "state_commit_rows"
    live = ops.slice_tensor(
        live_rows, [slice(None), (slice(0, count), out_dim)]
    )
    scratch = ops.slice_tensor(
        scratch_rows, [slice(None), (slice(0, count), out_dim)]
    )
    ops.inplace_custom(
        "state_space_copy_rows",
        live_pool.device,
        [
            live_pool,
            scratch_pool,
            live.cast(DType.uint32),
            scratch.cast(DType.uint32),
        ],
    )
