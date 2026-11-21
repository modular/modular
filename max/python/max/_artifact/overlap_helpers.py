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
"""Builds the overlap helper graphs.

The helpers feed each decode step's tokens on the device. An artifact stores
them as roles of its ``model`` component (:class:`HelperRole`), hashed like
every other graph; ``gen_overlap_helpers`` writes the same graphs beside a
registry MEF under the file names below.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any

from max import tree
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops import scatter_nd_skip_oob_indices
from max.nn.kv_cache import (
    CacheLeafParamInterface,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import TensorLayout
from max.tensor.sharding.types import as_layout

from .value_types import ArtifactError

# Token buffers and indices are int64 to match the engine's token dtype
# and the scheduler's slot-index type.
_TOK = DType.int64

# The names a registry engine looks for the helpers by beside a model MEF,
# where each carries the MEF's stem. The first five hold no geometry, so the
# shared name, as `gen_overlap_helpers` writes them, serves any export that has
# no file of its own.
REALIZE_MEF = "overlap-realize-future-tokens.mef"
REALIZE_INPLACE_MEF = "overlap-realize-future-tokens-inplace.mef"
REALIZE_DRAFTS_MEF = "overlap-realize-draft-tokens.mef"
UPDATE_CACHE_LENGTHS_MEF = "overlap-update-cache-lengths.mef"
REALIZE_SPEC_TOKENS_MEF = "overlap-realize-spec-next-tokens.mef"
# The spec feeds are built for one export's devices and KV groups, so they have
# no shared name.
SPEC_REALIZE_INPLACE_MEF = "overlap-spec-realize-inplace.mef"
SPEC_REALIZE_SHARED_INPLACE_MEF = "overlap-spec-realize-shared-inplace.mef"
SPEC_REALIZE_MIXED_INPLACE_MEF = "overlap-spec-realize-mixed-inplace.mef"


def _build_realize_future_tokens(device_ref: DeviceRef) -> Graph:
    """Builds the graph that realizes the next decode batch's token input.

    It reads the previous batch's sampled tokens (the async-overlap
    device-feedback primitive).

    This is the plain-decode simplification of MAX's
    ``build_realize_future_token_graph``
    (``pipelines/lib/pipeline_variants/overlap_text_generation.py``). Instead of
    a persistent per-slot accumulator buffer (which a functional scatter cannot
    update in-place under dispatch-before-commit), the previous batch's sampler
    output is threaded device-side and scattered directly into the current
    batch's token placeholders:

        realized[curr] = scatter(curr_tokens, prev_generated_tokens,
                                 indices=prev_to_curr_map)

    ``prev_to_curr_map[i]`` is the position in the current batch's flat token
    input where previous-batch row ``i``'s request starts (a token offset, which
    a widened row ahead of it moves past the row index), or an out-of-range
    sentinel when that request departed — those rows are skipped, leaving the
    current placeholder (e.g. a freshly-prefilled request's first decode token)
    untouched. Shapes are symbolic so one MEF serves every batch size.
    """
    curr_tokens_type = TensorType(_TOK, ["curr_batch"], device_ref)
    prev_tokens_type = TensorType(_TOK, ["prev_batch"], device_ref)
    prev_to_curr_type = TensorType(_TOK, ["prev_batch"], device_ref)
    with Graph(
        "realize_future_tokens",
        input_types=[curr_tokens_type, prev_tokens_type, prev_to_curr_type],
    ) as graph:
        curr_tokens = graph.inputs[0].tensor
        prev_generated_tokens = graph.inputs[1].tensor
        prev_to_curr_map = graph.inputs[2].tensor
        # scatter_nd indexes axis 0, so indices are shape [prev_batch, 1].
        realized = scatter_nd_skip_oob_indices(
            input=curr_tokens,
            updates=prev_generated_tokens,
            indices=ops.unsqueeze(prev_to_curr_map, axis=-1),
        )
        graph.output(realized)
    return graph


def _build_realize_future_tokens_inplace(device_ref: DeviceRef) -> Graph:
    """Builds the in-place form of :func:`_build_realize_future_tokens`.

    It runs overlap together with CUDA graph capture.

    A captured forward is replayed against the buffer addresses it was captured
    with, so the functional helper cannot feed it: that one returns a fresh
    tensor each step, which the launch path threads into the forward's token
    slot by name, and a replay has no way to rewire a slot. This form instead
    scatters into the token buffer the engine already wired, leaving the address
    the capture baked untouched.

    ``dest_tokens`` is both the base and the destination. The host has already
    copied this step's placeholder tokens into it (correct for freshly-prefilled
    rows), and the scatter overwrites exactly the rows continuing from the
    previous step, so there is no separate ``curr_tokens`` input:

        dest_tokens[prev_to_curr_map[i]] = prev_generated_tokens[i]

    Out-of-range map entries are skipped, as in the functional helper.
    """
    dest_type = BufferType(_TOK, ["curr_batch"], device_ref)
    prev_tokens_type = TensorType(_TOK, ["prev_batch"], device_ref)
    prev_to_curr_type = TensorType(_TOK, ["prev_batch"], device_ref)
    with Graph(
        "realize_future_tokens_inplace",
        input_types=[dest_type, prev_tokens_type, prev_to_curr_type],
    ) as graph:
        dest_tokens = graph.inputs[0].buffer
        prev_generated_tokens = graph.inputs[1].tensor
        prev_to_curr_map = graph.inputs[2].tensor
        realized = scatter_nd_skip_oob_indices(
            input=ops.buffer_load(dest_tokens),
            updates=prev_generated_tokens,
            indices=ops.unsqueeze(prev_to_curr_map, axis=-1),
        )
        ops.buffer_store(dest_tokens, realized)
        # No tensor outputs; `output()` packs the device chain, which is what
        # orders the store against the replay that reads the buffer.
        graph.output()
    return graph


def _build_realize_draft_tokens(device_ref: DeviceRef) -> Graph:
    """Builds the graph that realizes the next spec step's draft tokens.

    It reads the previous spec step's device-resident ``next_draft_tokens``
    (the spec-overlap draft feed-forward).

    A spec step's fused graph emits ``next_draft_tokens`` (``output2``, shape
    ``[batch, k]``) on device. To launch step N+1 before retiring N, those
    drafts must feed N+1's ``draft_tokens`` input WITHOUT a host round-trip.
    This scatters each previous row's whole ``k``-wide draft slice into its
    current-batch row (same KV slot), mirroring ``realize_future_tokens`` but
    for the 2-D ``[batch, k]`` draft tensor — ``scatter_nd`` with rank-1 index
    depth replaces the entire ``[k]`` row slice:

        realized[curr] = scatter(curr_draft, prev_next_draft,
                                 indices=prev_to_curr_map)

    Departed requests (OOB ``prev_to_curr_map``) are skipped, leaving a
    freshly-prefilled request's host-seeded initial drafts untouched. Shapes
    are symbolic so one MEF serves any batch / draft width.
    """
    # The fused spec graph emits `next_draft_tokens` (output2) as int64 on
    # device (verified against the MEF: out0/1/2 are all Int64), matching the
    # forward's int64 `draft_tokens` input slot. The chained device feed copies
    # the bytes verbatim, so `prev_next_draft` is int64 — no cast.
    curr_draft_type = TensorType(_TOK, ["curr_batch", "k"], device_ref)
    prev_draft_type = TensorType(_TOK, ["prev_batch", "k"], device_ref)
    prev_to_curr_type = TensorType(_TOK, ["prev_batch"], device_ref)
    with Graph(
        "realize_draft_tokens",
        input_types=[curr_draft_type, prev_draft_type, prev_to_curr_type],
    ) as graph:
        curr_draft = graph.inputs[0].tensor
        prev_next_draft = graph.inputs[1].tensor
        prev_to_curr_map = graph.inputs[2].tensor
        realized = scatter_nd_skip_oob_indices(
            input=curr_draft,
            updates=prev_next_draft,
            indices=ops.unsqueeze(prev_to_curr_map, axis=-1),
        )
        graph.output(realized)
    return graph


def _build_realize_spec_next_tokens(device_ref: DeviceRef) -> Graph:
    """Builds the graph that realizes the next spec step's decode-token input.

    It reads the previous spec step's device-resident ``next_tokens`` (the
    spec-overlap bonus-token feed-forward, the 1-D analogue of
    ``realize_draft_tokens``).

    A spec step's fused graph emits ``next_tokens`` (``output1``, the committed
    bonus token per row) as int64 on device (verified against the MEF), matching
    the forward's int64 ``tokens`` input. To launch step N+1 before retiring N,
    that token feeds N+1's ``tokens`` input WITHOUT a host round-trip:

        realized[curr] = scatter(curr_tokens, prev_next_tokens,
                                 indices=prev_to_curr_map)
    """
    curr_tokens_type = TensorType(_TOK, ["curr_batch"], device_ref)
    prev_tokens_type = TensorType(_TOK, ["prev_batch"], device_ref)
    prev_to_curr_type = TensorType(_TOK, ["prev_batch"], device_ref)
    with Graph(
        "realize_spec_next_tokens",
        input_types=[curr_tokens_type, prev_tokens_type, prev_to_curr_type],
    ) as graph:
        curr_tokens = graph.inputs[0].tensor
        prev_next_tokens = graph.inputs[1].tensor
        prev_to_curr_map = graph.inputs[2].tensor
        realized = scatter_nd_skip_oob_indices(
            input=curr_tokens,
            updates=prev_next_tokens,
            indices=ops.unsqueeze(prev_to_curr_map, axis=-1),
        )
        graph.output(realized)
    return graph


def spec_realize_inplace_graph_name(
    tp: int, kv_groups: int, positions: bool = False
) -> str:
    """Name of the fused spec feed graph for a ``(tp, kv_groups)`` geometry.

    The engine rebuilds this name from the geometry it is actually serving and
    requires the loaded MEF to declare it, which is the only way it can reject a
    helper whose input count matches by coincidence.

    The ``_narrow`` suffix marks the ABI whose previous drafts take their own
    width (see :func:`_feed_drafts`); the engine accepts the unsuffixed
    build only when no step verifies fewer drafts than it proposes. The
    ``_positions`` build also advances the verify step's 3-axis positions.
    """
    suffix = "_positions" if positions else ""
    return f"spec_realize_inplace_tp{tp}_kv{kv_groups}{suffix}_narrow"


def spec_realize_mixed_inplace_graph_name(
    dp: int,
    kv_groups: int,
    next_draft_dtype: DType = DType.int32,
    positions: bool = False,
) -> str:
    """Name of the mixed-step fused spec feed for a ``(dp, kv_groups)`` geometry.

    Same destination ABI as :func:`spec_realize_inplace_dp_graph_name`, plus the
    per-previous-row offsets the bonus token lands at. A registry engine
    requires the name, so it can never wire a decode feed built for one
    geometry in the mixed step's place.
    """
    suffix = "_i32drafts" if next_draft_dtype == DType.int32 else ""
    if positions:
        suffix += "_positions"
    return f"spec_realize_mixed_inplace_dp{dp}_kv{kv_groups}{suffix}"


def spec_realize_inplace_dp_graph_name(
    dp: int,
    kv_groups: int,
    next_draft_dtype: DType = DType.int64,
    positions: bool = False,
) -> str:
    """Returns the name of the data-parallel fused spec feed.

    The name is for a ``(dp, kv_groups)`` geometry with the served model's
    next-draft dtype.

    The DP variant keeps the block adapter's destination ABI — tokens, drafts,
    then each group's cache lengths per device, group-major — so the arity
    cannot tell it from the TP build at the same device count, and the draft
    dtype cannot be told from the input list at all: an int64-draft helper
    handed to a model whose fused graph emits int32 drafts declares the same
    inputs. The name is the only thing that can tell the builds apart, so a
    non-default draft dtype is spelled into it; the engine requires the
    loaded MEF to declare the one its geometry spells. The ``_narrow`` suffix
    is the TP build's, and ``_positions`` marks the build that also advances
    3-axis positions.
    """
    suffix = "" if next_draft_dtype == DType.int64 else "_i32drafts"
    if positions:
        suffix += "_positions"
    return f"spec_realize_inplace_dp{dp}_kv{kv_groups}{suffix}_narrow"


def _feed_drafts(
    dest_draft: BufferValue, prev_next_draft: TensorValue, indices: TensorValue
) -> None:
    """Scatters each previous row's leading drafts into its current row.

    A step proposes K drafts but may verify fewer
    (``--num-speculative-tokens-per-batch-size``), so the previous step's
    ``[prev_batch, prev_k]`` proposals are trimmed to the current
    ``[curr_batch, k]`` destination, ``k <= prev_k``, before the scatter.
    """
    curr_draft = ops.buffer_load(dest_draft)
    ops.buffer_store(
        dest_draft,
        scatter_nd_skip_oob_indices(
            input=curr_draft,
            updates=prev_next_draft[:, : curr_draft.shape[1]],
            indices=indices,
        ),
    )


def _advance_positions(
    dest_positions: BufferValue, increments: TensorValue
) -> None:
    """Advances each row's window of 3-axis positions by its increment.

    A verify step's rows each hold a window of ``k + 1`` positions::

        dest_positions[:, curr, :] += increments[curr]
    """
    current = ops.buffer_load(dest_positions)
    rows = current.shape[1]
    window = rows // Dim("curr_batch")
    windows = current.rebind([3, window * Dim("curr_batch")]).reshape(
        [3, "curr_batch", window]
    )
    shifted = windows + ops.reshape(increments, [1, -1, 1])
    ops.buffer_store(
        dest_positions,
        shifted.reshape([3, window * Dim("curr_batch")]).rebind([3, rows]),
    )


def _advance_packed_positions(
    dest_positions: BufferValue,
    increments: TensorValue,
    token_offsets: TensorValue,
    width: Dim,
) -> None:
    """Advances each fed row's window of 3-axis positions in a mixed step.

    A mixed step packs a prompt row's whole chunk beside the verify rows'
    ``k + 1`` windows, so the windows are not equal and a row index does not
    say where one starts. Each previous row's verify window starts at the
    token offset its bonus token lands at and spans ``width`` positions::

        dest_positions[:, offsets[p] : offsets[p] + width] += increments[p]

    A row with no current match carries the skip sentinel, the token count,
    so its whole window is out of range and skipped; a prompt row is never
    fed and keeps the positions the host staged.
    """
    current = ops.buffer_load(dest_positions)
    tokens = current.shape[1]
    device = token_offsets.device
    steps = ops.range(0, width, 1, out_dim=width, dtype=_TOK, device=device)
    window_offsets = ops.unsqueeze(token_offsets, -1) + ops.unsqueeze(steps, 0)
    window_increments = ops.broadcast_to(
        ops.unsqueeze(increments, -1), ["prev_batch", width]
    )
    zeros = ops.broadcast_to(
        ops.constant(0, dtype=_TOK, device=device), [tokens]
    )
    per_token = scatter_nd_skip_oob_indices(
        input=zeros,
        updates=window_increments.reshape([-1]),
        indices=window_offsets.reshape([-1, 1]),
    )
    ops.buffer_store(dest_positions, current + ops.unsqueeze(per_token, 0))


def _build_spec_realize_inplace(
    device_refs: list[DeviceRef],
    kv_groups: int,
    positions: bool = False,
    name: str | None = None,
) -> Graph:
    """Graph: the whole spec-overlap device feed, in place, in one dispatch.

    Under graph capture the fused spec forward is replayed against the buffer
    addresses it was captured with, so the three functional spec helpers cannot
    feed it: each returns a fresh tensor that the launch path would have to wire
    into an input slot by name, and a replay cannot rewire a slot. This form
    scatters into the captured input map's own buffers instead, which is the
    address the capture baked.

    It also fuses what the eager path dispatches separately. The eager chain runs
    ``update_cache_lengths`` once per KV group plus ``realize_spec_tokens`` and
    ``realize_drafts``; under capture that is four model launches whose host cost
    lands on the very step budget overlap is trying to reclaim. All three feeds
    read the same previous step's outputs and the same ``prev_to_curr_map``, so
    one graph does the lot::

        dest_tokens[curr]      = prev_next_tokens[prev]
        dest_draft[curr]       = prev_next_draft[prev]
        dest_cache[g][d][curr] += prev_num_accepted[prev] + 1

    Tensor-parallel serving replicates ``cache_lengths`` per device while the
    token and draft inputs stay device-0 only, so the destinations are
    ``2 + kv_groups * len(device_refs)`` buffers and the increment computed on
    device 0 is transferred to each other shard. That is what makes the feed
    device-major rather than dev0-only, which is what blocked spec overlap under
    tensor parallelism.

    With ``positions`` the verify step also takes ``[3, curr_batch * (k + 1)]``
    position ids (M-RoPE), which the host computes from the same lengths a
    step behind as the cache lengths. Each row's window advances by the same
    increment::

        dest_positions[:, curr, :] += prev_num_accepted[prev] + 1

    Input order is positional and the engine wires it by index:
    ``dest_tokens``, ``dest_draft``, ``dest_cache`` (group-major, device-minor),
    ``dest_positions`` when built with them, then ``prev_num_accepted``,
    ``prev_next_tokens``, ``prev_next_draft``, ``prev_to_curr_map``.

    The graph is named for its geometry (see
    :func:`spec_realize_inplace_graph_name`) because TP and the KV-group count
    enter the input list only as a product: ``--tp 1 --kv-groups 2`` and
    ``--tp 2 --kv-groups 1`` declare the same eight inputs but place and group
    the cache destinations differently, so arity alone cannot tell the engine it
    was handed the wrong helper. ``name`` replaces that name for an engine
    that requires another.
    """
    dev0 = device_refs[0]
    dest_types: list[BufferType] = [
        BufferType(_TOK, ["curr_batch"], dev0),
        BufferType(_TOK, ["curr_batch", "k"], dev0),
    ]
    for _group in range(kv_groups):
        for dev in device_refs:
            dest_types.append(BufferType(DType.uint32, ["curr_batch"], dev))
    if positions:
        dest_types.append(BufferType(_TOK, [3, "merged_seq_len"], dev0))
    prev_types = [
        TensorType(_TOK, ["prev_batch"], dev0),  # num_accepted
        TensorType(_TOK, ["prev_batch"], dev0),  # next_tokens
        TensorType(_TOK, ["prev_batch", "prev_k"], dev0),  # next_draft_tokens
        TensorType(_TOK, ["prev_batch"], dev0),  # prev_to_curr_map
    ]
    with Graph(
        name
        or spec_realize_inplace_graph_name(
            len(device_refs), kv_groups, positions
        ),
        input_types=[*dest_types, *prev_types],
    ) as graph:
        n_dest = len(dest_types)
        n_cache = kv_groups * len(device_refs)
        dest_tokens = graph.inputs[0].buffer
        dest_draft = graph.inputs[1].buffer
        dest_cache = [graph.inputs[2 + i].buffer for i in range(n_cache)]
        prev_num_accepted = graph.inputs[n_dest].tensor
        prev_next_tokens = graph.inputs[n_dest + 1].tensor
        prev_next_draft = graph.inputs[n_dest + 2].tensor
        prev_to_curr_map = graph.inputs[n_dest + 3].tensor

        # scatter_nd indexes axis 0, so indices are shape [prev_batch, 1]. The
        # same index tensor drives all three feeds: they are the same row remap.
        indices = ops.unsqueeze(prev_to_curr_map, axis=-1)

        ops.buffer_store(
            dest_tokens,
            scatter_nd_skip_oob_indices(
                input=ops.buffer_load(dest_tokens),
                updates=prev_next_tokens,
                indices=indices,
            ),
        )
        _feed_drafts(dest_draft, prev_next_draft, indices)

        # A spec step commits `num_accepted + 1` tokens per row; rows with no
        # previous match (departed, OOB index) are skipped and stay at 0.
        one = ops.broadcast_to(
            ops.constant(1, dtype=_TOK, device=dev0), ["prev_batch"]
        )
        zeros = ops.broadcast_to(
            ops.constant(0, dtype=_TOK, device=dev0), ["curr_batch"]
        )
        batch_increments = scatter_nd_skip_oob_indices(
            input=zeros, updates=prev_num_accepted + one, indices=indices
        )
        per_device_increments = [
            batch_increments
            if dev == dev0
            else ops.transfer_to(batch_increments, dev)
            for dev in device_refs
        ]
        for group in range(kv_groups):
            for device_index in range(len(device_refs)):
                dest = dest_cache[group * len(device_refs) + device_index]
                bumped = (
                    ops.buffer_load(dest).cast(_TOK)
                    + per_device_increments[device_index]
                ).cast(DType.uint32)
                ops.buffer_store(dest, bumped)
        if positions:
            _advance_positions(
                graph.inputs[2 + n_cache].buffer, batch_increments
            )

        # No tensor outputs; `output()` packs the device chain, which is what
        # orders the stores against the replay that reads the buffers.
        graph.output()
    return graph


def _build_spec_realize_inplace_dp(
    device_refs: list[DeviceRef],
    data_parallel_degree: int,
    kv_groups: int,
    next_draft_dtype: DType = _TOK,
    packed_token_offsets: bool = False,
    positions: bool = False,
    name: str | None = None,
) -> Graph:
    """Graph: the spec-overlap device feed, sliced by data-parallel replica.

    A data-parallel export serves each attention replica's rows on that
    replica's devices, so the ``cache_lengths``
    destinations hold one replica's rows, not the whole batch the TP build's
    do. The increment is still one global scatter — the row map and the
    previous step's outputs are global — but each (group, device)
    destination is bumped with its own replica's rows-portion of it, sliced
    by the CPU replica bounds the engine wires per step. All groups of one
    device share the same rows, so they share the same slice::

        dest_tokens[curr]      = prev_next_tokens[prev]
        dest_draft[curr]       = prev_next_draft[prev]
        dest_cache[g][d][r]   += increments[splits[r] .. splits[r + 1]]

    The destination order is the TP build's unchanged — tokens, drafts, then
    each group's cache lengths, group-major and device-minor — so the
    engine's wiring table is shared; only the increment's slicing is new.
    The graph is named for ``(dp, kv_groups)`` (see
    :func:`spec_realize_inplace_dp_graph_name`) because the arity alone
    cannot tell a DP build from a TP one at the same device count, nor one
    replica split from another.

    At one replica the slice is the whole batch, so the build serves a
    tensor-parallel export with the same inputs as a data-parallel one, and
    at one KV group it serves an engine that keeps one cache-length buffer
    per device for all its caches.

    ``next_draft_dtype`` is the dtype the served model's fused graph emits
    its ``next_draft_tokens`` at. The destination stays the engine's int64
    draft input either way; a narrower source is widened at the scatter.

    ``packed_token_offsets`` builds the mixed step's feed: the same
    destinations and the same row-indexed drafts and cache bump, plus one
    input holding, per previous row, the token offset its bonus token lands
    at. A mixed step packs a prompt row's whole chunk into the token input, so
    a row index is not a token offset there.

    With ``positions`` the step also takes ``[3, merged_seq_len]`` position
    ids (M-RoPE) after the cache destinations, each row's window advanced by
    its increment, as in :func:`_build_spec_realize_inplace`. The mixed build
    finds each verify row's window at its token offset
    (:func:`_advance_packed_positions`), since prompt chunks make the windows
    unequal. ``name``
    replaces the geometry's name for an engine that requires another.

    Input order: ``dest_tokens``, ``dest_draft``, ``dest_cache`` (group-major,
    device-minor), ``dest_positions`` when built with them, then
    ``prev_num_accepted``, ``prev_next_tokens``, ``prev_next_draft``,
    ``prev_to_curr_map``, the token offsets for the mixed build, and the
    ``[dp + 1]`` replica row bounds on the host.
    """
    n_devices = len(device_refs)
    if data_parallel_degree < 1 or n_devices % data_parallel_degree:
        raise ValueError(
            f"DP spec feedback needs a replica count dividing {n_devices}, "
            f"got {data_parallel_degree}"
        )
    per_replica = n_devices // data_parallel_degree
    dev0 = device_refs[0]
    dest_types: list[BufferType] = [
        # A mixed step packs prompt chunks, so its tokens outnumber its rows.
        BufferType(
            _TOK,
            ["curr_tokens" if packed_token_offsets else "curr_batch"],
            dev0,
        ),
        BufferType(_TOK, ["curr_batch", "k"], dev0),
    ]
    for _group in range(kv_groups):
        for i, dev in enumerate(device_refs):
            dest_types.append(
                BufferType(
                    DType.uint32, [f"replica_{i // per_replica}_batch"], dev
                )
            )
    if positions:
        dest_types.append(BufferType(_TOK, [3, "merged_seq_len"], dev0))
    prev_types = [
        TensorType(_TOK, ["prev_batch"], dev0),  # num_accepted
        TensorType(_TOK, ["prev_batch"], dev0),  # next_tokens
        TensorType(next_draft_dtype, ["prev_batch", "prev_k"], dev0),  # drafts
        TensorType(_TOK, ["prev_batch"], dev0),  # prev_to_curr_map
    ]
    if packed_token_offsets:
        prev_types.append(
            TensorType(_TOK, ["prev_batch"], dev0)
        )  # token offsets
    # The replica row bounds, one per edge: host memory the engine wires per
    # step, as the fused graph's own `data_parallel_splits` input.
    prev_types.append(
        TensorType(_TOK, [data_parallel_degree + 1], DeviceRef.CPU())
    )
    if name is None:
        name = (
            spec_realize_mixed_inplace_graph_name(
                data_parallel_degree, kv_groups, next_draft_dtype, positions
            )
            if packed_token_offsets
            else spec_realize_inplace_dp_graph_name(
                data_parallel_degree, kv_groups, next_draft_dtype, positions
            )
        )
    with Graph(name, input_types=[*dest_types, *prev_types]) as graph:
        n_dest = len(dest_types)
        n_cache = kv_groups * n_devices
        dest_tokens = graph.inputs[0].buffer
        dest_draft = graph.inputs[1].buffer
        dest_cache = [graph.inputs[2 + i].buffer for i in range(n_cache)]
        prev_num_accepted = graph.inputs[n_dest].tensor
        prev_next_tokens = graph.inputs[n_dest + 1].tensor
        prev_next_draft = graph.inputs[n_dest + 2].tensor
        prev_to_curr_map = graph.inputs[n_dest + 3].tensor
        splits = graph.inputs[n_dest + 4 + int(packed_token_offsets)].tensor

        indices = ops.unsqueeze(prev_to_curr_map, axis=-1)
        # The drafts and the cache bump stay row-indexed; only the bonus token
        # goes where the batch's packed token input holds its row.
        token_indices = (
            ops.unsqueeze(graph.inputs[n_dest + 4].tensor, axis=-1)
            if packed_token_offsets
            else indices
        )
        ops.buffer_store(
            dest_tokens,
            scatter_nd_skip_oob_indices(
                input=ops.buffer_load(dest_tokens),
                updates=prev_next_tokens,
                indices=token_indices,
            ),
        )
        # A narrower draft source widens here; the int64 default casts
        # nothing.
        draft_updates = (
            prev_next_draft
            if next_draft_dtype == _TOK
            else prev_next_draft.cast(_TOK)
        )
        _feed_drafts(dest_draft, draft_updates, indices)
        one = ops.broadcast_to(
            ops.constant(1, dtype=_TOK, device=dev0), ["prev_batch"]
        )
        zeros = ops.broadcast_to(
            ops.constant(0, dtype=_TOK, device=dev0), ["curr_batch"]
        )
        # One global increment per current row, scattered from the previous
        # step's accepted counts; each destination takes its replica's slice.
        increments = scatter_nd_skip_oob_indices(
            input=zeros, updates=prev_num_accepted + one, indices=indices
        )
        for group in range(kv_groups):
            for i, dev in enumerate(device_refs):
                replica = i // per_replica
                local = ops.slice_tensor(
                    increments,
                    [
                        (
                            slice(splits[replica], splits[replica + 1]),
                            f"replica_{replica}_batch",
                        )
                    ],
                )
                if dev != dev0:
                    local = ops.transfer_to(local, dev)
                dest = dest_cache[group * n_devices + i]
                bumped = (ops.buffer_load(dest).cast(_TOK) + local).cast(
                    DType.uint32
                )
                ops.buffer_store(dest, bumped)
        if positions and packed_token_offsets:
            # The verify rows' windows are `k + 1` wide, beside prompt
            # chunks of any length.
            _advance_packed_positions(
                graph.inputs[2 + n_cache].buffer,
                prev_num_accepted + one,
                graph.inputs[n_dest + 4].tensor,
                dest_draft.shape[1] + 1,
            )
        elif positions:
            _advance_positions(graph.inputs[2 + n_cache].buffer, increments)

        # No tensor outputs; `output()` packs the device chain, which is what
        # orders the stores against the replay that reads the buffers.
        graph.output()
    return graph


def _build_update_cache_lengths(device_ref: DeviceRef) -> Graph:
    """Builds the device-resident variable-accept KV cache-length update.

    The graph serves spec overlap.

    A speculative step commits ``num_accepted + 1`` tokens per row, where
    ``num_accepted`` (0..k) is produced ON DEVICE by the previous spec step
    (``output0``). To launch step N+1 before retiring N (the overlap
    dispatch-before-commit), the next step's ``cache_lengths`` must be advanced
    by ``num_accepted(N) + 1`` *without* the host reading ``num_accepted`` — the
    worst-case ``+(k+1)`` cannot be used because the spec forward would then
    attend over invalid KV positions. This graph applies that increment on the
    compute stream, mirroring the cache-length fold in MAX's
    ``build_realize_future_token_graph``
    (``pipeline_variants/overlap_text_generation.py``):

        increment[prev]            = prev_num_accepted + 1
        batch_increments[curr]     = scatter(zeros, increment, prev_to_curr_map)
        realized[curr]             = (curr_cache_lengths + batch_increments) as u32

    ``prev_to_curr_map[i]`` maps previous-batch row ``i`` to its current-batch
    row (same KV slot), or an out-of-range sentinel for a departed request
    (skipped). Increments are computed in int64 then folded into the paged-cache
    ``cache_lengths`` u32 dtype. Shapes are symbolic so one MEF serves any batch.
    """
    cache_lengths_type = TensorType(DType.uint32, ["curr_batch"], device_ref)
    num_accepted_type = TensorType(_TOK, ["prev_batch"], device_ref)
    prev_to_curr_type = TensorType(_TOK, ["prev_batch"], device_ref)
    with Graph(
        "update_cache_lengths",
        input_types=[cache_lengths_type, num_accepted_type, prev_to_curr_type],
    ) as graph:
        curr_cache_lengths = graph.inputs[0].tensor
        prev_num_accepted = graph.inputs[1].tensor
        prev_to_curr_map = graph.inputs[2].tensor
        # increment[prev] = num_accepted + 1 (the committed-token count per row).
        one = ops.broadcast_to(
            ops.constant(1, dtype=_TOK, device=device_ref), ["prev_batch"]
        )
        increment = prev_num_accepted + one
        # Scatter the increments into a zeroed curr-batch vector; OOB (departed)
        # rows are skipped, leaving their increment at 0.
        zeros = ops.broadcast_to(
            ops.constant(0, dtype=_TOK, device=device_ref), ["curr_batch"]
        )
        batch_increments = scatter_nd_skip_oob_indices(
            input=zeros,
            updates=increment,
            indices=ops.unsqueeze(prev_to_curr_map, axis=-1),
        )
        realized = (curr_cache_lengths.cast(_TOK) + batch_increments).cast(
            DType.uint32
        )
        graph.output(realized)
    return graph


def _on_device_zero(
    build: Callable[[DeviceRef], Graph], device_refs: list[DeviceRef]
) -> Graph:
    return build(device_refs[0])


@dataclass(frozen=True)
class SpecFeed:
    """The geometry an in-place spec feed is built for.

    ``dp`` selects the build that slices each device's cache-length bump by
    replica; 0 the tensor-parallel build, which takes no replica bounds. An
    engine names the build it needs from the geometry it serves and refuses
    any other.
    """

    tp: int
    """The devices the feed spans, every replica's."""
    kv_groups: int
    """The cache-length destinations per device."""
    dp: int = 0
    next_draft_dtype: DType = DType.int64
    positions: bool = False
    """Whether the feed also advances 3-axis positions."""
    graph_name: str | None = None
    """The name to build the graph under, for an engine that requires one
    other than the geometry's."""

    def build(self) -> Callable[[list[DeviceRef]], Graph]:
        """Returns the builder of the feed's graph across ``tp`` devices."""
        if self.dp:
            return partial(
                _build_spec_realize_inplace_dp,
                data_parallel_degree=self.dp,
                kv_groups=self.kv_groups,
                next_draft_dtype=self.next_draft_dtype,
                positions=self.positions,
                name=self.graph_name,
            )
        return partial(
            _build_spec_realize_inplace,
            kv_groups=self.kv_groups,
            positions=self.positions,
            name=self.graph_name,
        )

    def mixed_build(self) -> Callable[[list[DeviceRef]], Graph] | None:
        """Returns the mixed-step feed's builder for a registry engine.

        Returns ``None`` when a registry engine never arms a mixed chain for
        this geometry: it arms one only for a data-parallel feed whose
        drafts are int32. An artifact ships the mixed feed of every
        speculative export (:func:`artifact_helper_builds`).
        """
        if not self.dp or self.next_draft_dtype != DType.int32:
            return None
        return self.packed_build()

    def packed_build(self) -> Callable[[list[DeviceRef]], Graph]:
        """Returns the builder of the mixed-step feed.

        It also takes the token offset each previous row's bonus token lands
        at.
        """
        return partial(
            _build_spec_realize_inplace_dp,
            data_parallel_degree=max(self.dp, 1),
            kv_groups=self.kv_groups,
            next_draft_dtype=self.next_draft_dtype,
            packed_token_offsets=True,
            positions=self.positions,
        )


def attention_cache_count(params: CacheLeafParamInterface) -> int:
    """Counts the attention caches in a KV parameter tree.

    These are the KV groups whose cache lengths a speculative step advances.
    Recurrent state has none.
    """
    if isinstance(params, MultiKVCacheParams):
        return sum(
            attention_cache_count(child) for child in params.children.values()
        )
    return 0 if isinstance(params, RecurrentStateParams) else 1


@dataclass(frozen=True)
class HelperBuild:
    """One overlap helper an export ships, and how to build it."""

    name: str
    """The file name a registry engine looks for the helper by, or the
    helper's role in an artifact."""
    n_devices: int
    """The devices the graph spans."""
    build: Callable[[list[DeviceRef]], Graph]
    """Returns the graph across the first ``n_devices`` devices."""


_PLAIN_HELPERS: tuple[tuple[Callable[[DeviceRef], Graph], str], ...] = (
    (_build_realize_future_tokens, REALIZE_MEF),
    (_build_realize_future_tokens_inplace, REALIZE_INPLACE_MEF),
    (_build_realize_draft_tokens, REALIZE_DRAFTS_MEF),
    (_build_update_cache_lengths, UPDATE_CACHE_LENGTHS_MEF),
    (_build_realize_spec_next_tokens, REALIZE_SPEC_TOKENS_MEF),
)


def helper_builds(
    spec_feed: SpecFeed | None, shared_feed: SpecFeed | None = None
) -> list[HelperBuild]:
    """Returns every overlap helper a registry engine may load for an export.

    Each is named by the file name the engine looks for it by. The single-device helpers compile against device 0 alone; the spec
    feeds write one cache-length destination per device, so they span every
    device.

    Args:
        spec_feed: The in-place spec feed of a speculative export, or
            ``None``.
        shared_feed: The feed an engine that shares one cache-length buffer
            per device across its KV groups loads instead, or ``None``.

    Returns:
        The helpers, plain ones first.
    """
    builds = [
        HelperBuild(name, 1, partial(_on_device_zero, build))
        for build, name in _PLAIN_HELPERS
    ]
    if spec_feed is not None:
        builds.append(
            HelperBuild(
                SPEC_REALIZE_INPLACE_MEF,
                spec_feed.tp,
                spec_feed.build(),
            )
        )
        mixed = spec_feed.mixed_build()
        if mixed is not None:
            builds.append(
                HelperBuild(
                    SPEC_REALIZE_MIXED_INPLACE_MEF,
                    spec_feed.tp,
                    mixed,
                )
            )
    if shared_feed is not None:
        builds.append(
            HelperBuild(
                SPEC_REALIZE_SHARED_INPLACE_MEF,
                shared_feed.tp,
                shared_feed.build(),
            )
        )
    return builds


class HelperRole(str, Enum):
    """The roles of an artifact's overlap helpers.

    An artifact's ``model`` component stores the helpers under these roles,
    beside the pipeline's own graphs.
    """

    REALIZE_FUTURE_TOKENS = "realize_future_tokens"
    """This step's tokens, the previous step's sampled tokens, and where each
    previous row lands, to the realized tokens."""

    REALIZE_FUTURE_TOKENS_INPLACE = "realize_future_tokens_inplace"
    """:attr:`REALIZE_FUTURE_TOKENS` writing its first input, for a captured
    step."""

    SPEC_REALIZE_INPLACE = "spec_realize_inplace"
    """A verify step's whole device feed in one graph, in place: tokens,
    drafts, every (KV group, device) cache length and, for an M-RoPE graph,
    positions. Sliced by replica; see :func:`_build_spec_realize_inplace_dp`
    for its inputs."""

    SPEC_REALIZE_MIXED_INPLACE = "spec_realize_mixed_inplace"
    """:attr:`SPEC_REALIZE_INPLACE` for a mixed prefill and verify step,
    which also takes the token offset each bonus token lands at."""


_PLAIN_ROLES: dict[HelperRole, Callable[[DeviceRef], Graph]] = {
    HelperRole.REALIZE_FUTURE_TOKENS: _build_realize_future_tokens,
    HelperRole.REALIZE_FUTURE_TOKENS_INPLACE: (
        _build_realize_future_tokens_inplace
    ),
}


def artifact_helper_builds(spec_feed: SpecFeed | None) -> list[HelperBuild]:
    """Returns the overlap helpers an artifact stores, named by role.

    Every export gets the plain token feeds. A speculative one also gets its
    in-place verify and mixed-step feeds, both built sliced by replica, so
    one input list serves every parallel layout.

    Args:
        spec_feed: The geometry of a speculative export's feeds, or
            ``None``.

    Returns:
        The helpers, plain ones first.
    """
    builds = [
        HelperBuild(role.value, 1, partial(_on_device_zero, build))
        for role, build in _PLAIN_ROLES.items()
    ]
    if spec_feed is None:
        return builds
    sliced = SpecFeed(
        tp=spec_feed.tp,
        kv_groups=spec_feed.kv_groups,
        dp=max(spec_feed.dp, 1),
        next_draft_dtype=spec_feed.next_draft_dtype,
        positions=spec_feed.positions,
    )
    builds += [
        HelperBuild(
            HelperRole.SPEC_REALIZE_INPLACE.value, sliced.tp, sliced.build()
        ),
        HelperBuild(
            HelperRole.SPEC_REALIZE_MIXED_INPLACE.value,
            sliced.tp,
            sliced.packed_build(),
        ),
    ]
    return builds


def compile_helpers(
    devices: Sequence[Device], builds: Sequence[HelperBuild]
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles each helper for the first of ``devices`` it spans.

    Args:
        devices: The devices the export is compiled for.
        builds: The helpers, from :func:`artifact_helper_builds`.

    Returns:
        Each helper's graph keyed by its name, as a callable that describes
        the graph and exports its MEF.

    Raises:
        ArtifactError: If a helper spans more devices than the export.
    """
    compiled: dict[str, CompiledCallable[..., Any]] = {}
    for helper in builds:
        if helper.n_devices > len(devices):
            raise ArtifactError(
                f"overlap helper {helper.name} spans {helper.n_devices}"
                f" devices; the export has {len(devices)}"
            )
        span = list(devices[: helper.n_devices])
        graph = helper.build([DeviceRef.from_device(d) for d in span])
        in_layouts, in_tree = tree.flatten(
            (tuple(as_layout(value.type) for value in graph.inputs), {}),
            leaf=TensorLayout,
        )
        out_layouts, out_tree = tree.flatten(
            tree.map(as_layout, graph.output_tree), leaf=TensorLayout
        )
        compiled[helper.name] = CompiledCallable.from_compiled(
            InferenceSession(devices=span).compile(graph),
            None,
            in_layouts,
            in_tree,
            out_layouts,
            out_tree,
        )
    return compiled
