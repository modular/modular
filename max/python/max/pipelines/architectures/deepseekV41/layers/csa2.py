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
"""CSA2 sparse attention over quantized, paged window and compressed caches.

A layer keeps two byte caches. The window cache holds a 528-byte MXFP8
record per token: 512 E4M3 values, then 16 UE8M0 scales of 32 channels
each. The compressed cache holds a 288-byte NVFP4 record per group of
``ratio`` tokens: 256 bytes of E2M1 pairs, then 32 E4M3 scales of 16
channels each. Both rotate the last 64 channels with interleaved RoPE before
quantizing.

Each cache is a paged cache leaf whose ``uint8`` blocks,
``[blocks, 1, layers, slots, 1, pitch]``, several layers share. The ops view
it as pages of ``slots * pitch`` bytes, one per (block, layer): page
``block * layers + layer``. A page holds ``slots`` records, all of their
values and then all of their scales, so absolute slot ``s`` is offset
``s % slots`` of page ``s // slots``.

Each query token names up to 128 window slots and up to 512 compressed
slots in a ``[T, 640]`` int32 list, ``-1`` for none (a window-only call
may lead with more window slots, in whole tiles of 64), and its heads attend
over those records through one softmax, with a per-head sink logit that
joins the denominator only. Each record is both key and value.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    StaticDim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import PagedCacheValues

HEAD_DIM = 512
"""Channels of a query head, a latent and a dequantized record."""
WINDOW_RECORD_BYTES = 528
"""Bytes of an MXFP8 window record."""
COMPRESSED_RECORD_BYTES = 288
"""Bytes of an NVFP4 compressed record."""
WINDOW_DATA_BYTES = HEAD_DIM
"""Bytes of a window record's E4M3 values, before its scales."""
COMPRESSED_DATA_BYTES = HEAD_DIM // 2
"""Bytes of a compressed record's E2M1 values, before its scales."""
INDEX_WIDTH = 640
"""Entries of a token's index list: 128 window slots, then 512 compressed."""
WINDOW_ENTRIES = 128
"""The window slots a token's list names."""
MAX_COMPRESSED = 512
"""The most compressed slots a token's list names."""
PREFILL_ROWS = 1 << 18
"""Distinct named slots a model's prefill workspace holds: a forward whose
lists may name more runs the decode attention instead."""
DECODE_SPLIT_ROWS = 296
"""Partial rows (``splits * tokens``) a decode workspace holds by default,
two waves of a B200's CTAs; a decode splits a token's work only while its
partials fit."""


def cache_page_bytes(page_size: int, record_bytes: int, data_bytes: int) -> int:
    """Returns the bytes of a cache page of ``page_size`` records.

    The attention reads a multi-record page's values in rows of
    ``data_bytes``, so such a page rounds up to a multiple of it.

    Args:
        page_size: Records a page holds.
        record_bytes: Bytes of a record.
        data_bytes: Bytes of a record's values.
    """
    raw = page_size * record_bytes
    if page_size == 1:
        return -(-raw // 16) * 16
    return -(-raw // data_bytes) * data_bytes


def _host_f32(value: float) -> TensorValue:
    return ops.constant([value], DType.float32, DeviceRef.CPU())


def _packed_stride() -> TensorValue:
    return ops.constant([-1], DType.int64, DeviceRef.CPU())


def records_per_page(cache: BufferValue) -> int:
    """Returns the records a page of a cache leaf's blocks holds."""
    if cache.rank != 6:
        raise ValueError(
            "a CSA cache is a leaf's [blocks, 1, layers, slots, 1, pitch] "
            f"blocks, got rank {cache.rank}"
        )
    return int(cache.shape[3])


def cache_slots(cache: BufferValue) -> Dim:
    """Returns the absolute slots of every page of a cache leaf's blocks."""
    shape = cache.shape
    return shape[0] * shape[1] * shape[2] * shape[3]


def window_entries(
    positions: TensorValue,
    request_ids: TensorValue,
    valid_from: TensorValue | None = None,
) -> TensorValue:
    """Returns each query's 128 logical SWA entries, newest first.

    Args:
        positions: ``[T]`` absolute query positions.
        request_ids: ``[T]`` request indices into ``valid_from``.
        valid_from: Optional ``[batch]`` integer absolute start of each
            request's contiguous valid SWA suffix for this layer. Entries
            below it become ``-1``. The caller supplies cache validity,
            including any preceding chunk or restored prefix; this function
            neither advances nor persists the boundary. ``None`` leaves the
            window unrestricted. Negative entries are masked by
            :func:`leaf_slots` in either case.

    Returns:
        The ``[T, 128]`` int32 logical entries, with entries below an explicit
        floor set to -1.
    """
    device = positions.device
    back = ops.range(
        0,
        -WINDOW_ENTRIES,
        -1,
        out_dim=WINDOW_ENTRIES,
        dtype=DType.int32,
        device=device,
    )
    entries = ops.unsqueeze(
        ops.cast(positions, DType.int32), -1
    ) + ops.unsqueeze(back, 0)
    if valid_from is not None:
        floor = ops.gather(
            ops.cast(valid_from, DType.int32),
            ops.cast(request_ids, DType.int32),
            axis=0,
        )
        entries = ops.where(
            entries >= ops.unsqueeze(floor, -1),
            entries,
            ops.constant(-1, DType.int32, device),
        )
    return entries


def leaf_slots(
    leaf: PagedCacheValues,
    request: TensorValue,
    entry: TensorValue,
    layer: int,
) -> TensorValue:
    """Returns the absolute slots of a layer's logical entries in a leaf.

    Entry ``e`` of request ``r`` sits at offset ``e % slots`` of the
    request's block ``lookup_table[r, e // slots]``, so its slot is
    ``(block * layers + layer) * slots + e % slots``. Negative entries map
    to ``-1``.

    Args:
        leaf: The window or compressed cache leaf.
        request: Each entry's request row in the lookup table,
            broadcastable against ``entry``.
        entry: Logical entries: token positions in the window leaf, group
            ids in a compressed leaf.
        layer: The layer's index among the leaf's layers.

    Returns:
        int32 absolute slots, ``entry``'s shape.
    """
    blocks = leaf.kv_blocks
    slots = records_per_page(blocks)
    layers = int(blocks.shape[2])
    entry = ops.cast(entry, DType.int32)
    valid = entry >= 0
    e = ops.max(entry, ops.constant(0, DType.int32, entry.device))
    request = ops.broadcast_to(ops.cast(request, DType.int32), e.shape)
    block = ops.gather_nd(
        leaf.lookup_table,
        ops.stack(
            [request, ops.floor_div(e, slots)],
            axis=-1,
        ),
    )
    slot = (ops.cast(block, DType.int32) * layers + layer) * slots + e % slots
    return ops.where(valid, slot, ops.constant(-1, DType.int32, entry.device))


def index_list(
    positions: TensorValue,
    requests: TensorValue,
    window: PagedCacheValues,
    window_layer: int,
    *,
    valid_from: TensorValue | None = None,
    selected: TensorValue | None = None,
    comp: PagedCacheValues | None = None,
    comp_layer: int = 0,
    ratio: int = 1,
) -> TensorValue:
    """Returns each query's ``[T, 640]`` index list, built in one launch.

    The list :func:`window_entries` and :func:`leaf_slots` build, bit for
    bit: the window slots of positions ``p, p - 1, ..., p - 127``, then the
    compressed slots of groups ``selected // ratio``, ``-1``-padded to 512.
    A position below 0 or ``valid_from`` and a negative selection name
    ``-1``. GPU only.

    Args:
        positions: ``[T]`` absolute query positions.
        requests: ``[T]`` each query's row in the leaves' lookup tables.
        window: The window cache leaf.
        window_layer: The layer's index among the window leaf's layers.
        valid_from: Optional ``[batch]`` start of each request's valid
            window, as for :func:`window_entries`.
        selected: ``[T, k]`` raw positions of the selected groups' closing
            tokens, ``-1`` for none, ``k`` at most 512; ``None`` on a
            window-only layer.
        comp: The compressed cache leaf; required with ``selected``.
        comp_layer: The layer's index among the compressed leaf's layers.
        ratio: Positions pooled into one compressed record.

    Raises:
        ValueError: If ``selected`` is given without ``comp``.
    """
    device = positions.device
    positions = ops.cast(positions, DType.int32)
    window_blocks = window.kv_blocks
    parameters: dict[str, bool | int | float | str | DType] = {
        "window_page": records_per_page(window_blocks),
        "window_layers": int(window_blocks.shape[2]),
        "has_floor": valid_from is not None,
    }
    values = [
        positions,
        ops.cast(requests, DType.int32),
        # Unread without a floor.
        positions if valid_from is None else ops.cast(valid_from, DType.int32),
        ops.cast(window.lookup_table, DType.uint32),
    ]
    layers = [ops.constant(window_layer, DType.uint32, DeviceRef.CPU())]
    name = "mo.csa2.index_list.window_only"
    if selected is not None:
        if comp is None:
            raise ValueError("a compressed selection needs its cache leaf")
        name = "mo.csa2.index_list"
        parameters["comp_page"] = records_per_page(comp.kv_blocks)
        parameters["comp_layers"] = int(comp.kv_blocks.shape[2])
        parameters["ratio"] = ratio
        values += [
            ops.cast(comp.lookup_table, DType.uint32),
            ops.cast(selected, DType.int32),
        ]
        layers.append(ops.constant(comp_layer, DType.uint32, DeviceRef.CPU()))
    return ops.custom(
        name,
        device=device,
        values=[*values, *layers],
        out_types=[
            TensorType(DType.int32, [positions.shape[0], INDEX_WIDTH], device)
        ],
        parameters=parameters,
    )[0].tensor


def index_list_ops(
    positions: TensorValue,
    requests: TensorValue,
    window: PagedCacheValues,
    window_layer: int,
    *,
    valid_from: TensorValue | None = None,
    selected: TensorValue | None = None,
    comp: PagedCacheValues | None = None,
    comp_layer: int = 0,
    ratio: int = 1,
) -> TensorValue:
    """Returns the list :func:`index_list` builds, from graph ops.

    :func:`window_entries` and :func:`leaf_slots` for the window, the
    selection's groups through :func:`leaf_slots` for the compressed part,
    and concats; runs on any device. Takes :func:`index_list`'s arguments.

    Raises:
        ValueError: If ``selected`` is given without ``comp``.
    """
    device = positions.device
    t = positions.shape[0]
    request = ops.unsqueeze(ops.cast(requests, DType.int32), -1)
    window_slots = leaf_slots(
        window,
        request,
        window_entries(positions, requests, valid_from),
        window_layer,
    )
    none = ops.constant(-1, DType.int32, device)
    if selected is None:
        return ops.concat(
            [window_slots, ops.broadcast_to(none, [t, MAX_COMPRESSED])], axis=1
        )
    if comp is None:
        raise ValueError("a compressed selection needs its cache leaf")
    group = ops.where(selected >= 0, ops.floor_div(selected, ratio), none)
    lists = [window_slots, leaf_slots(comp, request, group, comp_layer)]
    pad = MAX_COMPRESSED - int(selected.shape[-1])
    if pad:
        lists.append(ops.broadcast_to(none, [t, pad]))
    return ops.concat(lists, axis=1)


def store_window_kv(
    cache: BufferValue,
    latent: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    slots: TensorValue,
    *,
    block_stride: TensorValue | None = None,
) -> None:
    """RoPEs, quantizes to MXFP8 and stores each row's window record.

    Args:
        cache: The window cache leaf's blocks.
        latent: ``[T, 512]`` bf16 KV latents before RoPE.
        positions: ``[T]`` token positions.
        rope_table: ``[rows, 64]`` float32 cosines then sines per position.
        slots: ``[T]`` absolute slots, negative to skip a row.
        block_stride: Host ``[1]`` int64 block stride in bytes, ``-1`` when
            the blocks are packed (the default); the op requires packed
            blocks.
    """
    ops.inplace_custom(
        "mo.csa.window_kv_store",
        device=latent.device,
        values=[
            ops.cast(latent, DType.bfloat16),
            ops.cast(positions, DType.int64),
            rope_table,
            ops.cast(slots, DType.int64),
            cache,
            _packed_stride() if block_stride is None else block_stride,
        ],
        parameters={"page_size": records_per_page(cache)},
    )


def store_compressed_kv(
    cache: BufferValue,
    latent: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    slots: TensorValue,
    *,
    ratio: int,
    block_stride: TensorValue | None = None,
) -> None:
    """RoPEs, quantizes to NVFP4 and stores each group-closing row's record.

    The trailing 64 channels rotate at the group's first position.

    Args:
        cache: The compressed cache leaf's blocks.
        latent: ``[T, 512]`` bf16 pooled latents before RoPE.
        positions: ``[T]`` token positions.
        rope_table: ``[rows, 64]`` float32 cosines then sines per position.
        slots: ``[T]`` absolute slots, negative to skip a row.
        ratio: Positions pooled into one record, 1 or 2.
        block_stride: As for :func:`store_window_kv`.
    """
    ops.inplace_custom(
        "mo.csa.compressed_kv_store",
        device=latent.device,
        values=[
            ops.cast(latent, DType.bfloat16),
            ops.cast(positions, DType.int64),
            rope_table,
            ops.cast(slots, DType.int64),
            cache,
            _packed_stride() if block_stride is None else block_stride,
        ],
        parameters={"page_size": records_per_page(cache), "ratio": ratio},
    )


def compressed_entries(compressed_rows: int) -> int:
    """Returns the compressed list entries a token walks, a multiple of 64."""
    return -(-min(compressed_rows, MAX_COMPRESSED) // 64) * 64


def decode_workspace_floats(n_heads: int, split_rows: int) -> int:
    """Returns the float32 values of a decode's split workspace."""
    return split_rows * n_heads * (HEAD_DIM + 1)


def prefill_workspace_values(rows: Dim | int, slots: Dim | int) -> Dim:
    """Returns the bf16 values of a prefill's workspace.

    The dequantized rows, then int32 words (two bf16 values each): a map
    entry per absolute slot of both caches, each row's slot and the row
    count.

    Args:
        rows: The most distinct slots the call's lists name.
        slots: The absolute slots of both caches.
    """
    return Dim(rows) * (HEAD_DIM + 2) + 2 * (Dim(slots) + 1)


def prefill_workspace_rows(
    tokens: Dim | int,
    entries: int,
    slots: Dim | int,
    max_distinct_rows: Dim | int | None = None,
) -> Dim:
    """Returns the rows a prefill's workspace holds.

    A call's lists name at most one distinct slot per walked list entry,
    and at most every slot of both caches, so by default the rows are the
    lesser of the two when both are static.

    Args:
        tokens: Query tokens ``T``.
        entries: List entries each token walks.
        slots: The absolute slots of both caches.
        max_distinct_rows: The rows to hold instead, if given.
    """
    if max_distinct_rows is not None:
        return Dim(max_distinct_rows)
    t, n = Dim(tokens), Dim(slots)
    if isinstance(t, StaticDim) and isinstance(n, StaticDim):
        return Dim(min(int(t) * entries, int(n)))
    return t * entries


def _check_dim(what: str, dim: Dim, want: int, *, least: bool = False) -> None:
    if not isinstance(dim, StaticDim):
        return
    if int(dim) < want or (not least and int(dim) != want):
        raise ValueError(f"CSA2 needs {want} {what}, got {int(dim)}")


def csa2_attention(
    q: TensorValue,
    idx: TensorValue,
    sink: TensorValue,
    kv_window: BufferValue,
    kv_comp: BufferValue | None,
    *,
    scale: float,
    window_stride: TensorValue | None = None,
    comp_stride: TensorValue | None = None,
    compressed_rows: int = MAX_COMPRESSED,
    prefill: bool | TensorValue = False,
    split_rows: int = DECODE_SPLIT_ROWS,
    max_distinct_rows: Dim | int | None = None,
    window_entries: int = WINDOW_ENTRIES,
) -> TensorValue:
    """Returns ``[T, n_heads, 512]`` bf16 sparse attention outputs.

    Decode reads the records in place and suits up to 8 query tokens a
    sequence. Prefill first dequantizes each distinct slot the lists name
    to one bf16 row, so its scratch and time grow with the slots named.
    A host ``prefill`` flag picks between them per call, inside the op, so
    a graph serving both needs no branch; its workspace fits either.

    Args:
        q: ``[T, n_heads, 512]`` bf16 queries.
        idx: ``[T, 640]`` int32 window then compressed absolute slots,
            ``-1`` for none, the compressed ones packed first.
        sink: ``[n_heads]`` float32 sink logits.
        kv_window: The window cache leaf's blocks.
        kv_comp: The compressed cache leaf's blocks, ``None`` on a
            window-only layer.
        scale: Softmax scale.
        window_stride: Host ``[1]`` int64 block stride of ``kv_window``,
            ``-1`` (the default) when packed; the op requires packed blocks.
        comp_stride: The same for ``kv_comp``.
        compressed_rows: The most compressed entries any list names, at
            most 512; it sets the tiles each token walks.
        prefill: Whether to dequantize the named cache slots first: a
            constant, or a host bool scalar read when the op runs.
        split_rows: Partial rows a decode's split workspace holds.
        max_distinct_rows: The most distinct slots a prefill's lists name
            across both caches, the rows its workspace holds; a call naming
            more aborts. Defaults to a row per walked list entry,
            ``T * (W + C)`` with ``W`` the window entries and ``C`` from
            :func:`compressed_entries`, capped at the caches' slots when
            both are static.
        window_entries: The window slots leading each list, a multiple of
            64; only a window-only call may name more than 128.

    Raises:
        ValueError: If ``window_entries`` is not a multiple of 64 within the
            list, or exceeds 128 with a compressed cache, or if a static
            dimension of ``q`` or ``idx`` differs from the shapes above,
            or ``sink`` has fewer logits than ``q`` has heads.
    """
    if (
        window_entries <= 0
        or window_entries % 64
        or window_entries > INDEX_WIDTH
        or (kv_comp is not None and window_entries != WINDOW_ENTRIES)
    ):
        raise ValueError(
            f"{window_entries} window entries do not fit a CSA2 list"
            + ("" if kv_comp is None else " with a compressed cache")
        )
    t, n_heads = q.shape[0], int(q.shape[1])
    _check_dim("query channels", q.shape[2], HEAD_DIM)
    _check_dim("index entries per token", idx.shape[1], INDEX_WIDTH)
    _check_dim("sink logits", sink.shape[0], n_heads, least=True)
    device = q.device
    flag = None if isinstance(prefill, bool) else prefill
    if flag is None:
        name = "mo.csa2." + ("prefill" if prefill else "decode")
    else:
        name = "mo.csa2.attention"
    parameters: dict[str, bool | int | float | str | DType] = {
        "num_heads": n_heads,
        "window_page": records_per_page(kv_window),
    }
    if flag is not None:
        parameters["split_rows"] = split_rows
    strides = [_packed_stride() if window_stride is None else window_stride]
    if kv_comp is None:
        name += ".window_only"
        parameters["window_tiles"] = window_entries // 64
    else:
        parameters["compressed_rows"] = compressed_rows
        parameters["comp_page"] = records_per_page(kv_comp)
        strides.append(_packed_stride() if comp_stride is None else comp_stride)
    decode_floats = decode_workspace_floats(n_heads, split_rows)
    if prefill is False:
        workspace = TensorType(DType.float32, [decode_floats], device)
    else:
        entries = window_entries
        slots = cache_slots(kv_window)
        if kv_comp is not None:
            entries += compressed_entries(compressed_rows)
            slots = slots + cache_slots(kv_comp)
        rows = prefill_workspace_rows(t, entries, slots, max_distinct_rows)
        values = prefill_workspace_values(rows, slots)
        if flag is not None:
            # Room for the decode too, which views the values as float32
            # partials.
            values = values + 2 * decode_floats
        workspace = TensorType(DType.bfloat16, [values], device)
    caches = [kv_window] if kv_comp is None else [kv_window, kv_comp]
    return ops.inplace_custom(
        name,
        device=device,
        values=[
            ops.cast(q, DType.bfloat16),
            *caches,
            ops.cast(idx, DType.int32),
            ops.cast(sink, DType.float32),
            _host_f32(scale),
            *strides,
            *(
                []
                if flag is None
                else [ops.reshape(ops.cast(flag, DType.bool), [1])]
            ),
        ],
        out_types=[
            TensorType(DType.bfloat16, [t, n_heads, HEAD_DIM], device),
            workspace,
        ],
        parameters=parameters,
    )[0].tensor
