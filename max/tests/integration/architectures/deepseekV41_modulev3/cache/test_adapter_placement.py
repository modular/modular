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
"""Regression: the thin CSA adapter's index-list placement contract.

The request ids must be one per query row, ``[T, 1]`` (V2
``CSA2Attention._index_list``): a flat ``[T]`` id would broadcast
against the ``[T, entries]`` lists. This test runs the same in-graph
construction -- V2
``window_entries``, one ``floor_div`` of closing positions to group ids, V2
``leaf_slots`` over a real two-request page table of distinct blocks per
request (the staging the block manager produces) -- and pins the slot ids it
must produce: per-request placement (the same entry id resolves to
different physical slots for different requests), the shared window leaf's
layer offset, and ``-1`` kept for below-floor window entries and unselected
compressed entries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    Type,
    ops,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    MAX_COMPRESSED,
    leaf_slots,
    window_entries,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.lib import KVCacheConfig

NH = 2
HEAD_DIM = 512
LAYER = 1  # a nonzero layer of the shared window leaf
VALID_FROM = (16, 0)  # per-request SWA begin floors
POSITIONS = (130, 5)  # r0 spans two pages; r1 is a short prefix
# Closing positions each request's compressed selection names; ratio 2.
SELECTIONS = ((1, 3, 129), (1, 65))
# Page tables: two logical pages per request, each row staging distinct
# blocks the way the block manager does (no aliasing within a row).
WINDOW_TABLE = ((2, 1), (1, 0))
COMP_TABLE = ((0, 1), (1, 0))


def _hf_config() -> dict[str, Any]:
    return {
        "text_config": {
            "vocab_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": NH,
            "head_dim": HEAD_DIM,
            "sliding_window": 128,
            "compress_ratios": [0, 2],
            "kv_source_layer_ids": [1],
            "index_head_dim": 128,
            "engram_num_embeddings": [2],
            "engram_head_dim": 256,
            "dspark_target_layer_ids": [],
        }
    }


def _shape(key: str) -> tuple[int, ...]:
    params = build_kv_params(
        V41Config.from_hf(_hf_config()),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    child = params.children[key]
    assert isinstance(child, KVCacheParams)
    return tuple(int(d) for d in child.shape_per_block)


def _expected() -> np.ndarray:
    """The slot ids the documented leaf contract must produce.

    Entry ``e`` of request ``r`` sits at slot
    ``(table[r, e // slots] * layers + layer) * slots + e % slots``, and a
    negative entry stays ``-1``.
    """
    interface = V41Config.from_hf(_hf_config())
    # The DSpark draft stages share the window leaf (``kv_leaf_specs``
    # sizes its num_layers with them), so the op strides a block's layers
    # across the draft rows too.
    w_layers = interface.num_hidden_layers + interface.num_dspark_stages
    w_slots = PAGE_SIZE
    c_slots = PAGE_SIZE // 2
    out = np.full((2, 128 + MAX_COMPRESSED), -1, np.int64)
    for r, pos in enumerate(POSITIONS):
        entries = np.arange(pos, pos - 128, -1)
        keep = entries >= VALID_FROM[r]
        e = np.where(keep, entries, -1)
        valid = e >= 0
        block = np.take(WINDOW_TABLE[r], np.maximum(e, 0) // w_slots)
        out[r, :128] = np.where(
            valid, (block * w_layers + LAYER) * w_slots + e % w_slots, -1
        )
        sel = np.full(MAX_COMPRESSED, -1, np.int64)
        sel[: len(SELECTIONS[r])] = SELECTIONS[r]
        group = np.where(sel >= 0, sel // 2, -1)
        valid = group >= 0
        block = np.take(COMP_TABLE[r], np.maximum(group, 0) // c_slots)
        out[r, 128:] = np.where(valid, block * c_slots + group % c_slots, -1)
    return out


def test_index_list_placement() -> None:
    """The adapter's index list places every entry through its request's
    row of the page table, at the addressed layer, keeping ``-1``."""
    cpu = DeviceRef.CPU()
    swa_blocks_shape = (3, *_shape("swa"))
    comp_blocks_shape = (2, *_shape("c2a"))
    assert swa_blocks_shape[3] == PAGE_SIZE
    assert comp_blocks_shape[3] == PAGE_SIZE // 2
    inputs: list[Type[Any]] = [
        TensorType(DType.int32, [2], cpu),  # positions
        TensorType(DType.int32, [2], cpu),  # request ids, 1-D
        TensorType(DType.int32, [2], cpu),  # per-request SWA floors
        TensorType(DType.int32, [2, MAX_COMPRESSED], cpu),  # selections
        BufferType(DType.uint8, list(swa_blocks_shape), cpu),
        BufferType(DType.uint8, list(comp_blocks_shape), cpu),
    ]
    with Graph("adapter_placement", input_types=inputs) as g:
        positions, request, valid_from, sel = (v.tensor for v in g.inputs[:4])
        swa_blocks, comp_blocks = (v.buffer for v in g.inputs[4:])
        packed = ops.constant([-1], DType.int64, cpu)

        def leaf(
            blocks: Any, table: tuple[tuple[int, ...], ...]
        ) -> PagedCacheValues:
            return PagedCacheValues(
                kv_blocks=blocks,
                cache_lengths=ops.constant([1, 1], DType.int64, cpu),
                lookup_table=ops.constant(
                    [list(r) for r in table], DType.int64, cpu
                ),
                max_prompt_length=ops.constant(1, DType.int64, cpu),
                max_cache_length=ops.constant(1, DType.int64, cpu),
                page_stride=packed,
            )

        swa = leaf(swa_blocks, WINDOW_TABLE)
        comp = leaf(comp_blocks, COMP_TABLE)
        # The V2 ``_index_list`` contract: ``window_entries`` takes the
        # request ids 1-D; ``leaf_slots`` takes them row-shaped, [T, 1].
        request_row = ops.unsqueeze(request, -1)
        window = leaf_slots(
            swa,
            request_row,
            window_entries(positions, request, valid_from),
            LAYER,
        )
        # Closing position -> group id, divided exactly once.
        group = ops.where(
            sel >= 0,
            ops.floor_div(sel, 2),
            ops.constant(-1, DType.int32, cpu),
        )
        compressed = leaf_slots(comp, request_row, group, 0)
        g.output(ops.concat([window, compressed], 1))

    model = InferenceSession(devices=[CPU()]).load(g)
    sel_host = np.full((2, MAX_COMPRESSED), -1, np.int64)
    for r, closing in enumerate(SELECTIONS):
        sel_host[r, : len(closing)] = closing
    host = [
        np.array(POSITIONS, np.int32),
        np.arange(2, dtype=np.int32),  # row 0 is request 0, row 1 request 1
        np.array(VALID_FROM, np.int32),
        sel_host.astype(np.int32),
        np.zeros(swa_blocks_shape, np.uint8),
        np.zeros(comp_blocks_shape, np.uint8),
    ]
    buffers = [Buffer.from_numpy(np.ascontiguousarray(a)) for a in host]
    (idx,) = model.execute(*buffers)
    np.testing.assert_array_equal(idx.to_numpy(), _expected().astype(np.int32))


def test_request_ids_must_be_one_per_row() -> None:
    """A ``[T]`` request id cannot name ``[T, entries]`` entries: the
    adapter must keep the request ids row-shaped, or graph construction
    fails."""
    cpu = DeviceRef.CPU()
    inputs: list[Type[Any]] = [
        TensorType(DType.int32, [2], cpu),
        TensorType(DType.int32, [2], cpu),
        TensorType(DType.int32, [2], cpu),
        BufferType(DType.uint8, [3, *_shape("swa")], cpu),
    ]
    with Graph("adapter_bad_request", input_types=inputs) as g:
        positions, request, valid_from = (v.tensor for v in g.inputs[:3])
        swa_blocks = g.inputs[3].buffer
        packed = ops.constant([-1], DType.int64, cpu)
        swa = PagedCacheValues(
            kv_blocks=swa_blocks,
            cache_lengths=ops.constant([1, 1], DType.int64, cpu),
            lookup_table=ops.constant([[0, 0], [0, 0]], DType.int64, cpu),
            max_prompt_length=ops.constant(1, DType.int64, cpu),
            max_cache_length=ops.constant(1, DType.int64, cpu),
            page_stride=packed,
        )
        try:
            leaf_slots(
                swa,
                request,
                window_entries(positions, request, valid_from),
                LAYER,
            )
        except Exception as err:
            assert "broadcast" in str(err).lower(), err
        else:
            raise AssertionError(
                "a [T] request id built a [T, 128] slot list; leaf_slots "
                "must refuse it"
            )
