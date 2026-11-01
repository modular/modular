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
"""GPU test for the attention index list on main's ``mo.csa2.index_list``.

The expected list is derived from the leaf-slot formula --
``(block * layers + layer) * slots + entry % slots`` with
``block = lookup_table[request, entry // slots]`` -- never from the kernel's
own arithmetic. Two requests of 60 live rows fill ``T == 128`` rows, so a
flat request axis would cross-broadcast against the 128-column window, and
the last eight rows are capture padding pinned to the last request: those
rows must name nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.module import Module
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    Attention,
    RaggedRows,
)
from max.tensor import Tensor

_TOTAL = 128
_LIVE = 60
_WINDOW = 128
_TOPK = 8
_LAYERS = 2
_LAYER_SWA = 1
_LAYER_COMP = 1
_PAGES = 2
_SWA_SLOTS = 128
_COMP_SLOTS = 64
_FLOORS = (0, 200)


def _gpu(t: np.ndarray, dtype: DType | None = None) -> Tensor:
    out = Tensor.from_dlpack(np.ascontiguousarray(t)).to(Accelerator())
    return out if dtype is None else out.cast(dtype)


def _leaf(slots: int, pages: int) -> tuple[PagedCacheValues, np.ndarray]:
    """A leaf with an identity lookup table: request ``r``'s page ``p`` is
    physical block ``r * pages + p``."""
    lut = np.arange(2 * pages, dtype=np.int64).reshape(2, pages)
    blocks = np.zeros((2 * pages, 1, _LAYERS, slots, 1, 68), dtype=np.uint8)
    leaf = PagedCacheValues(
        kv_blocks=_gpu(blocks),
        cache_lengths=_gpu(np.zeros(2, dtype=np.uint32)),
        lookup_table=_gpu(lut),
        max_prompt_length=Tensor.from_dlpack(np.array([1], dtype=np.uint32)),
        max_cache_length=Tensor.from_dlpack(np.array([1], dtype=np.uint32)),
        page_stride=Tensor.from_dlpack(np.array([-1], dtype=np.int64)),
    )
    return leaf, lut


def _slot(lut: np.ndarray, slots: int, req: int, entry: int, layer: int) -> int:
    block = int(lut[req, entry // slots])
    return (block * _LAYERS + layer) * slots + entry % slots


@pytest.mark.parametrize("ratio", [1, 2])
def test_index_list_matches_the_leaf_slot_formula(ratio: int) -> None:
    offsets = np.array([0, _LIVE, 2 * _LIVE], dtype=np.int32)
    bid = np.concatenate([np.zeros(_LIVE), np.ones(_TOTAL - _LIVE)]).astype(
        np.int32
    )
    # Request 0 runs positions 0..59, request 1 runs 192..251: both inside
    # the pages each request owns. The tail is padding.
    positions = np.concatenate(
        [
            np.arange(_LIVE),
            192 + np.arange(_LIVE),
            252 + np.arange(_TOTAL - 2 * _LIVE),
        ]
    ).astype(np.int32)
    rows = RaggedRows(
        offsets=_gpu(offsets, DType.uint32),
        bid=_gpu(bid),
        local=_gpu(np.zeros(_TOTAL, dtype=np.int32)),
        positions=_gpu(positions),
        lengths=_gpu(np.array([_LIVE, _LIVE], dtype=np.int32)),
    )
    swa, swa_lut = _leaf(_SWA_SLOTS, _PAGES)
    comp, comp_lut = _leaf(_COMP_SLOTS, _PAGES)

    rng = np.random.default_rng(ratio)
    entries = np.full((_TOTAL, _TOPK), -1, dtype=np.int32)
    for r in range(2 * _LIVE):
        # Ascending group ids inside the request's compressed pages,
        # ``-1`` padded.
        picked = np.sort(
            rng.choice(_PAGES * _COMP_SLOTS, rng.integers(0, _TOPK + 1), False)
        )
        entries[r, : len(picked)] = picked
    # A padded row's selection must be masked too.
    entries[2 * _LIVE :, :3] = [0, 1, 2]

    attention = object.__new__(Attention)
    Module.__init__(attention)
    attention.compress_ratio = ratio
    got = attention._index_list(
        swa,
        comp,
        Tensor.from_dlpack(np.array(_LAYER_SWA, dtype=np.uint32)),
        Tensor.from_dlpack(np.array(_LAYER_COMP, dtype=np.uint32)),
        rows,
        _gpu(np.array(_FLOORS, dtype=np.int32)),
        _gpu(entries),
    ).to_numpy()

    assert got.shape == (_TOTAL, 640)
    want = np.full((_TOTAL, 640), -1, dtype=np.int64)
    for r in range(2 * _LIVE):
        req = int(bid[r])
        for k in range(_WINDOW):
            entry = int(positions[r]) - k
            if entry >= max(_FLOORS[req], 0):
                want[r, k] = _slot(swa_lut, _SWA_SLOTS, req, entry, _LAYER_SWA)
        for j, group in enumerate(entries[r]):
            if group >= 0:
                want[r, _WINDOW + j] = _slot(
                    comp_lut, _COMP_SLOTS, req, int(group), _LAYER_COMP
                )
    # The compressed slots sit at the list's head after the window, one per
    # selected entry, ``-1`` for the rest.
    np.testing.assert_array_equal(got[: 2 * _LIVE], want[: 2 * _LIVE])
    assert (got[2 * _LIVE :] == -1).all()
