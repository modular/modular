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

"""One extent per symbolic page dim, across every input a pool binds.

Inputs sharing a symbolic dim are checked against each other by a *runtime*
assert the entry graph carries, so a pair that disagrees does not fail to
compile. The model loads, the server answers ``/v1/models``, and the first
real batch dies binding its inputs.

A Jenga pool is what makes them disagree. It hands every leaf a view of one
shared slab at that leaf's own page width, so a quantized cache -- two leaves,
a values one and a narrower scales one -- holds more scale pages than value
pages. A legacy pool allocates the two counts equal, which is why one symbol
covering both went unnoticed until a model reached Jenga.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    CacheLeafParamInterface,
    KVCacheBuffer,
    KVCacheParamInterface,
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheBuffer,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    plan_jenga_geometry,
)

PAGE_SIZE = 128

#: Sparse-attention layers GLM-5.3-Flash caches, its MTP draft layer included.
SPARSE_LAYERS = 11


def _mla(
    *,
    head_dim: int,
    quant: KVCacheQuantizationConfig | None,
    dtype: DType = DType.float8_e4m3fn,
) -> MLAKVCacheParams:
    """An MLA leaf shaped like a sparse-attention model's latent or indexer."""
    return MLAKVCacheParams(
        dtype=dtype,
        num_layers=SPARSE_LAYERS,
        head_dim=head_dim,
        num_q_heads=64,
        enable_prefix_caching=True,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.CPU()],
        kvcache_quant_config=quant,
    )


def _mla_latent() -> MLAKVCacheParams:
    # 576 is the padded latent width the SM100 kernels comptime-assert. The
    # int8 scale dtype is GLM-5.3-Flash's own, and ``quantized_kv_cache`` does
    # not accept it, so this leaf contributes no scales today -- which is why
    # the tree case below fails on the indexer's dim alone.
    return _mla(
        head_dim=576,
        quant=KVCacheQuantizationConfig(
            scale_dtype=DType.int8, quantization_granularity=32
        ),
    )


def _indexer() -> MLAKVCacheParams:
    # The DSA indexer holds one pooled key per ``index_kpool`` tokens, so its
    # head dim is ``index_head_dim // index_kpool``, and one f32 scale covers
    # the whole pooled key. That makes its scale page 8x narrower than its
    # value page -- the ratio this file exists to keep out of one symbol.
    return _mla(
        head_dim=32,
        quant=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=32
        ),
    )


def _mha_quantized() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        num_layers=2,
        n_kv_heads=4,
        head_dim=256,
        enable_prefix_caching=True,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.CPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        ),
    )


def _state() -> RecurrentStateParams:
    # Shaped like a hybrid model's linear-attention state, scaled down: the
    # real dimensions are exercised by the Jenga geometry tests, and what
    # matters here is only that a state child is present.
    return RecurrentStateParams(
        regions=(
            RecurrentStateRegion(
                leaf_id="linear_attn/conv",
                num_layers=2,
                row_shape=(8, 9),
                dtype=DType.float32,
            ),
            RecurrentStateRegion(
                leaf_id="linear_attn/recurrent",
                num_layers=2,
                row_shape=(4, 8, 9),
                dtype=DType.float32,
            ),
        ),
        devices=[DeviceRef.CPU()],
    )


def _sparse_tree(with_state: bool) -> MultiKVCacheParams:
    """A sparse-attention cache: the latent, its indexer, maybe a state."""
    children: dict[str, CacheLeafParamInterface] = {
        "mla": _mla_latent(),
        "indexer": _indexer(),
    }
    if with_state:
        children["state"] = _state()
    return MultiKVCacheParams.from_params(children)


def _jenga_slab(
    params: KVCacheParamInterface, huge_blocks: int = 4
) -> tuple[Buffer, Mapping[str, int]]:
    """Allocates the one slab a Jenga pool would, plus its padded page sizes.

    Mirrors ``JengaKVCacheManager.create``: a paged leaf declares a
    replica-wide page, so it is divided down to the one device here, while a
    row-addressed leaf is already per-device.
    """
    leaves = params.leaves()
    tp_degree = params.tensor_parallel_degree
    pages = {
        leaf_id: leaf.bytes_per_page
        if leaf.group_id.is_recurrent()
        else leaf.bytes_per_page // tp_degree
        for leaf_id, leaf in leaves.items()
    }
    rows = {leaf_id: leaf.row_bytes for leaf_id, leaf in leaves.items()}
    geometry = plan_jenga_geometry(
        huge_blocks * max(pages.values()), pages, rows
    )
    slab = Buffer.zeros(
        shape=(geometry.num_huge_blocks, geometry.huge_page_bytes),
        dtype=DType.uint8,
        device=CPU(),
    )
    return slab, geometry.padded_sizes


def _record(extents: dict[str, set[int]], declared: Any, bound: Buffer) -> None:
    extents[str(declared.shape[0])].add(int(bound.shape[0]))


def _collect(
    params: CacheLeafParamInterface,
    symbolic: Any,
    buffers: Any,
    slabs: Sequence[Buffer],
    extents: dict[str, set[int]],
) -> None:
    """Walks a params tree, pairing each declared page dim with its buffer."""
    if isinstance(params, MultiKVCacheParams):
        assert isinstance(buffers, MultiKVCacheBuffer)
        for key, child in params.children.items():
            _collect(
                child, symbolic[key], buffers.children[key], slabs, extents
            )
        return

    if isinstance(params, RecurrentStateParams):
        # A state leaf's kernels index the rows the pool view exposes, not the
        # pages, so that is what the graph binds.
        rows = params.slab_to_row_views(slabs)
        for device_idx, per_device in enumerate(symbolic):
            for region, leaf in zip(
                params.regions, per_device.leaves, strict=True
            ):
                _record(extents, leaf.pool, rows[region.leaf_id][device_idx])
        return

    assert isinstance(buffers, KVCacheBuffer)
    for device_idx, per_device in enumerate(symbolic):
        _record(extents, per_device.kv_blocks, buffers.values[device_idx])
        if per_device.kv_scales is not None:
            assert buffers.scales is not None
            _record(extents, per_device.kv_scales, buffers.scales[device_idx])


def _page_dim_extents(
    params: KVCacheParamInterface,
) -> dict[str, set[int]]:
    """Maps each symbolic page dim to the extents a Jenga pool binds to it."""
    slab, padded = _jenga_slab(params)
    extents: dict[str, set[int]] = defaultdict(set)
    _collect(
        params,
        params.get_symbolic_inputs(),
        params.slab_to_buffer_views([slab], padded),
        [slab],
        extents,
    )
    return extents


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(_mha_quantized(), id="mha-fp8"),
        pytest.param(_indexer(), id="mla-indexer-fp8"),
        pytest.param(_sparse_tree(with_state=False), id="latent-plus-indexer"),
        pytest.param(_sparse_tree(with_state=True), id="plus-recurrent-state"),
    ],
)
def test_jenga_binds_one_extent_per_page_dim(
    params: KVCacheParamInterface,
) -> None:
    """No symbolic dim may reach two inputs the pool sizes differently.

    Fails without a separate scales dim with
    ``{'total_num_pages': [values, scales]}``, naming the two counts the
    entry graph would have asserted equal.
    """
    ambiguous = {
        dim: sorted(seen)
        for dim, seen in _page_dim_extents(params).items()
        if len(seen) > 1
    }
    assert not ambiguous, (
        "these symbolic dims reach inputs the pool binds at different extents,"
        f" which the entry graph asserts against on the first batch: {ambiguous}"
    )


def test_a_quantized_leaf_really_does_hold_more_scale_pages() -> None:
    """The positive control: the two counts differ, so the case is live.

    Without this, a params set whose scale page happened to match its value
    page would pass the invariant above having tested nothing.
    """
    params = _indexer()
    slab, padded = _jenga_slab(params)

    buffers = params.slab_to_buffer_views([slab], padded)

    assert isinstance(buffers, KVCacheBuffer)
    assert buffers.scales is not None
    assert buffers.values[0].shape[0] < buffers.scales[0].shape[0]


def test_a_legacy_pool_still_binds_the_two_counts_equal() -> None:
    """A separate symbol costs the legacy allocator nothing.

    It hands values and scales one page count, which satisfies two dims as
    readily as one -- so this is not a reason to force the counts equal.
    """
    params = _indexer()

    (buffers,) = params.allocate_buffers(total_num_pages=8)

    assert isinstance(buffers, KVCacheBuffer)
    assert buffers.scales is not None
    assert buffers.values[0].shape[0] == buffers.scales[0].shape[0] == 8
