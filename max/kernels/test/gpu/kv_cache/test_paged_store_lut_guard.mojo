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

"""Regression test: a store resolved outside its pool drops the write.

The LUTs are staged graph inputs, so a cell the step's copy did not cover holds
whatever the arena was allocated with: zero, now that the staging arena is
zeroed when it is allocated, and anything else means a corrupted arena or a
caller staging an id outside the pool. A store that follows such a word
addresses outside the pool, so `store` and `store_scale` return without
writing.

This pins that each skips exactly that cell, and nothing else: an id past the
pool writes nothing at all, while the pool's last page — the null page the
manager reserves — still stores, as does an ordinary page. The scale arm runs
the same check against the scale pool, whose bound `store_scale` computes from
its own page count.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.utils.numerics import isnan, nan

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import padded_lut_cols

comptime _POOL_PAGES = 4
"""Real pages in each pool: the ones a store legitimately lands on."""

comptime _NULL_PAGE = _POOL_PAGES
"""The page `PagedKVCacheManager` appends after the pool for the null block.

The unassigned-slot fill `total_num_pages` resolves here, in bounds by design.
"""

comptime _CACHE_PAGES = _NULL_PAGE + 1
"""Pages the cache view covers: the pool plus its null page."""

comptime _CANARY_PAGE = _CACHE_PAGES
"""The page only a store resolved outside the pool can reach."""

comptime _STORAGE_PAGES = _CANARY_PAGE + 1
"""Pages allocated, so the canary exists to be clobbered."""

comptime _UNWRITTEN_SCALE = nan[DType.float32]()
"""What every scale page holds until a store lands on it.

A NaN is a value the stores below cannot produce, so a canary that reads back
finite was written by something this test did not intend.
"""


def _store_through_lut_kernel[
    cache_t: KVCacheT, page_size: Int
](kv: cache_t,):
    """One thread, three stores, each resolved through a different LUT cell."""
    if global_idx.x != 0:
        return
    comptime elem = cache_t.dtype
    # An ordinary page: the control proving the store path is live here.
    kv.store(0, 0, 0, 0, SIMD[elem, 1](1.0))
    # The pool's last page, which the manager reserves for the null block, is
    # in bounds and stores like any other.
    kv.store(0, 0, 2 * page_size, 0, SIMD[elem, 1](3.0))
    # A cell holding an id past the pool must not reach memory at all.
    kv.store(0, 0, page_size, 0, SIMD[elem, 1](2.0))


def _store_scale_through_lut_kernel[
    cache_t: KVCacheT, page_size: Int
](kv: cache_t,):
    """One thread, two scale stores, resolved through different LUT cells."""
    if global_idx.x != 0:
        return
    comptime scale = cache_t.scale_dtype
    # An ordinary scale page: the control proving the store path is live here.
    kv.store_scale(0, 0, 0, 0, SIMD[scale, 1](5.0))
    # A cell holding an id past the scale pool must not reach memory at all.
    kv.store_scale(0, 0, page_size, 0, SIMD[scale, 1](6.0))


def run_store_guard_test[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](ctx: DeviceContext) raises:
    """Checks that `store` drops only the write its pool cannot hold.

    Parameters:
        dtype: Element type of the value pool.
        kv_params: Static shapes of the cache.
        page_size: Tokens per page.

    Args:
        ctx: The device context to allocate on.
    """
    comptime unwritten = nan[dtype]()

    var blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(_STORAGE_PAGES),
            Idx[2],
            Int64(1),  # num_layers
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
        ctx,
    )
    _ = blocks.host_tensor().fill(unwritten)
    blocks.to_device()

    # Column `i` is the page slot a token with `tok_idx // page_size == i`
    # resolves through: column 0 holds an ordinary page, column 2 the null
    # page, and every other column an id past the pool.
    var lut_columns = padded_lut_cols(_CACHE_PAGES + 4)
    var lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(1), Int64(lut_columns)), ctx
    )
    var lut_host = lut.host_tensor()
    for c in range(lut_columns):
        lut_host[0, c] = UInt32(_CACHE_PAGES)
    lut_host[0, 0] = UInt32(0)
    lut_host[0, 2] = UInt32(_NULL_PAGE)
    lut.to_device()

    var cache_lengths = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    cache_lengths.host_tensor()[0] = UInt32(page_size)
    cache_lengths.to_device()

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    # The cache view covers every page but the canary, so the store through the
    # out-of-pool cell would land on it.
    var collection = Collection(
        rebind[Collection.blocks_tt_type](
            blocks.device_tensor()
            .tile(
                Coord(
                    Int64(_CACHE_PAGES),
                    Idx[2],
                    Int64(1),
                    Idx[page_size],
                    Idx[kv_params.num_heads],
                    Idx[kv_params.head_size],
                ),
                Coord(0, 0, 0, 0, 0, 0),
            )
            .as_unsafe_any_origin()
        ),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(page_size),
        UInt32(page_size),
    )
    var key_cache = collection.get_key_cache(0)

    ctx.enqueue_function[
        _store_through_lut_kernel[type_of(key_cache), page_size]
    ](
        key_cache,
        grid_dim=1,
        block_dim=1,
    )
    ctx.synchronize()

    blocks.to_host()
    var host = blocks.host_tensor()
    var written = 0
    for page in range(_STORAGE_PAGES):
        for plane in range(2):
            for tok in range(page_size):
                for head in range(kv_params.num_heads):
                    for dim in range(kv_params.head_size):
                        var value = Float32(
                            host[page, plane, 0, tok, head, dim]
                        )
                        if not isnan(value):
                            written += 1

    if Float32(host[0, 0, 0, 0, 0, 0]) != 1.0:
        raise Error(
            (
                "the store through an ordinary LUT cell wrote nothing: expected"
                " 1, got"
            ),
            Float32(host[0, 0, 0, 0, 0, 0]),
        )
    if Float32(host[_NULL_PAGE, 0, 0, 0, 0, 0]) != 3.0:
        raise Error(
            (
                "the store through the pool's last page (the null page) wrote"
                " nothing: expected 3, got"
            ),
            Float32(host[_NULL_PAGE, 0, 0, 0, 0, 0]),
        )
    if not isnan(Float32(host[_CANARY_PAGE, 0, 0, 0, 0, 0])):
        raise Error(
            (
                "the store through an out-of-pool LUT cell should have been"
                " skipped, but the page past the pool holds"
            ),
            Float32(host[_CANARY_PAGE, 0, 0, 0, 0, 0]),
        )
    if written != 2:
        raise Error(
            "expected the two in-pool stores to write, and only those;",
            written,
            "elements differ from the fill value",
        )


def run_scale_guard_test[
    kv_params: KVCacheStaticParams,
    page_size: Int,
    scale_granularity: Int,
](ctx: DeviceContext) raises:
    """Checks that `store_scale` drops only what its scale pool cannot hold.

    The scale view is one page shorter than the value pool here (five pages
    against the pool's six), so the same out-of-pool LUT cell resolves past the
    scales but not past the blocks: a bound computed from the value pool would
    not skip it, and would land on the scale canary page.

    Parameters:
        kv_params: Static shapes of the cache.
        page_size: Tokens per page.
        scale_granularity: Elements of a head that share one scale.

    Args:
        ctx: The device context to allocate on.
    """
    comptime head_dim_granularity = ceildiv(
        kv_params.head_size - kv_params.fp8_tail_size, scale_granularity
    )

    # `store_scale` resolves through the value pool's LUT unless the collection
    # is given a scale LUT of its own. Nothing writes the value pool here; it
    # only has to exist for the collection. It is deliberately left covering
    # all six pages: tiled to the scale view's five, an out-of-pool id would
    # skip under a bound taken from the value pool too, and this arm would stop
    # catching a revert to that bound.
    var blocks = HostDeviceTileTensor[DType.bfloat16](
        row_major(
            Int64(_STORAGE_PAGES),
            Idx[2],
            Int64(1),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
        ctx,
    )
    _ = blocks.host_tensor().fill(0)
    blocks.to_device()

    var scales = HostDeviceTileTensor[DType.float32](
        row_major(
            Int64(_STORAGE_PAGES),
            Idx[2],
            Int64(1),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[head_dim_granularity],
        ),
        ctx,
    )
    _ = scales.host_tensor().fill(_UNWRITTEN_SCALE)
    scales.to_device()

    var lut_columns = padded_lut_cols(_CACHE_PAGES + 4)
    var lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(1), Int64(lut_columns)), ctx
    )
    var lut_host = lut.host_tensor()
    for c in range(lut_columns):
        lut_host[0, c] = UInt32(_CACHE_PAGES)
    lut_host[0, 0] = UInt32(0)
    lut.to_device()

    var cache_lengths = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    cache_lengths.host_tensor()[0] = UInt32(page_size)
    cache_lengths.to_device()

    comptime Collection = PagedKVCacheCollection[
        DType.bfloat16,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=scale_granularity,
    ]
    # The scale view covers every page but the canary, so the store through the
    # out-of-pool cell would land on it.
    var scales_view = OptionalReg[Collection.scales_tt_type](
        rebind[Collection.scales_tt_type](
            scales.device_tensor()
            .tile(
                Coord(
                    Int64(_CACHE_PAGES),
                    Idx[2],
                    Int64(1),
                    Idx[page_size],
                    Idx[kv_params.num_heads],
                    Idx[head_dim_granularity],
                ),
                Coord(0, 0, 0, 0, 0, 0),
            )
            .as_unsafe_any_origin()
        )
    )
    var collection = Collection(
        rebind[Collection.blocks_tt_type](
            blocks.device_tensor().as_unsafe_any_origin()
        ),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(page_size),
        UInt32(page_size),
        scales_view,
    )
    var key_cache = collection.get_key_cache(0)

    ctx.enqueue_function[
        _store_scale_through_lut_kernel[type_of(key_cache), page_size]
    ](
        key_cache,
        grid_dim=1,
        block_dim=1,
    )
    ctx.synchronize()

    scales.to_host()
    var host = scales.host_tensor()
    var written = 0
    for page in range(_STORAGE_PAGES):
        for plane in range(2):
            for tok in range(page_size):
                for head in range(kv_params.num_heads):
                    for dim in range(head_dim_granularity):
                        var value = host[page, plane, 0, tok, head, dim]
                        if not isnan(value):
                            written += 1

    if host[0, 0, 0, 0, 0, 0] != 5.0:
        raise Error(
            (
                "the scale store through an ordinary LUT cell wrote nothing:"
                " expected 5, got"
            ),
            host[0, 0, 0, 0, 0, 0],
        )
    if not isnan(host[_CANARY_PAGE, 0, 0, 0, 0, 0]):
        raise Error(
            (
                "the scale store through an out-of-pool LUT cell should have"
                " been skipped, but the scale page past the pool holds"
            ),
            host[_CANARY_PAGE, 0, 0, 0, 0, 0],
        )
    if written != 1:
        raise Error(
            "expected the one in-pool scale store to write, and only it;",
            written,
            "scale elements differ from the fill value",
        )


def main() raises:
    comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=64)

    with DeviceContext() as ctx:
        run_store_guard_test[DType.bfloat16, kv_params, 128](ctx)
        run_scale_guard_test[kv_params, 128, 32](ctx)
        print("paged store LUT guard: passed")
