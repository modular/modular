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
"""A split paged KV cache: one plane whose rows pack a key and a value.

SM100 (B200). Shapes are MiMo-V2.6-Flash's: a 192-wide key and a 128-wide
value in a 320-wide row, 128-token pages, 1 to 8 KV heads per device.

Three properties, each bitwise:

- The element API round-trips. The key view writes columns `0..191`, the
  value view writes `v_col_base() + d`, and the result is the page the host
  builds by hand. A value written at its own column would land on the key.
- A TMA copy at a column base reads the columns it names. With no swizzle the
  shared-memory tile is checked against the host page directly; with the
  production 128B swizzle it is checked against the same copy from the padded
  256-wide cache the split cache replaces, for the key's 128-column main part,
  its 64-column tail and the value.
- The static params a graph op builds for one plane are a split cache only
  when the op names a value width, and an MLA latent otherwise.
"""

from std.math import align_up, ceildiv
from std.sys import size_of

from max.gpu import block_dim, global_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from kv_cache.types import KVCacheStaticParams, KVCacheT, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tma_async import SharedMemBarrier, TMATensorTile
from nn.attention.gpu.nvidia.sm100.attention_utils import elect
from nn.kv_cache import _paged_cache_static_params
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal, assert_false, assert_true

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime dtype = DType.bfloat16
comptime page_size = 128
comptime k_dim = 192
comptime v_dim = 128
comptime row_dim = k_dim + v_dim
comptime padded_dim = 256
comptime num_layers = 2
# A nonzero layer, so a view that ignored the layer offset reads zeros.
comptime layer = 1
comptime num_blocks = 20


def _split_params(num_heads: Int) -> KVCacheStaticParams:
    return KVCacheStaticParams(num_heads, row_dim, v_head_size=v_dim)


def _k_val[dt: DType = dtype](b: Int, h: Int, t: Int, d: Int) -> Scalar[dt]:
    # Multiples of 1/4 below 32 in magnitude: exact in bfloat16.
    return Scalar[dt](Float32((b * 31 + h * 17 + t * 7 + d) % 239 - 119) * 0.25)


def _v_val[dt: DType = dtype](b: Int, h: Int, t: Int, d: Int) -> Scalar[dt]:
    return Scalar[dt](
        Float32((b * 13 + h * 29 + t * 11 + d * 3) % 241 - 120) * -0.5
    )


def test_static_params() raises:
    comptime split = _split_params(4)
    assert_true(split.is_split())
    assert_equal(split.num_planes(), 1)
    assert_equal(split.k_head_size(), k_dim)
    assert_equal(split.v_col_base(), k_dim)

    comptime unsplit = KVCacheStaticParams(num_heads=4, head_size=128)
    assert_false(unsplit.is_split())
    assert_equal(unsplit.num_planes(), 2)
    assert_equal(unsplit.k_head_size(), 128)
    assert_equal(unsplit.v_col_base(), 0)

    comptime mla = KVCacheStaticParams(num_heads=1, head_size=576, is_mla=True)
    assert_equal(mla.num_planes(), 1)
    assert_equal(mla.k_head_size(), 576)


def test_graph_op_params() raises:
    """One plane is a split cache only when the op names a value width."""
    assert_true(
        _paged_cache_static_params(4, row_dim, 1, v_dim) == _split_params(4)
    )
    var latent = _paged_cache_static_params(1, 576, 1, -1)
    assert_true(latent.is_mla)
    assert_false(latent.is_split())
    var pair = _paged_cache_static_params(4, 128, 2, -1)
    assert_false(pair.is_mla)
    assert_equal(pair.num_planes(), 2)


def _store_kernel[
    cache_t: KVCacheT
](k_cache: cache_t, v_cache: cache_t, batch: Int32, tokens: Int32):
    comptime kv_params = cache_t.kv_params
    var batch_idx = Int(batch)
    var num_tokens = Int(tokens)
    var idx = global_idx.x
    if idx >= num_tokens * kv_params.num_heads:
        return
    var t, h = divmod(idx, kv_params.num_heads)
    for d in range(kv_params.k_head_size()):
        k_cache.store(
            batch_idx, h, t, d, _k_val[cache_t.dtype](batch_idx, h, t, d)
        )
    for d in range(kv_params.v_head_size):
        v_cache.store(
            batch_idx,
            h,
            t,
            kv_params.v_col_base() + d,
            _v_val[cache_t.dtype](batch_idx, h, t, d),
        )


def _load_kernel[
    cache_t: KVCacheT
](
    k_cache: cache_t,
    v_cache: cache_t,
    batch: Int32,
    tokens: Int32,
    k_out: MutPointer[Scalar[cache_t.dtype], MutAnyOrigin],
    v_out: MutPointer[Scalar[cache_t.dtype], MutAnyOrigin],
):
    """Writes the request's key and value rows densely, `[t, h, d]`."""
    comptime kv_params = cache_t.kv_params
    comptime k_width = kv_params.k_head_size()
    comptime v_width = kv_params.v_head_size
    var batch_idx = Int(batch)
    var num_tokens = Int(tokens)
    var idx = global_idx.x
    if idx >= num_tokens * kv_params.num_heads:
        return
    var t, h = divmod(idx, kv_params.num_heads)
    for d in range(k_width):
        k_out[unsafe_offset=idx * k_width + d] = k_cache.load[width=1](
            batch_idx, h, t, d
        )
    for d in range(v_width):
        v_out[unsafe_offset=idx * v_width + d] = v_cache.load[width=1](
            batch_idx, h, t, kv_params.v_col_base() + d
        )


@__llvm_arg_metadata(tma, `nvvm.grid_constant`)
def _tma_tile_kernel[
    cache_t: KVCacheT,
    tile_shape: Coord,
    desc_shape: Coord,
    BN: Int,
    BK: Int,
](
    tma: TMATensorTile[cache_t.dtype, tile_shape, desc_shape, True],
    cache: cache_t,
    batch_idx: UInt32,
    start_tok: UInt32,
    head_idx: UInt32,
    depth_offset: UInt32,
    dst: MutPointer[Scalar[cache_t.dtype], MutAnyOrigin],
):
    """Copies one `BN x BK` tile through the production paged issue path and
    writes the shared-memory bytes out as they landed."""
    var smem = external_memory[
        Scalar[cache_t.dtype],
        address_space=.SHARED,
        alignment=1024,
        name="split_head_tma_smem",
    ]()
    var mbar = unsafe_stack_allocation[
        1, SharedMemBarrier, alignment=8, address_space=.SHARED
    ]()
    var e = elect()
    if thread_idx.x == 0:
        mbar[].init(1)
        mbar[].expect_bytes(Int32(BN * BK * size_of[cache_t.dtype]()))
    barrier()

    var rows = cache.populate[BN, BN](batch_idx, start_tok)
    rows.tma_copy_k[needs_partial=False](
        tma,
        smem,
        mbar[],
        kv_head_idx=head_idx,
        elect=e,
        depth_offset=depth_offset,
    )
    mbar[].wait(0)
    for i in range(thread_idx.x, BN * BK, block_dim.x):
        dst[unsafe_offset=i] = smem[unsafe_offset=i]


def _copy_tile[
    cache_t: KVCacheT, //, swizzle: TensorMapSwizzle, BN: Int, BK: Int
](
    ctx: DeviceContext,
    cache: cache_t,
    batch_idx: Int,
    start_tok: Int,
    head_idx: Int,
    depth_offset: Int,
) raises -> List[Scalar[cache_t.dtype]]:
    comptime box_rows = min(BN, page_size)
    var tma = cache.create_tma_tile[swizzle, BN=box_rows, BK=BK](ctx)
    var out = ctx.enqueue_create_buffer[cache_t.dtype](BN * BK)
    comptime smem_bytes = BN * BK * size_of[cache_t.dtype]()
    comptime kernel = _tma_tile_kernel[
        cache_t, type_of(tma).tile_shape, type_of(tma).desc_shape, BN, BK
    ]
    ctx.enqueue_function[kernel](
        tma,
        cache,
        UInt32(batch_idx),
        UInt32(start_tok),
        UInt32(head_idx),
        UInt32(depth_offset),
        out.unsafe_ptr(),
        # One warp: every warp's elected lane would issue the copy.
        grid_dim=1,
        block_dim=32,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )
    var host = ctx.enqueue_create_host_buffer[cache_t.dtype](BN * BK)
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    var result = List[Scalar[cache_t.dtype]](capacity=BN * BK)
    for i in range(BN * BK):
        result.append(host[i])
    return result^


def _assert_bitwise_equal[
    dtype: DType
](got: List[Scalar[dtype]], expected: List[Scalar[dtype]], what: String) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        if got[i].to_bits() != expected[i].to_bits():
            raise Error(
                t"{what}: element {i} is {got[i]}, expected {expected[i]}"
            )


def test_split_cache[num_heads: Int](ctx: DeviceContext) raises:
    comptime split_params = _split_params(num_heads)
    comptime padded_params = KVCacheStaticParams(num_heads, padded_dim)
    comptime SplitCollection = PagedKVCacheCollection[
        dtype,
        split_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime PaddedCollection = PagedKVCacheCollection[
        dtype,
        padded_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    # Whole pages, a page boundary inside a request, and a partial last page.
    var lengths: List[Int] = [256, 600, 1000]
    var no_prompt: List[Int] = [0, 0, 0]
    var batch_size = len(lengths)
    var table = CacheLengthsTable.build(no_prompt, lengths, ctx)
    var max_length = 1000
    var lut = PagedLookupTable[page_size].build(
        no_prompt, lengths, max_length, num_blocks, ctx
    )
    var lut_host = lut.host_tensor()
    comptime assert lut_host.flat_rank == 2

    var split_layout = row_major(
        Int64(num_blocks),
        Idx[1],
        Int64(num_layers),
        Idx[page_size],
        Idx[num_heads],
        Idx[row_dim],
    )
    var expected = HostDeviceTileTensor[dtype](split_layout, ctx)
    var stored = HostDeviceTileTensor[dtype](split_layout, ctx)
    var padded = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_heads],
            Idx[padded_dim],
        ),
        ctx,
    )
    var expected_host = expected.host_tensor().fill(0)
    _ = stored.host_tensor().fill(0)
    var padded_host = padded.host_tensor().fill(0)
    comptime assert expected_host.flat_rank == 6
    comptime assert padded_host.flat_rank == 6
    for b in range(batch_size):
        for t in range(lengths[b]):
            var page, slot = divmod(t, page_size)
            var block = Int(lut_host[b, page])
            for h in range(num_heads):
                for d in range(k_dim):
                    expected_host[block, 0, layer, slot, h, d] = _k_val(
                        b, h, t, d
                    )
                    padded_host[block, 0, layer, slot, h, d] = _k_val(
                        b, h, t, d
                    )
                for d in range(v_dim):
                    expected_host[block, 0, layer, slot, h, k_dim + d] = _v_val(
                        b, h, t, d
                    )
                    padded_host[block, 1, layer, slot, h, d] = _v_val(
                        b, h, t, d
                    )
    expected.to_device()
    stored.to_device()
    padded.to_device()

    # The element API writes exactly the host-built page.
    var store_into = SplitCollection(
        stored.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(0),
        UInt32(max_length),
    )
    var k_store = store_into.get_key_cache(layer)
    var v_store = store_into.get_value_cache(layer)
    for b in range(batch_size):
        ctx.enqueue_function[_store_kernel[type_of(k_store)]](
            k_store,
            v_store,
            Int32(b),
            Int32(lengths[b]),
            grid_dim=ceildiv(lengths[b] * num_heads, 128),
            block_dim=128,
        )
    ctx.synchronize()
    stored.to_host()
    var stored_host = stored.host_tensor()
    comptime assert stored_host.flat_rank == 6
    for blk in range(num_blocks):
        for l in range(num_layers):
            for s in range(page_size):
                for h in range(num_heads):
                    for c in range(row_dim):
                        var got = stored_host[blk, 0, l, s, h, c]
                        var want = expected_host[blk, 0, l, s, h, c]
                        if got.to_bits() != want.to_bits():
                            raise Error(
                                t"store: block {blk} layer {l} slot {s}"
                                t" head {h} col {c} is {got}, expected {want}"
                            )

    # The element API reads the host-built page back.
    var read_from = SplitCollection(
        expected.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(0),
        UInt32(max_length),
    )
    var k_read = read_from.get_key_cache(layer)
    var v_read = read_from.get_value_cache(layer)
    for b in range(batch_size):
        var n = lengths[b] * num_heads
        var k_out = ctx.enqueue_create_buffer[dtype](n * k_dim)
        var v_out = ctx.enqueue_create_buffer[dtype](n * v_dim)
        ctx.enqueue_function[_load_kernel[type_of(k_read)]](
            k_read,
            v_read,
            Int32(b),
            Int32(lengths[b]),
            k_out.unsafe_ptr(),
            v_out.unsafe_ptr(),
            grid_dim=ceildiv(n, 128),
            block_dim=128,
        )
        var k_host = ctx.enqueue_create_host_buffer[dtype](n * k_dim)
        var v_host = ctx.enqueue_create_host_buffer[dtype](n * v_dim)
        ctx.enqueue_copy(k_host, k_out)
        ctx.enqueue_copy(v_host, v_out)
        ctx.synchronize()
        for t in range(lengths[b]):
            for h in range(num_heads):
                var i = t * num_heads + h
                for d in range(k_dim):
                    assert_equal(
                        k_host[i * k_dim + d].to_bits(),
                        _k_val(b, h, t, d).to_bits(),
                    )
                for d in range(v_dim):
                    assert_equal(
                        v_host[i * v_dim + d].to_bits(),
                        _v_val(b, h, t, d).to_bits(),
                    )

    # TMA copies at the key's, its tail's and the value's column bases.
    var padded_collection = PaddedCollection(
        padded.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(0),
        UInt32(max_length),
    )
    var padded_k = padded_collection.get_key_cache(layer)
    var padded_v = padded_collection.get_value_cache(layer)
    comptime k_main = 128
    comptime k_tail = k_dim - k_main
    comptime v_base = split_params.v_col_base()
    comptime sw128 = TensorMapSwizzle.SWIZZLE_128B
    comptime no_sw = TensorMapSwizzle.SWIZZLE_NONE

    comptime for pages_per_tile in range(1, 3):
        comptime BN = pages_per_tile * page_size
        for b in range(batch_size):
            # Whole tiles over allocated pages only: the LUT past them is
            # uninitialized.
            var allocated = align_up(lengths[b], page_size)
            for start in range(0, allocated - BN + 1, BN):
                for h in range(num_heads):
                    # Production swizzle, against the padded cache's planes.
                    _assert_bitwise_equal(
                        _copy_tile[sw128, BN, k_main](
                            ctx, k_read, b, start, h, 0
                        ),
                        _copy_tile[sw128, BN, k_main](
                            ctx, padded_k, b, start, h, 0
                        ),
                        "key main tile",
                    )
                    _assert_bitwise_equal(
                        _copy_tile[sw128, BN, k_tail](
                            ctx, k_read, b, start, h, k_main
                        ),
                        _copy_tile[sw128, BN, k_tail](
                            ctx, padded_k, b, start, h, k_main
                        ),
                        "key tail tile",
                    )
                    _assert_bitwise_equal(
                        _copy_tile[sw128, BN, v_dim](
                            ctx, v_read, b, start, h, v_base
                        ),
                        _copy_tile[sw128, BN, v_dim](
                            ctx, padded_v, b, start, h, 0
                        ),
                        "value tile",
                    )

        # Unswizzled, the tile is `[chunk][row][gran]` and checkable against
        # the host page. One request, every head, the value's columns.
        comptime gran = no_sw.bytes() // size_of[dtype]()
        var b = batch_size - 1
        for h in range(num_heads):
            var tile = _copy_tile[no_sw, BN, v_dim](
                ctx, v_read, b, BN, h, v_base
            )
            for j in range(v_dim // gran):
                for r in range(BN):
                    var t = BN + r
                    for c in range(gran):
                        var d = j * gran + c
                        var want = _v_val(b, h, t, d) if t < lengths[
                            b
                        ] else Scalar[dtype](0)
                        assert_equal(
                            tile[j * BN * gran + r * gran + c].to_bits(),
                            want.to_bits(),
                        )


def main() raises:
    test_static_params()
    test_graph_op_params()
    with DeviceContext() as ctx:
        test_split_cache[1](ctx)
        test_split_cache[2](ctx)
        test_split_cache[4](ctx)
        test_split_cache[8](ctx)
