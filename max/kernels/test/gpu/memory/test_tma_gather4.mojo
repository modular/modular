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
"""Tests for TMA gather4 instruction (SM100/Blackwell).

Test groups:

1. raw smoke test -- a single `cp_async_bulk_tensor_2d_gather4` call
   to verify the intrinsic works (4 rows, one dtype, minimal code).

2. While-loop tests using actual KV cache types -- constructs real
   `PagedKVCacheCollection` / `ContinuousBatchingKVCacheCollection` objects,
   calls `kv_cache.create_gather4_tma_tile[row_width](ctx)` to create the
   TMA descriptor, and uses `kv_cache.row_idx(batch, tok)` to compute
   physical row indices for gather4.

   Variants:
   - Paged BF16 (PagedKVCacheCollection)
   - Paged FP8 (PagedKVCacheCollection, float8_e4m3fn)
   - Paged INT64-packed FP8 (PagedKVCacheCollection with dtype=int64,
     simulating the DeepSeek V3.2 576-byte row packing)
   - Contiguous BF16 (ContinuousBatchingKVCacheCollection)

Note: Production dispatch code accesses gather4 through the MHAOperand trait
layer, i.e. ``k.create_gather4_tma_tile[row_width](ctx)`` where ``k`` is a
``KVCacheMHAOperand``, ``LayoutTensorMHAOperand``, or ``RaggedMHAOperand``.
The MHAOperand implementations delegate to the underlying cache or buffer.
See ``nn/mha_operand.mojo`` for the trait definition and implementations.
"""

from std.math import ceildiv
from std.sys.info import size_of

from std.gpu import block_dim_uint as block_dim, thread_idx_uint as thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from std.gpu.memory import (
    AddressSpace,
    cp_async_bulk_tensor_2d_gather4,
    external_memory,
)
from std.gpu.sync import (
    barrier,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
)
from std.memory import alloc, stack_allocation
from std.random import rand, seed
from std.utils.index import IndexList

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile_gather4,
)
from layout.swizzle import make_swizzle
from nn.attention.mha_operand import KVCacheMHAOperand
from std.testing import assert_equal


# ===========================================================================
# GPU kernels
# ===========================================================================


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
def gather4_raw_smoke_kernel[
    dtype: DType, row_width: Int
](
    descriptor: TMADescriptor,
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    row0: Int32,
    row1: Int32,
    row2: Int32,
    row3: Int32,
):
    """Minimal kernel: single gather4 via raw intrinsic, 4 rows."""
    var shmem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]()

    var mbar = stack_allocation[1, Int64, address_space=AddressSpace.SHARED]()
    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    if thread_idx.x == 0:
        var expected_bytes = 4 * row_width * size_of[dtype]()
        mbarrier_arrive_expect_tx_shared(mbar, Int32(expected_bytes))
        cp_async_bulk_tensor_2d_gather4(
            shmem,
            descriptor_ptr,
            mbar,
            col_idx=Int32(0),
            row0=row0,
            row1=row1,
            row2=row2,
            row3=row3,
        )

    mbarrier_try_wait_parity_shared(mbar, 0, 10_000_000)
    barrier()

    var total_elems = 4 * row_width
    var tid = Int(thread_idx.x)
    var num_threads = Int(block_dim.x)
    for i in range(tid, total_elems, num_threads):
        d_out[i] = shmem[i]


# ===========================================================================
# Host-side verification
# ===========================================================================


def _verify_gathered_rows[
    dtype: DType, row_width: Int
](
    h_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    h_source: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    h_indices: UnsafePointer[Int32, MutAnyOrigin],
    num_rows: Int,
) raises:
    """Verifies gathered output rows match the source buffer at the expected
    physical row offsets."""
    for gather_idx in range(num_rows):
        var src_row = Int(h_indices[gather_idx])
        for col in range(row_width):
            var got = h_out[gather_idx * row_width + col]
            var expected = h_source[src_row * row_width + col]
            assert_equal(
                got.cast[DType.float32](),
                expected.cast[DType.float32](),
                msg=String(
                    "Mismatch at gathered row ",
                    gather_idx,
                    " (source row ",
                    src_row,
                    "), col ",
                    col,
                ),
            )


# ===========================================================================
# Host-side test drivers
# ===========================================================================


def test_raw_smoke[
    dtype: DType, row_width: Int, num_tokens: Int
](ctx: DeviceContext) raises:
    """Single gather4 via raw intrinsic -- minimal smoke test."""
    print(
        "== test_raw_smoke [",
        dtype,
        ", row_width=",
        row_width,
        ", num_tokens=",
        num_tokens,
        "]",
    )

    var num_elems = num_tokens * row_width
    var h_data = alloc[Scalar[dtype]](num_elems)
    rand[dtype](h_data, num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    var tma_desc = create_tma_descriptor[dtype, 2](
        d_data,
        (num_tokens, row_width),
        (row_width, 1),
        (1, row_width),
    )

    var r0 = Int32(3)
    var r1 = Int32(min(500, num_tokens - 1))
    var r2 = Int32(17)
    var r3 = Int32(min(999, num_tokens - 1))

    var output_elems = 4 * row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var shared_mem_bytes = 4 * row_width * size_of[dtype]()

    comptime kernel = gather4_raw_smoke_kernel[dtype, row_width]
    ctx.enqueue_function[kernel, kernel](
        tma_desc,
        d_out,
        r0,
        r1,
        r2,
        r3,
        grid_dim=1,
        block_dim=128,
        shared_mem_bytes=shared_mem_bytes,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Build a 4-element index array for the unified verification helper.
    var h_indices = alloc[Int32](4)
    h_indices[0] = r0
    h_indices[1] = r1
    h_indices[2] = r2
    h_indices[3] = r3
    _verify_gathered_rows[dtype, row_width](h_out, h_data, h_indices, 4)
    print("  PASSED: all 4 gathered rows match expected data")

    h_data.free()
    h_out.free()
    h_indices.free()
    _ = d_data
    _ = d_out


def _run_paged_gather4_test[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    use_mha_operand: Bool,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Shared implementation for paged KV cache gather4 tests.

    When `use_mha_operand` is True, the TMA tile is created through
    KVCacheMHAOperand; otherwise it is created directly from the cache.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_heads),
        head_size=UInt(head_size),
    )
    comptime row_width = num_heads * head_size

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, kv_dim, num_layers, page_size, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    # Fill entire buffer with random data.
    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_len_layout
    ](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table with shuffled page assignments.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[DType.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache for layer 0.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Create the TMA tile -- either directly or through MHAOperand.
    # The tile type encodes box_width in tile_shape[1]; no need to
    # compute it separately.
    var kv_tile = kv_cache.create_gather4_tma_tile[row_width, swizzle_mode](ctx)
    comptime if use_mha_operand:
        var operand = KVCacheMHAOperand(kv_cache)
        kv_tile = operand.create_gather4_tma_tile[row_width, swizzle_mode](ctx)

    # Build gather indices on host.
    # Physical row = phys_block * stride + offset_in_page
    # where stride = kv_dim * num_layers * page_size.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify by reading expected data directly from the host blocks buffer.
    # The 6D tensor is row-major, so for kv_idx=0, layer_idx=0:
    #   offset(block, tok, h, d) = block*s0 + tok*s3 + h*s4 + d
    # which equals phys_row * row_width + col.
    _verify_gathered_rows[dtype, row_width](
        h_out, blocks_host.ptr, h_indices, topk
    )

    comptime if use_mha_operand:
        print(
            "  PASSED: all",
            topk,
            "rows from KVCacheMHAOperand (",
            topk // 4,
            "iterations, page_size=",
            page_size,
            ") match",
        )
    else:
        print(
            "  PASSED: all",
            topk,
            "rows from PagedKVCache (",
            topk // 4,
            "iterations, page_size=",
            page_size,
            ") match",
        )

    h_indices.free()
    h_out.free()
    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_paged_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 with a real PagedKVCacheCollection."""
    _run_paged_gather4_test[
        dtype,
        num_heads,
        head_size,
        page_size,
        num_blocks,
        num_layers,
        batch_size,
        tokens_per_seq,
        topk,
        kv_dim,
        use_mha_operand=False,
        swizzle_mode=swizzle_mode,
    ](ctx)


def test_continuous_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    max_seq_len: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 with a real ContinuousBatchingKVCacheCollection."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_heads), head_size=UInt(head_size)
    )
    comptime row_width = num_heads * head_size

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, max_seq_len, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    # Fill entire buffer with random data.
    var block_elems = (
        num_blocks * 2 * num_layers * max_seq_len * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_len_layout
    ](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table (1D: one block per batch entry).
    var lookup_managed = ManagedLayoutTensor[DType.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_host = lookup_managed.tensor[update=False]()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)

    # Construct the ContinuousBatchingKVCacheCollection.
    var collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lookup_managed.device_tensor(),
        UInt32(max_seq_len),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)
    var kv_tile = kv_cache.create_gather4_tma_tile[row_width, swizzle_mode](ctx)

    # Build gather indices on host.
    # Physical row = block_id * stride + tok_idx
    # where stride = 2 * num_layers * max_seq_len.
    comptime cont_stride = 2 * num_layers * max_seq_len
    var h_indices = alloc[Int32](topk)
    var lookup_ptr = lookup_host.ptr
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var block_id = Int(lookup_ptr[seq_idx])
        var phys_row = block_id * cont_stride + tok_idx
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, row_width](
        h_out, blocks_host.ptr, h_indices, topk
    )
    print(
        "  PASSED: all",
        topk,
        "rows from ContinuousBatchingKVCache (",
        topk // 4,
        "iterations) match",
    )

    h_indices.free()
    h_out.free()
    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lookup_managed


def test_device_buffer_overload[
    dtype: DType, row_width: Int, num_tokens: Int, topk: Int
](ctx: DeviceContext) raises:
    """Tests the DeviceBuffer overload of create_tma_tile_gather4."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    print(
        "== test_device_buffer_overload [",
        dtype,
        ", row_width=",
        row_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    var num_elems = num_tokens * row_width
    var h_data = alloc[Scalar[dtype]](num_elems)
    rand[dtype](h_data, num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Use the DeviceBuffer overload of create_tma_tile_gather4.
    var kv_tile = create_tma_tile_gather4[dtype, row_width](
        ctx, d_data, num_tokens
    )

    # Build gather indices on host.
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch while-loop kernel.
    var output_elems = topk * row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, row_width](h_out, h_data, h_indices, topk)
    print("  PASSED: all", topk, "rows from DeviceBuffer overload match")

    h_data.free()
    h_out.free()
    h_indices.free()
    _ = d_data
    _ = d_out
    _ = d_indices


def test_mha_operand_gather4[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 through the MHAOperand trait via KVCacheMHAOperand."""
    _run_paged_gather4_test[
        dtype,
        num_heads,
        head_size,
        page_size,
        num_blocks,
        num_layers,
        batch_size,
        tokens_per_seq,
        topk,
        kv_dim,
        use_mha_operand=True,
        swizzle_mode=swizzle_mode,
    ](ctx)


@__llvm_arg_metadata(kv_tile, `nvvm.grid_constant`)
def gather4_kernel[
    dtype: DType,
    global_row_width: Int,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    desc_shape_param: IndexList[tile_rank],
    swizzle_mode: TensorMapSwizzle,
](
    kv_tile: TMATensorTile[
        dtype, tile_rank, tile_shape_param, desc_shape_param
    ],
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    num_tiles: Int32,
):
    """While-loop gather4 loader using the level 3 TMATensorTile API.

    Handles both narrow (global_row_width == box_width, loop=1) and wide
    (global_row_width > box_width, loop>1) rows via a comptime for-loop
    over column groups.  The box width and number of column groups are
    derived from the tile's compile-time shape ``tile_shape_param[1]``.
    """
    comptime box_width = tile_shape_param[1]
    comptime num_col_groups = ceildiv(global_row_width, box_width)
    comptime smem_layout = Layout.row_major(4, box_width)
    var smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space=AddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init(1)
        var desc_ptr = UnsafePointer(to=kv_tile.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)
    barrier()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    var tid = Int(thread_idx.x)
    var num_threads = Int(block_dim.x)
    var phase = UInt32(0)

    var tile_idx: Int32 = 0
    while tile_idx < num_tiles:
        var idx_base = Int(tile_idx) * 4
        var row0 = d_indices[idx_base + 0]
        var row1 = d_indices[idx_base + 1]
        var row2 = d_indices[idx_base + 2]
        var row3 = d_indices[idx_base + 3]

        # Load each column group separately.
        comptime for cg in range(num_col_groups):
            if thread_idx.x == 0:
                mbar[0].expect_bytes(Int32(4 * box_width * size_of[dtype]()))
                kv_tile.async_copy_gather4(
                    smem_tile,
                    mbar[0],
                    col_idx=Int32(cg * box_width),
                    row0=row0,
                    row1=row1,
                    row2=row2,
                    row3=row3,
                )

            mbar[0].wait(phase)

            # Copy from SMEM to output, applying de-swizzle.
            # Output is laid out as [tile_idx * 4 rows, global_row_width].
            var out_tile_base = Int(tile_idx) * 4 * global_row_width
            for i in range(tid, 4 * box_width, num_threads):
                var row_in_tile = i // box_width
                var col_in_group = i % box_width
                var out_idx = (
                    out_tile_base
                    + row_in_tile * global_row_width
                    + cg * box_width
                    + col_in_group
                )
                d_out[out_idx] = smem_tile.ptr[Int(swizzle(i))]

            barrier()
            phase ^= 1

        tile_idx += 1


def test_wide_gather4_device_buffer[
    dtype: DType,
    global_row_width: Int,
    num_tokens: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 with DeviceBuffer: global_row_width > box_width."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    print(
        "== test_wide_gather4_device_buffer [",
        dtype,
        ", global_row_width=",
        global_row_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    var num_elems = num_tokens * global_row_width
    var h_data = alloc[Scalar[dtype]](num_elems)
    rand[dtype](h_data, num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Create the TMA tile -- box_width is encoded in tile_shape[1].
    var kv_tile = create_tma_tile_gather4[
        dtype, global_row_width, swizzle_mode
    ](ctx, d_data, num_tokens)

    # Build gather indices.
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * global_row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        global_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify: output row i should match source row h_indices[i].
    _verify_gathered_rows[dtype, global_row_width](
        h_out, h_data, h_indices, topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows match expected data",
    )

    h_data.free()
    h_out.free()
    h_indices.free()
    _ = d_data
    _ = d_out
    _ = d_indices


def test_wide_gather4_paged_kv[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through PagedKVCache.create_gather4_tma_tile."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_heads),
        head_size=UInt(head_size),
    )
    comptime global_row_width = num_heads * head_size

    print(
        "== test_wide_gather4_paged_kv [",
        dtype,
        ", global_row_width=",
        global_row_width,
        "]",
    )

    # Build the 6D blocks tensor.
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_len_layout
    ](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[DType.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Use the unified gather4 API (wide rows handled automatically).
    var kv_tile = kv_cache.create_gather4_tma_tile[
        global_row_width, swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * global_row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        global_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, global_row_width](
        h_out, blocks_host.ptr, h_indices, topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from PagedKVCache match",
    )

    h_indices.free()
    h_out.free()
    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_wide_gather4_continuous_kv[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    max_seq_len: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through ContinuousBatchingKVCache.create_gather4_tma_tile.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_heads),
        head_size=UInt(head_size),
    )
    comptime global_row_width = num_heads * head_size

    print(
        "== test_wide_gather4_continuous_kv [",
        dtype,
        ", global_row_width=",
        global_row_width,
        "]",
    )

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, max_seq_len, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * 2 * num_layers * max_seq_len * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_len_layout
    ](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table (1D: one block per batch entry).
    var lookup_managed = ManagedLayoutTensor[DType.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_host = lookup_managed.tensor[update=False]()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)

    # Construct the ContinuousBatchingKVCacheCollection.
    var collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lookup_managed.device_tensor(),
        UInt32(max_seq_len),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Use the unified gather4 API (wide rows handled automatically).
    var kv_tile = kv_cache.create_gather4_tma_tile[
        global_row_width, swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime cont_stride = 2 * num_layers * max_seq_len
    var h_indices = alloc[Int32](topk)
    var lookup_ptr = lookup_host.ptr
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var block_id = Int(lookup_ptr[seq_idx])
        var phys_row = block_id * cont_stride + tok_idx
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * global_row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        global_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, global_row_width](
        h_out, blocks_host.ptr, h_indices, topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from ContinuousBatchingKVCache match",
    )

    h_indices.free()
    h_out.free()
    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lookup_managed


def test_wide_gather4_mha_operand[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through KVCacheMHAOperand wrapping PagedKVCache."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_heads),
        head_size=UInt(head_size),
    )
    comptime global_row_width = num_heads * head_size

    print(
        "== test_wide_gather4_mha_operand [",
        dtype,
        ", global_row_width=",
        global_row_width,
        "]",
    )

    # Build the 6D blocks tensor.
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_len_layout
    ](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[DType.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Create TMA tile through MHAOperand trait.
    var operand = KVCacheMHAOperand(kv_cache)
    var kv_tile = operand.create_gather4_tma_tile[
        global_row_width, swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * global_row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        global_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, global_row_width](
        h_out, blocks_host.ptr, h_indices, topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from KVCacheMHAOperand match",
    )

    h_indices.free()
    h_out.free()
    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_non_divisible_width[
    dtype: DType,
    global_row_width: Int,
    num_tokens: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests gather4 where global_row_width is not a multiple of box_width.

    TMA hardware zero-fills out-of-bounds elements in the last column group.
    We verify that in-bounds elements match and out-of-bounds elements are zero.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    # Allocate global memory with the actual (non-padded) row width.
    var num_elems = num_tokens * global_row_width
    var h_data = alloc[Scalar[dtype]](num_elems)
    rand[dtype](h_data, num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Create the TMA tile -- works even with non-divisible width.
    # box_width and num_col_groups are derived from the tile's type.
    var kv_tile = create_tma_tile_gather4[
        dtype, global_row_width, swizzle_mode
    ](ctx, d_data, num_tokens)

    comptime box_width = type_of(kv_tile).tile_shape[1]
    comptime num_col_groups = ceildiv(global_row_width, box_width)
    comptime padded_row_width = num_col_groups * box_width

    print(
        "== test_non_divisible_width [",
        dtype,
        ", global_row_width=",
        global_row_width,
        ", box_width=",
        box_width,
        ", num_col_groups=",
        num_col_groups,
        ", padded_row_width=",
        padded_row_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    # Build gather indices.
    var h_indices = alloc[Int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[DType.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Output uses padded_row_width so the kernel writes full column groups.
    var output_elems = topk * padded_row_width
    var h_out = alloc[Scalar[dtype]](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    # Use the gather4_kernel which iterates over column groups.
    # NOTE: we pass padded_row_width as global_row_width to the kernel so
    # that its output indexing accounts for the full padded layout.
    comptime kernel = gather4_kernel[
        dtype,
        padded_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel, kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify: for each gathered row, in-bounds columns must match source,
    # and out-of-bounds columns (global_row_width..padded_row_width) must
    # be zero (TMA zero-fill).
    for gather_idx in range(topk):
        var src_row = Int(h_indices[gather_idx])
        # Check in-bounds columns.
        for col in range(global_row_width):
            var got = h_out[gather_idx * padded_row_width + col]
            var expected = h_data[src_row * global_row_width + col]
            assert_equal(
                got.cast[DType.float32](),
                expected.cast[DType.float32](),
                msg=String(
                    "Mismatch at gathered row ",
                    gather_idx,
                    " (source row ",
                    src_row,
                    "), col ",
                    col,
                ),
            )
        # Check out-of-bounds columns are zero-filled by TMA.
        for col in range(global_row_width, padded_row_width):
            var got = h_out[gather_idx * padded_row_width + col]
            assert_equal(
                got.cast[DType.float32](),
                Scalar[DType.float32](0),
                msg=String(
                    "Expected zero at OOB col ",
                    col,
                    " of gathered row ",
                    gather_idx,
                ),
            )

    print(
        "  PASSED: all",
        topk,
        "rows verified (",
        num_col_groups,
        "col groups, last group zero-fills",
        padded_row_width - global_row_width,
        "elements)",
    )

    h_data.free()
    h_out.free()
    h_indices.free()
    _ = d_data
    _ = d_out
    _ = d_indices


# ===========================================================================
# Main
# ===========================================================================


def main() raises:
    seed()

    with DeviceContext() as ctx:
        # Level 2 raw smoke test.
        print("--- Kernel 1: Level 2 raw smoke test ---")
        test_raw_smoke[DType.bfloat16, 128, 1024](ctx)

        # DeviceBuffer overload of create_tma_tile_gather4.
        print(
            "\n--- DeviceBuffer overload of create_tma_tile_gather4:"
            " bfloat16 ---"
        )
        test_device_buffer_overload[
            DType.bfloat16,
            128,  # row_width
            1024,  # num_tokens
            64,  # topk
        ](ctx)

        # Paged KV cache tests.
        print("\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16 ---")
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16,"
            " kv_dim=2 ---"
        )
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            2,  # kv_dim (standard separate K/V layout)
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): float8_e4m3fn ---"
        )
        test_paged_kv_cache[
            DType.float8_e4m3fn,
            1,  # num_heads
            128,  # head_size (row_width = 128)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): int64-packed"
            " FP8 (72 int64 = 576 bytes) ---"
        )
        test_paged_kv_cache[
            DType.int64,
            1,  # num_heads
            72,  # head_size (72 INT64 = 576 bytes)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        # Contiguous KV cache test.
        print(
            "\n--- Contiguous KV Cache"
            " (ContinuousBatchingKVCacheCollection): bfloat16 ---"
        )
        test_continuous_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 256)
            512,  # max_seq_len
            4,  # num_blocks (one per batch entry)
            1,  # num_layers
            4,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
        ](ctx)

        # Swizzled paged KV cache tests (FlashMLA-style swizzle modes).
        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16,"
            " SWIZZLE_128B ---"
        )
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            64,  # head_size (row_width = 64, 128 bytes = SWIZZLE_128B)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): float8_e4m3fn,"
            " SWIZZLE_64B ---"
        )
        test_paged_kv_cache[
            DType.float8_e4m3fn,
            1,  # num_heads
            64,  # head_size (row_width = 64, 64 bytes = SWIZZLE_64B)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # MHAOperand trait layer test.
        print(
            "\n--- MHAOperand trait layer"
            " (KVCacheMHAOperand wrapping PagedKVCache): bfloat16 ---"
        )
        test_mha_operand_gather4[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        # Wide gather4 tests: row_width > box_width with swizzle.
        print(
            "\n--- Wide gather4: DeviceBuffer, bfloat16,"
            " global_row_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_device_buffer[
            DType.bfloat16,
            512,  # global_row_width
            1024,  # num_tokens
            16,  # topk (4 tiles of 4 rows)
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Wide gather4: DeviceBuffer, float8_e4m3fn,"
            " global_row_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_device_buffer[
            DType.float8_e4m3fn,
            512,  # global_row_width (512 bytes; box_width = 128)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Wide gather4: DeviceBuffer, bfloat16,"
            " global_row_width=256, SWIZZLE_64B ---"
        )
        test_wide_gather4_device_buffer[
            DType.bfloat16,
            256,  # global_row_width
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # Wide gather4 through PagedKVCache.
        print(
            "\n--- Wide gather4: PagedKVCache, bfloat16,"
            " global_row_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_paged_kv[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (row_width = 1*512 = 512)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # --- Test 1: BF16 + SWIZZLE_64B + row_width=64 (box=32, 2 col groups).
        print(
            "\n--- Paged KV Cache: bfloat16, SWIZZLE_64B,"
            " row_width=64 (box=32, 2 col groups) ---"
        )
        test_wide_gather4_paged_kv[
            DType.bfloat16,
            1,  # num_heads
            64,  # head_size (row_width = 1*64 = 64)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # --- Test 2: FP8 + SWIZZLE_64B + wide via DeviceBuffer (row_width=256).
        print(
            "\n--- Wide gather4: DeviceBuffer, float8_e4m3fn,"
            " global_row_width=256, SWIZZLE_64B ---"
        )
        test_wide_gather4_device_buffer[
            DType.float8_e4m3fn,
            256,  # global_row_width (box=64, 4 col groups)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # --- Test 3: Wide via ContinuousBatchingKVCache.
        print(
            "\n--- Wide gather4: ContinuousBatchingKVCache, bfloat16,"
            " global_row_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_continuous_kv[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (row_width = 1*512 = 512)
            512,  # max_seq_len
            4,  # num_blocks (one per batch entry)
            1,  # num_layers
            4,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # --- Test 4: Wide via MHAOperand.
        print(
            "\n--- Wide gather4: KVCacheMHAOperand, bfloat16,"
            " global_row_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_mha_operand[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (row_width = 1*512 = 512)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # Non-divisible width test: global_row_width=520, box_width=64,
        # ceildiv(520, 64)=9 col groups, last group extends 56 elements past
        # the boundary. TMA hardware zero-fills those 56 elements.
        # NOTE: global_row_width must be 8-element aligned for BF16 with
        # swizzle modes (16-byte stride alignment required by TMA hardware).
        print(
            "\n--- Non-divisible width: DeviceBuffer, bfloat16,"
            " global_row_width=520, SWIZZLE_128B ---"
        )
        test_non_divisible_width[
            DType.bfloat16,
            520,  # global_row_width (not a multiple of 64, but 8-aligned)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
