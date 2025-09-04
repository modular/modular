# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from buffer import NDBuffer
from collections import Set
from gpu import barrier
from gpu.memory import fence_async_view_proxy
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx
from internal_utils import HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    PagedKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor
from layout.tensor_core_async import tile_layout_mn_major, tile_layout_k_major
from layout.tma_async import (
    SharedMemBarrier,
    TMANestedTensorTile,
)
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from nn.mha_operand import (
    MHAOperand,
    KVCacheMHAOperand,
    NDBufferMHAOperand,
    RaggedMHAOperand,
)
from random import random_ui64, seed
from testing import assert_equal
from utils import IndexList
from sys import size_of


@__llvm_arg_metadata(src_tma_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(dst_tma_tile, `nvvm.grid_constant`)
fn mha_operand_tma_copy_kernel[
    tile_m: Int,
    head_size: Int,
    kv_t: MHAOperand,
    swizzle_mode: TensorMapSwizzle,
    is_k_major: Bool,
](
    src_tma_tile: TMANestedTensorTile[
        kv_t.dtype,
        tile_m,
        head_size,
        swizzle_mode,
        is_k_major,
    ],
    dst_tma_tile: TMANestedTensorTile[
        kv_t.dtype,
        tile_m,
        head_size,
        swizzle_mode,
        is_k_major,
    ],
    src_operand: kv_t,
    dst_operand: kv_t,
):
    # Map block indices to MHA parameters
    batch_idx = UInt32(block_idx.z)
    head_idx = UInt32(block_idx.y)

    num_keys = src_operand.cache_length(Int(batch_idx))

    # Allocate shared memory tile
    smem_tile = LayoutTensor[
        kv_t.dtype,
        __type_of(src_tma_tile).layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    # Initialize barrier
    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    phase: UInt32 = 0

    # Calculate col coordinates
    src_col = src_operand.col_idx(head_idx)
    dst_col = dst_operand.col_idx(head_idx)
    # Declare row coordinates

    # Loop over columns to copy full head size
    for kv_tile_start_row in range(0, num_keys, tile_m):
        if thread_idx.x == 0:
            src_row = src_operand.row_idx(batch_idx, kv_tile_start_row)
            mbar[0].expect_bytes(tile_m * head_size * size_of[kv_t.dtype]())

            # Initiate TMA load
            src_tma_tile.async_copy(
                smem_tile, mbar[0], (UInt(src_col), UInt(src_row))
            )

        # Synchronize all threads
        barrier()
        mbar[0].wait(phase)
        phase ^= 1

        # Ensure data is visible before store
        barrier()

        # Ensures all previous shared memory stores are completed.
        fence_async_view_proxy()
        # Store to destination
        if thread_idx.x == 0:
            dst_row = dst_operand.row_idx(batch_idx, kv_tile_start_row)

            # Initiate TMA store
            dst_tma_tile.async_store(smem_tile, (UInt(dst_col), UInt(dst_row)))

            dst_tma_tile.commit_group()
            dst_tma_tile.wait_group()


def mha_operand_copy[
    kv_t: MHAOperand, //,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
    *,
    is_k_major: Bool,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ctx: DeviceContext, src: kv_t, dst: kv_t, batch_size: Int, max_seq_len: Int,):
    # Each kernel call copies the entire array, iterating over `tile_m` chunks
    # along the rows. `head_size` is the continugous axis.
    # For `Q @ K'`, `is_k_major` is True.
    # For `P @ V`, `is_k_major` is False.
    # We emulate both, using the appropriate smem layout for the (wg/u)mma instructions.
    alias head_size = kv_params.head_size

    # Create TMA tiles
    src_tma = src.create_tma_tile[
        tile_m, head_size, swizzle_mode, is_k_major=is_k_major
    ](ctx)
    dst_tma = dst.create_tma_tile[
        tile_m, head_size, swizzle_mode, is_k_major=is_k_major
    ](ctx)

    # Calculate grid dimensions
    # NOTE: In context encoding, we would have grid_x = ceildiv(max_prompt_len, BM)
    # Each of these, as well as `q_num_heads // kv_num_heads` represent multicast
    # opportunities.
    grid_x = 1
    alias grid_y = kv_params.num_heads
    grid_z = batch_size

    alias kernel = mha_operand_tma_copy_kernel[
        tile_m,
        head_size,
        kv_t,
        swizzle_mode,
        is_k_major,
    ]

    # Launch kernel with block_dim=32
    ctx.enqueue_function[kernel](
        src_tma,
        dst_tma,
        src,
        dst,
        grid_dim=(grid_x, grid_y, grid_z),
        block_dim=(32,),
    )

    ctx.synchronize()


def test_mha_host_operand[
    kv_t: MHAOperand, //,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](src: kv_t, dst: kv_t, batch_size: Int):
    """Test function that compares two MHAOperands using block_paged_ptr."""
    alias kv_row_stride = Int(kv_params.head_size * kv_params.num_heads)
    # Iterate over all batch entries and tokens
    for b in range(batch_size):
        seq_len = src.cache_length(b)
        for s in range(0, seq_len, tile_m):
            actual_tokens = min(tile_m, seq_len - s)
            for h in range(Int(kv_params.num_heads)):
                # Get pointers using block_paged_ptr
                src_ptr = src.block_paged_ptr[tile_m](
                    UInt32(b), UInt32(s), UInt32(h), UInt32(0)
                )
                dst_ptr = dst.block_paged_ptr[tile_m](
                    UInt32(b), UInt32(s), UInt32(h), UInt32(0)
                )

                # Compare values for the actual number of tokens
                for tok in range(actual_tokens):
                    for hd in range(Int(kv_params.head_size)):
                        offset = tok * kv_row_stride + hd
                        src_val = src_ptr[offset]
                        dst_val = dst_ptr[offset]
                        if src_val != dst_val:
                            print(b, s, h, tok, hd, src_val, dst_val)
                        assert_equal(src_val, dst_val)


def test_continuous_kv_cache[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](ctx: DeviceContext, batch_size: Int, max_seq_len: Int, num_layers: Int,):
    alias msg = "  Testing ContinuousBatchingKVCache with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Initialize cache blocks
    num_blocks = batch_size + 2
    dyn_shape = IndexList[6](
        num_blocks,
        2,  # key and value
        num_layers,
        max_seq_len,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    kv_block_host = HostNDBuffer[dtype, 6](dyn_shape)
    random(kv_block_host.tensor)
    kv_block_device = kv_block_host.copy_to_device(ctx)

    # Set up lookup table and cache lengths
    lookup_table_host = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))
    cache_lengths_host = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))

    for i in range(batch_size):
        lookup_table_host.tensor[i] = i
        cache_lengths_host.tensor[i] = max_seq_len // 2  # Half filled caches

    lookup_table_device = lookup_table_host.copy_to_device(ctx)
    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    # Create source and destination collections
    src_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )

    # Test copying key cache at layer 0
    src_key = KVCacheMHAOperand(src_collection.get_key_cache(0))
    src_host_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )

    src_host_key = KVCacheMHAOperand(src_host_collection.get_key_cache(0))

    for is_k_major in range(2):
        dst_block_host = HostNDBuffer[dtype, 6](dyn_shape)
        dst_block_device = dst_block_host.copy_to_device(ctx)

        dst_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
            dst_block_device.tensor,
            cache_lengths_device.tensor,
            lookup_table_device.tensor,
            UInt32(max_seq_len),
            UInt32(max_seq_len),
        )
        dst_key = KVCacheMHAOperand(dst_collection.get_key_cache(0))

        if is_k_major == 0:  # unswitch
            mha_operand_copy[tile_m, kv_params, is_k_major=False](
                ctx,
                src_key,
                dst_key,
                batch_size,
                max_seq_len,
            )
        else:
            mha_operand_copy[tile_m, kv_params, is_k_major=True](
                ctx,
                src_key,
                dst_key,
                batch_size,
                max_seq_len,
            )

        # Verify results - copy device data back to host
        ctx.enqueue_copy(dst_block_host.tensor.data, dst_block_device.buffer)
        ctx.synchronize()

        # Create host-side MHAOperands for verification
        dst_host_collection = ContinuousBatchingKVCacheCollection[
            dtype, kv_params
        ](
            dst_block_host.tensor,
            cache_lengths_host.tensor,
            lookup_table_host.tensor,
            UInt32(max_seq_len),
            UInt32(max_seq_len),
        )
        dst_host_key = KVCacheMHAOperand(dst_host_collection.get_key_cache(0))
        # Verify using block_paged_ptr
        test_mha_host_operand[tile_m, kv_params](
            src_host_key, dst_host_key, batch_size
        )
        _ = dst_block_device^
        _ = dst_block_host^

    print("    ContinuousBatchingKVCache test passed!")

    _ = kv_block_host^
    _ = kv_block_device^
    _ = lookup_table_host^
    _ = lookup_table_device^
    _ = cache_lengths_host^
    _ = cache_lengths_device^


def test_paged_kv_cache[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](ctx: DeviceContext, batch_size: Int, max_seq_len: Int, num_layers: Int,):
    alias msg = "  Testing PagedKVCache with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Calculate number of pages needed
    pages_per_seq = (max_seq_len + page_size - 1) // page_size
    num_blocks = batch_size * pages_per_seq + 10  # Extra blocks

    # Initialize paged cache blocks
    dyn_shape = IndexList[6](
        num_blocks,
        2,  # key and value
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )

    kv_block_host = HostNDBuffer[dtype, 6](dyn_shape)
    random(kv_block_host.tensor)
    kv_block_device = kv_block_host.copy_to_device(ctx)

    # Set up page lookup table
    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, pages_per_seq)
    )
    paged_lut_set = Set[Int]()

    for bs in range(batch_size):
        for page_idx in range(pages_per_seq):
            block_idx = Int(random_ui64(0, num_blocks - 1))
            while block_idx in paged_lut_set:
                block_idx = Int(random_ui64(0, num_blocks - 1))
            paged_lut_set.add(block_idx)
            paged_lut_host.tensor[bs, page_idx] = block_idx

    paged_lut_device = paged_lut_host.copy_to_device(ctx)

    # Set up cache lengths
    cache_lengths_host = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = max_seq_len // 2  # Half filled

    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    # Create source and destination collections
    src_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )
    src_key = KVCacheMHAOperand(src_collection.get_key_cache(0))

    # Create host-side MHAOperands for verification
    src_host_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        paged_lut_host.tensor,
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )

    src_host_key = KVCacheMHAOperand(src_host_collection.get_key_cache(0))

    for is_k_major in range(2):
        # Create destination with zeroed blocks
        dst_block_host = HostNDBuffer[dtype, 6](dyn_shape)
        dst_block_device = dst_block_host.copy_to_device(ctx)

        dst_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
            dst_block_device.tensor,
            cache_lengths_device.tensor,
            paged_lut_device.tensor,
            UInt32(max_seq_len),
            UInt32(max_seq_len),
        )
        # Test copying key cache at layer 0
        dst_key = KVCacheMHAOperand(dst_collection.get_key_cache(0))

        if is_k_major == 0:
            mha_operand_copy[tile_m, kv_params, is_k_major=False](
                ctx,
                src_key,
                dst_key,
                batch_size,
                max_seq_len,
            )
        else:
            mha_operand_copy[tile_m, kv_params, is_k_major=True](
                ctx,
                src_key,
                dst_key,
                batch_size,
                max_seq_len,
            )

        # Verify results - copy device data back to host
        ctx.enqueue_copy(dst_block_host.tensor.data, dst_block_device.buffer)
        ctx.synchronize()

        dst_host_collection = PagedKVCacheCollection[
            dtype, kv_params, page_size
        ](
            dst_block_host.tensor,
            cache_lengths_host.tensor,
            paged_lut_host.tensor,
            UInt32(max_seq_len),
            UInt32(max_seq_len),
        )
        dst_host_key = KVCacheMHAOperand(dst_host_collection.get_key_cache(0))

        # Verify using block_paged_ptr
        test_mha_host_operand[tile_m, kv_params](
            src_host_key, dst_host_key, batch_size
        )
        _ = dst_block_device^
        _ = dst_block_host^

    print("    PagedKVCache test passed!")

    _ = kv_block_host^
    _ = kv_block_device^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = cache_lengths_host^
    _ = cache_lengths_device^


def test_ndbuffer[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](ctx: DeviceContext, batch_size: Int, max_seq_len: Int,):
    alias msg = "  Testing NDBuffer with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Create source and destination buffers with BSHD layout
    dyn_shape = IndexList[4](
        batch_size,
        max_seq_len,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )

    src_host = HostNDBuffer[dtype, 4](dyn_shape)
    random(src_host.tensor)
    src_device = src_host.copy_to_device(ctx)

    # Create MHAOperands
    src_operand = NDBufferMHAOperand(src_device.tensor)
    for is_k_major in range(2):
        dst_host = HostNDBuffer[dtype, 4](dyn_shape)

        dst_device = dst_host.copy_to_device(ctx)
        dst_operand = NDBufferMHAOperand(dst_device.tensor)

        if is_k_major == 0:  # unswitch
            mha_operand_copy[tile_m, kv_params, is_k_major=False](
                ctx,
                src_operand,
                dst_operand,
                batch_size,
                max_seq_len,
            )
        else:
            mha_operand_copy[tile_m, kv_params, is_k_major=True](
                ctx,
                src_operand,
                dst_operand,
                batch_size,
                max_seq_len,
            )

        # Verify results - copy device data back to host
        ctx.enqueue_copy(dst_host.tensor.data, dst_device.buffer)
        ctx.synchronize()

        # Create host-side MHAOperands for verification
        src_host_operand = NDBufferMHAOperand(src_host.tensor)
        dst_host_operand = NDBufferMHAOperand(dst_host.tensor)

        # Verify using block_paged_ptr
        test_mha_host_operand[tile_m, kv_params](
            src_host_operand, dst_host_operand, batch_size
        )
        _ = dst_host^
        _ = dst_device^

    print("    NDBuffer test passed!")

    _ = src_host^
    _ = src_device^


def test_ragged[
    dtype: DType, tile_m: Int, kv_params: KVCacheStaticParams
](ctx: DeviceContext, batch_size: Int):
    alias msg = "  Testing RaggedTensor with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Create variable length sequences
    seq_lens = List[Int]()
    total_tokens = 0
    # Create cache row offsets
    cache_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    for i in range(batch_size):
        seq_len = Int(random_ui64(100, 500))
        seq_lens.append(seq_len)
        cache_row_offsets_host.tensor[i] = total_tokens
        total_tokens += seq_len

    cache_row_offsets_host.tensor[batch_size] = total_tokens

    cache_row_offsets_device = cache_row_offsets_host.copy_to_device(ctx)

    # Create ragged buffers
    dyn_shape = IndexList[3](
        total_tokens, Int(kv_params.num_heads), Int(kv_params.head_size)
    )

    src_host = HostNDBuffer[dtype, 3](dyn_shape)
    random(src_host.tensor)
    src_device = src_host.copy_to_device(ctx)

    # Create MHAOperands
    src_operand = RaggedMHAOperand(
        src_device.tensor, cache_row_offsets_device.tensor
    )

    # Find max sequence length for grid calculation
    max_seq_len = 0
    for i in range(batch_size):
        max_seq_len = max(max_seq_len, seq_lens[i])

    # Create host-side MHAOperands for verification
    src_host_operand = RaggedMHAOperand(
        src_host.tensor, cache_row_offsets_host.tensor
    )

    for is_k_major in range(2):
        dst_host = HostNDBuffer[dtype, 3](dyn_shape)
        dst_device = dst_host.copy_to_device(ctx)

        dst_operand = RaggedMHAOperand(
            dst_device.tensor, cache_row_offsets_device.tensor
        )

        if is_k_major == 0:
            mha_operand_copy[tile_m, kv_params, is_k_major=False](
                ctx,
                src_operand,
                dst_operand,
                batch_size,
                max_seq_len,
            )
        else:
            mha_operand_copy[tile_m, kv_params, is_k_major=True](
                ctx,
                src_operand,
                dst_operand,
                batch_size,
                max_seq_len,
            )

        # Verify results - copy device data back to host
        ctx.enqueue_copy(dst_host.tensor.data, dst_device.buffer)
        ctx.synchronize()

        dst_host_operand = RaggedMHAOperand(
            dst_host.tensor, cache_row_offsets_host.tensor
        )

        # Verify using block_paged_ptr
        test_mha_host_operand[tile_m, kv_params](
            src_host_operand, dst_host_operand, batch_size
        )

        _ = dst_host^
        _ = dst_device^

    print("    RaggedTensor test passed!")

    _ = src_host^
    _ = src_device^
    _ = cache_row_offsets_host^
    _ = cache_row_offsets_device^


def main():
    seed(42)
    with DeviceContext() as ctx:
        alias batch_size = 4
        alias max_seq_len = 1024
        alias num_layers = 2
        alias page_size = 512
        alias dtype = DType.bfloat16

        print("Testing TMA copy with different tile configurations")

        @parameter
        for i in range(6, 9):
            alias head_size = 1 << i  # 64, 128, 256
            alias kv_params = KVCacheStaticParams(
                num_heads=8, head_size=UInt(head_size)
            )

            @parameter
            for j in range(6, 15 - i):
                alias block_m = 1 << j  # 64, ..., (64 * 256) // block_m

                alias msg = "\nTesting block_m=" + String(
                    block_m
                ) + ", head_size=" + String(head_size)
                print(msg)

                test_continuous_kv_cache[dtype, block_m, kv_params](
                    ctx, batch_size, max_seq_len, num_layers
                )
                test_paged_kv_cache[dtype, block_m, kv_params, page_size](
                    ctx,
                    batch_size,
                    max_seq_len,
                    num_layers,
                )
                test_ndbuffer[dtype, block_m, kv_params](
                    ctx, batch_size, max_seq_len
                )
                test_ragged[dtype, block_m, kv_params](ctx, batch_size)
