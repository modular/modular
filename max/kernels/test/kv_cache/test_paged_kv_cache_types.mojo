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

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import alloc
from testing import assert_true

from utils.index import Index, IndexList
from collections import OptionalReg

comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=16)


def do_test[
    page_size: Int, layout_block_size: Int, scale_dtype: DType = DType.invalid
]():
    comptime batch_size = 16
    comptime max_num_blocks = 100
    comptime shape = IndexList[6](
        100,
        2,
        1,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )

    var blocks_ptr = alloc[Float32](shape.flattened_length())
    var blocks = LayoutTensor[DType.float32, Layout.row_major[6]()](
        blocks_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape)
    ).fill(0)
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths_ptr = alloc[UInt32](batch_size)
    var cache_lengths = LayoutTensor[DType.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    ).fill(0)
    comptime layout_2d = Layout.row_major[2]()
    var lookup_table_ptr = alloc[UInt32](batch_size * max_num_blocks)
    var lookup_table = LayoutTensor[DType.uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](batch_size, max_num_blocks)
        ),
    ).fill(0)
    for i in range(batch_size):
        cache_lengths[i] = UInt32(i)
        for j in range(max_num_blocks):
            lookup_table[i, j] = UInt32(j)

    var max_seq_length = UInt32(2048)
    var max_cache_length = UInt32(2048)

    var scales: OptionalReg[
        LayoutTensor[scale_dtype, Layout.row_major[6](), MutAnyOrigin]
    ] = None

    comptime if scale_dtype == DType.float8_e4m3fn:
        # Use the same shape as the blocks
        var scales_ptr = alloc[Scalar[scale_dtype]](shape.flattened_length())
        scales = LayoutTensor[scale_dtype, Layout.row_major[6](), MutAnyOrigin](
            scales_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape)
        ).fill(0)

    var collection = PagedKVCacheCollection[
        DType.float32,
        kv_params,
        page_size,
        scale_dtype,
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6](), MutAnyOrigin](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[lookup_table.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            lookup_table.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        max_seq_length,
        max_cache_length,
        scales,
    )

    comptime layout = Layout(
        IntTuple(layout_block_size, Int(kv_params.head_size)),
        IntTuple(Int(kv_params.num_heads * kv_params.head_size), 1),
    )

    var cache = collection.get_key_cache(1)
    _ = cache.block_paged_ptr[layout_block_size](1, layout_block_size, 0)

    # Clean up heap allocations
    blocks_ptr.free()
    cache_lengths_ptr.free()
    lookup_table_ptr.free()


fn test_paged_kv_cache_stride_is_unknown() raises:
    """Test that PagedKVCache has UNKNOWN stride[0] for view tensor correctness.

    PagedKVCache is a 4D view of a 6D parent tensor. The outer stride depends
    on num_layers from the parent tensor, which is unknown at compile time.
    Setting stride[0] = UNKNOWN_VALUE ensures offset calculations use the
    runtime stride rather than an incorrect compile-time value.

    This is a regression test - previously Layout.row_major() computed stride[0]
    from just the 4D shape, which was incorrect for view tensors.
    """
    comptime CacheType = PagedKVCache[DType.float32, kv_params, 16]

    # Verify stride[0] is UNKNOWN_VALUE
    comptime stride_0 = CacheType.blocks_layout.stride[0].value()
    assert_true(
        stride_0 == UNKNOWN_VALUE,
        String("PagedKVCache stride[0] should be UNKNOWN_VALUE (-1), got ")
        + String(stride_0),
    )

    # Verify inner strides are still known (enables partial constant folding)
    comptime stride_1 = CacheType.blocks_layout.stride[1].value()
    comptime stride_2 = CacheType.blocks_layout.stride[2].value()
    comptime stride_3 = CacheType.blocks_layout.stride[3].value()

    comptime expected_stride_1 = Int(kv_params.num_heads * kv_params.head_size)
    comptime expected_stride_2 = Int(kv_params.head_size)
    comptime expected_stride_3 = 1

    assert_true(
        stride_1 == expected_stride_1,
        String("PagedKVCache stride[1] should be ")
        + String(expected_stride_1)
        + ", got "
        + String(stride_1),
    )
    assert_true(
        stride_2 == expected_stride_2,
        String("PagedKVCache stride[2] should be ")
        + String(expected_stride_2)
        + ", got "
        + String(stride_2),
    )
    assert_true(
        stride_3 == expected_stride_3,
        String("PagedKVCache stride[3] should be ")
        + String(expected_stride_3)
        + ", got "
        + String(stride_3),
    )


fn test_paged_kv_cache_offset_correctness() raises:
    """Test that PagedKVCache offset calculations use correct runtime strides.

    This test verifies that when accessing elements through a PagedKVCache view,
    the correct values from the underlying 6D tensor are returned. This catches
    bugs where compile-time strides don't match the actual memory layout.

    The 6D parent tensor has shape:
        [num_blocks, 2, num_layers, page_size, num_heads, head_size]

    The 4D PagedKVCache view has shape:
        [num_blocks, page_size, num_heads, head_size]

    The view's stride[0] must account for the skipped dimensions (2, num_layers).
    """
    comptime num_blocks = 4
    comptime num_layers = 3
    comptime page_size = 2
    comptime num_heads = 2
    comptime head_size = 4

    comptime kv_params_small = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_size
    )

    # 6D shape: [num_blocks, 2, num_layers, page_size, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, page_size, num_heads, head_size
    )
    comptime total_elems = shape_6d.flattened_length()

    # Allocate and fill with unique values (value = flattened index)
    var blocks_ptr = alloc[Float32](total_elems)
    for i in range(total_elems):
        blocks_ptr[i] = Float32(i)

    var blocks = LayoutTensor[DType.float32, Layout.row_major[6]()](
        blocks_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape_6d)
    )

    # Create minimal supporting tensors
    comptime batch_size = 1
    var cache_lengths_ptr = alloc[UInt32](batch_size)
    cache_lengths_ptr[0] = 0
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths = LayoutTensor[DType.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    comptime layout_2d = Layout.row_major[2]()
    var lookup_table_ptr = alloc[UInt32](batch_size * num_blocks)
    for i in range(num_blocks):
        lookup_table_ptr[i] = UInt32(i)  # Identity mapping
    var lookup_table = LayoutTensor[DType.uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
    )

    # Create collection
    var collection = PagedKVCacheCollection[
        DType.float32, kv_params_small, page_size
    ](
        LayoutTensor[DType.float32, Layout.row_major[6](), MutAnyOrigin](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            lookup_table.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        UInt32(page_size),
        UInt32(page_size),
    )

    # Test: Access element through key cache view's blocks tensor and verify correct value
    # Get key cache for layer 1 (kv_idx=0, layer_idx=1)
    var key_cache = collection.get_key_cache(1)

    # Directly access the blocks tensor using IndexList
    # 4D coords [block=1, page=0, head=1, dim=2]
    # In 6D, this corresponds to [block=1, kv=0, layer=1, page=0, head=1, dim=2]
    #
    # 6D strides (row-major):
    #   stride[0] = 2 * 3 * 2 * 2 * 4 = 96
    #   stride[1] = 3 * 2 * 2 * 4 = 48
    #   stride[2] = 2 * 2 * 4 = 16
    #   stride[3] = 2 * 4 = 8
    #   stride[4] = 4
    #   stride[5] = 1
    #
    # Expected 6D offset = 1*96 + 0*48 + 1*16 + 0*8 + 1*4 + 2*1 = 96 + 16 + 4 + 2 = 118
    comptime expected_6d_offset = (
        1 * (2 * num_layers * page_size * num_heads * head_size)
        + 0 * (num_layers * page_size * num_heads * head_size)
        + 1 * (page_size * num_heads * head_size)
        + 0 * (num_heads * head_size)
        + 1 * head_size
        + 2
    )

    # The 4D view's runtime stride[0] should be 2 * num_layers * page_size * num_heads * head_size
    # If the bug existed (using compile-time stride[0] = page_size * num_heads * head_size),
    # we'd compute wrong offset: 1*16 + 0*8 + 1*4 + 2 = 22 (wrong!)

    # Access via the blocks LayoutTensor using IndexList - this tests _offset(IndexList)
    var value = key_cache.blocks.load[1](Index(1, 0, 1, 2))
    var expected_value = Float32(expected_6d_offset)

    assert_true(
        value == expected_value,
        String("PagedKVCache returned wrong value! Got ")
        + String(value)
        + " but expected "
        + String(expected_value)
        + " (at 6D offset "
        + String(expected_6d_offset)
        + "). This indicates stride[0] is using incorrect compile-time value.",
    )

    # Clean up
    blocks_ptr.free()
    cache_lengths_ptr.free()
    lookup_table_ptr.free()


fn test_paged_kv_cache_quantization() raises:
    comptime CacheType = PagedKVCache[
        DType.float32, kv_params, 16, DType.float8_e4m3fn, 256
    ]
    assert_true(CacheType.quantization_enabled, "Quantization not enabled")
    assert_true(
        CacheType.quantization_granularity == 256,
        "Incorrect quantization granularity",
    )


def main():
    test_paged_kv_cache_stride_is_unknown()
    test_paged_kv_cache_offset_correctness()
    test_paged_kv_cache_quantization()
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
    do_test[128, 64, DType.float8_e4m3fn]()
