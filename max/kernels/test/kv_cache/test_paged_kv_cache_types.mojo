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

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import alloc

from utils.index import IndexList

comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=16)


def do_test[page_size: Int, layout_block_size: Int]():
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
        cache_lengths[i] = i
        for j in range(max_num_blocks):
            lookup_table[i, j] = j

    var max_seq_length = UInt32(2048)
    var max_cache_length = UInt32(2048)

    var collection = PagedKVCacheCollection[
        DType.float32, kv_params, page_size
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
    )

    comptime layout = Layout(
        IntTuple(layout_block_size, Int(kv_params.head_size)),
        IntTuple(Int(kv_params.num_heads * kv_params.head_size), 1),
    )

    var cache = collection.get_key_cache(1)
    var layout_tensor = cache.block_paged_ptr[layout_block_size](
        1, layout_block_size, 0
    )

    # Clean up heap allocations
    blocks_ptr.free()
    cache_lengths_ptr.free()
    lookup_table_ptr.free()


def main():
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
