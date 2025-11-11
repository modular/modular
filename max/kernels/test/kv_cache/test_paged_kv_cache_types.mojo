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

from internal_utils import HostNDBuffer
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

from utils.index import IndexList

alias kv_params = KVCacheStaticParams(num_heads=16, head_size=16)


def do_test[page_size: Int, layout_block_size: Int]():
    var batch_size = 16
    var max_num_blocks = 100
    var blocks = HostNDBuffer[DType.float32, 6](
        IndexList[6](
            100,
            2,
            1,
            page_size,
            Int(kv_params.num_heads),
            Int(kv_params.head_size),
        )
    )
    var cache_lengths = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))
    var lookup_table = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, max_num_blocks)
    )
    for i in range(batch_size):
        cache_lengths.tensor[i] = i
        for j in range(max_num_blocks):
            lookup_table.tensor[i, j] = j

    var max_seq_length = UInt32(2048)
    var max_cache_length = UInt32(2048)

    var collection = PagedKVCacheCollection[
        DType.float32, kv_params, page_size
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6](), MutAnyOrigin](
            blocks.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.to_layout_tensor().runtime_layout.shape.value,
                blocks.to_layout_tensor().runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths.to_layout_tensor().ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.to_layout_tensor().runtime_layout.shape.value,
                cache_lengths.to_layout_tensor().runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[lookup_table.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            lookup_table.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.to_layout_tensor().runtime_layout.shape.value,
                lookup_table.to_layout_tensor().runtime_layout.stride.value,
            ),
        ),
        max_seq_length,
        max_cache_length,
    )

    alias layout = Layout(
        IntTuple(layout_block_size, Int(kv_params.head_size)),
        IntTuple(Int(kv_params.num_heads * kv_params.head_size), 1),
    )

    var cache = collection.get_key_cache(1)
    var layout_tensor = cache.block_paged_ptr[layout_block_size](
        1, layout_block_size, 0
    )
    print(layout_tensor)


def main():
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
