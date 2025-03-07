# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from internal_utils import HostNDBuffer
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import IntTuple, Layout
from memory import UnsafePointer

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
            kv_params.num_heads,
            kv_params.head_size,
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
        blocks.tensor,
        cache_lengths.tensor,
        lookup_table.tensor,
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
