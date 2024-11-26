# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from kv_cache.types import (
    PagedKVCacheCollection,
    PagedKVCache,
    KVCacheStaticParams,
)
from buffer import NDBuffer
from layout import Layout, IntTuple
from utils.index import IndexList
from memory import UnsafePointer
from internal_utils import HostNDBuffer

alias kv_params = KVCacheStaticParams(num_heads=16, head_size=16)


def do_test[cache_block_size: Int, layout_block_size: Int]():
    var batch_size = 16
    var max_num_blocks = 100
    var blocks = HostNDBuffer[DType.float32, 6](
        IndexList[6](
            1,
            2,
            100,
            cache_block_size,
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

    var is_cache_empty = Bool(False)
    var seq_ids = List[Int]()
    for i in range(batch_size):
        seq_ids.append(i)

    var collection = PagedKVCacheCollection[
        DType.float32, kv_params, cache_block_size
    ](
        blocks.tensor,
        cache_lengths.tensor,
        lookup_table.tensor,
        is_cache_empty,
        seq_ids,
    )

    alias layout = Layout(
        IntTuple(layout_block_size, int(kv_params.head_size)),
        IntTuple(int(kv_params.num_heads * kv_params.head_size), 1),
    )

    var cache = collection.get_key_cache[collection.CacheType](1)
    var layout_tensor = cache.block_paged_ptr[DType.float32, layout_block_size](
        1, layout_block_size, 0
    )
    print(layout_tensor)


def main():
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
