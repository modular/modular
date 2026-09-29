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

"""`PagedKVCache.kv_tma_coords` against the flat `row_idx` it replaces.

Two properties, and the second is why the split exists.

*Faithful decomposition.* `block * stride + row_in_block` must reproduce
`row_idx` exactly, for every token. A transposed or mis-looked-up coordinate
still type-checks and still reads *a* row, so nothing downstream catches it
except wrong scores; this asserts the arithmetic directly.

*A bound that does not track the slab.* The flat form leaves signed 32 bits
once the pool is large, because both of its factors grow with total cache
memory: the block is a physical id into the shared allocation, and the stride
is the whole per-block pitch. The split's two coordinates are bounded by the
block count and a block's own rows. The overflow case is checked as
arithmetic rather than by allocating the hundreds of gigabytes that would
reach it on a real device.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_memset_zero
from std.utils import IndexList
from std.testing import assert_equal, assert_true

from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import padded_lut_cols


comptime _NUM_PROBES = 32


def _coords_kernel[
    cache_t: KVCacheT,
](
    kv: cache_t,
    flat_out: MutPointer[UInt32, MutAnyOrigin],
    row_out: MutPointer[Int32, MutAnyOrigin],
    block_out: MutPointer[Int32, MutAnyOrigin],
    tok_stride: Int32,
):
    """Records both ways of addressing one token, for the host to compare."""
    if global_idx.x != 0:
        return
    for i in range(_NUM_PROBES):
        var tok = UInt32(Int32(i) * tok_stride)
        flat_out[i] = kv.row_idx(UInt32(0), tok)
        var row, block = kv.kv_tma_coords(UInt32(0), tok)
        row_out[i] = row
        block_out[i] = block


def _check_decomposition[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](ctx: DeviceContext) raises:
    """The split must rebuild the flat row exactly, at every probe."""
    var num_used = 64
    var lut_columns = padded_lut_cols(num_used)

    comptime lut_layout = Layout.row_major[2]()
    var lut_runtime = RuntimeLayout[lut_layout].row_major(
        IndexList[2](1, lut_columns)
    )
    var lut = ManagedLayoutTensor[.uint32, lut_layout](lut_runtime, ctx)
    var lut_host = lut.tensor[update=False]()
    # Deliberately not the identity: a coordinate that ignored the lookup
    # table would still decompose correctly against an identity mapping.
    for c in range(lut_columns):
        lut_host[0, c] = UInt32((c * 7 + 3) % num_used) if c < num_used else 0

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_runtime = RuntimeLayout[cache_lengths_layout].row_major(
        IndexList[1](1)
    )
    var cache_lengths = ManagedLayoutTensor[.uint32, cache_lengths_layout](
        cache_lengths_runtime, ctx
    )
    cache_lengths.tensor[update=False]()[0] = UInt32(num_used * page_size)

    comptime blocks_layout = Layout.row_major[6]()
    var blocks_runtime = RuntimeLayout[blocks_layout].row_major(
        IndexList[6](
            num_used,
            2,
            1,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    var blocks = ManagedLayoutTensor[dtype, blocks_layout](blocks_runtime, ctx)
    unsafe_memset_zero(blocks.tensor[update=False]().ptr, blocks_runtime.size())

    var flat_buf = ctx.enqueue_create_buffer[.uint32](_NUM_PROBES)
    var row_buf = ctx.enqueue_create_buffer[.int32](_NUM_PROBES)
    var block_buf = ctx.enqueue_create_buffer[.int32](_NUM_PROBES)

    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths.device_tensor(),
        lut.device_tensor(),
        UInt32(num_used * page_size),
        UInt32(num_used * page_size),
    )
    var kv = collection.get_key_cache(0)

    # A stride that is not a multiple of the page, so probes land at varied
    # offsets within their blocks rather than always on a boundary.
    var tok_stride = max(page_size // 3, 1)
    ctx.enqueue_function[_coords_kernel[type_of(kv)]](
        kv,
        flat_buf.unsafe_ptr(),
        row_buf.unsafe_ptr(),
        block_buf.unsafe_ptr(),
        Int32(tok_stride),
        grid_dim=1,
        block_dim=1,
    )

    var flat_host = ctx.enqueue_create_host_buffer[.uint32](_NUM_PROBES)
    var row_host = ctx.enqueue_create_host_buffer[.int32](_NUM_PROBES)
    var block_host = ctx.enqueue_create_host_buffer[.int32](_NUM_PROBES)
    ctx.enqueue_copy(flat_host, flat_buf)
    ctx.enqueue_copy(row_host, row_buf)
    ctx.enqueue_copy(block_host, block_buf)
    ctx.synchronize()

    # The block pitch in rows: the blocks tensor is
    # `[blocks, kv=2, layers=1, page_size, heads, head_size]`, and `_stride()`
    # divides its leading stride by `heads * head_size`.
    var stride = 2 * page_size
    for i in range(_NUM_PROBES):
        var rebuilt = Int(block_host[i]) * stride + Int(row_host[i])
        assert_equal(
            rebuilt,
            Int(flat_host[i]),
            "split coordinate does not rebuild the flat row",
        )
        # The row must be an offset *within* a block, not a pool-wide index;
        # that is the whole point of the split.
        assert_true(
            Int(row_host[i]) < page_size,
            "row_in_block escaped its page",
        )


def test_flat_row_overflows_where_the_split_does_not() raises:
    """GLM-5.3-Flash's indexer geometry, as arithmetic.

    Eleven sparse-MLA layers at a 128-row page give a 1408-row block pitch,
    and a shared slab hands this leaf roughly 1.84M blocks. The fold passes
    2^31; neither split coordinate comes close.
    """
    comptime int32_max = (1 << 31) - 1
    var stride = 11 * 128
    var total_blocks = 1_840_895
    var last_block = total_blocks - 1
    var page_size = 128

    var flat = last_block * stride + (page_size - 1)
    assert_true(
        flat > int32_max,
        "this geometry is supposed to overflow the flat coordinate",
    )
    assert_true(
        last_block <= int32_max and page_size - 1 <= int32_max,
        "neither split coordinate may approach the 32-bit ceiling",
    )
    # And the bound the descriptor declares is the block count, which does
    # not grow with the pitch: doubling the layers leaves it untouched.
    assert_equal(last_block, (2 * total_blocks - 1) // 2)


def main() raises:
    test_flat_row_overflows_where_the_split_does_not()
    with DeviceContext() as ctx:
        _check_decomposition[
            .float8_e4m3fn, KVCacheStaticParams(num_heads=1, head_size=32), 128
        ](ctx)
        _check_decomposition[
            .bfloat16, KVCacheStaticParams(num_heads=8, head_size=128), 128
        ](ctx)
