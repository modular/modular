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
"""Ragged cross-attention on zero query rows returns without launching.

A speculative-decoding step that skips its drafter hands the drafter's
cross-attention zero sequences (query offsets ``[0]``), while the KV side
still describes the whole target batch. The query and output buffers are
sized for one row so the device allocations are real. The views are zero-row,
and the output must keep its sentinel since nothing may be written.
"""

from std.math import ceildiv, rsqrt
from std.testing import assert_equal
from max.gpu.host import DeviceContext

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import generic_cross_attention_kv_cache


def execute_cross_attention_zero_rows[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](kv_lengths: List[Int], ctx: DeviceContext) raises:
    comptime page_size = 128
    comptime num_layers = 1

    var batch_size = len(kv_lengths)
    var max_kv_length = 0
    for i in range(batch_size):
        max_kv_length = max(max_kv_length, kv_lengths[i])

    var one_row = row_major(
        Int64(1), Idx[num_q_heads], Idx[kv_params.head_size]
    )
    var zero_row = row_major(
        Int64(0), Idx[num_q_heads], Idx[kv_params.head_size]
    )
    var q = HostDeviceTileTensor[dtype](one_row, ctx)
    var output = HostDeviceTileTensor[dtype](one_row, ctx)
    random(q.host_tensor())
    q.to_device()

    var sentinel = Scalar[dtype](7.0)
    var output_host = output.host_tensor()
    for h in range(num_q_heads):
        for d in range(kv_params.head_size):
            output_host[0, h, d] = sentinel
    output.to_device()

    var q_offsets = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    q_offsets.host_tensor()[0] = 0
    q_offsets.to_device()

    var kv_offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var kv_offsets_host = kv_offsets.host_tensor()
    var cache_lengths_host = cache_lengths.host_tensor()
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        kv_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(kv_lengths[i])
        running_offset += UInt32(kv_lengths[i])
    kv_offsets_host[batch_size] = running_offset
    kv_offsets.to_device()
    cache_lengths.to_device()

    # A sequential drafter's step reads a query length of one.
    var q_max_seq_len = Array[UInt32, 1](fill=UInt32(1))

    var pages_per_seq = ceildiv(max_kv_length, page_size)
    var num_paged_blocks = pages_per_seq * batch_size
    var kv_block_paged = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_paged_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
        ctx,
    )
    random(kv_block_paged.host_tensor())
    kv_block_paged.to_device()
    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size), Int64(pages_per_seq)), ctx
    )
    var paged_lut_host = paged_lut.host_tensor()
    for bs in range(batch_size):
        for page in range(pages_per_seq):
            paged_lut_host[bs, page] = UInt32(bs * pages_per_seq + page)
    paged_lut.to_device()

    comptime PagedCollection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = PagedCollection(
        kv_block_paged.device_tensor().as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(max_kv_length),
    )

    generic_cross_attention_kv_cache[target="gpu", mask_str="causal"](
        TileTensor(q.device_tensor().as_imm().unsafe_ptr(), zero_row),
        q_offsets.device_tensor().as_imm(),
        TileTensor(q_max_seq_len, row_major[1]()),
        kv_offsets.device_tensor().as_imm(),
        kv_collection,
        UInt32(0),
        rsqrt(Float32(kv_params.head_size)),
        TileTensor(output.device_tensor().unsafe_ptr(), zero_row),
        ctx,
    )
    ctx.synchronize()

    output.to_host()
    var output_after = output.host_tensor()
    for h in range(num_q_heads):
        for d in range(kv_params.head_size):
            assert_equal(
                output_after[0, h, d],
                sentinel,
                "a zero-row cross-attention call wrote its output",
            )


def main() raises:
    with DeviceContext() as ctx:
        comptime kv_params = KVCacheStaticParams(num_heads=4, head_size=128)
        var kv_lengths: List = [37, 129]

        execute_cross_attention_zero_rows[16, DType.bfloat16, kv_params](
            kv_lengths, ctx
        )

        print("PASS")
