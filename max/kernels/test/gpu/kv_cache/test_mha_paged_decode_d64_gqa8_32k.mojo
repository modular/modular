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
# Paged decode at head_dim 64, 16 q / 2 kv heads (GQA 8), page 128 and 32K
# context against `mha_gpu_naive`, called with the decode dispatch metadata
# serving passes. Batch 1 takes the workspace split-K arm and its combine
# kernel, and a ragged case covers partial tiles and pages.
#
# Sequences draw their pages with replacement from a small shared pool, so
# 32K contexts need little memory. The kernel and the reference read separate
# copies of the page table.

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import assert_no_nan_inf, padded_lut_cols
from layout import Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import (
    MHADecodeDispatchMetadata,
    flash_attention,
    mha_gpu_naive,
)
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

comptime num_q_heads = 16
comptime kv_params = KVCacheStaticParams(num_heads=2, head_size=64)
comptime group = num_q_heads // kv_params.num_heads
comptime page_size = 128
comptime pool_pages = 1024
comptime dtype = DType.bfloat16


def run_case(context_lengths: List[Int], ctx: DeviceContext) raises:
    var batch_size = len(context_lengths)
    var max_context = 0
    for c in context_lengths:
        max_context = max(max_context, c)
    print("batch", batch_size, "max context", max_context)

    var q_layout = row_major(
        Int64(batch_size), Idx[num_q_heads], Idx[kv_params.head_size]
    )
    var row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    for i in range(batch_size):
        row_offsets.host_tensor()[i] = UInt32(i)
        cache_lengths.host_tensor()[i] = UInt32(context_lengths[i] - 1)
    row_offsets.host_tensor()[batch_size] = UInt32(batch_size)
    row_offsets.to_device()
    cache_lengths.to_device()

    # A wide Q range peaks the softmax, so a wrong key range moves the output
    # by far more than the tolerance; zero-mean K/V keep the output's sign and
    # size dependent on which keys were attended.
    var q = HostDeviceTileTensor[dtype](q_layout, ctx)
    random(q.host_tensor(), Scalar[dtype](-8), Scalar[dtype](8))
    q.to_device()

    var blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(pool_pages),
            Idx[2],
            Idx[1],
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
        ctx,
    )
    random(blocks.host_tensor(), Scalar[dtype](-1), Scalar[dtype](1))
    blocks.to_device()

    var lut_layout = row_major(
        Int64(batch_size),
        Int64(padded_lut_cols(ceildiv(max_context, page_size))),
    )
    var ref_lut = HostDeviceTileTensor[.uint32](lut_layout, ctx)
    var test_lut = HostDeviceTileTensor[.uint32](lut_layout, ctx)
    for b in range(batch_size):
        for p in range(ceildiv(context_lengths[b], page_size)):
            var page = UInt32(random_ui64(0, pool_pages - 1))
            ref_lut.host_tensor()[b, p] = page
            test_lut.host_tensor()[b, p] = page
    ref_lut.to_device()
    test_lut.to_device()

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var cache_lengths_dev = (
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin()
    )
    var ref_kv = Collection(
        blocks.device_tensor().as_unsafe_any_origin(),
        cache_lengths_dev,
        ref_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(max_context),
    )
    var test_kv = Collection(
        blocks.device_tensor().as_unsafe_any_origin(),
        cache_lengths_dev,
        test_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(max_context),
    )

    var output = HostDeviceTileTensor[dtype](q_layout, ctx)
    var ref_output = HostDeviceTileTensor[dtype](q_layout, ctx)
    var scale = rsqrt(Float32(kv_params.head_size))

    flash_attention[ragged=True](
        output.device_tensor(),
        q.device_tensor(),
        test_kv.get_key_cache(0),
        test_kv.get_value_cache(0),
        CausalMask(),
        row_offsets.device_tensor(),
        scale,
        ctx,
        decode_dispatch_metadata=MHADecodeDispatchMetadata.from_runtime_values[
            num_q_heads, group
        ](batch_size, 1, max_context, ctx),
    )
    mha_gpu_naive[ragged=True](
        q.device_tensor(),
        ref_kv.get_key_cache(0),
        ref_kv.get_value_cache(0),
        CausalMask(),
        ref_output.device_tensor(),
        row_offsets.device_tensor(),
        scale,
        batch_size,
        1,
        max_context,
        num_q_heads,
        kv_params.head_size,
        group,
        ctx,
    )

    assert_no_nan_inf(output, "output")
    assert_no_nan_inf(ref_output, "ref_output")
    var out_host = output.host_tensor()
    var ref_host = ref_output.host_tensor()
    var max_err = Float32(0)
    var max_ref = Float32(0)
    for b in range(batch_size):
        for h in range(num_q_heads):
            for d in range(kv_params.head_size):
                var got = out_host[b, h, d].cast[.float32]()
                var want = ref_host[b, h, d].cast[.float32]()
                max_err = max(max_err, abs(got - want))
                max_ref = max(max_ref, abs(want))
    print("max |ref|", max_ref, "max |err|", max_err)
    for b in range(batch_size):
        for h in range(num_q_heads):
            for d in range(kv_params.head_size):
                assert_almost_equal(
                    out_host[b, h, d],
                    ref_host[b, h, d],
                    atol=2e-3,
                    rtol=1e-2,
                )


def main() raises:
    seed(7)
    with DeviceContext() as ctx:
        run_case([32768], ctx)
        run_case([32768, 32767, 16385, 4097, 129, 1], ctx)
