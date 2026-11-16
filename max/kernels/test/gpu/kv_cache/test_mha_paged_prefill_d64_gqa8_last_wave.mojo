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
# Paged prefill at head_dim 64, 16 q / 2 kv heads (GQA 8), page 128, over a
# 32K or longer cache against `mha_gpu_naive`. On B200 these prompts leave
# the 2Q grid's last wave nearly empty, so FA4 takes its 1Q config instead; a
# partial last tile and a batch of two equal prompts are covered too.
#
# Pages are drawn with replacement from a small shared pool.

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import assert_no_nan_inf, padded_lut_cols
from layout import Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

comptime num_q_heads = 16
comptime kv_params = KVCacheStaticParams(num_heads=2, head_size=64)
comptime group = num_q_heads // kv_params.num_heads
comptime page_size = 128
comptime pool_pages = 1024
comptime dtype = DType.bfloat16


def run_case(
    batch_size: Int, prompt: Int, cache: Int, window: Int, ctx: DeviceContext
) raises:
    """Checks the first and last `window` query rows of each prompt, so the
    naive reference's score buffer stays small at long contexts."""
    var rows = batch_size * prompt
    var max_context = cache + prompt
    print("batch", batch_size, "prompt", prompt, "cache", cache)

    var q_layout = row_major(
        Int64(rows), Idx[num_q_heads], Idx[kv_params.head_size]
    )
    var row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    for i in range(batch_size):
        row_offsets.host_tensor()[i] = UInt32(i * prompt)
        cache_lengths.host_tensor()[i] = UInt32(cache)
    row_offsets.host_tensor()[batch_size] = UInt32(rows)
    row_offsets.to_device()
    cache_lengths.to_device()

    # A wide Q range peaks the softmax, so a wrong key range moves the output
    # by far more than the tolerance.
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

    var lut_cols = padded_lut_cols(ceildiv(max_context, page_size))
    var lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size), Int64(lut_cols)), ctx
    )
    for b in range(batch_size):
        for p in range(ceildiv(max_context, page_size)):
            lut.host_tensor()[b, p] = UInt32(random_ui64(0, pool_pages - 1))
    lut.to_device()

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv = Collection(
        blocks.device_tensor().as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(prompt),
        UInt32(max_context),
    )

    var output = HostDeviceTileTensor[dtype](q_layout, ctx)
    var scale = rsqrt(Float32(kv_params.head_size))
    flash_attention[ragged=True](
        output.device_tensor(),
        q.device_tensor(),
        kv.get_key_cache(0),
        kv.get_value_cache(0),
        CausalMask(),
        row_offsets.device_tensor(),
        scale,
        ctx,
    )
    assert_no_nan_inf(output, "output")
    var out_host = output.host_tensor()
    var q_host = q.host_tensor()

    # The reference sees each window as its own one-row batch whose cache
    # holds the prompt rows before it, over a separate copy of the table.
    var starts = List[Int]()
    starts.append(0)
    if prompt > window:
        starts.append(max(window, prompt - window))
    var max_err = Float32(0)
    for b in range(batch_size):
        for start in starts:
            var n = min(window, prompt - start)
            var win_layout = row_major(
                Int64(n), Idx[num_q_heads], Idx[kv_params.head_size]
            )
            var win_q = HostDeviceTileTensor[dtype](win_layout, ctx)
            var win_ref = HostDeviceTileTensor[dtype](win_layout, ctx)
            for r in range(n):
                for h in range(num_q_heads):
                    for d in range(kv_params.head_size):
                        win_q.host_tensor()[r, h, d] = q_host[
                            b * prompt + start + r, h, d
                        ]
            win_q.to_device()
            var win_offsets = HostDeviceTileTensor[.uint32](
                row_major(Int64(2)), ctx
            )
            win_offsets.host_tensor()[0] = 0
            win_offsets.host_tensor()[1] = UInt32(n)
            win_offsets.to_device()
            var win_cache = HostDeviceTileTensor[.uint32](
                row_major(Int64(1)), ctx
            )
            win_cache.host_tensor()[0] = UInt32(cache + start)
            win_cache.to_device()
            var win_lut = HostDeviceTileTensor[.uint32](
                row_major(Int64(1), Int64(lut_cols)), ctx
            )
            for p in range(lut_cols):
                win_lut.host_tensor()[0, p] = lut.host_tensor()[b, p]
            win_lut.to_device()
            var win_context = cache + start + n
            var ref_kv = Collection(
                blocks.device_tensor().as_unsafe_any_origin(),
                win_cache.device_tensor().as_imm().as_unsafe_any_origin(),
                win_lut.device_tensor().as_imm().as_unsafe_any_origin(),
                UInt32(n),
                UInt32(win_context),
            )
            mha_gpu_naive[ragged=True](
                win_q.device_tensor(),
                ref_kv.get_key_cache(0),
                ref_kv.get_value_cache(0),
                CausalMask(),
                win_ref.device_tensor(),
                win_offsets.device_tensor(),
                scale,
                1,
                n,
                win_context,
                num_q_heads,
                kv_params.head_size,
                group,
                ctx,
            )
            assert_no_nan_inf(win_ref, "ref_output")
            var ref_host = win_ref.host_tensor()
            for r in range(n):
                for h in range(num_q_heads):
                    for d in range(kv_params.head_size):
                        var got = out_host[b * prompt + start + r, h, d]
                        var want = ref_host[r, h, d]
                        max_err = max(
                            max_err,
                            abs(got.cast[.float32]() - want.cast[.float32]()),
                        )
                        assert_almost_equal(got, want, atol=2e-3, rtol=1e-2)
    print("max |err|", max_err)


def main() raises:
    seed(7)
    with DeviceContext() as ctx:
        run_case(1, 2400, 32768, 1200, ctx)
        run_case(1, 2390, 32768, 1200, ctx)
        run_case(2, 1200, 32768, 1200, ctx)
        # An 8192-token prefill chunk at 102400 cached tokens.
        run_case(1, 8192, 102400, 512, ctx)
