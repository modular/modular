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

# Decode with attention sinks. The FA2 decode kernels seeded the sink softmax
# state on warp 0 only, so the depth columns owned by the other warps
# normalized their output by the key-only sum.
#
# One decode step over a populated cache, compared against a float64 reference
# built from the same tensors. Sink logits differ sharply across query heads,
# so the case covers both sink-dominated and key-dominated heads.

from std.collections import OptionalReg
from std.math import abs, exp, rsqrt
from std.random import seed

from max.gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import flash_attention
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal


def test_decode_with_sink[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    dtype: DType,
](cache_len: Int, sink_logit: Float64, ctx: DeviceContext,) raises:
    comptime batch_size = 1
    comptime num_layers = 2
    comptime layer_idx = 1
    comptime head_size = kv_params.head_size
    comptime group = num_q_heads // kv_params.num_heads

    print(
        "decode sink: heads=",
        num_q_heads,
        "kv_heads=",
        kv_params.num_heads,
        "head_size=",
        head_size,
        "cache_len=",
        cache_len,
        "sink_logit=",
        sink_logit,
    )

    var total_length = 1
    var max_context = cache_len + 1
    var q_layout = row_major(total_length, Idx[num_q_heads], Idx[head_size])

    var row_offsets = HostDeviceTileTensor[.uint32](row_major(Int64(2)), ctx)
    var row_offsets_host = row_offsets.host_tensor()
    row_offsets_host[0] = 0
    row_offsets_host[1] = 1
    row_offsets.to_device()

    var cache_lens = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    var cache_lens_host = cache_lens.host_tensor()
    cache_lens_host[0] = UInt32(cache_len)
    cache_lens.to_device()

    var q = HostDeviceTileTensor[dtype](q_layout, ctx)
    random(q.host_tensor())
    q.to_device()

    var output = HostDeviceTileTensor[dtype](q_layout, ctx)

    var num_blocks = batch_size + 2
    var kv_block = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Int64(2),
            Int64(num_layers),
            Int64(max_context),
            Idx[kv_params.num_heads],
            Idx[head_size],
        ),
        ctx,
    )
    random(kv_block.host_tensor())
    kv_block.to_device()

    var lookup_table = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    lookup_table.host_tensor()[0] = 0
    lookup_table.to_device()

    comptime Collection = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, MutAnyOrigin, ImmutAnyOrigin, ImmutAnyOrigin
    ]
    var kv_collection = Collection(
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lens.device_tensor().as_imm().as_unsafe_any_origin(),
        lookup_table.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(max_context),
    )

    var sinks = HostDeviceTileTensor[dtype](row_major(Int64(num_q_heads)), ctx)
    # Alternating regimes, flipped every band of 8, so every 8-row band keeps a
    # mix of sink-dominated and key-dominated heads and the bands do not line up
    # head for head.
    var sinks_host = sinks.host_tensor()
    for h in range(num_q_heads):
        var dominant = (h + h // 8) % 2 == 0
        sinks_host[h] = Scalar[dtype](
            sink_logit if dominant else sink_logit - 20.0
        )
    sinks.to_device()

    var sink_view = OptionalReg[ImmutTileTensor1D[dtype]](
        sinks.device_tensor().as_imm().as_unsafe_any_origin()
    )

    var scale = rsqrt(Float32(head_size))
    flash_attention[ragged=True, sink=True](
        output.device_tensor(),
        q.device_tensor(),
        kv_collection.get_key_cache(layer_idx),
        kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        row_offsets.device_tensor(),
        scale,
        ctx,
        sink_weights=sink_view,
    )
    output.to_host()

    var out_host = output.host_tensor()
    var q_host = q.host_tensor()
    var kv_host = kv_block.host_tensor()
    var sink_host = sinks.host_tensor()

    # Softmax over the sink logit and every key the query sees, then the value
    # average. The query's own K/V sits at `cache_len`, so it attends
    # `cache_len + 1` keys; the sink has no value row.
    var num_keys = cache_len + 1
    var worst_err = Float64(0)
    var probs = List[Float64]()
    for h in range(num_q_heads):
        var kv_head = h // group
        var sink_logit_h = Float64(sink_host[h].cast[.float64]())
        var scores = List[Float64]()
        var max_score = sink_logit_h
        for j in range(num_keys):
            var dot = Float64(0)
            for d in range(head_size):
                dot += Float64(q_host[0, h, d].cast[.float64]()) * Float64(
                    kv_host[0, 0, layer_idx, j, kv_head, d].cast[.float64]()
                )
            var score = Float64(scale) * dot
            scores.append(score)
            if score > max_score:
                max_score = score

        var denom = exp(sink_logit_h - max_score)
        for j in range(num_keys):
            denom += exp(scores[j] - max_score)

        probs.clear()
        for j in range(num_keys):
            probs.append(exp(scores[j] - max_score) / denom)

        for d in range(head_size):
            var acc = Float64(0)
            for j in range(num_keys):
                acc += probs[j] * Float64(
                    kv_host[0, 1, layer_idx, j, kv_head, d].cast[.float64]()
                )
            var actual = Float64(out_host[0, h, d].cast[.float64]())
            var err = abs(actual - acc)
            if err > worst_err:
                worst_err = err
            assert_almost_equal(actual, acc, atol=5e-3, rtol=2e-2)

    print("worst abs err=", worst_err)


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Flint shape: 32 query heads over 4 KV heads, depth 128.
        test_decode_with_sink[
            32,
            KVCacheStaticParams(num_heads=4, head_size=128),
            DType.bfloat16,
        ](65, 20.0, ctx)

        # Single KV head, longer cache that spans several key tiles.
        test_decode_with_sink[
            8,
            KVCacheStaticParams(num_heads=1, head_size=128),
            DType.bfloat16,
        ](257, 20.0, ctx)

        # Same shape in fp32. `fa4_route` only claims half-float and fp8 decode,
        # so this case keeps the FA2 decode ladder on SM100, where the bf16 cases
        # above route to the FA4 kernel and never reach the seeded code.
        test_decode_with_sink[
            32,
            KVCacheStaticParams(num_heads=4, head_size=128),
            DType.float32,
        ](65, 20.0, ctx)

        print("PASS")
