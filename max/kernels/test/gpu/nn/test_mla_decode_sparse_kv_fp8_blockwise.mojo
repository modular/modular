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
"""Numerical test of bf16-Q sparse MLA decode over a blockwise-scaled cache.

`flare_mla_decoding` sends this configuration to the unified sparse kernel's
in-place conversion. Every scale block of every cached row carries its own
power-of-two scale (the kernel stages scales as e8m0), and each token
reads many 64-key tiles, so a tile that reads another tile's scales fails.
"""

from std.math import ceildiv, exp, exp2, floor
from std.random import rand, randn, seed

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from internal_utils import coprime_multiplier
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig, NonNullPointer
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)

comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1


def _reference[
    num_heads: Int, top_k: Int, scale_block_size: Int
](
    q_device: DeviceBuffer[DType.bfloat16],
    blocks_device: DeviceBuffer[DType.float8_e4m3fn],
    scales_device: DeviceBuffer[DType.float32],
    d_indices_device: DeviceBuffer[DType.int32],
    topk_lengths_device: DeviceBuffer[DType.uint32],
    total_q_tokens: Int,
    scale: Float32,
) raises -> List[Float32]:
    comptime scales_per_token = ceildiv(Q_DEPTH, scale_block_size)
    var out = List[Float32](length=total_q_tokens * num_heads * V_DEPTH, fill=0)
    with q_device.map_to_host() as q_h, blocks_device.map_to_host() as kv_h, scales_device.map_to_host() as s_h, d_indices_device.map_to_host() as idx_h, topk_lengths_device.map_to_host() as tl_h:
        var p = List[Float64](length=top_k, fill=0)
        var acc = List[Float64](length=V_DEPTH, fill=0)
        for g in range(total_q_tokens):
            var n = Int(tl_h[g])
            for h in range(num_heads):
                var qb = (g * num_heads + h) * Q_DEPTH
                var mx = Float64(-1e300)
                for j in range(n):
                    var r = Int(idx_h[g * top_k + j])
                    var dot = Float64(0)
                    for d in range(Q_DEPTH):
                        dot += (
                            q_h[qb + d].cast[.float64]()
                            * kv_h[r * Q_DEPTH + d].cast[.float64]()
                            * s_h[
                                r * scales_per_token + d // scale_block_size
                            ].cast[.float64]()
                        )
                    p[j] = dot * Float64(scale)
                    mx = max(mx, p[j])
                var den = Float64(0)
                for d in range(V_DEPTH):
                    acc[d] = 0
                for j in range(n):
                    var w = exp(p[j] - mx)
                    den += w
                    var r = Int(idx_h[g * top_k + j])
                    for d in range(V_DEPTH):
                        acc[d] += (
                            w
                            * kv_h[r * Q_DEPTH + d].cast[.float64]()
                            * s_h[
                                r * scales_per_token + d // scale_block_size
                            ].cast[.float64]()
                        )
                var ob = (g * num_heads + h) * V_DEPTH
                for d in range(V_DEPTH):
                    out[ob + d] = Float32(acc[d] / den)
    return out^


def run_test[
    num_heads: Int,
    top_k: Int,
    skip_masked_tiles: Bool,
    scale_block_size: Int = 32,
](ctx: DeviceContext, batch_size: Int, seq_len: Int, cache_len: Int) raises:
    comptime scales_per_token = ceildiv(Q_DEPTH, scale_block_size)
    comptime q_dtype = DType.bfloat16
    comptime kv_dtype = DType.float8_e4m3fn
    comptime scale = Float32(0.125)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    print(
        "num_heads=",
        num_heads,
        " top_k=",
        top_k,
        " skip_masked_tiles=",
        skip_masked_tiles,
        " batch_size=",
        batch_size,
        " seq_len=",
        seq_len,
        " cache_len=",
        cache_len,
        sep="",
    )

    var total_q_tokens = batch_size * seq_len
    var num_keys = cache_len + seq_len
    var pages_per_seq = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_seq

    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * Q_DEPTH
    var blocks_device = ctx.enqueue_create_buffer[kv_dtype](block_elems)
    with blocks_device.map_to_host() as blocks_host:
        randn(blocks_host.as_span(), mean=0.0, standard_deviation=2.0)
    var scales_elems = total_pages * NUM_LAYERS * PAGE_SIZE * scales_per_token
    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    with scales_device.map_to_host() as scales_host:
        rand(scales_host.as_span(), min=0.0, max=3.0)
        for i in range(scales_elems):
            scales_host[i] = exp2(-floor(scales_host[i]))

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](total_pages)
    with lookup_table_device.map_to_host() as lut_host:
        for i in range(total_pages):
            lut_host[i] = UInt32(i)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_len)

    var q_device = ctx.enqueue_create_buffer[q_dtype](
        total_q_tokens * num_heads * Q_DEPTH
    )
    with q_device.map_to_host() as q_host:
        randn(q_host.as_span(), mean=0.0, standard_deviation=1.0)

    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_elems)

    # The token at position p draws min(top_k, p + 1) distinct keys from
    # [0, p], spread across the sequence's pages, and pads the rest with -1.
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * top_k
    )
    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    with d_indices_device.map_to_host() as idx_host:
        with topk_lengths_device.map_to_host() as tl_host:
            for bi in range(batch_size):
                for s in range(seq_len):
                    var g = bi * seq_len + s
                    var ring = cache_len + s + 1
                    var n_valid = min(top_k, ring)
                    tl_host[g] = UInt32(n_valid)
                    var mult = coprime_multiplier(ring)
                    for i in range(top_k):
                        if i >= n_valid:
                            idx_host[g * top_k + i] = -1
                            continue
                        var t = (i * mult + 1 + s) % ring
                        var block_id = bi * pages_per_seq + t // PAGE_SIZE
                        idx_host[g * top_k + i] = Int32(
                            block_id * PAGE_SIZE + t % PAGE_SIZE
                        )

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with row_offsets_device.map_to_host() as ro_host:
        for i in range(batch_size + 1):
            ro_host[i] = UInt32(i * seq_len)
    ctx.synchronize()

    comptime Collection = PagedKVCacheCollection[
        kv_dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=scale_block_size,
    ]
    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var scales_tt = TileTensor(
        scales_device,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[scales_per_token],
        ),
    )
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(pages_per_seq)),
        )
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(cache_len),
        rebind[Collection.scales_tt_type](scales_tt.as_unsafe_any_origin()),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major(total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])
    )
    var out_tt = TileTensor(
        out_device, row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH])
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )
    var mla_args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, seq_len, ctx
    )
    var d_indices = rebind[MutPointer[Int32, MutAnyOrigin]](
        d_indices_device.unsafe_ptr()
    )

    comptime if skip_masked_tiles:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_dtype](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
            skip_masked_tiles=True,
            kernels=SM100MLA,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            mla_args.gpu_tile_tensor(),
            d_indices=d_indices,
            indices_stride=top_k,
            topk_lengths=NonNullPointer[DType.int32](
                rebind[Pointer[Int32, ImmutAnyOrigin]](
                    topk_lengths_device.unsafe_ptr().unsafe_bitcast[Int32]()
                )
            ),
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_dtype](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
            kernels=SM100MLA,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            mla_args.gpu_tile_tensor(),
            d_indices=d_indices,
            indices_stride=top_k,
        )
    ctx.synchronize()

    var expected = _reference[num_heads, top_k, scale_block_size](
        q_device,
        blocks_device,
        scales_device,
        d_indices_device,
        topk_lengths_device,
        total_q_tokens,
        scale,
    )
    # bf16 output and bf16-rounded K/V products; the scales themselves are
    # exact in e8m0.
    comptime atol = Float32(5e-2)
    comptime rtol = Float32(2e-2)
    var num_bad = 0
    var first_bad = -1
    with out_device.map_to_host() as out_host:
        for i in range(out_elems):
            var got = out_host[i].cast[.float32]()
            if not (abs(got - expected[i]) <= atol + rtol * abs(expected[i])):
                if first_bad < 0:
                    first_bad = i
                num_bad += 1
        if num_bad > 0:
            raise Error(
                String(
                    num_bad,
                    " of ",
                    out_elems,
                    " outputs out of tolerance; first at ",
                    first_bad,
                    ": got ",
                    out_host[first_bad],
                    ", expected ",
                    expected[first_bad],
                )
            )

    _ = mla_args
    _ = blocks_device
    _ = scales_device
    _ = lookup_table_device
    _ = cache_lengths_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        # Decode-shaped: every token reads all 32 tiles of a long context.
        run_test[8, 2048, False](ctx, 2, 3, 4096)
        run_test[64, 2048, False](ctx, 1, 2, 4096)
        # Coarser scale blocks.
        run_test[16, 2048, False, 128](ctx, 3, 1, 4096)
        run_test[64, 2048, False, 64](ctx, 2, 1, 2048)
        # Prefill-shaped on an empty context: per-token lengths skip tiles,
        # and short rows end in `-1` slots that must read at unit scale.
        run_test[8, 2048, True](ctx, 1, 160, 0)
        run_test[8, 2048, True](ctx, 2, 3, 4096)
        run_test[16, 2048, True](ctx, 2, 3, 4096)
        run_test[32, 2048, True](ctx, 1, 64, 0)
        run_test[64, 2048, True](ctx, 1, 160, 0)
