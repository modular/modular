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

"""Correctness test for GQA packing in the SM100 d256 shared-key decode kernel.

The shared-key BM=32 vehicle packs a GQA group that does not divide 32
(Qwen3.8: 24 q heads / 4 kv heads, group 6) into one tile of
`token * group + head` rows, leaving `32 % group` pad rows. This test runs
the native-fp8 paged ragged `flash_attention` against a bf16 paged reference
built from the same (losslessly upcast) data, over ragged batches with 1-4
query tokens, cache lengths straddling page boundaries, and a per-(token,
head) cosine floor so a single wrong head or pad-row leak cannot hide in the
aggregate.

Target hardware family: NVIDIA SM100 (B200).
"""

from std.collections import Set
from std.math import ceildiv, sqrt
from std.random import random_ui64, randn, seed

from max.gpu.host import DeviceContext
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)

from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, MHAMask

from std.testing import assert_true

# `PagedKVCache`'s SIMD lookup needs the LUT row stride to be a multiple of 8
# and at least `cols + 15`.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def execute_pack_gqa_test[
    MaskType: MHAMask,
    *,
    num_q_heads: Int,
    group: Int,
    mask_name: StaticString,
    page_size: Int = 256,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    mask: MaskType,
    ctx: DeviceContext,
    cos_bar: Float64 = 0.9997,
) raises:
    comptime head_dim = 256
    comptime kv_num_heads = num_q_heads // group
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime bf16_dtype = DType.bfloat16
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    var batch_size = len(valid_lengths)
    var total_length = 0
    var max_ctx = 0
    var max_prompt = 0
    for i in range(batch_size):
        max_ctx = max(max_ctx, cache_lengths[i] + valid_lengths[i])
        max_prompt = max(max_prompt, valid_lengths[i])
        total_length += valid_lengths[i]

    print(
        "test_mha_sm100_d256_pack_gqa: mask=",
        mask_name,
        " group=",
        group,
        " n_q=",
        num_q_heads,
        " n_kv=",
        kv_num_heads,
        " bs=",
        batch_size,
        " max_ctx=",
        max_ctx,
        " max_prompt=",
        max_prompt,
    )

    var q_layout = row_major(total_length, Idx[num_q_heads], Idx[head_dim])

    var input_row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size + 1))), ctx
    )
    var cache_lengths_managed = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size))), ctx
    )
    var row_offsets_host = input_row_offsets.host_tensor()
    var cache_lengths_host = cache_lengths_managed.host_tensor()
    var running: UInt32 = 0
    for i in range(batch_size):
        row_offsets_host[i] = running
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running += UInt32(valid_lengths[i])
    row_offsets_host[batch_size] = running
    input_row_offsets.to_device()
    cache_lengths_managed.to_device()

    var q_fp8 = HostDeviceTileTensor[fp8_dtype](q_layout, ctx)
    var q_bf16 = HostDeviceTileTensor[bf16_dtype](q_layout, ctx)
    var q_fp8_host = q_fp8.host_tensor().unsafe_ptr()
    var q_bf16_host = q_bf16.host_tensor().unsafe_ptr()
    randn(q_fp8_host, total_length * num_q_heads * head_dim)
    for i in range(total_length * num_q_heads * head_dim):
        q_bf16_host[i] = q_fp8_host[i].cast[bf16_dtype]()
    q_fp8.to_device()
    q_bf16.to_device()

    var num_blocks = ceildiv(max_ctx, page_size) * batch_size + 4
    var kv_layout = row_major(
        Coord(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[kv_num_heads],
            Idx[head_dim],
        )
    )
    var kv_fp8 = HostDeviceTileTensor[fp8_dtype](kv_layout, ctx)
    var kv_bf16 = HostDeviceTileTensor[bf16_dtype](kv_layout, ctx)
    var kv_fp8_host = kv_fp8.host_tensor().unsafe_ptr()
    var kv_bf16_host = kv_bf16.host_tensor().unsafe_ptr()

    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(
            Coord(
                Int64(batch_size),
                Int64(padded_lut_cols(ceildiv(max_ctx, page_size))),
            )
        ),
        ctx,
    )
    var paged_lut_host = paged_lut.host_tensor()
    var used = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for blk in range(ceildiv(seq_len, page_size)):
            var r = Int(random_ui64(0, UInt64(num_blocks - 1)))
            while r in used:
                r = Int(random_ui64(0, UInt64(num_blocks - 1)))
            used.add(r)
            paged_lut_host[bs, blk] = UInt32(r)

    var kv_pool = (
        num_blocks * 2 * num_layers * page_size * kv_num_heads * head_dim
    )
    randn(kv_fp8_host, kv_pool)
    for i in range(kv_pool):
        kv_bf16_host[i] = kv_fp8_host[i].cast[bf16_dtype]()
    kv_fp8.to_device()
    kv_bf16.to_device()
    paged_lut.to_device()

    var out_fp8 = HostDeviceTileTensor[bf16_dtype](q_layout, ctx)
    var out_ref = HostDeviceTileTensor[bf16_dtype](q_layout, ctx)

    comptime Fp8Collection = PagedKVCacheCollection[
        fp8_dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime Bf16Collection = PagedKVCacheCollection[
        bf16_dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var cache_lengths_device = (
        cache_lengths_managed.device_tensor().as_imm().as_unsafe_any_origin()
    )
    var paged_lut_device = (
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin()
    )
    var coll_fp8 = Fp8Collection(
        kv_fp8.device_tensor().as_unsafe_any_origin(),
        cache_lengths_device,
        paged_lut_device,
        UInt32(max_prompt),
        UInt32(max_ctx),
    )
    var coll_bf16 = Bf16Collection(
        kv_bf16.device_tensor().as_unsafe_any_origin(),
        cache_lengths_device,
        paged_lut_device,
        UInt32(max_prompt),
        UInt32(max_ctx),
    )

    flash_attention[ragged=True](
        out_ref.device_tensor(),
        q_bf16.device_tensor(),
        coll_bf16.get_key_cache(layer_idx),
        coll_bf16.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        scale,
        ctx,
    )
    flash_attention[ragged=True](
        out_fp8.device_tensor(),
        q_fp8.device_tensor(),
        coll_fp8.get_key_cache(layer_idx),
        coll_fp8.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        scale,
        ctx,
    )
    ctx.synchronize()

    out_fp8.to_host()
    out_ref.to_host()
    var out_fp8_h = out_fp8.host_tensor()
    var out_ref_h = out_ref.host_tensor()
    var offs_h = input_row_offsets.host_tensor()

    comptime rtol = 5e-2
    comptime atol = 3e-1
    var num_mismatches = 0
    var num_compared = 0
    var max_abs_diff: Float64 = 0.0
    var dot: Float64 = 0.0
    var aa: Float64 = 0.0
    var bb: Float64 = 0.0
    var min_head_cos: Float64 = 1.0
    for bs in range(batch_size):
        var off = Int(offs_h[bs])
        for s in range(valid_lengths[bs]):
            for h in range(num_q_heads):
                var hdot: Float64 = 0.0
                var haa: Float64 = 0.0
                var hbb: Float64 = 0.0
                for d in range(head_dim):
                    var e = out_ref_h[off + s, h, d].cast[DType.float64]()[0]
                    var a = out_fp8_h[off + s, h, d].cast[DType.float64]()[0]
                    var diff = abs(a - e)
                    max_abs_diff = max(max_abs_diff, diff)
                    num_compared += 1
                    if not (diff <= atol + rtol * abs(e)):
                        if num_mismatches < 16:
                            print("mismatch bs=", bs, "s=", s, "h=", h, "d=", d)
                            print("  actual=", a, "expect=", e)
                        num_mismatches += 1
                    hdot += a * e
                    haa += a * a
                    hbb += e * e
                dot += hdot
                aa += haa
                bb += hbb
                var hc: Float64 = 0.0
                if haa > 0.0 and hbb > 0.0:
                    hc = hdot / (sqrt(haa) * sqrt(hbb))
                if not (hc >= min_head_cos):
                    min_head_cos = hc
                    if hc < 0.99:
                        print("low head cosine bs=", bs, "s=", s, "h=", h, hc)

    var cos: Float64 = 0.0
    if aa > 0.0 and bb > 0.0:
        cos = dot / (sqrt(aa) * sqrt(bb))
    print(
        "  mismatches=",
        num_mismatches,
        "/",
        num_compared,
        " max_abs_diff=",
        max_abs_diff,
        " cosine=",
        cos,
        " min_head_cosine=",
        min_head_cos,
    )
    # The 0.9997 aggregate bar is the fp8-vs-bf16 convention from the
    # short-context fp8 tests. It does not hold at long context for the
    # UNPACKED kernel either (measured on this test's data, group 6: unpacked
    # 0.99969 at 4k, ~0.99965 at 64k; packed 0.99971-0.99973 at 4k, 0.99968 at
    # 64k), so long-context callers pass a floor measured under both paths.
    # The per-element bars above and the per-head cosine stay at full strength.
    assert_true(cos >= cos_bar, "cosine below the aggregate bar")
    assert_true(min_head_cos >= 0.995, "a (token, head) cosine below 0.995")
    assert_true(num_mismatches == 0, "elements outside atol/rtol")
    print("  PASSED")


def ragged_batch(
    batch: Int, ctx_choices: List[Int]
) -> Tuple[List[Int], List[Int]]:
    """Batch of rows cycling 1..4 query tokens and `ctx_choices` cache lengths.
    """
    var valid = List[Int]()
    var cache = List[Int]()
    for i in range(batch):
        valid.append(1 + (i % 4))
        cache.append(ctx_choices[i % len(ctx_choices)])
    return (valid^, cache^)


def main() raises:
    seed(0xC0FFEE)
    with DeviceContext() as ctx:
        var causal = CausalMask()

        # Qwen3.8 shape (24 q / 4 kv, group 6): every token count x every
        # page-boundary cache length, batch 1.
        comptime for t in range(1, 5):
            for c in [1, 255, 256, 257, 4096]:
                execute_pack_gqa_test[
                    CausalMask,
                    num_q_heads=24,
                    group=6,
                    mask_name="CAUSAL_g6_single",
                ]([t], [c], causal, ctx)

        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_64k"
        ]([4], [65536 - 4], causal, ctx, cos_bar=0.9995)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_64k_1q"
        ]([1], [65535], causal, ctx, cos_bar=0.9995)

        # 5 tokens x 6 heads = 30 real rows: the largest decode row the route
        # admits (max_prompt_len * group <= 32) and the exact BM-minus-pad fill.
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_5tok"
        ]([5], [4091], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_5tok_pg"
        ]([5], [254], causal, ctx)
        execute_pack_gqa_test[
            CausalMask,
            num_q_heads=24,
            group=6,
            mask_name="CAUSAL_g6_mixed_5tok",
        ]([5, 1, 4, 2], [256, 1, 4096, 255], causal, ctx)

        # Ragged batches mixing token counts and cache lengths (the short
        # rows are the padded rows of a 4-token tile).
        var b19 = ragged_batch(19, [1, 255, 256, 257, 4096])
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_bs19"
        ](b19[0], b19[1], causal, ctx)
        var b32 = ragged_batch(32, [4096, 257, 1, 256, 255])
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_bs32"
        ](b32[0], b32[1], causal, ctx)
        var b22 = ragged_batch(22, [8192, 35000, 20000, 1000])
        execute_pack_gqa_test[
            CausalMask, num_q_heads=24, group=6, mask_name="CAUSAL_g6_bs22_long"
        ](b22[0], b22[1], causal, ctx, cos_bar=0.9995)

        # Other groups that do not divide 32 take the same packed path.
        execute_pack_gqa_test[
            CausalMask, num_q_heads=12, group=3, mask_name="CAUSAL_g3"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=20, group=5, mask_name="CAUSAL_g5"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=28, group=7, mask_name="CAUSAL_g7"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)

        # Divisor groups keep the pre-existing fuse path.
        execute_pack_gqa_test[
            CausalMask, num_q_heads=32, group=4, mask_name="CAUSAL_g4"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=32, group=8, mask_name="CAUSAL_g8"
        ]([1, 2, 3, 4], [255, 256, 1, 4096], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=32, group=2, mask_name="CAUSAL_g2"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)
        execute_pack_gqa_test[
            CausalMask, num_q_heads=8, group=1, mask_name="CAUSAL_g1"
        ]([1, 4, 3, 2, 4], [255, 256, 1, 4096, 257], causal, ctx)

        print("test_mha_sm100_d256_pack_gqa: ALL PASSED")
