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

"""Bit-exact-ish correctness test for SM100 MHA with FP8 KV cache (d=512) — PREFILL.

Prefill-only sibling of `test_mha_sm100_fp8_kv_d512_decode.mojo`. Both
files were split out of the original `test_mha_sm100_fp8_kv_d512.mojo`
so the prefill and decode paths can advance independently. Mirrors the
same split done for d256.

Slice 3 of the Gemma4 FP8 KV cache design. Exercises the **same**
`dequant_paged_fp8_kv_to_bf16` kernel surface at head_dim=512 — the
shape used by Gemma4's global (full-attention) layers
(`gemma4/model_config.py:90,93`):

    n_kv_heads          = num_global_key_value_heads = 4
    global_head_dim     = 512

`num_attention_heads` is HF-config-driven (not hardcoded in MAX); we
test group ∈ {4, 8} which spans the realistic n_q_heads ∈ {16, 32}
range for 31B-class models.

Multi-token Q cases (`seq_len > 1`) at head_dim=512 land in
`mha_sm100_depth512_dispatch` (the depth=256/512 pair-CTA prefill
kernel) — the in-scope target for the FP8 fusion (Tier 1 REVISED).

Acceptance bar: cosine ≥ 0.9995 between bf16-reference and
fp8-via-staging attention outputs.

Target hardware family: NVIDIA SM100 (B200).
"""

from std.math import ceildiv, sqrt
from std.memory import memset_zero
from std.random import seed
from std.collections import Set

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from kv_cache.types import KVCacheStaticParams

from nn.attention.gpu.mha import flash_attention
from nn.attention.gpu.nvidia.sm100.mha_fp8_kv import (
    dequant_paged_fp8_kv_to_bf16,
)
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    SlidingWindowCausalMask,
)

from std.testing import assert_true


# ===-----------------------------------------------------------------------===#
# Helpers (duplicated from test_mha_sm100_fp8_kv_d256_{prefill,decode}.mojo,
# parametric on head_dim).
# ===-----------------------------------------------------------------------===#


@always_inline
def _cosine_similarity(
    a_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    b_ptr: UnsafePointer[Scalar[DType.bfloat16], _],
    n: Int,
) -> Float64:
    var dot: Float64 = 0.0
    var aa: Float64 = 0.0
    var bb: Float64 = 0.0
    for i in range(n):
        var a = a_ptr[i].cast[DType.float64]()
        var b = b_ptr[i].cast[DType.float64]()
        dot += a * b
        aa += a * a
        bb += b * b
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return dot / (sqrt(aa) * sqrt(bb))


def _quantize_bf16_to_fp8_blockwise[
    head_dim: Int,
    granularity: Int,
](
    src_bf16: UnsafePointer[Scalar[DType.bfloat16], _],
    dst_fp8: UnsafePointer[mut=True, Scalar[DType.float8_e4m3fn], _],
    dst_scales: UnsafePointer[mut=True, Scalar[DType.float32], _],
    n_rows: Int,
):
    comptime fp8_max = Float32(448.0)
    comptime gran = granularity
    comptime n_blocks = ceildiv(head_dim, gran)

    for r in range(n_rows):
        for b in range(n_blocks):
            var lo = b * gran
            var hi = min(lo + gran, head_dim)
            var absmax: Float32 = 0.0
            for j in range(lo, hi):
                var v = abs(src_bf16[r * head_dim + j].cast[DType.float32]())
                if v > absmax:
                    absmax = v
            var scale = absmax / fp8_max
            if scale < 1e-12:
                scale = 1e-12
            dst_scales[r * n_blocks + b] = scale
            for j in range(lo, hi):
                var v = src_bf16[r * head_dim + j].cast[DType.float32]() / scale
                dst_fp8[r * head_dim + j] = v.cast[DType.float8_e4m3fn]()


# ===-----------------------------------------------------------------------===#
# Core test (parametric on head_dim)
# ===-----------------------------------------------------------------------===#


def execute_fp8_kv_test[
    MaskType: MHAMask,
    *,
    head_dim: Int,
    num_q_heads: Int,
    group: Int,
    seq_len: Int,
    num_keys: Int,
    mask_name: StaticString,
](mask: MaskType, ctx: DeviceContext,) raises:
    comptime assert seq_len > 1, (
        "Prefill test file must run with seq_len > 1. Use the _decode test"
        " file for seq_len == 1 cases (which route through mha_1q.mojo)."
    )
    comptime g = 64
    comptime page_size = 128
    comptime kv_num_heads = num_q_heads // group
    comptime batch_size = 1
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))

    print(
        "test_mha_sm100_fp8_kv_d",
        head_dim,
        "_prefill:",
        " mask=",
        mask_name,
        " group=",
        group,
        " n_q_heads=",
        num_q_heads,
        " n_kv_heads=",
        kv_num_heads,
        " seq_len=",
        seq_len,
        " num_keys=",
        num_keys,
        " head_dim=",
        head_dim,
        " g=",
        g,
        " page_size=",
        page_size,
    )

    seed(0xC0FFEE)

    comptime in_dtype = DType.bfloat16
    comptime kv_dtype = DType.float8_e4m3fn
    comptime scale_dtype = DType.float32
    comptime head_dim_gran = ceildiv(head_dim, g)
    var n_blocks_per_seq = ceildiv(num_keys, page_size)
    # Slice 4: no even/odd parity constraint on physical blocks.
    var num_paged_blocks = 2 * n_blocks_per_seq + 8
    if num_paged_blocks < 16:
        num_paged_blocks = 16

    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    var q_size = batch_size * seq_len * num_q_heads * head_dim
    var k_size = batch_size * num_keys * kv_num_heads * head_dim
    var v_size = k_size
    var o_size = q_size

    var q_host = alloc[Scalar[in_dtype]](q_size)
    var k_host = alloc[Scalar[in_dtype]](k_size)
    var v_host = alloc[Scalar[in_dtype]](v_size)
    var k_host_tt = TileTensor(k_host, row_major((k_size,)))
    var v_host_tt = TileTensor(v_host, row_major((v_size,)))
    var q_host_tt = TileTensor(q_host, row_major((q_size,)))
    random(q_host_tt)
    random(k_host_tt)
    random(v_host_tt)

    var k_fp8_host = alloc[Scalar[kv_dtype]](k_size)
    var v_fp8_host = alloc[Scalar[kv_dtype]](v_size)
    var k_scales_size = batch_size * num_keys * kv_num_heads * head_dim_gran
    var v_scales_size = k_scales_size
    var k_scales_host = alloc[Scalar[scale_dtype]](k_scales_size)
    var v_scales_host = alloc[Scalar[scale_dtype]](v_scales_size)

    _quantize_bf16_to_fp8_blockwise[head_dim, g](
        k_host, k_fp8_host, k_scales_host, batch_size * num_keys * kv_num_heads
    )
    _quantize_bf16_to_fp8_blockwise[head_dim, g](
        v_host, v_fp8_host, v_scales_host, batch_size * num_keys * kv_num_heads
    )

    comptime num_layers = 1
    # HBM staging buffer scales linearly with head_dim. At d=512 this is
    # 2x what Slice 2a saw at d=256: BN×512×2 = 64 KiB/page × num_pages.
    # The test's largest shape (4096 keys, 32 pages, n_kv_heads=4) uses
    # ~32 MiB for paged_fp8 and another ~64 MiB for paged_bf16 shadow,
    # which fits comfortably in B200's 80 GiB HBM.
    var paged_fp8_size = (
        num_paged_blocks * 2 * num_layers * page_size * kv_num_heads * head_dim
    )
    var paged_scales_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_num_heads
        * head_dim_gran
    )

    var paged_fp8_host = alloc[Scalar[kv_dtype]](paged_fp8_size)
    var paged_scales_host = alloc[Scalar[scale_dtype]](paged_scales_size)
    memset_zero(paged_fp8_host, paged_fp8_size)
    memset_zero(paged_scales_host, paged_scales_size)

    # Arbitrary mixed-parity physical blocks (Slice 4: aliasing bug fixed).
    var k_lut = alloc[UInt32](n_blocks_per_seq)
    var v_lut = alloc[UInt32](n_blocks_per_seq)
    var used = Set[Int]()
    for i in range(n_blocks_per_seq):
        k_lut[i] = UInt32(i)
        used.add(i)
    for i in range(n_blocks_per_seq):
        var v_phys = n_blocks_per_seq + i
        v_lut[i] = UInt32(v_phys)
        used.add(v_phys)

    for tok in range(num_keys):
        var blk_seq = tok // page_size
        var off_in_block = tok % page_size
        var k_phys = Int(k_lut[blk_seq])
        var v_phys = Int(v_lut[blk_seq])
        for h in range(kv_num_heads):
            var src_k_off = (tok * kv_num_heads + h) * head_dim
            var dst_k_off = (
                (((k_phys * 2 + 0) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim
            for d in range(head_dim):
                paged_fp8_host[dst_k_off + d] = k_fp8_host[src_k_off + d]
            var src_ks_off = (tok * kv_num_heads + h) * head_dim_gran
            var dst_ks_off = (
                (((k_phys * 2 + 0) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim_gran
            for b in range(head_dim_gran):
                paged_scales_host[dst_ks_off + b] = k_scales_host[
                    src_ks_off + b
                ]
            var src_v_off = (tok * kv_num_heads + h) * head_dim
            var dst_v_off = (
                (((v_phys * 2 + 1) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim
            for d in range(head_dim):
                paged_fp8_host[dst_v_off + d] = v_fp8_host[src_v_off + d]
            var src_vs_off = (tok * kv_num_heads + h) * head_dim_gran
            var dst_vs_off = (
                (((v_phys * 2 + 1) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim_gran
            for b in range(head_dim_gran):
                paged_scales_host[dst_vs_off + b] = v_scales_host[
                    src_vs_off + b
                ]

    var paged_fp8_dev = ctx.enqueue_create_buffer[kv_dtype](paged_fp8_size)
    var paged_scales_dev = ctx.enqueue_create_buffer[scale_dtype](
        paged_scales_size
    )
    var paged_bf16_dev = ctx.enqueue_create_buffer[in_dtype](paged_fp8_size)
    ctx.enqueue_copy(paged_fp8_dev, paged_fp8_host)
    ctx.enqueue_copy(paged_scales_dev, paged_scales_host)

    dequant_paged_fp8_kv_to_bf16[
        kv_params=kv_params,
        page_size=page_size,
        quantization_granularity=g,
    ](
        paged_fp8_dev.unsafe_ptr(),
        paged_scales_dev.unsafe_ptr(),
        paged_bf16_dev.unsafe_ptr(),
        num_paged_blocks=num_paged_blocks,
        num_layers=num_layers,
        layer_idx=0,
        ctx=ctx,
    )
    ctx.synchronize()

    var paged_bf16_host = alloc[Scalar[in_dtype]](paged_fp8_size)
    ctx.enqueue_copy(paged_bf16_host, paged_bf16_dev)
    ctx.synchronize()

    var k_dequant_host = alloc[Scalar[in_dtype]](k_size)
    var v_dequant_host = alloc[Scalar[in_dtype]](v_size)
    for tok in range(num_keys):
        var blk_seq = tok // page_size
        var off_in_block = tok % page_size
        var k_phys = Int(k_lut[blk_seq])
        var v_phys = Int(v_lut[blk_seq])
        for h in range(kv_num_heads):
            var src_k_off = (
                (((k_phys * 2 + 0) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim
            var dst_k_off = (tok * kv_num_heads + h) * head_dim
            for d in range(head_dim):
                k_dequant_host[dst_k_off + d] = paged_bf16_host[src_k_off + d]
            var src_v_off = (
                (((v_phys * 2 + 1) * num_layers + 0) * page_size + off_in_block)
                * kv_num_heads
                + h
            ) * head_dim
            var dst_v_off = (tok * kv_num_heads + h) * head_dim
            for d in range(head_dim):
                v_dequant_host[dst_v_off + d] = paged_bf16_host[src_v_off + d]

    var q_dev = ctx.enqueue_create_buffer[in_dtype](q_size)
    var k_ref_dev = ctx.enqueue_create_buffer[in_dtype](k_size)
    var v_ref_dev = ctx.enqueue_create_buffer[in_dtype](v_size)
    var k_dq_dev = ctx.enqueue_create_buffer[in_dtype](k_size)
    var v_dq_dev = ctx.enqueue_create_buffer[in_dtype](v_size)
    var out_ref_dev = ctx.enqueue_create_buffer[in_dtype](o_size)
    var out_dq_dev = ctx.enqueue_create_buffer[in_dtype](o_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_ref_dev, k_host)
    ctx.enqueue_copy(v_ref_dev, v_host)
    ctx.enqueue_copy(k_dq_dev, k_dequant_host)
    ctx.enqueue_copy(v_dq_dev, v_dequant_host)

    var q_lt = TileTensor(
        q_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_q_heads],
                Idx[head_dim],
            )
        ),
    )
    var k_ref_lt = TileTensor(
        k_ref_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var v_ref_lt = TileTensor(
        v_ref_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var k_dq_lt = TileTensor(
        k_dq_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var v_dq_lt = TileTensor(
        v_dq_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var out_ref_lt = TileTensor(
        out_ref_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_q_heads],
                Idx[head_dim],
            )
        ),
    )
    var out_dq_lt = TileTensor(
        out_dq_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_q_heads],
                Idx[head_dim],
            )
        ),
    )

    flash_attention(out_ref_lt, q_lt, k_ref_lt, v_ref_lt, mask, scale, ctx)
    flash_attention(out_dq_lt, q_lt, k_dq_lt, v_dq_lt, mask, scale, ctx)
    ctx.synchronize()

    var out_ref_host = alloc[Scalar[in_dtype]](o_size)
    var out_dq_host = alloc[Scalar[in_dtype]](o_size)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.enqueue_copy(out_dq_host, out_dq_dev)
    ctx.synchronize()

    var cos = _cosine_similarity(out_ref_host, out_dq_host, o_size)
    print("  cosine(bf16_ref, fp8_via_staging) =", cos)

    assert_true(cos >= 0.9995, "cosine below 0.9995 bar")


# ===-----------------------------------------------------------------------===#
# Entry point — prefill cases (seq_len > 1) only
# ===-----------------------------------------------------------------------===#


def main() raises:
    with DeviceContext() as ctx:
        var causal = CausalMask()

        # Gemma4-global: n_kv_heads=4, head_dim=512. Group depends on
        # num_attention_heads (HF-config-driven). Test group ∈ {4, 8}
        # which spans realistic n_q_heads ∈ {16, 32} for ~31B models.

        # CAUSAL_MASK at group=4 (smaller; n_q_heads=16).
        execute_fp8_kv_test[
            CausalMask,
            head_dim=512,
            num_q_heads=16,
            group=4,
            seq_len=256,
            num_keys=256,
            mask_name="CAUSAL_g4_s256",
        ](causal, ctx)
        execute_fp8_kv_test[
            CausalMask,
            head_dim=512,
            num_q_heads=16,
            group=4,
            seq_len=1024,
            num_keys=1024,
            mask_name="CAUSAL_g4_s1024",
        ](causal, ctx)

        # CAUSAL_MASK at group=8 (Gemma4-global-like; n_q_heads=32, n_kv_heads=4).
        execute_fp8_kv_test[
            CausalMask,
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=256,
            num_keys=256,
            mask_name="CAUSAL_g8_s256",
        ](causal, ctx)
        execute_fp8_kv_test[
            CausalMask,
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=1024,
            num_keys=1024,
            mask_name="CAUSAL_g8_s1024",
        ](causal, ctx)
        execute_fp8_kv_test[
            CausalMask,
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=2048,
            num_keys=2048,
            mask_name="CAUSAL_g8_s2048",
        ](causal, ctx)

        # Sliding-window variants at the Gemma4-global shape. Although
        # Gemma4 global layers do not use sliding-window in production
        # (they use full attention), this exercises the mask interaction
        # with the FA4 d=512 path under fp8-staging.
        var sw_1024 = SlidingWindowCausalMask[1024]()
        var sw_4096 = SlidingWindowCausalMask[4096]()
        execute_fp8_kv_test[
            SlidingWindowCausalMask[1024],
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=1024,
            num_keys=1024,
            mask_name="SW1024_g8_s1024",
        ](sw_1024, ctx)
        execute_fp8_kv_test[
            SlidingWindowCausalMask[1024],
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=2048,
            num_keys=2048,
            mask_name="SW1024_g8_s2048",
        ](sw_1024, ctx)
        execute_fp8_kv_test[
            SlidingWindowCausalMask[4096],
            head_dim=512,
            num_q_heads=32,
            group=8,
            seq_len=2048,
            num_keys=2048,
            mask_name="SW4096_g8_s2048",
        ](sw_4096, ctx)

        print("test_mha_sm100_fp8_kv_d512_prefill: ALL PASSED")
