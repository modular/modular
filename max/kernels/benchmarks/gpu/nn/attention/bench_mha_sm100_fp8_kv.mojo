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
"""Microbench: SM100 MHA with FP8 KV cache via dequant-staging.

Compares two decode-shape configurations side by side, both ending in
the same `flash_attention` SM100 path on bf16:

    bf16:      bf16 K,V              -> flash_attention
    fp8-stage: fp8 K,V + fp32 scales -> dequant_paged_fp8_kv_to_bf16
                                     -> flash_attention

The dequant-then-attention path is expected to be slower than bf16
because every layer pays an HBM-resident O(kv_bytes) dequant pass.
This benchmark records the gap a future fused convert+attention path
would have to close.

Shape sweep: M ∈ {1, 16, 32, 64, 128}, n_kv_heads=4, head_dim=256,
group=8 (Gemma4-local-like), CAUSAL_MASK, seq_len ∈ {2048, 16384},
page_size=128.

Target hardware family: NVIDIA SM100 (B200).
"""

from std.math import ceildiv, rsqrt, sqrt

from std.benchmark import (
    Bench,
    Bencher,
    BenchConfig,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams
from layout import Idx, TileTensor, row_major

from nn.attention.gpu.mha import flash_attention
from nn.attention.gpu.nvidia.sm100.mha_fp8_kv import (
    dequant_paged_fp8_kv_to_bf16,
)
from nn.attention.mha_mask import CausalMask


# ---------------------------------------------------------------------------
# bf16 baseline
# ---------------------------------------------------------------------------


def bench_bf16[
    num_heads: Int,
    group: Int,
    head_dim: Int,
](mut m: Bench, seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    comptime kv_num_heads = num_heads // group
    var scale = rsqrt(Float32(head_dim))
    var batch_size = 1
    var q_size = batch_size * seq_len * num_heads * head_dim
    var k_size = batch_size * num_keys * kv_num_heads * head_dim
    var v_size = k_size
    var o_size = q_size

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_size)
    var v_dev = ctx.enqueue_create_buffer[DType.bfloat16](v_size)
    var o_dev = ctx.enqueue_create_buffer[DType.bfloat16](o_size)
    q_dev.enqueue_fill(BFloat16(0.01))
    k_dev.enqueue_fill(BFloat16(0.02))
    v_dev.enqueue_fill(BFloat16(0.03))

    var q = TileTensor(
        q_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[head_dim],
            )
        ),
    )
    var k = TileTensor(
        k_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var v = TileTensor(
        v_dev,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[head_dim],
            )
        ),
    )
    var o = TileTensor(
        o_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[head_dim],
            )
        ),
    )

    @parameter
    @always_inline
    @__copy_capture(q, k, v, o, scale)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            flash_attention(o, q, k, v, CausalMask(), scale, ctx)

        b.iter_custom[_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "mha_sm100_bf16_baseline",
            input_id=String(
                "M=",
                seq_len,
                "/n_kv_heads=",
                kv_num_heads,
                "/group=",
                group,
                "/head_dim=",
                head_dim,
                "/seq_len=",
                seq_len,
                "/num_keys=",
                num_keys,
            ),
        ),
    )
    ctx.synchronize()


# ---------------------------------------------------------------------------
# FP8-via-staging path: dequant kernel + bf16 flash_attention
# ---------------------------------------------------------------------------


def bench_fp8_staged[
    num_heads: Int,
    group: Int,
    head_dim: Int,
    g: Int,
    page_size: Int,
](mut m: Bench, seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    comptime kv_num_heads = num_heads // group
    comptime head_dim_gran = ceildiv(head_dim, g)
    var scale = rsqrt(Float32(head_dim))
    var batch_size = 1
    var q_size = batch_size * seq_len * num_heads * head_dim
    var k_size = batch_size * num_keys * kv_num_heads * head_dim
    var o_size = q_size

    var n_blocks_per_seq = ceildiv(num_keys, page_size)
    var num_paged_blocks = max(2 * n_blocks_per_seq + 4, 8)

    var paged_fp8_size = (
        num_paged_blocks * 2 * page_size * kv_num_heads * head_dim
    )
    var paged_scales_size = (
        num_paged_blocks * 2 * page_size * kv_num_heads * head_dim_gran
    )

    var fp8_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](paged_fp8_size)
    var scales_dev = ctx.enqueue_create_buffer[DType.float32](paged_scales_size)
    var bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](paged_fp8_size)
    fp8_dev.enqueue_fill(Scalar[DType.float8_e4m3fn](0.1))
    scales_dev.enqueue_fill(Float32(0.01))

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var o_dev = ctx.enqueue_create_buffer[DType.bfloat16](o_size)
    var k_dq_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_size)
    var v_dq_dev = ctx.enqueue_create_buffer[DType.bfloat16](k_size)
    q_dev.enqueue_fill(BFloat16(0.01))
    k_dq_dev.enqueue_fill(BFloat16(0.02))
    v_dq_dev.enqueue_fill(BFloat16(0.03))

    var q = TileTensor(
        q_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[head_dim],
            )
        ),
    )
    var k = TileTensor(
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
    var v = TileTensor(
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
    var o = TileTensor(
        o_dev,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[head_dim],
            )
        ),
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    @parameter
    @always_inline
    @__copy_capture(q, k, v, o, scale, fp8_dev, scales_dev, bf16_dev)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            # Step 1: dequantize the paged FP8 cache to BF16.
            dequant_paged_fp8_kv_to_bf16[
                kv_params=kv_params,
                page_size=page_size,
                quantization_granularity=g,
            ](
                fp8_dev.unsafe_ptr(),
                scales_dev.unsafe_ptr(),
                bf16_dev.unsafe_ptr(),
                num_paged_blocks=num_paged_blocks,
                num_layers=1,
                layer_idx=0,
                ctx=ctx,
            )
            # Step 2: attention on the bf16 contiguous K, V (already a
            # rebuild of the dequant result; in production a separate
            # copy-from-paged-via-LUT happens here too, but we omit it
            # here since the dequant pass is the new HBM cost the
            # benchmark is meant to expose).
            flash_attention(o, q, k, v, CausalMask(), scale, ctx)

        b.iter_custom[_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "mha_sm100_fp8_via_staging",
            input_id=String(
                "M=",
                seq_len,
                "/n_kv_heads=",
                kv_num_heads,
                "/group=",
                group,
                "/head_dim=",
                head_dim,
                "/g=",
                g,
                "/seq_len=",
                seq_len,
                "/num_keys=",
                num_keys,
                "/page_size=",
                page_size,
            ),
        ),
    )
    ctx.synchronize()


# ---------------------------------------------------------------------------
# LUT-aware dequant micro
#
# Routes the dequant kernel through the lookup table so grid_z =
# lut_extent * 2 instead of num_paged_blocks * 2.  This micro measures
# the per-kernel dequant time at a production-shape pool
# (num_paged_blocks = 1938, matching the Gemma4-31B-NVFP4 single-B200
# page-pool size on the bf16 baseline run) while varying `lut_extent`
# from 32 (decode batch_size=4 × padded 8 LUT entries) up to 1938 (worst
# case = full pool).  The bench reports just the dequant kernel time so
# the LUT-aware win is visible directly without bf16 attention noise.
# `lut_extent=0` triggers the legacy full-pool path.
# ---------------------------------------------------------------------------


def bench_dequant_lut_aware[
    head_dim: Int,
    kv_num_heads: Int,
    g: Int,
    page_size: Int,
](mut m: Bench, pool_size: Int, lut_extent: Int, ctx: DeviceContext,) raises:
    comptime head_dim_gran = ceildiv(head_dim, g)
    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    var paged_fp8_size = pool_size * 2 * page_size * kv_num_heads * head_dim
    var paged_scales_size = (
        pool_size * 2 * page_size * kv_num_heads * head_dim_gran
    )

    var fp8_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](paged_fp8_size)
    var scales_dev = ctx.enqueue_create_buffer[DType.float32](paged_scales_size)
    var bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](paged_fp8_size)
    fp8_dev.enqueue_fill(Scalar[DType.float8_e4m3fn](0.1))
    scales_dev.enqueue_fill(Float32(0.01))

    # LUT contents: entries 0..lut_extent-1 point to physical blocks
    # 0..lut_extent-1 (simulating a freshly-allocated batch). Real production
    # LUTs have padded sentinel entries; here all entries are valid.
    var lut_size = max(lut_extent, 1)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    # We rely on the kernel reading only entries < lut_extent and skipping
    # sentinels; for the micro we just fill the LUT with sequential IDs.
    var lut_host = alloc[Scalar[DType.uint32]](lut_size)
    for i in range(lut_size):
        lut_host[i] = UInt32(i)
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.synchronize()

    @parameter
    @always_inline
    @__copy_capture(fp8_dev, scales_dev, bf16_dev, lut_dev)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            dequant_paged_fp8_kv_to_bf16[
                kv_params=kv_params,
                page_size=page_size,
                quantization_granularity=g,
            ](
                fp8_dev.unsafe_ptr(),
                scales_dev.unsafe_ptr(),
                bf16_dev.unsafe_ptr(),
                num_paged_blocks=pool_size,
                num_layers=1,
                layer_idx=0,
                ctx=ctx,
                dst_num_layers=1,
                dst_layer_idx=0,
                kv_lookup_table_ptr=lut_dev.unsafe_ptr(),
                lut_extent=lut_extent,
            )

        b.iter_custom[_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "dequant_lut_aware",
            input_id=String(
                "head_dim=",
                head_dim,
                "/pool=",
                pool_size,
                "/lut_extent=",
                lut_extent,
                "/n_kv_heads=",
                kv_num_heads,
                "/g=",
                g,
            ),
        ),
    )
    ctx.synchronize()
    lut_host.free()


# ---------------------------------------------------------------------------
# Compact-then-dequant micro
#
# Measures the new compact-aware path that bench-operator's report
# identified as the only path to the <50ms target.  The path is a
# 2-kernel sequence: (1) preprocessor scans LUT + cache_lengths, writes
# live physical_block IDs to a scratch buffer; (2) dequant kernel
# iterates that buffer.  Grid_z is sized at num_paged_blocks * 2 (static)
# but most blocks early-return cheaply when compact_count is small.
#
# This is the production-realistic regime the brief flagged: batch=5
# active sequences with padded LUT~1960 per seq (=lut_extent=9800) but
# only ~256 true live blocks (each seq has ~50 pages of ~6400 tokens of
# the 14k input averaged).
# ---------------------------------------------------------------------------


def bench_dequant_compact_aware[
    head_dim: Int,
    kv_num_heads: Int,
    g: Int,
    page_size: Int,
](
    mut m: Bench,
    pool_size: Int,
    batch_size: Int,
    max_blocks_per_seq: Int,
    true_live_blocks: Int,
    ctx: DeviceContext,
) raises:
    """Bench the compact-then-dequant path.

    Sets up a LUT of shape `[batch_size, max_blocks_per_seq]` where the
    first `true_live_blocks` entries (distributed across rows) hold
    sequential physical block IDs.  cache_lengths is set so that exactly
    `true_live_blocks / batch_size` blocks per seq are live (= rounded);
    remaining slots within each row are LIVE-IN-LUT but pruned by the
    cache_lengths check.

    For simplicity: distribute `true_live_blocks` evenly across the
    batch_size rows by setting `cache_lengths[s] = (true_live_blocks /
    batch_size) * page_size` for each seq, with rounding for the
    remainder.
    """
    comptime head_dim_gran = ceildiv(head_dim, g)
    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    var paged_fp8_size = pool_size * 2 * page_size * kv_num_heads * head_dim
    var paged_scales_size = (
        pool_size * 2 * page_size * kv_num_heads * head_dim_gran
    )

    var fp8_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](paged_fp8_size)
    var scales_dev = ctx.enqueue_create_buffer[DType.float32](paged_scales_size)
    var bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](paged_fp8_size)
    fp8_dev.enqueue_fill(Scalar[DType.float8_e4m3fn](0.1))
    scales_dev.enqueue_fill(Float32(0.01))

    # LUT + cache_lengths setup.  Distribute true_live_blocks evenly
    # across rows.  Set LUT entries with sentinel values (= pool_size)
    # for positions beyond the per-seq live count.
    var lut_size = batch_size * max_blocks_per_seq
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    var lut_host = alloc[Scalar[DType.uint32]](lut_size)
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var cache_lengths_host = alloc[Scalar[DType.uint32]](batch_size)

    # All LUT entries default to sentinel (= pool_size).  Live entries
    # get a sequential physical block ID.
    for i in range(lut_size):
        lut_host[i] = UInt32(pool_size)
    var live_per_seq = true_live_blocks // batch_size
    var phys_counter = 0
    for s in range(batch_size):
        var seq_live = live_per_seq
        if s < (true_live_blocks - live_per_seq * batch_size):
            seq_live += 1  # distribute remainder
        # cache_lengths[s] = seq_live * page_size (exact boundary).
        cache_lengths_host[s] = UInt32(seq_live * page_size)
        for b in range(seq_live):
            lut_host[s * max_blocks_per_seq + b] = UInt32(phys_counter)
            phys_counter += 1
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.synchronize()

    var compact_buf_dev = ctx.enqueue_create_buffer[DType.uint32](pool_size)
    var compact_count_dev = ctx.enqueue_create_buffer[DType.uint32](1)

    @parameter
    @always_inline
    @__copy_capture(
        fp8_dev,
        scales_dev,
        bf16_dev,
        lut_dev,
        cache_lengths_dev,
        compact_buf_dev,
        compact_count_dev,
    )
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            dequant_paged_fp8_kv_to_bf16[
                kv_params=kv_params,
                page_size=page_size,
                quantization_granularity=g,
            ](
                fp8_dev.unsafe_ptr(),
                scales_dev.unsafe_ptr(),
                bf16_dev.unsafe_ptr(),
                num_paged_blocks=pool_size,
                num_layers=1,
                layer_idx=0,
                ctx=ctx,
                dst_num_layers=1,
                dst_layer_idx=0,
                kv_lookup_table_ptr=lut_dev.unsafe_ptr(),
                lut_extent=lut_size,
                compact_buf_ptr=compact_buf_dev.unsafe_ptr(),
                compact_count_ptr=compact_count_dev.unsafe_ptr(),
                batch_size=batch_size,
                max_blocks_per_seq=max_blocks_per_seq,
                # Matches the Mogg op's DEQUANT_GRID_Z_CAP for production
                # parity (cap=128, see MOGGKernelAPI.mojo:~8166).
                dequant_grid_z_cap=128,
            )

        b.iter_custom[_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "dequant_compact_aware",
            input_id=String(
                "head_dim=",
                head_dim,
                "/pool=",
                pool_size,
                "/batch=",
                batch_size,
                "/max_bps=",
                max_blocks_per_seq,
                "/live=",
                true_live_blocks,
            ),
        ),
    )
    ctx.synchronize()
    lut_host.free()
    cache_lengths_host.free()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() raises:
    var m = Bench(BenchConfig(num_warmup_iters=2, num_repetitions=5))
    with DeviceContext() as ctx:
        # ---- d=256: Gemma4-local layers (head_dim=256, n_kv_heads=4) ----
        for M in [1, 16, 32, 64, 128]:
            for sl in [2048, 16384]:
                # Decode shape: seq_len = M, num_keys = sl + M.
                try:
                    bench_bf16[
                        num_heads=32,
                        group=8,
                        head_dim=256,
                    ](m, M, sl + M, ctx)
                except e:
                    print("bf16 d=256 bench failed:", e)
                try:
                    bench_fp8_staged[
                        num_heads=32,
                        group=8,
                        head_dim=256,
                        g=64,
                        page_size=128,
                    ](m, M, sl + M, ctx)
                except e:
                    print("fp8-staged d=256 bench failed:", e)

        # ---- d=512: Gemma4-global layers (head_dim=512, n_kv_heads=4) ----
        # Same (M, seq_len) grid as d=256 so the gap a future fused
        # convert path would have to close is comparable.
        for M in [1, 16, 32, 64, 128]:
            for sl in [2048, 16384]:
                try:
                    bench_bf16[
                        num_heads=32,
                        group=8,
                        head_dim=512,
                    ](m, M, sl + M, ctx)
                except e:
                    print("bf16 d=512 bench failed:", e)
                try:
                    bench_fp8_staged[
                        num_heads=32,
                        group=8,
                        head_dim=512,
                        g=64,
                        page_size=128,
                    ](m, M, sl + M, ctx)
                except e:
                    print("fp8-staged d=512 bench failed:", e)

        # ---- LUT-aware dequant micro ----
        # Production Gemma4-31B-NVFP4 single-B200 pool size = 1938 blocks.
        # Sweep lut_extent to measure the per-kernel speedup at typical
        # decode batch sizes:
        #   - lut_extent=250880: production warmup shape — batch_size=128
        #                       (max), padded_lut=1960.  total_pairs =
        #                       501760 exceeds the CUDA grid.z cap
        #                       (65535) and exercises the grid-stride
        #                       loop in the kernel.  Must be present so
        #                       this overflow regime is covered by CI.
        #   - lut_extent=1938:  legacy full-pool baseline.
        #   - lut_extent=512:   batch_size=8, padded 64 LUT entries per seq.
        #   - lut_extent=128:   batch_size=4, padded 32 entries per seq.
        #   - lut_extent=32:    batch_size=4, padded 8 entries per seq —
        #                       production single-stream short-context shape.
        # head_dim is comptime so we unroll the two Gemma4 layer types.
        # lut_extent=0 triggers the legacy full-pool path explicitly so
        # we can confirm legacy behaviour is preserved.
        for ext in [0, 250880, 9800, 1938, 512, 128, 32]:
            try:
                bench_dequant_lut_aware[
                    head_dim=256,
                    kv_num_heads=4,
                    g=64,
                    page_size=128,
                ](m, 1938, ext, ctx)
            except e:
                print("dequant_lut_aware d=256 bench failed:", e)
            try:
                bench_dequant_lut_aware[
                    head_dim=512,
                    kv_num_heads=4,
                    g=64,
                    page_size=128,
                ](m, 1938, ext, ctx)
            except e:
                print("dequant_lut_aware d=512 bench failed:", e)

        # ---- Compact-then-dequant micro ----
        # Production-realistic workload:
        #   batch_size=5, max_blocks_per_seq=1960 (= ceildiv(248064/128)),
        #   true_live_blocks=256 (each seq has ~50 pages × 128 tokens of
        #   ~6400 tokens of the p95 14k input averaged).
        # bench-operator's observed step time pre-Slice-6d was 2460ms.
        # Target: <50ms/step for the dequant overhead.  At 60 layers,
        # that's <0.83ms per dequant kernel.
        #
        # We also include the warmup batch=128 worst case to confirm no
        # regression from the existing lut_aware path.
        for cfg in [
            (5, 1960, 256),  # production single-stream
            (5, 1960, 512),  # production 2x
            (5, 1960, 1024),  # production 4x (saturating)
            (8, 1960, 256),  # 8-seq batch
            (128, 1960, 1024),  # warmup batch=128
        ]:
            try:
                bench_dequant_compact_aware[
                    head_dim=256,
                    kv_num_heads=4,
                    g=64,
                    page_size=128,
                ](m, 1938, cfg[0], cfg[1], cfg[2], ctx)
            except e:
                print("dequant_compact_aware d=256 bench failed:", e)
            try:
                bench_dequant_compact_aware[
                    head_dim=512,
                    kv_num_heads=4,
                    g=64,
                    page_size=128,
                ](m, 1938, cfg[0], cfg[1], cfg[2], ctx)
            except e:
                print("dequant_compact_aware d=512 bench failed:", e)

    m.dump_report()
