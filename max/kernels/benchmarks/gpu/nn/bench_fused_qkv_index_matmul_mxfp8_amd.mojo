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
"""Kernel-level perf benchmark: MXFP8 QKV + indexer-QKV on CDNA4, fused vs not.

MXFP8 sibling of `bench_fused_qkv_index_matmul.mojo` (which covers the BF16
kernel). Compares, at the raw-kernel level:

  * FUSED   : ONE call to
              `generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4`
              over the stacked weight [Wq|Wk|Wv|Wiq|Wik] (N_total=2560), whose
              CDNA4 path scatters K/V/IndexK from the GEMM's own epilogue.
  * UNFUSED : the path MXFP8-on-AMD bring-up left behind — FIVE separate
              `block_scaled_matmul_amd` calls, one per output band
              (N = 2048, 128, 128, 128, 128).

Both do the same 2*M*N_total*K FLOPs against the same cold weight bytes, so the
difference is the GEMM count plus the three paged `kv_cache_store` launches
the unfused path needs to place K/V/IndexK, which the fused epilogue does from
the GEMM's own store path. Both are inside the timed closure.

Shapes: M3 per-device (TP=4), MXFP8 operands with E8M0 scales over 32-element K
blocks. Sweeps the DECODE regime (total_seq == decode batch size, one token
each) across {1, 8, 16, 32, 64, 128, 256}, plus one PREFILL shape (2 prompts x
256 tokens). Cache topologies match the differential test: MAIN = non-MLA GQA
(K+V, 1 KV head); INDEX = MLA (K-only, 1 latent head).

Timing: stdlib `benchmark` `Bench` / `iter_custom`. Operands and both scale
tensors are cache-busted (`CacheBustingBuffer` + per-iteration `offset_ptr`) so
each iteration reads cold HBM -- decode QKV is weight-bandwidth-bound. Reports
per-iteration mean plus GFLOP/s and GB/s via `ThroughputMeasure`.

CDNA4 only: the fused epilogue and the block-scaled AMD matmul are MI355X paths.

Run directly:  mojo max/kernels/benchmarks/gpu/nn/bench_fused_qkv_index_matmul_mxfp8_amd.mojo
Or via bazel:  ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qkv_index_matmul_mxfp8_amd
"""

from std.random import seed

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from layout.tile_tensor import lt_to_tt
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from linalg.matmul.gpu.amd import block_scaled_matmul_amd
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    kv_cache_store_ragged,
)

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType

from std.math import ceildiv
from std.sys import size_of
from std.utils import IndexList

comptime OPERAND_DTYPE = DType.float8_e4m3fn
comptime OUT_DTYPE = DType.bfloat16
comptime SCALE_DTYPE = DType.float8_e8m0fnu
comptime SF_VECTOR_SIZE = 32

comptime HEAD_SIZE = 128
comptime NUM_Q_HEADS = 16  # q_dim = 2048
comptime MAIN_KV_HEADS = 1  # kv_dim = 128
comptime NUM_INDEX_HEADS = 1  # iq_dim = 128

comptime hidden = 6144  # K
comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 2048
comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 128
comptime ik_dim = HEAD_SIZE  # 128
comptime n_total = q_dim + 2 * kv_dim + iq_dim + ik_dim  # 2560
comptime k_scales = hidden // SF_VECTOR_SIZE  # 192

comptime page_size = 512
comptime num_pages = 512
comptime num_layers = 1
comptime layer_idx = 0

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(num_heads=1, head_size=HEAD_SIZE)
comptime MainCollection = PagedKVCacheCollection[
    OUT_DTYPE, main_kv_params, page_size, ...
]
comptime IndexCollection = PagedKVCacheCollection[
    OUT_DTYPE, index_kv_params, page_size, ...
]


@always_inline
def _any(
    ptr: UnsafePointer[Scalar[OUT_DTYPE], ...],
) -> UnsafePointer[Scalar[OUT_DTYPE], MutAnyOrigin]:
    """Erase a pointer's origin so one helper serves all three bands."""
    return UnsafePointer[Scalar[OUT_DTYPE], MutAnyOrigin](
        unsafe_from_address=Int(ptr)
    )


# Column offset of each band in the stacked weight, for the unfused slices.
comptime k_off = q_dim
comptime v_off = k_off + kv_dim
comptime iq_off = v_off + kv_dim
comptime ik_off = iq_off + iq_dim


def bench_shape(
    ctx: DeviceContext,
    mut m: Bench,
    prompt_lens: List[Int],
    regime: String,
) raises:
    """Build device inputs / caches for `prompt_lens`, register both entries."""
    var batch_size = len(prompt_lens)

    var total_seq = 0
    var max_seq = 0
    var iro_host = List[Scalar[DType.uint32]](
        length=batch_size + 1, fill=Scalar[DType.uint32](0)
    )
    for i in range(batch_size):
        iro_host[i] = UInt32(total_seq)
        total_seq += prompt_lens[i]
        max_seq = max(max_seq, prompt_lens[i])
    iro_host[batch_size] = UInt32(total_seq)
    var max_ctx = max_seq

    var iro_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_copy(iro_dev, iro_host)
    var iro_tensor = LayoutTensor[
        mut=False, DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        iro_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    var cache_lengths_host = List[Scalar[DType.uint32]](
        length=batch_size, fill=Scalar[DType.uint32](0)
    )
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_tensor = LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_dev.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
    )

    var lut_cols = ((ceildiv(max_ctx, page_size) + 7) // 8) * 8 + 16
    var lut_host = List[Scalar[DType.uint32]](
        length=batch_size * lut_cols, fill=Scalar[DType.uint32](0)
    )
    var block_counter = 0
    for b in range(batch_size):
        var pages = ceildiv(prompt_lens[b], page_size)
        for p in range(pages):
            lut_host[b * lut_cols + p] = UInt32(block_counter)
            block_counter += 1
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size * lut_cols)
    ctx.enqueue_copy(lut_dev, lut_host)
    var lut_tensor = LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[2]()
    ](
        lut_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[2]()].row_major(
            IndexList[2](batch_size, lut_cols)
        ),
    )

    # ---- cache-busting inputs: fp8 operands plus both E8M0 scale tensors ----
    comptime simd_size = 4
    var cb_hs = CacheBustingBuffer[OPERAND_DTYPE](
        total_seq * hidden, simd_size, ctx
    )
    var cb_w = CacheBustingBuffer[OPERAND_DTYPE](
        n_total * hidden, simd_size, ctx
    )
    # E8M0 has no zero encoding, so a `CacheBustingBuffer` of it trips an
    # APFloat assertion in the compiler; hold the scale bytes as uint8 and
    # reinterpret them at the tensor seam instead.
    var cb_asf = CacheBustingBuffer[DType.uint8](
        total_seq * k_scales, simd_size, ctx
    )
    var cb_bsf = CacheBustingBuffer[DType.uint8](
        n_total * k_scales, simd_size, ctx
    )
    cb_hs.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_w.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_asf.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_bsf.init_on_device(InitializationType.uniform_distribution, ctx)

    # ---- outputs: fused writes Q / IndexQ; unfused writes one per band ----
    var q_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * q_dim)
    var q_out = LayoutTensor[OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, q_dim)](
        q_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, q_dim)].row_major(
            IndexList[2](total_seq, q_dim)
        ),
    )
    var iq_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * iq_dim)
    var iq_out = LayoutTensor[
        OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, iq_dim)
    ](
        iq_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, iq_dim)].row_major(
            IndexList[2](total_seq, iq_dim)
        ),
    )
    # K, V and IndexK land in dense buffers on the unfused path; the fused
    # epilogue scatters them straight into the caches instead.
    var kv_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        3 * total_seq * kv_dim
    )
    var kv_out_ptr = kv_out_dev.unsafe_ptr()

    # ---- KV cache blocks (main: K+V; index: MLA K-only) ----
    comptime block_layout = Layout.row_major[6]()
    var main_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        main_block_shape.flattened_length()
    )
    var main_blocks = LayoutTensor[OUT_DTYPE, block_layout](
        main_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(main_block_shape),
    )
    var index_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        index_block_shape.flattened_length()
    )
    var index_blocks = LayoutTensor[OUT_DTYPE, block_layout](
        index_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(index_block_shape),
    )

    var main_collection = MainCollection(
        main_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor,
        lut_tensor,
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        index_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor,
        lut_tensor,
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    var flops = 2 * total_seq * n_total * hidden
    # Both paths read the same cold weight and scale bytes; the unfused chain
    # re-reads the activation once per band.
    var operand_bytes = n_total * hidden + n_total * k_scales
    var act_bytes = total_seq * hidden + total_seq * k_scales
    var write_elems = (
        total_seq * (q_dim + iq_dim)
        + 2 * total_seq * kv_dim
        + total_seq * ik_dim
    )
    var fused_bytes = (
        operand_bytes + act_bytes + write_elems * size_of[OUT_DTYPE]()
    )
    var unfused_bytes = (
        operand_bytes + 5 * act_bytes + write_elems * size_of[OUT_DTYPE]()
    )

    # ============ FUSED: one GEMM, scatter from the epilogue ============
    @parameter
    @__copy_capture(
        cb_hs,
        cb_w,
        cb_asf,
        cb_bsf,
        iro_tensor,
        main_collection,
        index_collection,
        q_out,
        iq_out,
        total_seq,
    )
    @always_inline
    def fused_launch(ctx: DeviceContext, iteration: Int) raises:
        var hs = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var w = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(n_total, hidden)
        ](
            cb_w.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(n_total, hidden)].row_major(
                IndexList[2](n_total, hidden)
            ),
        )
        var asf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(UNKNOWN_VALUE, k_scales)
        ](
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, k_scales)].row_major(
                IndexList[2](total_seq, k_scales)
            ),
        )
        var bsf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(n_total, k_scales)
        ](
            cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(n_total, k_scales)].row_major(
                IndexList[2](n_total, k_scales)
            ),
        )
        generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
        ](
            hs,
            iro_tensor,
            w,
            asf,
            bsf,
            Float32(1.0),
            main_collection,
            index_collection,
            UInt32(layer_idx),
            iq_dim,
            q_out,
            iq_out,
            ctx,
        )

    @parameter
    @always_inline
    def fused_bench(mut b: Bencher) raises:
        bencher_iter_custom[fused_launch](b, ctx)

    m.bench_function[fused_bench](
        BenchId("fused   " + regime + " total_seq=" + String(total_seq)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, fused_bytes),
        ],
    )

    # ============ UNFUSED: one dense GEMM per output band (5 calls) ==========
    @parameter
    @__copy_capture(
        cb_hs,
        cb_w,
        cb_asf,
        cb_bsf,
        iro_tensor,
        main_collection,
        index_collection,
        q_out,
        iq_out,
        kv_out_ptr,
        total_seq,
    )
    @always_inline
    def unfused_launch(ctx: DeviceContext, iteration: Int) raises:
        var hs = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var asf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(UNKNOWN_VALUE, k_scales)
        ](
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, k_scales)].row_major(
                IndexList[2](total_seq, k_scales)
            ),
        )
        var hs_tt = lt_to_tt(hs).bitcast[DType.uint8]()
        var asf_tt = lt_to_tt(asf)

        # Q band: the only wide one (N=2048); the other four are N=128.
        @parameter
        @always_inline
        def band[
            band_n: Int
        ](
            col_off: Int,
            out_ptr: UnsafePointer[Scalar[OUT_DTYPE], MutAnyOrigin],
        ) raises:
            var w = LayoutTensor[
                mut=False, OPERAND_DTYPE, Layout.row_major(band_n, hidden)
            ](
                cb_w.offset_ptr(iteration) + col_off * hidden,
                RuntimeLayout[Layout.row_major(band_n, hidden)].row_major(
                    IndexList[2](band_n, hidden)
                ),
            )
            var bsf = LayoutTensor[
                mut=False, SCALE_DTYPE, Layout.row_major(band_n, k_scales)
            ](
                cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]]()
                + col_off * k_scales,
                RuntimeLayout[Layout.row_major(band_n, k_scales)].row_major(
                    IndexList[2](band_n, k_scales)
                ),
            )
            var c = LayoutTensor[
                OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, band_n)
            ](
                out_ptr,
                RuntimeLayout[
                    Layout.row_major(UNKNOWN_VALUE, band_n)
                ].row_major(IndexList[2](total_seq, band_n)),
            )
            block_scaled_matmul_amd[lane_bytes=32](
                lt_to_tt(c),
                hs_tt,
                lt_to_tt(w).bitcast[DType.uint8](),
                asf_tt,
                lt_to_tt(bsf),
                ctx,
            )

        band[q_dim](0, _any(q_out.ptr))
        band[kv_dim](k_off, _any(kv_out_ptr))
        band[kv_dim](v_off, _any(kv_out_ptr) + total_seq * kv_dim)
        band[iq_dim](iq_off, _any(iq_out.ptr))
        band[ik_dim](ik_off, _any(kv_out_ptr) + 2 * total_seq * kv_dim)

        # Placing K/V/IndexK is the other half of what the fused epilogue does,
        # so the unfused path pays for three paged-store launches on top of its
        # five GEMMs.
        @parameter
        @always_inline
        def k_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (_any(kv_out_ptr) + idx[0] * kv_dim + idx[2]).load[
                width=width
            ]()

        @parameter
        @always_inline
        def v_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (
                _any(kv_out_ptr) + total_seq * kv_dim + idx[0] * kv_dim + idx[2]
            ).load[width=width]()

        @parameter
        @always_inline
        def ik_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (
                _any(kv_out_ptr)
                + 2 * total_seq * kv_dim
                + idx[0] * ik_dim
                + idx[2]
            ).load[width=width]()

        kv_cache_store_ragged[target="gpu", input_fn=k_in](
            main_collection.get_key_cache(layer_idx),
            IndexList[3](total_seq, MAIN_KV_HEADS, HEAD_SIZE),
            iro_tensor,
            ctx,
        )
        kv_cache_store_ragged[target="gpu", input_fn=v_in](
            main_collection.get_value_cache(layer_idx),
            IndexList[3](total_seq, MAIN_KV_HEADS, HEAD_SIZE),
            iro_tensor,
            ctx,
        )
        kv_cache_store_ragged[target="gpu", input_fn=ik_in](
            index_collection.get_key_cache(layer_idx),
            IndexList[3](total_seq, 1, HEAD_SIZE),
            iro_tensor,
            ctx,
        )

    @parameter
    @always_inline
    def unfused_bench(mut b: Bencher) raises:
        bencher_iter_custom[unfused_launch](b, ctx)

    m.bench_function[unfused_bench](
        BenchId("unfused " + regime + " total_seq=" + String(total_seq)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, unfused_bytes),
        ],
    )

    _ = cb_hs^
    _ = cb_w^
    _ = cb_asf^
    _ = cb_bsf^
    _ = iro_dev^
    _ = cache_lengths_dev^
    _ = lut_dev^
    _ = q_out_dev^
    _ = iq_out_dev^
    _ = kv_out_dev^
    _ = main_blocks_dev^
    _ = index_blocks_dev^


def main() raises:
    seed(0)
    var m = Bench()
    with DeviceContext() as ctx:
        for bs in [1, 8, 16, 32, 64, 128, 256, 512]:
            var decode_lens = List[Int](length=bs, fill=1)
            bench_shape(ctx, m, decode_lens, "decode")

        bench_shape(ctx, m, [256, 256], "prefill")

        # Speculative decoding: the target verifies num_spec+1 tokens per
        # sequence in one pass, so M = bs * (num_spec+1). 3 is what M3
        # actually runs on AMD.
        var num_spec = 3
        for bs in [1, 8, 16, 32, 64, 128, 256]:
            var spec_lens = List[Int](length=bs, fill=num_spec + 1)
            bench_shape(ctx, m, spec_lens, "decode_spec" + String(num_spec))
    m.dump_report()
