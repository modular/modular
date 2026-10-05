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
"""Kernel-level perf benchmark: MXFP8 QKV projections on CDNA4, fused vs not.

Covers both attention flavours behind one `has_indexer` argument, since the
harness (ragged offsets, paged caches, cache-busted operands, band slicing) is
identical and only the band list and the fused entry point differ:

  * DENSE  (`has_indexer=False`) -- layers with no sparse indexer. Bands
    `[Wq|Wk|Wv]`, N_total=2304. FUSED is one
    `generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4`; UNFUSED is
    three `block_scaled_matmul_amd` calls (N = 2048, 128, 128) plus two paged
    `kv_cache_store` launches to place K/V.
  * SPARSE (`has_indexer=True`) -- layers that also project the indexer's
    IndexQ/IndexK. Bands `[Wq|Wk|Wv|Wiq|Wik]`, N_total=2560. FUSED is one
    `generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4`;
    UNFUSED is five band GEMMs (N = 2048, 128, 128, 128, 128) plus three paged
    stores.

Either way both paths do the same 2*M*N_total*K FLOPs against the same cold
weight bytes, so the difference is the GEMM count plus the paged stores the
unfused path needs to place K/V (and IndexK), which the fused epilogue does from
the GEMM's own store path. Everything is inside the timed closure.

Shapes: M3 per-device (TP=4), MXFP8 operands with E8M0 scales over 32-element K
blocks. `batch_size` prompts of `seq_len` tokens each, so decode is seq_len=1
across a batch sweep and prefill is a few long prompts; the regime label follows
from seq_len. Cache topologies: MAIN = non-MLA GQA (K+V, 1 KV head); INDEX = MLA
(K-only, 1 latent head).

Timing: stdlib `benchmark` `Bench` / `iter_custom`. Operands and both scale
tensors are cache-busted (`CacheBustingBuffer` + per-iteration `offset_ptr`) so
each iteration reads cold HBM -- decode QKV is weight-bandwidth-bound. Reports
per-iteration mean plus GFLOP/s and GB/s via `ThroughputMeasure`.

CDNA4 only: the fused epilogue and the block-scaled AMD matmul are MI355X paths.

A run covers ONE variant at ONE shape, defaulting to dense at decode batch 1.
The sibling yaml holds the sweep: `$has_indexer` over both variants crossed with
`$batch_size` / `$seq_len` over the decode, verify, and prefill shapes.

Run the default (dense, decode bs=1):
    ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qkv_matmul_mxfp8_amd
One other point (sparse, decode bs=64):
    ... -- --has_indexer=True --batch_size=64 --seq_len=1
The whole sweep:
    python max/kernels/benchmarks/autotune/kbench.py \\
        max/kernels/benchmarks/gpu/nn/bench_fused_qkv_matmul_mxfp8_amd.yaml
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
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from linalg.matmul.gpu.amd import block_scaled_matmul_amd
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
    kv_cache_store_ragged,
)

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType, arg_parse

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
    OUT_DTYPE,
    main_kv_params,
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]
comptime IndexCollection = PagedKVCacheCollection[
    OUT_DTYPE,
    index_kv_params,
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]


@inline(.always)
def _any(
    ptr: MutPointer[Scalar[OUT_DTYPE], ...],
) -> MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin]:
    """Erase a pointer's origin so one helper serves every band."""
    return MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin](
        unsafe_from_address=Int(ptr)
    )


# Column offset of each band in the stacked weight, for the unfused slices. The
# index bands sit past the QKV ones, so the dense layout is a prefix of the
# sparse one and these offsets serve both variants.
comptime k_off = q_dim
comptime v_off = k_off + kv_dim
comptime iq_off = v_off + kv_dim
comptime ik_off = iq_off + iq_dim


def bench_shape[
    HAS_INDEXER: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    prompt_lens: List[Int],
    regime: String,
) raises:
    """Build device inputs / caches for `prompt_lens`, register both entries."""
    # Sparse stacks IndexQ/IndexK past the QKV bands; dense stops after V.
    comptime n_total = q_dim + 2 * kv_dim + (iq_dim + ik_dim) * Int(HAS_INDEXER)
    comptime variant = "sparse" if HAS_INDEXER else "dense "
    comptime num_bands = 3 + 2 * Int(HAS_INDEXER)

    var batch_size = len(prompt_lens)

    var total_seq = 0
    var max_seq = 0
    var iro_host = List[UInt32](length=batch_size + 1, fill=UInt32(0))
    for i in range(batch_size):
        iro_host[i] = UInt32(total_seq)
        total_seq += prompt_lens[i]
        max_seq = max(max_seq, prompt_lens[i])
    iro_host[batch_size] = UInt32(total_seq)
    var max_ctx = max_seq

    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(iro_dev, iro_host)
    var iro_tt = TileTensor(iro_dev, row_major(Int64(batch_size + 1)))

    var cache_lengths_host = List[UInt32](length=batch_size, fill=UInt32(0))
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_tensor = TileTensor(
        cache_lengths_dev, row_major(Int64(batch_size))
    )

    var lut_cols = ((ceildiv(max_ctx, page_size) + 7) // 8) * 8 + 16
    var lut_host = List[UInt32](length=batch_size * lut_cols, fill=UInt32(0))
    var block_counter = 0
    for b in range(batch_size):
        var pages = ceildiv(prompt_lens[b], page_size)
        for p in range(pages):
            lut_host[b * lut_cols + p] = UInt32(block_counter)
            block_counter += 1
    var lut_dev = ctx.enqueue_create_buffer[.uint32](batch_size * lut_cols)
    ctx.enqueue_copy(lut_dev, lut_host)
    var lut_tensor = TileTensor(
        lut_dev, row_major(Int64(batch_size), Int64(lut_cols))
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
    var cb_asf = CacheBustingBuffer[.uint8](
        total_seq * k_scales, simd_size, ctx
    )
    var cb_bsf = CacheBustingBuffer[.uint8](n_total * k_scales, simd_size, ctx)
    cb_hs.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_w.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_asf.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_bsf.init_on_device(InitializationType.uniform_distribution, ctx)

    # ---- outputs: fused writes Q (+ IndexQ); unfused writes one per band ----
    var q_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * q_dim)
    var q_out = TileTensor(q_out_dev, row_major(total_seq, Idx[q_dim]))
    # IndexQ and the index cache are allocated for both variants: they are a few
    # hundred KiB, sit outside the timed closure, and keeping them unconditional
    # lets one closure body serve dense and sparse alike.
    var iq_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * iq_dim)
    var iq_out = TileTensor(iq_out_dev, row_major(total_seq, Idx[iq_dim]))
    # K, V and IndexK land in dense buffers on the unfused path; the fused
    # epilogue scatters them straight into the caches instead.
    var kv_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        3 * total_seq * kv_dim
    )
    var kv_out_ptr = kv_out_dev.unsafe_ptr()

    # ---- KV cache blocks (main: K+V; index: MLA K-only) ----
    var main_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        main_block_shape.flattened_length()
    )
    comptime main_layout = MainCollection.blocks_tt_layout
    var main_shape = Coord[*main_layout.shape_types]()
    main_shape[0] = Int64(num_pages)
    main_shape[2] = Int64(num_layers)
    var main_strides = Coord[*main_layout.stride_types]()
    main_strides[0] = Int64(
        main_block_shape[1]
        * num_layers
        * page_size
        * main_block_shape[4]
        * HEAD_SIZE
    )
    main_strides[1] = Int64(
        num_layers * page_size * main_block_shape[4] * HEAD_SIZE
    )
    var main_blocks = TileTensor(
        main_blocks_dev, main_layout(main_shape, main_strides)
    )
    var index_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        index_block_shape.flattened_length()
    )
    comptime index_layout = IndexCollection.blocks_tt_layout
    var index_shape = Coord[*index_layout.shape_types]()
    index_shape[0] = Int64(num_pages)
    index_shape[2] = Int64(num_layers)
    var index_strides = Coord[*index_layout.stride_types]()
    index_strides[0] = Int64(
        index_block_shape[1]
        * num_layers
        * page_size
        * index_block_shape[4]
        * HEAD_SIZE
    )
    index_strides[1] = Int64(
        num_layers * page_size * index_block_shape[4] * HEAD_SIZE
    )
    var index_blocks = TileTensor(
        index_blocks_dev, index_layout(index_shape, index_strides)
    )

    var main_collection = MainCollection(
        main_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor.as_imm().as_unsafe_any_origin(),
        lut_tensor.as_imm().as_unsafe_any_origin(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        index_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor.as_imm().as_unsafe_any_origin(),
        lut_tensor.as_imm().as_unsafe_any_origin(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    var flops = 2 * total_seq * n_total * hidden
    # Both paths read the same cold weight and scale bytes; the unfused chain
    # re-reads the activation once per band. Every band is written exactly once,
    # so the write volume is N_total wide either way.
    var operand_bytes = n_total * hidden + n_total * k_scales
    var act_bytes = total_seq * hidden + total_seq * k_scales
    var write_elems = total_seq * n_total
    var fused_bytes = (
        operand_bytes + act_bytes + write_elems * size_of[OUT_DTYPE]()
    )
    var unfused_bytes = (
        operand_bytes
        + num_bands * act_bytes
        + write_elems * size_of[OUT_DTYPE]()
    )

    # ============ FUSED: one GEMM, scatter from the epilogue ============
    @inline(.always)
    def fused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_hs,
        mut cb_w,
        mut cb_asf,
        mut cb_bsf,
        mut q_out,
        mut iq_out,
        imm,
    }:
        var hs = (
            TileTensor(
                cb_hs.offset_ptr(iteration),
                row_major(total_seq, Idx[hidden]),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )
        var w = (
            TileTensor(
                cb_w.offset_ptr(iteration),
                row_major(Idx[n_total], Idx[hidden]),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )
        var asf = (
            TileTensor(
                cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
                row_major(total_seq, Idx[k_scales]),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )
        var bsf = (
            TileTensor(
                cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
                row_major(Idx[n_total], Idx[k_scales]),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )
        comptime if HAS_INDEXER:
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
            ](
                hs,
                iro_tt,
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
        else:
            generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
            ](
                hs,
                iro_tt,
                w,
                asf,
                bsf,
                Float32(1.0),
                main_collection,
                UInt32(layer_idx),
                q_out,
                ctx,
            )

    @inline(.always)
    def fused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, fused_launch, ctx)

    m.bench_function(
        fused_bench,
        BenchId(
            "fused   "
            + variant
            + " "
            + regime
            + " total_seq="
            + String(total_seq)
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, fused_bytes),
        ],
    )

    # ============ UNFUSED: one dense GEMM per output band ============
    @inline(.always)
    def unfused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_hs,
        mut cb_w,
        mut cb_asf,
        mut cb_bsf,
        mut q_out,
        mut iq_out,
        imm,
    }:
        var hs_tt = TileTensor(
            cb_hs.offset_ptr(iteration),
            row_major(total_seq, Idx[hidden]),
        ).bitcast[.uint8]()
        var asf_tt = TileTensor(
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            row_major(total_seq, Idx[k_scales]),
        )

        # Q band: the only wide one (N=2048); the rest are N=128.
        @inline(.always)
        def band[
            band_n: Int
        ](
            col_off: Int,
            out_ptr: MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin],
        ) raises {mut cb_w, mut cb_bsf, imm}:
            var w = TileTensor(
                cb_w.offset_ptr(iteration) + col_off * hidden,
                row_major(Idx[band_n], Idx[hidden]),
            )
            var bsf = TileTensor(
                cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]]()
                + col_off * k_scales,
                row_major(Idx[band_n], Idx[k_scales]),
            )
            var c = TileTensor(out_ptr, row_major(total_seq, Idx[band_n]))
            block_scaled_matmul_amd[lane_bytes=32](
                c,
                hs_tt,
                w.bitcast[.uint8](),
                asf_tt,
                bsf,
                ctx,
            )

        band[q_dim](0, _any(q_out.unsafe_ptr()))
        band[kv_dim](k_off, _any(kv_out_ptr))
        band[kv_dim](v_off, _any(kv_out_ptr) + total_seq * kv_dim)
        comptime if HAS_INDEXER:
            band[iq_dim](iq_off, _any(iq_out.unsafe_ptr()))
            band[ik_dim](ik_off, _any(kv_out_ptr) + 2 * total_seq * kv_dim)

        # Placing K/V (and IndexK) is the other half of what the fused epilogue
        # does, so the unfused path pays for those paged-store launches on top
        # of its band GEMMs.
        @inline(.always)
        @__copy_capture(kv_out_ptr)
        def k_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (_any(kv_out_ptr) + idx[0] * kv_dim + idx[2]).load[
                width=width
            ]()

        @inline(.always)
        @__copy_capture(kv_out_ptr, total_seq)
        def v_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (
                _any(kv_out_ptr) + total_seq * kv_dim + idx[0] * kv_dim + idx[2]
            ).load[width=width]()

        @inline(.always)
        @__copy_capture(kv_out_ptr, total_seq)
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
            iro_tt,
            ctx,
        )
        kv_cache_store_ragged[target="gpu", input_fn=v_in](
            main_collection.get_value_cache(layer_idx),
            IndexList[3](total_seq, MAIN_KV_HEADS, HEAD_SIZE),
            iro_tt,
            ctx,
        )
        comptime if HAS_INDEXER:
            kv_cache_store_ragged[target="gpu", input_fn=ik_in](
                index_collection.get_key_cache(layer_idx),
                IndexList[3](total_seq, 1, HEAD_SIZE),
                iro_tt,
                ctx,
            )

    @inline(.always)
    def unfused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, unfused_launch, ctx)

    m.bench_function(
        unfused_bench,
        BenchId(
            "unfused "
            + variant
            + " "
            + regime
            + " total_seq="
            + String(total_seq)
        ),
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
    # One variant at one shape per run; the yaml holds the sweep, which is what
    # keeps a default invocation short. All three knobs are runtime args, so a
    # sweep reuses one build: `bench_shape` is instantiated for both variants
    # and selected here rather than behind a `-D` define.
    var has_indexer = arg_parse("has_indexer", False)
    var batch_size = Int(arg_parse("batch_size", 1))
    var seq_len = Int(arg_parse("seq_len", 1))
    var is_verify = arg_parse("is_verify", False)

    seed(0)
    var m = Bench()
    with DeviceContext() as ctx:
        var prompt_lens = List[Int](length=batch_size, fill=seq_len)
        var regime = "prefill"
        if is_verify:
            regime = "verify"
        elif seq_len == 1:
            regime = "decode"
        if has_indexer:
            bench_shape[True](ctx, m, prompt_lens, regime)
        else:
            bench_shape[False](ctx, m, prompt_lens, regime)
    m.dump_report()
