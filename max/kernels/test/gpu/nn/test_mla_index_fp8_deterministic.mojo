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
"""Run-to-run reproducibility of the FP8 indexer's top-k and its consumer.

The sparse MLA reduces over a selection in its column order, so a top-k that
returns the right keys in a launch-dependent order still changes the low bits
of attention from launch to launch. This launches the indexer several times on
one input, through the physical-row remap production uses, and feeds each
selection to the sparse MLA decode:

- `BlockSplitTopk[deterministic=True]` must give the same selection, column for
  column, and a bit-identical decode output on every launch. Where it takes
  the deterministic cluster kernel, its selection must also equal, slot for
  slot, the one the ranked streaming select it replaced writes through the
  same scorer, bound and remap.
- `BlockSplitTopk[]`, the default, promises neither; its differences are
  printed, not checked, since a launch may happen to agree.

Every launch of either contract must also be a well-formed selection: no slot
left unwritten, `min(top_k, visible)` distinct keys of the row's own sequence,
all visible to its query, then `-1`. Which keys win is checked against a
reference in `test_topk_bitonic.mojo`, where the scores are known.

The indexer runs at Mistral Large 4's geometry: 32 index heads of 128, top-2048,
pages of 128. A captured decode graph freezes the cache's length bound at the
served maximum, so the decode and verify cases set it to 262144: that row
stride, over few enough rows for a cluster of CTAs each, is what sends
production decode to the deterministic cluster top-k. One case keeps the
live bound for the eager decode's resident arm, one has fewer live keys than
`top_k` under the frozen stride, and the prefill case has more rows than
SMs.
"""

from std.collections import OptionalReg, Set
from std.math import align_up, ceildiv
from std.memory import bitcast
from std.random import rand, randn, seed
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext
from kv_cache.paged_sparse_kv_index_remap import PagedKVRemap
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.gpu.mla_index_fp8 import (
    BlockSplitTopk,
    IndexerTopk,
    mla_indexer_ragged_float8_paged,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_mask import CausalMask, MaskName
from nn.attention.mha_utils import MHAConfig
from nn.topk_bitonic import (
    RowBound,
    topk_det_cluster_route,
    _histsel_topk_bounded_kernel,
    _HSEL_RANK_BITS,
    _HSEL_RES_MAX_WIDE,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
)
from max.gpu.host import FuncAttribute
from structured_kernels.trace_buf import NullTrace

comptime INDEX_HEADS = 32
comptime INDEX_DEPTH = 128
comptime TOP_K = 2048
comptime PAGE_SIZE = 128
# One tensor-parallel rank's share of the MLA heads.
comptime MLA_HEADS = 8
comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime SCALE = Float32(0.125)
comptime LAUNCHES = 4
# The served `--max-length`, which capture freezes the indexer's bound to.
comptime FROZEN_CACHE_LEN = 262144


struct RankedTopk(IndexerTopk):
    """The ranked streaming select, as `BlockSplitTopk[True]` launched it for
    few rows over a long stride before the cluster kernel took them."""

    @staticmethod
    def select[
        in_dtype: DType, RB: RowBound
    ](
        ctx: DeviceContext,
        scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
        out_idxs: MutPointer[Int32, MutAnyOrigin],
        n: Int,
        k: Int,
        rows: Int,
        row_bounds: RB,
    ) raises:
        ctx.enqueue_function[
            _histsel_topk_bounded_kernel[
                NullTrace,
                RB=RB,
                prefetch=True,
                tail_bits=_HSEL_TAIL_BITS,
                rank_bits=_HSEL_RANK_BITS,
                rank_slots=True,
                sel_cap=_HSEL_SEL_CAP,
                in_dtype=in_dtype,
            ]
        ](
            scores,
            out_idxs,
            Int32(n),
            Int32(k),
            row_bounds,
            NullTrace(),
            grid_dim=rows,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )


@fieldwise_init
struct _Drift(Movable):
    """How far the launches after the first strayed from it, and the first
    launch's physical selection."""

    var first_sel: List[Int32]
    var selection_rows: Int
    """Most selection rows, over the launches, that differ in any column."""
    var output_values: Int
    """Most decode output values, over the launches, that differ in any bit."""


# The top-k kernel families `persistent_topk_block_split` picks between.
comptime ARM_RESIDENT = 0
comptime ARM_RANKED_STREAMING = 1
comptime ARM_FILLS_GPU = 2
comptime ARM_CLUSTER = 3


def _arm_name(arm: Int) -> String:
    if arm == ARM_RESIDENT:
        return "resident"
    if arm == ARM_RANKED_STREAMING:
        return "ranked streaming"
    if arm == ARM_CLUSTER:
        return "deterministic cluster"
    return "streaming, rows fill the GPU"


def _check_selection(
    sel: List[Int32],
    phys_to_token: List[Int],
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    label: String,
) raises:
    """Checks one launch's physical selection is well formed.

    Row `(b, s)` sees `cache_len + s + 1` keys, so it holds exactly
    `min(TOP_K, visible)` distinct physical rows, each the row of a token of
    sequence `b` below that count, and `-1` in every slot after them. The
    `-2` fill must not survive anywhere.
    """
    for b in range(batch_size):
        for s in range(q_len):
            var row = b * q_len + s
            var visible = cache_len + s + 1
            var expect = min(TOP_K, visible)
            var seen = Set[Int]()
            for k in range(TOP_K):
                var v = Int(sel[row * TOP_K + k])
                assert_true(
                    v != -2,
                    String(label, ": row ", row, " slot ", k, " never written"),
                )
                if k >= expect:
                    assert_true(
                        v == -1,
                        String(label, ": row ", row, " slot ", k, " = ", v),
                    )
                    continue
                assert_true(
                    v >= 0 and v < len(phys_to_token),
                    String(label, ": row ", row, " slot ", k, " = ", v),
                )
                var token = phys_to_token[v]
                assert_true(
                    token >= 0
                    and token // (cache_len + q_len) == b
                    and token % (cache_len + q_len) < visible,
                    String(
                        label,
                        ": row ",
                        row,
                        " selected physical row ",
                        v,
                        ", not a key it sees",
                    ),
                )
                assert_true(
                    v not in seen,
                    String(label, ": row ", row, " repeats ", v),
                )
                seen.add(v)


def _run_case[
    Topk: IndexerTopk
](
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    frozen_cache_len: Int,
    run_decode: Bool,
    arm: Int,
    ctx: DeviceContext,
) raises -> _Drift:
    """Launches the indexer and its decode `LAUNCHES` times on one input.

    Every launch sees the same bytes: the generator is reseeded per case, so
    both contracts also see the same input. `frozen_cache_len` is the cache
    length bound the collection reports, `0` for the live one.
    """
    seed(1234)
    var num_keys = cache_len + q_len
    var total_q = batch_size * q_len
    var real_pages = ceildiv(num_keys, PAGE_SIZE)
    var num_blocks = batch_size * real_pages
    var bound_len = frozen_cache_len if frozen_cache_len > 0 else cache_len
    # The page table spans the bound, as a captured graph's does; pages past a
    # sequence's real ones point at block 0 and are never read.
    var lut_pages = ceildiv(bound_len + q_len, PAGE_SIZE)
    var sel_size = total_q * TOP_K

    # The top-k sees a row stride of at least the bound plus the query, so this
    # is the kernel family it takes; a case that drifted to another arm would
    # stop covering what it names.
    var stride = bound_len + q_len
    var sm_count = ctx.device().multiprocessor_count()
    var took: Int
    if total_q >= sm_count:
        took = ARM_FILLS_GPU
    elif stride <= _HSEL_RES_MAX_WIDE:
        took = ARM_RESIDENT
    elif (
        topk_det_cluster_route[
            DType.bfloat16, ordered=False, deterministic=True
        ](ctx, align_up(stride, 8), TOP_K, total_q, True)
        > 0
    ):
        took = ARM_CLUSTER
    else:
        took = ARM_RANKED_STREAMING
    assert_true(
        stride > TOP_K and took == arm,
        String(
            "case meant for the ",
            _arm_name(arm),
            " arm takes the ",
            _arm_name(took),
            " arm",
        ),
    )

    # One shuffled page table serves both caches, so a physical row is a
    # valid row of the MLA cache too.
    var phys_to_token = List[Int](length=num_blocks * PAGE_SIZE, fill=-1)
    var lut_device = ctx.enqueue_create_buffer[.uint32](batch_size * lut_pages)
    with lut_device.map_to_host() as lut_host:
        for b in range(batch_size):
            for p in range(lut_pages):
                var block = 0
                if p < real_pages:
                    block = b * real_pages + (p * 7 + 1) % real_pages
                    for o in range(PAGE_SIZE):
                        if p * PAGE_SIZE + o < num_keys:
                            phys_to_token[block * PAGE_SIZE + o] = (
                                b * num_keys + p * PAGE_SIZE + o
                            )
                lut_host[b * lut_pages + p] = UInt32(block)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for b in range(batch_size):
            cl_host[b] = UInt32(cache_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with row_offsets_device.map_to_host() as ro_host:
        for b in range(batch_size + 1):
            ro_host[b] = UInt32(b * q_len)

    # The indexer's FP8 query, per-head weights, keys and key scales. Uniform
    # FP8 values make exact score ties routine, which is the regime where the
    # default contract's surviving set also varies.
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        total_q * INDEX_HEADS * INDEX_DEPTH
    )
    with q_device.map_to_host() as h:
        rand(h.as_span())
    var qs_device = ctx.enqueue_create_buffer[.float32](total_q * INDEX_HEADS)
    with qs_device.map_to_host() as h:
        rand(h.as_span())
    var k_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        num_blocks * PAGE_SIZE * INDEX_DEPTH
    )
    with k_device.map_to_host() as h:
        rand(h.as_span())
    var ks_device = ctx.enqueue_create_buffer[.float32](num_blocks * PAGE_SIZE)
    with ks_device.map_to_host() as h:
        rand(h.as_span())

    comptime index_params = KVCacheStaticParams(
        num_heads=1, head_size=INDEX_DEPTH, is_mla=True
    )
    comptime IndexCollection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        index_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ]
    var index_collection = IndexCollection(
        rebind[IndexCollection.blocks_tt_type](
            TileTensor(
                k_device,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[INDEX_DEPTH],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_device, row_major(Int64(batch_size), Int64(lut_pages)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(bound_len),
        rebind[IndexCollection.scales_tt_type](
            TileTensor(
                ks_device,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[1],
                ),
            ).as_unsafe_any_origin()
        ),
    )
    var q_tile = TileTensor(
        q_device, row_major(total_q, INDEX_HEADS, INDEX_DEPTH)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_q, INDEX_HEADS))
    var row_offsets_tile = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    # The MLA cache and query the decode reads.
    comptime mla_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var mla_blocks_device = ctx.enqueue_create_buffer[.bfloat16](
        num_blocks * PAGE_SIZE * Q_DEPTH
    )
    with mla_blocks_device.map_to_host() as h:
        randn(h.as_span(), mean=0.0, standard_deviation=0.5)
    var mla_q_size = total_q * MLA_HEADS * Q_DEPTH
    var mla_q_device = ctx.enqueue_create_buffer[.bfloat16](mla_q_size)
    with mla_q_device.map_to_host() as h:
        randn(h.as_span(), mean=0.0, standard_deviation=0.5)
    comptime MLACollection = PagedKVCacheCollection[
        DType.bfloat16,
        mla_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var mla_collection = MLACollection(
        rebind[MLACollection.blocks_tt_type](
            TileTensor(
                mla_blocks_device,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[Q_DEPTH],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_device, row_major(Int64(batch_size), Int64(lut_pages)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(cache_len),
    )
    var mla_args = MLADispatchScalarArgs[num_heads=MLA_HEADS, is_fp8_kv=False](
        batch_size, cache_len, q_len, ctx
    )
    var out_size = total_q * MLA_HEADS * V_DEPTH
    var label = String("selection")

    var first_sel = List[Int32]()
    var first_out = List[UInt16]()
    var drift = _Drift(List[Int32](), 0, 0)
    for launch in range(LAUNCHES):
        # Production writes the physical rows only; the logical output is
        # left alone.
        var logical_device = ctx.enqueue_create_buffer[.int32](sel_size)
        var phys_device = ctx.enqueue_create_buffer[.int32](sel_size)
        phys_device.enqueue_fill(Int32(-2))
        mla_indexer_ragged_float8_paged[
            DType.float8_e4m3fn,
            type_of(index_collection),
            INDEX_HEADS,
            INDEX_DEPTH,
            TOP_K,
            MaskName.CAUSAL.name,
            write_logical=False,
            Topk=Topk,
        ](
            TileTensor(logical_device, row_major(total_q, TOP_K)),
            q_tile,
            qs_tile,
            row_offsets_tile,
            index_collection,
            UInt32(0),
            ctx,
            PagedKVRemap[PAGE_SIZE](
                phys_device.unsafe_ptr(),
                lut_device.unsafe_ptr(),
                lut_pages,
                batch_size,
                UInt32(num_blocks),
            ),
        )

        var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
        if run_decode:
            flare_mla_decoding[
                rank=3,
                config=MHAConfig[DType.bfloat16](MLA_HEADS, Q_DEPTH),
                ragged=True,
                sparse=True,
                kernels=SM100MLA,
            ](
                TileTensor(
                    out_device,
                    row_major(total_q, Idx[MLA_HEADS], Idx[V_DEPTH]),
                ),
                TileTensor(
                    mla_q_device,
                    row_major(total_q, Idx[MLA_HEADS], Idx[Q_DEPTH]),
                ),
                mla_collection.get_key_cache(0),
                CausalMask(),
                row_offsets_tile,
                SCALE,
                ctx,
                mla_args.gpu_tile_tensor(),
                d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                    phys_device.unsafe_ptr()
                ),
                indices_stride=TOP_K,
            )

        var sel_host = ctx.enqueue_create_host_buffer[.int32](sel_size)
        ctx.enqueue_copy(sel_host, phys_device)
        var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
        if run_decode:
            ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()

        var sel = List[Int32](capacity=sel_size)
        for i in range(sel_size):
            sel.append(sel_host[i])
        _check_selection(
            sel,
            phys_to_token,
            batch_size,
            cache_len,
            q_len,
            String(label, " launch ", launch),
        )

        if launch == 0:
            first_sel = sel^
            if run_decode:
                for i in range(out_size):
                    first_out.append(bitcast[DType.uint16, 1](out_host[i]))
            continue
        var rows = 0
        for r in range(total_q):
            for c in range(TOP_K):
                if sel[r * TOP_K + c] != first_sel[r * TOP_K + c]:
                    rows += 1
                    break
        var values = 0
        if run_decode:
            for i in range(out_size):
                if bitcast[DType.uint16, 1](out_host[i]) != first_out[i]:
                    values += 1
        drift.selection_rows = max(drift.selection_rows, rows)
        drift.output_values = max(drift.output_values, values)
        _ = logical_device
        _ = phys_device
        _ = out_device

    _ = mla_args
    _ = lut_device
    _ = cache_lengths_device
    _ = row_offsets_device
    _ = q_device
    _ = qs_device
    _ = k_device
    _ = ks_device
    _ = mla_blocks_device
    _ = mla_q_device
    drift.first_sel = first_sel^
    return drift^


def _check_case(
    name: StaticString,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    frozen_cache_len: Int,
    run_decode: Bool,
    arm: Int,
    ctx: DeviceContext,
) raises:
    var loose = _run_case[BlockSplitTopk[]](
        batch_size, cache_len, q_len, frozen_cache_len, run_decode, arm, ctx
    )
    var strict = _run_case[BlockSplitTopk[True]](
        batch_size, cache_len, q_len, frozen_cache_len, run_decode, arm, ctx
    )
    if arm == ARM_CLUSTER:
        var ranked = _run_case[RankedTopk](
            batch_size, cache_len, q_len, frozen_cache_len, False, arm, ctx
        )
        var differ = 0
        for i in range(len(strict.first_sel)):
            if strict.first_sel[i] != ranked.first_sel[i]:
                differ += 1
        print(
            t"{name}: cluster against ranked select:"
            t" {differ} of {len(strict.first_sel)} slots differ"
        )
        assert_equal(
            differ,
            0,
            String(
                name, ": the cluster kernel's selection is not the ranked's"
            ),
        )
    print(
        t"{name}: batch={batch_size} cache={cache_len} q={q_len}"
        t" bound={frozen_cache_len} arm={_arm_name(arm)}; over {LAUNCHES}"
        t" launches,"
        t" default contract: {loose.selection_rows} rows reordered,"
        t" {loose.output_values} output values changed; deterministic:"
        t" {strict.selection_rows} rows, {strict.output_values} values"
    )
    assert_equal(
        strict.selection_rows,
        0,
        String(name, ": a deterministic selection changed between launches"),
    )
    assert_equal(
        strict.output_values,
        0,
        String(name, ": the decode over a deterministic selection changed"),
    )


def main() raises:
    with DeviceContext() as ctx:
        # Captured decode and speculative verify: few rows over the frozen
        # bound.
        _check_case(
            "decode",
            4,
            8191,
            1,
            FROZEN_CACHE_LEN,
            True,
            ARM_CLUSTER,
            ctx,
        )
        _check_case(
            "verify",
            2,
            4092,
            4,
            FROZEN_CACHE_LEN,
            True,
            ARM_CLUSTER,
            ctx,
        )
        # Fewer live keys than `top_k` under the frozen bound: the `-1` tail a
        # request's first 2048 tokens of decode see.
        _check_case(
            "short",
            2,
            100,
            1,
            FROZEN_CACHE_LEN,
            True,
            ARM_CLUSTER,
            ctx,
        )
        # Eager decode, whose bound is the batch's real length.
        _check_case("decode_eager", 4, 8191, 1, 0, True, ARM_RESIDENT, ctx)
        # Prefill rows, each with more candidates than slots. The decode
        # kernel does not take a prompt this long, so only the selection is
        # compared.
        _check_case("prefill", 1, 2048, 768, 0, False, ARM_FILLS_GPU, ctx)
