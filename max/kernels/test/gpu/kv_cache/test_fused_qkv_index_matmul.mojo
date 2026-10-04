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
"""Numerical-equivalence test for the BF16 dual-cache fused QKV + index matmul.

Hardware-agnostic BF16 (non-scaled). Drives the new dual-cache fused op
(`generic_fused_qkv_index_matmul_kv_cache_paged_ragged`, which fuses
MiniMax-M3's 5 projections Q/K/V/IndexQ/IndexK into ONE plain BF16 GEMM over the
concatenated weight `[Wq|Wk|Wv|Wiq|Wik]`) and asserts its output matches running
the EXISTING single-cache fused BF16 op
(`generic_fused_qkv_matmul_kv_cache_paged_ragged`) TWICE:

  1. `[Wq|Wk|Wv]` -> MAIN cache (K/V) + Q output.
  2. `[Wiq|Wik]`  -> INDEX cache (IndexK, MLA single head) + IndexQ output.

Both runs read the SAME hidden state and the SAME weight rows, so the fused
result must equal the two unfused matmuls up to floating-point reassociation
(the fused GEMM may pick a different N-tiling / tactic than the two smaller
GEMMs). The test reports the actual max abs / rel deltas for Q, IndexQ, main
K/V, and index K, and asserts they stay within `|a-b| <= atol + rtol*|b|`; a
mis-routed column would show O(1) relative error, far above the reassociation
floor.

Unlike the MXFP8 variant, this path is dtype/hardware-agnostic (plain
`_matmul_common`; runs on AMD CDNA4 / MI355 and NVIDIA), so there is NO
scale-factor machinery and NO SF-atom band-alignment constraint.
"""

from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from std.memory import unsafe_memset_zero
from std.testing import assert_true

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    row_major,
)
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
)


from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

# M3 per-device (TP=4) BF16 parameters. Hidden/weight/Q outputs stay BF16.
# IndexK may be e4m3 (`index_kv_dtype`); main KV stays BF16.
comptime DATA_DTYPE = DType.bfloat16  # hidden state + weight
comptime OUT_DTYPE = DType.bfloat16  # combined output + KV caches
comptime KV_DTYPE = DType.bfloat16

comptime HEAD_SIZE = 128
# Main cache: GQA/MHA (non-MLA). Q has `NUM_Q_HEADS`, main KV has
# `MAIN_KV_HEADS`. q_dim = 2048, kv_dim = 128.
comptime NUM_Q_HEADS = 16
comptime MAIN_KV_HEADS = 1
# Index cache: MLA — single latent head, K only (M3: is_mla=True, n_kv_heads=1).
# IndexQ has `NUM_INDEX_HEADS` heads; here 1 -> iq_dim = 128 (M3 per-TP4-device
# indexer projection is 256 = IndexQ 128 + IndexK 128).
comptime NUM_INDEX_HEADS = 1

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(
    num_heads=1, head_size=HEAD_SIZE, is_mla=True
)


def execute_dual_cache_fused_bf16[
    index_kv_dtype: DType = DType.bfloat16,
    rtol: Float32 = 1e-2,
    atol: Float32 = 1e-2,
](
    prompt_lens: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    """Build small BF16 weights/caches and assert the dual-cache fused output
    matches the two single-cache fused ops within `atol + rtol*|ref|`."""
    comptime hidden = 6144  # K (M3 per-TP4-device hidden)
    comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 2048
    comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
    comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 128
    comptime ik_dim = HEAD_SIZE  # 128 (single index K head)

    comptime qkv_n = q_dim + 2 * kv_dim  # 2304 (main matmul N)
    comptime idx_n = iq_dim + ik_dim  # 256 (index matmul N)
    comptime n_total = qkv_n + idx_n  # 2560 (concatenated / stacked N)
    comptime combined_out = q_dim + iq_dim  # 2176 (dual-cache visible output)

    var batch_size = len(prompt_lens)
    var cache_sizes = List[Int]()
    for _ in range(batch_size):
        cache_sizes.append(0)

    comptime num_paged_blocks = 32
    comptime page_size = 512

    comptime MainCollection = PagedKVCacheCollection[
        KV_DTYPE,
        main_kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime IndexCollection = PagedKVCacheCollection[
        index_kv_dtype,
        index_kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    # ---- ragged inputs ----
    var clt = CacheLengthsTable.build(prompt_lens, cache_sizes, ctx)
    var total_length = clt.total_length
    var max_seq = clt.max_seq_length_batch
    var max_ctx = clt.max_full_context_length
    var input_row_offsets_tensor = clt.input_row_offsets.device_tile_tensor()

    # ---- hidden state (M, K) bf16 ----
    var hs = HostDeviceTileTensor[DATA_DTYPE](
        row_major(Coord(total_length, Idx[hidden])), ctx
    )
    random(hs.host_tensor())
    var hs_dev = hs.device_tensor()

    # ---- concatenated / stacked weight (N_total, K) bf16 ----
    var w = HostDeviceTileTensor[DATA_DTYPE](
        row_major(Coord(Idx[n_total], Idx[hidden])), ctx
    )
    random(w.host_tensor())
    var w_dev = w.device_tensor()

    # ---- KV cache blocks ----
    var main_blocks = HostDeviceTileTensor[KV_DTYPE](
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[MAIN_KV_HEADS],
                Idx[HEAD_SIZE],
            )
        ),
        ctx,
    )
    var main_blocks_ref = HostDeviceTileTensor[KV_DTYPE](
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[MAIN_KV_HEADS],
                Idx[HEAD_SIZE],
            )
        ),
        ctx,
    )
    # The MLA collection derives packed K-only pages; the oversized
    # allocation retains trailing capacity for the untouched-region check.
    var index_blocks = HostDeviceTileTensor[index_kv_dtype](
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[1],
                Idx[HEAD_SIZE],
            )
        ),
        ctx,
    )
    var index_blocks_ref = HostDeviceTileTensor[index_kv_dtype](
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[1],
                Idx[HEAD_SIZE],
            )
        ),
        ctx,
    )

    # Zero-initialize ALL cache buffers identically. `enqueue_create_buffer`
    # returns uninitialized device memory, so without this the verify loop would
    # compare independent garbage in slots neither run writes (the index cache's
    # unused trailing capacity, padding rows beyond `total_length`, etc.) and report
    # spurious diffs. Writing the host buffer here, then calling
    # `to_device()` below, guarantees unwritten slots match.
    var main_n0 = main_blocks.host_tensor().num_elements()
    unsafe_memset_zero(main_blocks.host_tensor().ptr, main_n0)
    unsafe_memset_zero(main_blocks_ref.host_tensor().ptr, main_n0)
    var index_n0 = index_blocks.host_tensor().num_elements()
    unsafe_memset_zero(index_blocks.host_tensor().ptr, index_n0)
    unsafe_memset_zero(index_blocks_ref.host_tensor().ptr, index_n0)

    var main_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )
    var index_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )

    var main_collection = MainCollection(
        rebind[MainCollection.blocks_tt_type](
            main_blocks.device_tensor().as_unsafe_any_origin()
        ),
        clt.cache_lengths.device_tile_tensor(),
        main_lut.device_tile_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var main_collection_ref = MainCollection(
        rebind[MainCollection.blocks_tt_type](
            main_blocks_ref.device_tensor().as_unsafe_any_origin()
        ),
        clt.cache_lengths.device_tile_tensor(),
        main_lut.device_tile_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        rebind[IndexCollection.blocks_tt_type](
            index_blocks.device_tensor()
            .tile(
                Coord(
                    Int64(num_paged_blocks),
                    Idx[1],
                    Int64(num_layers),
                    Idx[page_size],
                    Idx[1],
                    Idx[HEAD_SIZE],
                ),
                Coord(0, 0, 0, 0, 0, 0),
            )
            .as_unsafe_any_origin()
        ),
        clt.cache_lengths.device_tile_tensor(),
        index_lut.device_tile_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection_ref = IndexCollection(
        rebind[IndexCollection.blocks_tt_type](
            index_blocks_ref.device_tensor()
            .tile(
                Coord(
                    Int64(num_paged_blocks),
                    Idx[1],
                    Int64(num_layers),
                    Idx[page_size],
                    Idx[1],
                    Idx[HEAD_SIZE],
                ),
                Coord(0, 0, 0, 0, 0, 0),
            )
            .as_unsafe_any_origin()
        ),
        clt.cache_lengths.device_tile_tensor(),
        index_lut.device_tile_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    # ---- combined output buffer (Q then IndexQ) ----
    var fused_out = HostDeviceTileTensor[OUT_DTYPE](
        row_major(Coord(total_length, Idx[combined_out])), ctx
    )

    # Q-only and IndexQ-only reference outputs.
    var q_out = HostDeviceTileTensor[OUT_DTYPE](
        row_major(Coord(total_length, Idx[q_dim])), ctx
    )
    var iq_out = HostDeviceTileTensor[OUT_DTYPE](
        row_major(Coord(total_length, Idx[iq_dim])), ctx
    )

    # ============ DUAL-CACHE FUSED RUN (one GEMM over stacked weight) ========
    hs.to_device()
    w.to_device()
    fused_out.to_device()
    q_out.to_device()
    iq_out.to_device()
    main_blocks.to_device()
    main_blocks_ref.to_device()
    index_blocks.to_device()
    index_blocks_ref.to_device()
    ctx.synchronize()

    generic_fused_qkv_index_matmul_kv_cache_paged_ragged[target="gpu",](
        hs_dev,
        input_row_offsets_tensor,
        w_dev,
        main_collection,
        index_collection,
        UInt32(layer_idx),
        iq_dim,
        fused_out.device_tensor(),
        ctx,
    )

    # ============ REFERENCE 1: QKV -> main cache + Q output ============
    # Sub-weight rows [0, qkv_n): [Wq|Wk|Wv].
    var w_qkv = w_dev.slice[0:qkv_n, 0:hidden]()

    generic_fused_qkv_matmul_kv_cache_paged_ragged[target="gpu",](
        hs_dev,
        input_row_offsets_tensor,
        w_qkv,
        main_collection_ref,
        UInt32(layer_idx),
        q_out.device_tensor(),
        ctx,
    )

    # ============ REFERENCE 2: IndexQK -> index cache + IndexQ output ======
    # Sub-weight rows [qkv_n, n_total): [Wiq|Wik]. The index cache is MLA, so
    # the single-cache op routes cols [0, iq_dim) -> IndexQ output and
    # [iq_dim, iq_dim+ik_dim) -> index K cache (head 0), matching the fused
    # kernel's IndexQ/IndexK bands.
    var w_idx = w_dev.slice[qkv_n:n_total, 0:hidden]()

    generic_fused_qkv_matmul_kv_cache_paged_ragged[target="gpu",](
        hs_dev,
        input_row_offsets_tensor,
        w_idx,
        index_collection_ref,
        UInt32(layer_idx),
        iq_out.device_tensor(),
        ctx,
    )

    ctx.synchronize()

    # ============ VERIFY ============
    fused_out.to_host()
    q_out.to_host()
    iq_out.to_host()
    main_blocks.to_host()
    main_blocks_ref.to_host()
    index_blocks.to_host()
    index_blocks_ref.to_host()
    ctx.synchronize()

    var fused_host = fused_out.host_tensor()
    var q_host = q_out.host_tensor()
    var iq_host = iq_out.host_tensor()
    var main_host = main_blocks.host_tensor()
    var main_ref_host = main_blocks_ref.host_tensor()
    var index_host = index_blocks.host_tensor()
    var index_ref_host = index_blocks_ref.host_tensor()

    # ---- Q output region: fused[:, 0:q_dim] vs single-cache Q ----
    var q_mism = 0
    var q_viol = 0
    var q_max_abs = Float32(0)
    var q_max_rel = Float32(0)
    for m in range(total_length):
        for c in range(q_dim):
            var a = rebind[Float32](fused_host[m, c].cast[.float32]())
            var b = rebind[Float32](q_host[m, c].cast[.float32]())
            var d = abs(a - b)
            if a != b:
                q_mism += 1
            if d > atol + rtol * abs(b):
                q_viol += 1
            q_max_abs = max(q_max_abs, d)
            if abs(b) > 1e-30:
                q_max_rel = max(q_max_rel, d / abs(b))

    # ---- IndexQ output region: fused[:, q_dim:q_dim+iq_dim] vs IndexQ ----
    var iq_mism = 0
    var iq_viol = 0
    var iq_max_abs = Float32(0)
    var iq_max_rel = Float32(0)
    for m in range(total_length):
        for c in range(iq_dim):
            var a = rebind[Float32](fused_host[m, q_dim + c].cast[.float32]())
            var b = rebind[Float32](iq_host[m, c].cast[.float32]())
            var d = abs(a - b)
            if a != b:
                iq_mism += 1
            if d > atol + rtol * abs(b):
                iq_viol += 1
            iq_max_abs = max(iq_max_abs, d)
            if abs(b) > 1e-30:
                iq_max_rel = max(iq_max_rel, d / abs(b))

    # ---- main cache (K + V) flat compare ----
    var main_n = main_host.num_elements()
    var main_mism = 0
    var main_viol = 0
    var main_max_abs = Float32(0)
    var main_max_rel = Float32(0)
    for i in range(main_n):
        var a = main_host.ptr[i].cast[.float32]()
        var b = main_ref_host.ptr[i].cast[.float32]()
        var d = abs(a - b)
        if a != b:
            main_mism += 1
        if d > atol + rtol * abs(b):
            main_viol += 1
        main_max_abs = max(main_max_abs, d)
        if abs(b) > 1e-30:
            main_max_rel = max(main_max_rel, d / abs(b))

    # ---- index cache (K only) flat compare ----
    var index_n = index_host.num_elements()
    var index_mism = 0
    var index_viol = 0
    var index_max_abs = Float32(0)
    var index_max_rel = Float32(0)
    for i in range(index_n):
        var a = index_host.ptr[i].cast[.float32]()
        var b = index_ref_host.ptr[i].cast[.float32]()
        var d = abs(a - b)
        if a != b:
            index_mism += 1
        if d > atol + rtol * abs(b):
            index_viol += 1
        index_max_abs = max(index_max_abs, d)
        if abs(b) > 1e-30:
            index_max_rel = max(index_max_rel, d / abs(b))

    print(
        "  Q      : ",
        q_mism,
        "/",
        total_length * q_dim,
        " differ, ",
        q_viol,
        " beyond tol, max_abs=",
        q_max_abs,
        " max_rel=",
        q_max_rel,
        sep="",
    )
    print(
        "  IndexQ : ",
        iq_mism,
        "/",
        total_length * iq_dim,
        " differ, ",
        iq_viol,
        " beyond tol, max_abs=",
        iq_max_abs,
        " max_rel=",
        iq_max_rel,
        sep="",
    )
    print(
        "  mainKV : ",
        main_mism,
        "/",
        main_n,
        " differ, ",
        main_viol,
        " beyond tol, max_abs=",
        main_max_abs,
        " max_rel=",
        main_max_rel,
        sep="",
    )
    print(
        "  indexK : ",
        index_mism,
        "/",
        index_n,
        " differ, ",
        index_viol,
        " beyond tol, max_abs=",
        index_max_abs,
        " max_rel=",
        index_max_rel,
        sep="",
    )

    # The accuracy check: no element may exceed |a-b| <= atol + rtol*|ref|.
    # A mis-routed column would show O(1) relative error (a full-magnitude diff
    # or a zero-vs-value diff), far above the bf16 reassociation floor.
    assert_true(q_viol == 0, "Q output beyond tolerance vs single-cache Q")
    assert_true(
        iq_viol == 0, "IndexQ output beyond tolerance vs single-cache IndexQ"
    )
    assert_true(main_viol == 0, "main K/V cache beyond tolerance vs reference")
    assert_true(index_viol == 0, "index K cache beyond tolerance vs reference")

    _ = clt^
    _ = main_lut^
    _ = index_lut^


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Context-encoding (prefill): a couple of small ragged prompts.
        print("[case ce] ragged prefill, 2 prompts")
        var ce_lens = List[Int]()
        for _ in range(2):
            ce_lens.append(Int(random_ui64(8, 64)))
        execute_dual_cache_fused_bf16(ce_lens, 4, 1, ctx)

        # Single-token (decode-like) batch.
        print("[case tg] decode-like, 4 prompts of length 1")
        var tg_lens = List[Int]()
        for _ in range(4):
            tg_lens.append(1)
        execute_dual_cache_fused_bf16(tg_lens, 4, 2, ctx)

        print("[case ce-e4m3-indexk] ragged prefill, e4m3 IndexK")
        execute_dual_cache_fused_bf16[DType.float8_e4m3fn](ce_lens, 4, 1, ctx)
    print("\n=== ALL TESTS PASSED ===\n")
