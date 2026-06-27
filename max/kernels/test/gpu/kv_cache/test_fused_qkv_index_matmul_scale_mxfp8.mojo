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
"""Numerical-equivalence test for the dual-cache fused QKV + index matmul.

SM100 (B200) MXFP8 (E8M0 scales, SF_VECTOR_SIZE=32). Drives the new dual-cache
fused op
(`generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4`, which
fuses MiniMax-M3's 5 projections Q/K/V/IndexQ/IndexK into one block-scaled GEMM
over the concatenated weight `[Wq|Wk|Wv|Wiq|Wik]`) and asserts that its output
bit-matches running the EXISTING single-cache fused MXFP8 op
(`generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4`) twice:

  1. `[Wq|Wk|Wv]` -> MAIN cache (K/V) + Q output.
  2. `[Wiq|Wik]`  -> INDEX cache (IndexK) + IndexQ output.

Because every output band boundary is a multiple of SF_MN_GROUP_SIZE (128), the
per-column scale lookup in the fused matmul is identical to the two unfused
matmuls, so the fused result is bit-exact (`assert_equal`) to the reference.

This test is GPU-only (SM100): the block-scaled matmul asserts on SM100. Run
via `bt-b200 //max/kernels/test/gpu/kv_cache:test_fused_qkv_index_matmul_scale_mxfp8`.
"""

from std.math import ceildiv
from std.random import random_ui64, seed

from std.gpu.host import DeviceContext
from std.memory import memset_zero
from std.testing import assert_equal

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
)

from std.utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

# M3-shaped (per-TP8-device) parameters, scaled down for a unit test.
# All band widths are multiples of SF_MN_GROUP_SIZE == 128.
comptime DATA_DTYPE = DType.float8_e4m3fn
comptime SCALE_DTYPE = MXFP8_SF_DTYPE  # float8_e8m0fnu
comptime OUT_DTYPE = DType.bfloat16  # KV cache + combined output dtype
comptime KV_DTYPE = DType.bfloat16
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE  # 32

comptime HEAD_SIZE = 128
# Main cache: GQA/MHA (non-MLA), `MAIN_KV_HEADS` KV head(s); Q has `NUM_Q_HEADS`.
comptime NUM_Q_HEADS = 8
comptime MAIN_KV_HEADS = 1
# Index cache: MLA — single latent head, K only (matches M3 model_config.py:
# is_mla=True, n_kv_heads=1). IndexQ has `NUM_INDEX_HEADS` heads (M3:
# sparse_num_index_heads=4), which is independent of the cache's num_heads (1).
# Using NUM_INDEX_HEADS=4 here exercises the `iq_dim != index cache num_heads`
# path that the earlier non-MLA single-head test missed.
comptime NUM_INDEX_HEADS = 4

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(
    num_heads=1, head_size=HEAD_SIZE, is_mla=True
)


def execute_dual_cache_fused[
    rtol: Float64 = 0.0,
    atol: Float64 = 0.0,
](
    prompt_lens: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    """Build small fake weights/caches and assert the dual-cache fused output
    bit-matches the two single-cache fused ops."""
    comptime hidden = 256  # K, multiple of SF_VECTOR_SIZE * SF_ATOM_K (128)
    comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 1024
    comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
    comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 128
    comptime ik_dim = HEAD_SIZE  # 128 (single index K head)

    comptime qkv_n = q_dim + 2 * kv_dim  # main matmul N
    comptime idx_n = iq_dim + ik_dim  # index matmul N
    comptime n_total = qkv_n + idx_n  # concatenated N
    comptime combined_out = q_dim + iq_dim  # dual-cache output width

    # Every band boundary must land on an SF-atom row group for bit-exactness.
    comptime assert q_dim % SF_MN_GROUP_SIZE == 0
    comptime assert kv_dim % SF_MN_GROUP_SIZE == 0
    comptime assert iq_dim % SF_MN_GROUP_SIZE == 0
    comptime assert ik_dim % SF_MN_GROUP_SIZE == 0

    var batch_size = len(prompt_lens)
    var cache_sizes = List[Int]()
    for _ in range(batch_size):
        cache_sizes.append(0)

    comptime num_paged_blocks = 32
    comptime page_size = 512

    comptime MainCollection = PagedKVCacheCollection[
        KV_DTYPE, main_kv_params, page_size, ...
    ]
    comptime IndexCollection = PagedKVCacheCollection[
        KV_DTYPE, index_kv_params, page_size, ...
    ]

    # ---- ragged inputs ----
    var clt = CacheLengthsTable.build(prompt_lens, cache_sizes, ctx)
    var total_length = clt.total_length
    var max_seq = clt.max_seq_length_batch
    var max_ctx = clt.max_full_context_length
    var input_row_offsets_tensor = clt.input_row_offsets.device_tensor()

    # ---- hidden state (M, K) fp8 ----
    comptime hs_layout = Layout.row_major(UNKNOWN_VALUE, hidden)
    var hs = ManagedLayoutTensor[DATA_DTYPE, hs_layout](
        RuntimeLayout[hs_layout].row_major(IndexList[2](total_length, hidden)),
        ctx,
    )
    random(hs.tensor[update=False]())
    var hs_dev = hs.device_tensor()

    # ---- concatenated weight (N_total, K) fp8 ----
    comptime w_layout = Layout.row_major(n_total, hidden)
    var w = ManagedLayoutTensor[DATA_DTYPE, w_layout](ctx)
    random(w.tensor[update=False]())
    var w_dev = w.device_tensor()

    # ---- scales (rank-5 SF-atom layout) for input and weight ----
    # Input scale shape: (ceil(M/128), ceil(K/(V*ATOM_K)), 32, 4, ATOM_K).
    comptime k_sf = ceildiv(hidden, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime input_sf_layout = Layout.row_major(
        UNKNOWN_VALUE, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var m_sf = ceildiv(total_length, SF_MN_GROUP_SIZE)
    var input_scale = ManagedLayoutTensor[SCALE_DTYPE, input_sf_layout](
        RuntimeLayout[input_sf_layout].row_major(
            IndexList[5](m_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
        ctx,
    )
    random(input_scale.tensor[update=False]())
    var input_scale_dev = input_scale.device_tensor()

    # Weight scale shape: (N_total/128, k_sf, 32, 4, ATOM_K).
    comptime n_sf = n_total // SF_MN_GROUP_SIZE
    comptime weight_sf_layout = Layout.row_major(
        n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var weight_scale = ManagedLayoutTensor[SCALE_DTYPE, weight_sf_layout](ctx)
    random(weight_scale.tensor[update=False]())
    var weight_scale_dev = weight_scale.device_tensor()

    # ---- KV cache blocks ----
    comptime kv_block_layout = Layout.row_major[6]()
    var main_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(main_block_shape), ctx
    )
    var main_blocks_ref = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(main_block_shape), ctx
    )
    # MLA index cache: single latent KV head (num_heads == 1), K only. The
    # 6D block tensor keeps the K/V axis (size 2) but only the K half is
    # written; the head axis is 1.
    var index_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(index_block_shape), ctx
    )
    var index_blocks_ref = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(index_block_shape), ctx
    )

    # Zero-initialize ALL cache buffers identically. `enqueue_create_buffer`
    # returns uninitialized device memory, so without this the verify loop would
    # compare independent garbage in slots neither run writes (the index cache's
    # unused V half, padding rows beyond `total_length`, etc.) and report
    # spurious sub-bf16-resolution diffs. Writing the host buffer here, then
    # syncing via `device_tensor()` below, guarantees unwritten slots match.
    var main_n0 = main_blocks.tensor[update=False]().runtime_layout.size()
    memset_zero(main_blocks.tensor[update=False]().ptr, main_n0)
    memset_zero(main_blocks_ref.tensor[update=False]().ptr, main_n0)
    var index_n0 = index_blocks.tensor[update=False]().runtime_layout.size()
    memset_zero(index_blocks.tensor[update=False]().ptr, index_n0)
    memset_zero(index_blocks_ref.tensor[update=False]().ptr, index_n0)

    var main_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )
    var index_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )

    var main_collection = MainCollection(
        main_blocks.device_tensor(),
        clt.cache_lengths.device_tensor(),
        main_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var main_collection_ref = MainCollection(
        main_blocks_ref.device_tensor(),
        clt.cache_lengths.device_tensor(),
        main_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        index_blocks.device_tensor(),
        clt.cache_lengths.device_tensor(),
        index_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection_ref = IndexCollection(
        index_blocks_ref.device_tensor(),
        clt.cache_lengths.device_tensor(),
        index_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    # ---- combined output buffers ----
    comptime out_layout = Layout.row_major(UNKNOWN_VALUE, combined_out)
    var fused_out = ManagedLayoutTensor[OUT_DTYPE, out_layout](
        RuntimeLayout[out_layout].row_major(
            IndexList[2](total_length, combined_out)
        ),
        ctx,
    )

    # Q-only and IndexQ-only reference outputs.
    comptime q_out_layout = Layout.row_major(UNKNOWN_VALUE, q_dim)
    var q_out = ManagedLayoutTensor[OUT_DTYPE, q_out_layout](
        RuntimeLayout[q_out_layout].row_major(
            IndexList[2](total_length, q_dim)
        ),
        ctx,
    )
    comptime iq_out_layout = Layout.row_major(UNKNOWN_VALUE, iq_dim)
    var iq_out = ManagedLayoutTensor[OUT_DTYPE, iq_out_layout](
        RuntimeLayout[iq_out_layout].row_major(
            IndexList[2](total_length, iq_dim)
        ),
        ctx,
    )

    # ============ DUAL-CACHE FUSED RUN ============
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_dev,
        input_scale_dev,
        weight_scale_dev,
        Float32(1.0),
        main_collection,
        index_collection,
        UInt32(layer_idx),
        iq_dim,
        fused_out.device_tensor(),
        ctx,
    )

    # ============ REFERENCE 1: QKV -> main cache + Q output ============
    # Sub-weight rows [0, qkv_n); weight-scale dim0 slice [0, qkv_n/128).
    var w_qkv = LayoutTensor[DATA_DTYPE, Layout.row_major(qkv_n, hidden)](
        w_dev.ptr,
        RuntimeLayout[Layout.row_major(qkv_n, hidden)].row_major(
            IndexList[2](qkv_n, hidden)
        ),
    )
    comptime qkv_n_sf = qkv_n // SF_MN_GROUP_SIZE
    var ws_qkv = LayoutTensor[
        SCALE_DTYPE,
        Layout.row_major(qkv_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K),
    ](
        weight_scale_dev.ptr,
        RuntimeLayout[
            Layout.row_major(
                qkv_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
            )
        ].row_major(
            IndexList[5](qkv_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )

    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_qkv,
        input_scale_dev,
        ws_qkv,
        Float32(1.0),
        main_collection_ref,
        UInt32(layer_idx),
        q_out.device_tensor(),
        ctx,
    )

    # ============ REFERENCE 2: IndexQK -> index cache + IndexQ output ======
    # Sub-weight rows [qkv_n, n_total); weight-scale dim0 slice
    # [qkv_n/128, n_total/128). Because qkv_n is a multiple of 128, the slice
    # starts on an atom boundary.
    var w_idx = LayoutTensor[DATA_DTYPE, Layout.row_major(idx_n, hidden)](
        w_dev.ptr + qkv_n * hidden,
        RuntimeLayout[Layout.row_major(idx_n, hidden)].row_major(
            IndexList[2](idx_n, hidden)
        ),
    )
    comptime idx_n_sf = idx_n // SF_MN_GROUP_SIZE
    var ws_idx = LayoutTensor[
        SCALE_DTYPE,
        Layout.row_major(idx_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K),
    ](
        weight_scale_dev.ptr
        + qkv_n_sf * k_sf * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K,
        RuntimeLayout[
            Layout.row_major(
                idx_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
            )
        ].row_major(
            IndexList[5](idx_n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )

    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_idx,
        input_scale_dev,
        ws_idx,
        Float32(1.0),
        index_collection_ref,
        UInt32(layer_idx),
        iq_out.device_tensor(),
        ctx,
    )

    ctx.synchronize()

    # ============ VERIFY ============
    var fused_host = fused_out.tensor[update=True]()
    var q_host = q_out.tensor[update=True]()
    var iq_host = iq_out.tensor[update=True]()
    var main_host = main_blocks.tensor[update=True]()
    var main_ref_host = main_blocks_ref.tensor[update=True]()
    var index_host = index_blocks.tensor[update=True]()
    var index_ref_host = index_blocks_ref.tensor[update=True]()

    # Q / IndexQ output regions must be bit-exact (these matched in the first
    # GPU run; keep them strict).
    for m in range(total_length):
        for c in range(q_dim):
            assert_equal(fused_host[m, c], q_host[m, c])
        # IndexQ region (columns [q_dim, q_dim+iq_dim)) == single-cache IndexQ.
        for c in range(iq_dim):
            assert_equal(fused_host[m, q_dim + c], iq_host[m, c])

    # Cache comparisons: count mismatches and track the max abs diff across ALL
    # slots (do not stop at the first). With both buffers zero-initialized, a
    # benign accumulation-order/denormal difference shows O(1) tiny diffs; a
    # mis-route would show O(written-elements) diffs. We assert bit-exactness
    # since QKV->main and IndexQK->index run the same block-scaled matmul over
    # the same per-band scales, but report the stats either way.
    var main_n = main_host.runtime_layout.size()
    var main_mismatches = 0
    var main_max_diff = Float32(0.0)
    for i in range(main_n):
        var a = main_host.ptr[i].cast[DType.float32]()
        var b = main_ref_host.ptr[i].cast[DType.float32]()
        if a != b:
            main_mismatches += 1
            main_max_diff = max(main_max_diff, abs(a - b))

    var index_n = index_host.runtime_layout.size()
    var index_mismatches = 0
    var index_max_diff = Float32(0.0)
    for i in range(index_n):
        var a = index_host.ptr[i].cast[DType.float32]()
        var b = index_ref_host.ptr[i].cast[DType.float32]()
        if a != b:
            index_mismatches += 1
            index_max_diff = max(index_max_diff, abs(a - b))

    print(
        "main cache: ",
        main_mismatches,
        " mismatches / ",
        main_n,
        " slots, max_abs_diff=",
        main_max_diff,
        sep="",
    )
    print(
        "index cache: ",
        index_mismatches,
        " mismatches / ",
        index_n,
        " slots, max_abs_diff=",
        index_max_diff,
        sep="",
    )

    assert_equal(main_mismatches, 0)
    assert_equal(index_mismatches, 0)

    _ = clt^
    _ = main_lut^
    _ = index_lut^


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Context-encoding (prefill): a couple of small ragged prompts.
        var ce_lens = List[Int]()
        for _ in range(2):
            ce_lens.append(Int(random_ui64(8, 64)))
        execute_dual_cache_fused(ce_lens, 4, 1, ctx)

        # Single-token (decode-like) batch.
        var tg_lens = List[Int]()
        for _ in range(4):
            tg_lens.append(1)
        execute_dual_cache_fused(tg_lens, 4, 2, ctx)
    print("\n=== ALL TESTS PASSED ===\n")
