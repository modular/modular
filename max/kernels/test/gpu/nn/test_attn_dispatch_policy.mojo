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

"""Keep the generic attention dispatch equal to the functions it wraps."""

from max.gpu.host import DeviceContext
from nn.attention.gpu.attn_dispatch import (
    ATTN_DISPATCH_UNSET,
    ATTN_FLAG_FIXED_TOPK,
    ATTN_FLAG_PHYSICAL_INDICES,
    ATTN_FLAG_ROW_INVARIANT_VERIFY,
    ATTN_FLAG_SPLIT_HEAD_DIMS,
    ATTN_KV_DTYPE_FP8,
    ATTN_KV_DTYPE_OTHER,
    AttnDispatchDesc,
    attn_dispatch_resolve,
    mha_split_kv_decode_num_partitions_runtime,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_mla_decode_dispatch_scalars,
    mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_split_kv_decode_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.fa4_splitk_combine import P_MAX
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLA_SPARSE_KV_FORM_NVFP4,
    MLA_SPARSE_KV_FORM_PLAIN,
    MLA_SPARSE_KV_FORM_SCALED_FP8,
    compute_mla_dispatch_key_runtime,
)
from std.testing import assert_equal, assert_raises


def _batch_sizes() -> List[Int]:
    return [1, 2, 8, 32, 128]


def _cache_lengths() -> List[Int]:
    return [0, 1, 127, 512, 2048, 8191, 65536, 1000000]


def _desc(kind: String) -> AttnDispatchDesc:
    return AttnDispatchDesc(
        kind=kind,
        num_q_heads=ATTN_DISPATCH_UNSET,
        num_kv_heads=ATTN_DISPATCH_UNSET,
        head_dim=ATTN_DISPATCH_UNSET,
        page_size=ATTN_DISPATCH_UNSET,
        window_size=ATTN_DISPATCH_UNSET,
        sparse_indices_stride=ATTN_DISPATCH_UNSET,
        kv_dtype=ATTN_DISPATCH_UNSET,
        kv_form=ATTN_DISPATCH_UNSET,
        flags=0,
    )


def _check_mha(ctx: DeviceContext) raises:
    var kv_head_counts = [1, 2, 8, 16]
    for kv_heads in kv_head_counts:
        var desc = _desc("mha")
        desc.num_kv_heads = kv_heads
        for batch_size in _batch_sizes():
            for cache_length in _cache_lengths():
                var want = mha_decoding_num_partitions(
                    batch_size, cache_length, kv_heads, ctx
                )
                var got = attn_dispatch_resolve(
                    ctx, desc, batch_size, 1, cache_length
                )
                assert_equal(len(got.key), 1)
                assert_equal(got.key[0], want)
                assert_equal(len(got.metadata), 4)
                assert_equal(got.metadata[0], batch_size)
                assert_equal(got.metadata[1], 1)
                assert_equal(got.metadata[2], want)
                assert_equal(got.metadata[3], cache_length)


def _check_mha_split_head_dims(ctx: DeviceContext) raises:
    var sm_count = ctx.device().multiprocessor_count()
    var windows = [ATTN_DISPATCH_UNSET, 1024]
    for window in windows:
        var desc = _desc("mha")
        desc.num_kv_heads = 8
        desc.window_size = window
        desc.flags = ATTN_FLAG_SPLIT_HEAD_DIMS
        for batch_size in _batch_sizes():
            for cache_length in _cache_lengths():
                var want = mha_split_kv_decode_num_partitions_runtime(
                    batch_size,
                    cache_length,
                    8,
                    window if window >= 0 else 0,
                    sm_count,
                )
                var got = attn_dispatch_resolve(
                    ctx, desc, batch_size, 1, cache_length
                )
                assert_equal(got.key[0], want)
                assert_equal(got.metadata[2], want)


def _check_split_ladder() raises:
    """Checks the runtime SM count selects that count's own policy."""
    var windows = [0, 1024]
    for window in windows:
        for batch_size in _batch_sizes():
            for cache_length in _cache_lengths():
                comptime for sm_count in range(1, P_MAX + 1):
                    comptime if (
                        sm_count == 1
                        or sm_count == 74
                        or sm_count == 148
                        or sm_count == P_MAX
                    ):
                        assert_equal(
                            mha_split_kv_decode_num_partitions_runtime(
                                batch_size, cache_length, 8, window, sm_count
                            ),
                            mha_split_kv_decode_num_partitions[sm_count](
                                batch_size, cache_length, 8, window
                            ),
                        )
                # A device past the ladder takes its top rung.
                assert_equal(
                    mha_split_kv_decode_num_partitions_runtime(
                        batch_size, cache_length, 8, window, P_MAX + 40
                    ),
                    mha_split_kv_decode_num_partitions[P_MAX](
                        batch_size, cache_length, 8, window
                    ),
                )


def _mla_reference(
    ctx: DeviceContext,
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    stride: Int,
    kv_form: Int,
    flags: Int,
) raises -> Tuple[Int, Int, Int, Int]:
    if ctx.api() == "hip":
        var amd = amd_mla_decode_dispatch_scalars(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            ctx,
            sparse_indices_stride=stride,
        )
        return (amd[0], amd[1], amd[2], 0)
    return compute_mla_dispatch_key_runtime(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        stride,
        ctx.device().multiprocessor_count(),
        kv_form=kv_form,
        page_size=128,
        fixed_topk=(flags & ATTN_FLAG_FIXED_TOPK) != 0,
        physical_indices=(flags & ATTN_FLAG_PHYSICAL_INDICES) != 0,
        depth=576,
        row_invariant_verify=(flags & ATTN_FLAG_ROW_INVARIANT_VERIFY) != 0,
    )


def _check_mla_shape(
    ctx: DeviceContext,
    num_heads: Int,
    is_fp8_kv: Bool,
    stride: Int,
    kv_form: Int,
    flags: Int,
) raises:
    var desc = _desc("mla")
    desc.num_q_heads = num_heads
    desc.head_dim = 576
    desc.page_size = 128
    desc.kv_dtype = ATTN_KV_DTYPE_FP8 if is_fp8_kv else ATTN_KV_DTYPE_OTHER
    desc.kv_form = kv_form
    desc.flags = flags
    if stride > 0:
        desc.sparse_indices_stride = stride
    var q_lens = [1, 4]
    for q_len in q_lens:
        for batch_size in _batch_sizes():
            for cache_length in _cache_lengths():
                var want = _mla_reference(
                    ctx,
                    batch_size,
                    cache_length,
                    q_len,
                    num_heads,
                    is_fp8_kv,
                    stride,
                    kv_form,
                    flags,
                )
                var got = attn_dispatch_resolve(
                    ctx, desc, batch_size, q_len, cache_length
                )
                assert_equal(len(got.key), 2)
                assert_equal(got.key[0], want[2])
                assert_equal(got.key[1], want[3])
                assert_equal(len(got.metadata), 3)
                assert_equal(got.metadata[0], want[0])
                assert_equal(got.metadata[1], want[1])
                assert_equal(got.metadata[2], want[2])


def _check_mla(ctx: DeviceContext) raises:
    # 96 and 128 heads take the two-head-group split on SM100; the sparse
    # form has no 96-head launch and falls back to the dense key.
    var head_counts = [8, 16, 24, 32, 40, 48, 56, 64, 128]
    for heads in head_counts:
        _check_mla_shape(ctx, heads, True, 2048, MLA_SPARSE_KV_FORM_PLAIN, 0)
        _check_mla_shape(
            ctx,
            heads,
            True,
            2048,
            MLA_SPARSE_KV_FORM_SCALED_FP8,
            ATTN_FLAG_FIXED_TOPK,
        )
        _check_mla_shape(
            ctx,
            heads,
            False,
            2048,
            MLA_SPARSE_KV_FORM_NVFP4,
            ATTN_FLAG_FIXED_TOPK | ATTN_FLAG_PHYSICAL_INDICES,
        )
        _check_mla_shape(
            ctx,
            heads,
            True,
            2048,
            MLA_SPARSE_KV_FORM_SCALED_FP8,
            ATTN_FLAG_FIXED_TOPK | ATTN_FLAG_ROW_INVARIANT_VERIFY,
        )
    var dense_head_counts = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    for heads in dense_head_counts:
        _check_mla_shape(ctx, heads, True, 0, ATTN_DISPATCH_UNSET, 0)
        _check_mla_shape(ctx, heads, False, 0, ATTN_DISPATCH_UNSET, 0)


def _check_msa(ctx: DeviceContext) raises:
    var msa = attn_dispatch_resolve(ctx, _desc("msa"), 4, 1, 4096)
    assert_equal(len(msa.key), 0)
    assert_equal(len(msa.metadata), 2)
    assert_equal(msa.metadata[0], 4)
    assert_equal(msa.metadata[1], 4096)
    var index = attn_dispatch_resolve(ctx, _desc("msa_index"), 4, 1, 4096)
    assert_equal(len(index.key), 0)
    assert_equal(index.metadata[1], 4096)


def _check_dsa_indexer(ctx: DeviceContext) raises:
    # The key moves where the cache bound and the query rows pass the
    # single-block top-k's 2,048 columns, and nowhere else.
    var desc = _desc("dsa_indexer")
    var cases = [
        (1, 1, 1),
        (2047, 1, 1),
        (2048, 1, 0),
        (2044, 4, 1),
        (2045, 4, 0),
        (1000000, 1, 0),
    ]
    for shape in cases:
        for batch_size in _batch_sizes():
            var got = attn_dispatch_resolve(
                ctx, desc, batch_size, shape[1], shape[0]
            )
            assert_equal(len(got.key), 1)
            assert_equal(got.key[0], shape[2])
            assert_equal(len(got.metadata), 0)


def _check_errors(ctx: DeviceContext) raises:
    with assert_raises(contains="unknown attention dispatch kind"):
        _ = attn_dispatch_resolve(ctx, _desc("mamba"), 1, 1, 128)
    # A caller that omits a field the policy reads gets an error, not a key.
    with assert_raises(contains="num_kv_heads"):
        _ = attn_dispatch_resolve(ctx, _desc("mha"), 1, 1, 128)
    with assert_raises(contains="num_q_heads"):
        _ = attn_dispatch_resolve(ctx, _desc("mla"), 1, 1, 128)
    with assert_raises(contains="batch_size"):
        _ = attn_dispatch_resolve(ctx, _desc("msa"), 0, 1, 128)
    # A zero-initialized descriptor is refused like an unset one.
    var no_heads = _desc("mha")
    no_heads.num_kv_heads = 0
    with assert_raises(contains="positive descriptor field 'num_kv_heads'"):
        _ = attn_dispatch_resolve(ctx, no_heads, 1, 1, 128)
    var odd_dtype = _desc("mla")
    odd_dtype.num_q_heads = 64
    odd_dtype.kv_dtype = 7
    with assert_raises(contains="kv_dtype"):
        _ = attn_dispatch_resolve(ctx, odd_dtype, 1, 1, 128)
    var odd_flags = _desc("msa")
    odd_flags.flags = 1 << 12
    with assert_raises(contains="unknown attention dispatch flags"):
        _ = attn_dispatch_resolve(ctx, odd_flags, 1, 1, 128)


def main() raises:
    with DeviceContext() as ctx:
        _check_mha(ctx)
        _check_msa(ctx)
        _check_dsa_indexer(ctx)
        _check_errors(ctx)
        _check_split_ladder()
        var api = ctx.api()
        if api == "cuda" or api == "hip":
            _check_mha_split_head_dims(ctx)
            _check_mla(ctx)
