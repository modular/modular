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
"""Numerical test of sparse MLA decode with `skip_masked_tiles`.

This is the configuration the `mla_prefill_decode_graph_*` ops run sparse
prefill on for up to 64 heads, over a unit-scale FP8 cache (the native-FP8
kernel) or a bf16 cache (the KV_BF16 kernel): one CTA row per query token over
a ragged prompt, each
token's row packed with its causally visible keys first. Rows here are padded
two ways, so skipping is checked against both bounds it applies:

  * `-1` padding past the token's causal key count, with no lengths given;
  * per-token `topk_lengths` shorter than the causal count, with the slots
    past each length holding live keys that only the length excludes.

A prompt longer than 65535 tokens checks that the grid addresses every token,
and 128 heads check a grid with two head tiles per token. The FP8 128-head
cells whose CTAs walk more than 384 keys run the CTA pair, and under
`MLA_SPARSE_CG2_ROUTE=0` every FP8 128-head cell does.
"""

from std.testing import assert_equal
from std.math import ceildiv, exp, sqrt
from std.random import randn, seed
from std.sys import get_defined_int
from internal_utils import arg_parse, coprime_multiplier

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig, maybe_null_pointer
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    _MLA_SPARSE_CG2_ROUTE,
)
from std.utils.numerics import isnan, nan

comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
# Reachability control for the FP8 h=128 pair cells (`MLA_CG2_SLICE_BREAK`,
# see mla_decode_utils.mojo): only those cells run, and each must fail exactly
# when it runs the broken pair (`_run_pair_reachability`).
comptime CG2_SLICE_BREAK = get_defined_int["MLA_CG2_SLICE_BREAK", 0]() != 0
# Planted fault for the head-padding cells (`MLA_CORR_BREAK=1` skips the
# correction's O-rescale walk, see mla_decode_utils.mojo): they walk two key
# tiles, so the running max grows past the first and a stale O fails them.
comptime CORR_BREAK = get_defined_int["MLA_CORR_BREAK", 0]() != 0


def attention_reference[
    dtype: DType, num_heads: Int, top_k: Int, sink: Bool = False
](
    q_host: List[Scalar[dtype]],
    kv_host: List[Scalar[dtype]],
    indices: List[Int32],
    slot_counts: List[Int],
    tokens: List[Int],
    scale: Float32,
) -> List[Float32]:
    """Returns attention of each of `tokens` over its first `slot_counts`
    slots."""
    var out = List[Float32](length=len(tokens) * num_heads * V_DEPTH, fill=0)
    var scores = List[Float64](length=top_k, fill=0)
    for ti in range(len(tokens)):
        var g = tokens[ti]
        for h in range(num_heads):
            var q_base = (g * num_heads + h) * Q_DEPTH
            var row_max = Float64(-1e30)
            for i in range(slot_counts[g]):
                var k_base = Int(indices[g * top_k + i]) * Q_DEPTH
                var acc = Float64(0)
                for d in range(Q_DEPTH):
                    acc += (
                        q_host[q_base + d].cast[.float64]()
                        * kv_host[k_base + d].cast[.float64]()
                    )
                scores[i] = acc * Float64(scale)
                row_max = max(row_max, scores[i])
            var denom = Float64(0)
            for i in range(slot_counts[g]):
                scores[i] = exp(scores[i] - row_max)
                denom += scores[i]
            comptime if sink:
                denom += exp(
                    Float64(0.5 + Float64(h) / Float64(num_heads)) - row_max
                )
            var o_base = (ti * num_heads + h) * V_DEPTH
            for d in range(V_DEPTH):
                var acc = Float64(0)
                for i in range(slot_counts[g]):
                    acc += (
                        scores[i]
                        * kv_host[
                            Int(indices[g * top_k + i]) * Q_DEPTH + d
                        ].cast[.float64]()
                    )
                out[o_base + d] = Float32(acc / denom) if denom > 0 else 0
    return out^


def agreement[
    num_heads: Int
](
    out_host: Span[BFloat16, _], reference: List[Float32], tokens: List[Int]
) raises -> Tuple[Float64, Float64]:
    """Returns the max abs error and cosine of `tokens`' rows of `out_host` vs
    `reference`."""
    comptime row = num_heads * V_DEPTH
    var max_err = Float64(0)
    var dot = Float64(0)
    var norm_a = Float64(0)
    var norm_r = Float64(0)
    for i in range(len(reference)):
        var a = out_host[tokens[i // row] * row + i % row].cast[.float64]()
        var r = Float64(reference[i])
        if isnan(a):
            raise Error(String("NaN at ", i))
        max_err = max(max_err, abs(a - r))
        dot += a * r
        norm_a += a * a
        norm_r += r * r
    var cosine = dot / sqrt(
        norm_a * norm_r
    ) if norm_a * norm_r > 0 else Float64(norm_a == norm_r)
    return (max_err, cosine)


def agrees(max_err: Float64, cosine: Float64) -> Bool:
    # FP8 P quantization bounds the agreement.
    return cosine >= 0.999 and max_err <= 0.1


def run_case[
    num_heads: Int,
    top_k: Int,
    per_token_lengths: Bool,
    dtype: DType = DType.float8_e4m3fn,
    zero_lengths: Bool = False,
    logical: Bool = False,
    sink: Bool = False,
    selection_lag: Int = 0,
](
    name: StringLiteral,
    seq_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    edge_tokens: Int = 0,
    forced_np: Int = 0,
    dispatch_bound: Int = 0,
) raises:
    """Runs one case and checks it against the CPU reference.

    Args:
        name: Case name for the log.
        seq_lens: Query tokens per sequence.
        cache_lens: Cached tokens per sequence.
        ctx: The device context.
        edge_tokens: When positive, checks only the first and last
            `edge_tokens` tokens of each sequence, since the reference is too
            slow for a long prompt.
        forced_np: When positive, forces this many split-K partitions.
        dispatch_bound: Optional padded query maximum for capture replay.
    """
    _ = _run_case_impl[
        num_heads,
        top_k,
        per_token_lengths,
        dtype,
        zero_lengths,
        logical,
        sink,
        selection_lag,
    ](name, seq_lens, cache_lens, ctx, edge_tokens, forced_np, dispatch_bound)


def _run_case_impl[
    num_heads: Int,
    top_k: Int,
    per_token_lengths: Bool,
    dtype: DType = DType.float8_e4m3fn,
    zero_lengths: Bool = False,
    logical: Bool = False,
    sink: Bool = False,
    selection_lag: Int = 0,
    canary_head: Int = -1,
    poison_heads_from: Int = -1,
](
    name: StringLiteral,
    seq_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    edge_tokens: Int = 0,
    forced_np: Int = 0,
    dispatch_bound: Int = 0,
) raises -> List[BFloat16]:
    """Runs one case; see `run_case`.

    With `canary_head` nonnegative it returns that head's output rows instead
    of checking the reference, so two runs can be compared bit for bit.
    `poison_heads_from` replaces every head from that index up with NaN, which
    a leaky kernel would let into `canary_head`'s rows.
    """
    comptime scale = Float32(0.0625)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var batch_size = len(seq_lens)

    var row_starts = List[Int]()
    var total_tokens = 0
    var max_seq_len = 0
    var max_cache_len = 0
    var pages_per_seq = 0
    for bi in range(batch_size):
        row_starts.append(total_tokens)
        total_tokens += seq_lens[bi]
        max_seq_len = max(max_seq_len, seq_lens[bi])
        max_cache_len = max(max_cache_len, cache_lens[bi])
        pages_per_seq = max(
            pages_per_seq, ceildiv(cache_lens[bi] + seq_lens[bi], PAGE_SIZE)
        )
    row_starts.append(total_tokens)
    max_seq_len = max(max_seq_len, dispatch_bound)
    var total_pages = batch_size * pages_per_seq
    var kv_rows = total_pages * PAGE_SIZE

    # Zero-mean data, so a dropped or extra key moves the output.
    var kv_host = List[Scalar[dtype]](capacity=kv_rows * Q_DEPTH)
    var q_host = List[Scalar[dtype]](
        capacity=total_tokens * num_heads * Q_DEPTH
    )
    var src = ctx.enqueue_create_host_buffer[.float32](
        max(kv_rows, total_tokens * num_heads) * Q_DEPTH
    )
    ctx.synchronize()
    var q_elems = total_tokens * num_heads * Q_DEPTH
    randn(src.as_span()[: kv_rows * Q_DEPTH], mean=0.0, standard_deviation=1.0)
    for i in range(kv_rows * Q_DEPTH):
        kv_host.append(src[i].cast[dtype]())
    randn(src.as_span()[:q_elems], mean=0.0, standard_deviation=1.0)
    for i in range(q_elems):
        q_host.append(src[i].cast[dtype]())
    comptime if poison_heads_from >= 0:
        for g in range(total_tokens):
            for h in range(poison_heads_from, num_heads):
                for d in range(Q_DEPTH):
                    q_host[(g * num_heads + h) * Q_DEPTH + d] = nan[dtype]()

    # The token at position p sees ring = cache_len + p + 1 keys and packs
    # min(top_k, ring) of them first. With per-token lengths, each length
    # stops 0-75% short of that count and the row keeps live keys past it.
    var indices = List[Int32](capacity=total_tokens * top_k)
    var lengths = List[Int32](capacity=total_tokens)
    var causal_counts = List[Int](capacity=total_tokens)
    for bi in range(batch_size):
        for s in range(seq_lens[bi]):
            var g = row_starts[bi] + s
            var ring = max(cache_lens[bi] + s + 1 - selection_lag, 1)
            var n_valid = min(top_k, ring)
            var n_len = n_valid
            comptime if per_token_lengths:
                n_len = max(n_valid - (g % 4) * n_valid // 4, 1)
            comptime if zero_lengths:
                if g % 4 == 0:
                    n_len = 0
            lengths.append(Int32(n_len))
            causal_counts.append(n_valid)
            var mult = coprime_multiplier(ring)
            for i in range(top_k):
                var t = (i * mult + 1 + s) % ring
                if i >= n_valid:
                    comptime if per_token_lengths:
                        t = i % ring
                    else:
                        indices.append(-1)
                        continue
                indices.append(
                    Int32(
                        (bi * pages_per_seq + t // PAGE_SIZE) * PAGE_SIZE
                        + t % PAGE_SIZE
                    )
                )

    var blocks_device = ctx.enqueue_create_buffer[dtype](kv_rows * Q_DEPTH)
    var q_device = ctx.enqueue_create_buffer[dtype](
        total_tokens * num_heads * Q_DEPTH
    )
    var out_device = ctx.enqueue_create_buffer[.bfloat16](
        total_tokens * num_heads * V_DEPTH
    )
    var indices_device = ctx.enqueue_create_buffer[.int32](total_tokens * top_k)
    var sinks_device = ctx.enqueue_create_buffer[.float32](num_heads)
    with sinks_device.map_to_host() as host:
        for h in range(num_heads):
            host[h] = Float32(0.5 + Float64(h) / Float64(num_heads))
    var lengths_device = ctx.enqueue_create_buffer[.int32](total_tokens)
    var lut_device = ctx.enqueue_create_buffer[.uint32](total_pages)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with blocks_device.map_to_host() as host:
        for i in range(kv_rows * Q_DEPTH):
            host[i] = kv_host[i]
    with q_device.map_to_host() as host:
        for i in range(total_tokens * num_heads * Q_DEPTH):
            host[i] = q_host[i]
    with indices_device.map_to_host() as host:
        for i in range(total_tokens * top_k):
            var value = indices[i]
            comptime if logical:
                if value >= 0:
                    value = Int32(Int(value) % (pages_per_seq * PAGE_SIZE))
            host[i] = value
    with lengths_device.map_to_host() as host:
        for i in range(total_tokens):
            host[i] = lengths[i]
    with lut_device.map_to_host() as host:
        for i in range(total_pages):
            host[i] = UInt32(i)
    with cache_lengths_device.map_to_host() as host:
        for i in range(batch_size):
            host[i] = UInt32(cache_lens[i])
    with row_offsets_device.map_to_host() as host:
        for i in range(batch_size + 1):
            host[i] = UInt32(row_starts[i])

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime blocks_layout_type = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout_type.shape_types]()
    blocks_shape[0] = Int64(total_pages)
    blocks_shape[2] = Int64(NUM_LAYERS)
    var blocks_strides = Coord[*blocks_layout_type.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]
    var blocks = TileTensor(
        blocks_device, blocks_layout_type(blocks_shape, blocks_strides)
    ).as_unsafe_any_origin()
    var kv_collection = Collection(
        blocks,
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(
            lut_device, row_major(Int64(batch_size), Int64(pages_per_seq))
        )
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_seq_len),
        UInt32(max_cache_len),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads, is_fp8_kv=dtype.is_float8()
    ](batch_size, max_cache_len, max_seq_len, ctx)
    var d_indices = rebind[MutPointer[Int32, MutAnyOrigin]](
        indices_device.unsafe_ptr()
    )
    var out_tt = TileTensor(
        out_device, row_major(total_tokens, Idx[num_heads], Idx[V_DEPTH])
    )
    var q_tt = TileTensor(
        q_device, row_major(total_tokens, Idx[num_heads], Idx[Q_DEPTH])
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[dtype](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        skip_masked_tiles=True,
        physical_indices=not logical,
        kernels=SM100MLA,
    ](
        out_tt,
        q_tt,
        kv_collection.get_key_cache(0),
        CausalMask(),
        row_offsets_tt,
        scale,
        ctx,
        mla_args.gpu_tile_tensor(),
        d_indices=d_indices,
        indices_stride=top_k,
        selection_lag=selection_lag,
        attn_sink_ptr=maybe_null_pointer[sink](sinks_device.unsafe_ptr()),
        topk_lengths=maybe_null_pointer[per_token_lengths](
            lengths_device.unsafe_ptr()
        ),
        num_partitions_in=Optional[Int](forced_np) if forced_np
        > 0 else Optional[Int](None),
    )

    comptime if canary_head >= 0:
        var picked = List[BFloat16](capacity=total_tokens * V_DEPTH)
        with out_device.map_to_host() as out_host:
            for g in range(total_tokens):
                for d in range(V_DEPTH):
                    picked.append(
                        out_host[(g * num_heads + canary_head) * V_DEPTH + d]
                    )
        _ = mla_args
        _ = blocks_device
        _ = q_device
        _ = indices_device
        _ = lengths_device
        _ = sinks_device
        _ = lut_device
        _ = cache_lengths_device
        _ = row_offsets_device
        return picked^

    var tokens = List[Int]()
    for bi in range(batch_size):
        for s in range(seq_lens[bi]):
            if (
                edge_tokens <= 0
                or s < edge_tokens
                or s >= seq_lens[bi] - edge_tokens
            ):
                tokens.append(row_starts[bi] + s)
    var slot_counts = List[Int](capacity=total_tokens)
    for g in range(total_tokens):
        slot_counts.append(Int(lengths[g]))
    var reference = attention_reference[dtype, num_heads, top_k, sink](
        q_host, kv_host, indices, slot_counts, tokens, scale
    )
    with out_device.map_to_host() as out_host:
        comptime if zero_lengths:
            for g in tokens:
                if lengths[g] == 0:
                    for j in range(num_heads * V_DEPTH):
                        assert_equal(
                            out_host[g * num_heads * V_DEPTH + j], BFloat16(0)
                        )
        var max_err, cosine = agreement[num_heads](
            out_host.as_span(), reference, tokens
        )
        print(name, ": max_err =", max_err, " cosine =", cosine)
        if not agrees(max_err, cosine):
            raise Error(
                String(name, ": mismatch, max_err=", max_err, " cos=", cosine)
            )
        # Canary: the slots past each length hold live keys, so a kernel that
        # read the batch's length or the stride instead would match this.
        comptime if per_token_lengths:
            var differs = False
            for g in tokens:
                differs = differs or Int(lengths[g]) != causal_counts[g]
            if differs:
                var unbounded = attention_reference[
                    dtype, num_heads, top_k, sink
                ](q_host, kv_host, indices, causal_counts, tokens, scale)
                var c_err, c_cos = agreement[num_heads](
                    out_host.as_span(), unbounded, tokens
                )
                print(
                    name,
                    ": vs causal count, max_err =",
                    c_err,
                    " cosine =",
                    c_cos,
                )
                if agrees(c_err, c_cos):
                    raise Error(
                        String(name, ": per-token lengths have no effect")
                    )

    _ = mla_args
    _ = blocks_device
    _ = q_device
    _ = indices_device
    _ = lengths_device
    _ = sinks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = row_offsets_device
    return List[BFloat16]()


def _expect_break(name: StringLiteral, failed: Bool, on_pair: Bool) raises:
    if failed != on_pair:
        raise Error(
            name,
            " under MLA_CG2_SLICE_BREAK: ",
            "failed off the pair" if failed else "passed on the broken pair",
        )


def _run_pair_reachability(ctx: DeviceContext) raises:
    """Runs the FP8 h=128 cells under `MLA_CG2_SLICE_BREAK`.

    Under the default routing the short cells (at most 384 keys per CTA)
    run the unpaired kernel and pass, the route-boundary cells pin where
    the switch happens, and the `h128_pair_*` cells run the broken pair
    and fail. With `MLA_SPARSE_CG2_ROUTE=0` every cell fails.
    """
    var short_on_pair = _MLA_SPARSE_CG2_ROUTE == 0
    var seq_lens: List[Int] = [70, 33, 130]
    var cache_lens: List[Int] = [0, 200, 40]
    var short_seq: List[Int] = [12]
    var short_cache: List[Int] = [500]
    var ragged_seq: List[Int] = [9, 0, 1]
    var long_pad_seq: List[Int] = [3, 4]
    var long_pad_cache: List[Int] = [400, 1000]
    var long_ragged_cache: List[Int] = [600, 0, 700]
    var long_split_cache: List[Int] = [1100, 0, 1200]

    var failed = False
    try:
        run_case[128, 256, False]("h128_pad", seq_lens, cache_lens, ctx)
    except e:
        print(e)
        failed = True
    _expect_break("h128_pad", failed, short_on_pair)

    failed = False
    try:
        run_case[128, 512, True](
            "h128_splitk", short_seq, short_cache, ctx, forced_np=2
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_splitk", failed, short_on_pair)

    failed = False
    try:
        run_case[128, 512, False](
            "h128_pair_pad", long_pad_seq, long_pad_cache, ctx, forced_np=1
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_pair_pad", failed, True)

    failed = False
    try:
        run_case[128, 512, True, DType.float8_e4m3fn, True](
            "h128_pair_zero_lengths",
            ragged_seq,
            long_ragged_cache,
            ctx,
            forced_np=1,
            dispatch_bound=64,
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_pair_zero_lengths", failed, True)

    failed = False
    try:
        run_case[128, 1024, True, sink=True](
            "h128_pair_split_sink",
            ragged_seq,
            long_split_cache,
            ctx,
            forced_np=2,
            dispatch_bound=64,
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_pair_split_sink", failed, True)

    # Either side of the route's 384-key walk, so a CI run of this arm
    # pins the boundary itself: 384 keys stay on the unpaired kernel and
    # pass, 385 take the pair and fail. Split, the walk rounds up to whole
    # split pages: two splits of 768 keys walk 384 each and pass, of 769
    # keys one walks 512 and fails.
    var one_seq: List[Int] = [1]
    var route_cache: List[Int] = [4096]
    failed = False
    try:
        run_case[128, 384, False](
            "h128_route_topk384", one_seq, route_cache, ctx, forced_np=1
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_route_topk384", failed, short_on_pair)

    failed = False
    try:
        run_case[128, 385, False](
            "h128_route_topk385", one_seq, route_cache, ctx, forced_np=1
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_route_topk385", failed, True)

    failed = False
    try:
        run_case[128, 768, False](
            "h128_route_np2_topk768", one_seq, route_cache, ctx, forced_np=2
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_route_np2_topk768", failed, short_on_pair)

    failed = False
    try:
        run_case[128, 769, False](
            "h128_route_np2_topk769", one_seq, route_cache, ctx, forced_np=2
        )
    except e:
        print(e)
        failed = True
    _expect_break("h128_route_np2_topk769", failed, True)
    print("pair reachability: all expectations met")


def _count_moved(a: List[BFloat16], b: List[BFloat16]) -> Int:
    """Returns how many elements of two same-shaped outputs differ."""
    var moved = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            moved += 1
    return moved


def _run_head_pad_case[
    num_heads: Int, dtype: DType = DType.float8_e4m3fn
](name: StringLiteral, ctx: DeviceContext, expect_fail: Bool) raises:
    """Runs one head-padding cell against the CPU reference.

    `num_heads` does not fill the kernel's head tile (Layout G's BM=32 above 24
    rows, Layout E's BM=64 above 40), so its padded rows must not reach the
    real heads' output. `expect_fail` turns the cell into its negative control:
    under the planted fault it must report a mismatch.
    """
    var seq_lens: List[Int] = [8]
    var cache_lens: List[Int] = [512]
    var failure = String()
    try:
        _ = _run_case_impl[num_heads, 128, True, dtype](
            name, seq_lens, cache_lens, ctx, forced_np=1
        )
    except e:
        failure = String(e)
    if expect_fail:
        if failure.byte_length() == 0:
            raise Error(String("planted fault passed ", name))
        print("  EXPECTED_FAIL:", name, failure)
    elif failure.byte_length() > 0:
        raise Error(failure)


def _run_head_canary[
    num_heads: Int,
    first_poisoned: Int,
    dtype: DType = DType.float8_e4m3fn,
](name: StringLiteral, ctx: DeviceContext, rng_seed: Int = 7) raises:
    """Checks that the heads from `first_poisoned` up cannot move head 0.

    Every run draws the same Q, K/V and indices from `rng_seed`, so the runs
    differ only in the poisoned heads' Q. The third run poisons head 0 too and
    must move it: a canary whose comparison cannot see a change in the head it
    watches would pass vacuously.
    """
    var seq_lens: List[Int] = [8]
    var cache_lens: List[Int] = [512]
    seed(rng_seed)
    var clean = _run_case_impl[num_heads, 128, True, dtype, canary_head=0](
        name, seq_lens, cache_lens, ctx, forced_np=1
    )
    seed(rng_seed)
    var poisoned = _run_case_impl[
        num_heads,
        128,
        True,
        dtype,
        canary_head=0,
        poison_heads_from=first_poisoned,
    ](name, seq_lens, cache_lens, ctx, forced_np=1)
    seed(rng_seed)
    var self_poisoned = _run_case_impl[
        num_heads, 128, True, dtype, canary_head=0, poison_heads_from=0
    ](name, seq_lens, cache_lens, ctx, forced_np=1)
    var moved = _count_moved(clean, poisoned)
    var self_moved = _count_moved(clean, self_poisoned)
    print(name, ": moved=", moved, " self_moved=", self_moved)
    if moved != 0:
        raise Error(
            String(
                name,
                ": head 0 moved when heads ",
                first_poisoned,
                "..",
                num_heads - 1,
                " were poisoned",
            )
        )
    if self_moved == 0:
        raise Error(String(name, ": poisoning head 0 moved nothing"))


def _run_head_cell_negative_control(ctx: DeviceContext) raises:
    """Planted fault for the head-padding cells.

    `MLA_CORR_BREAK=1` drops the correction's O-rescale, so a cell whose
    running max grows after its first key tile reads a stale O.
    """
    _run_head_pad_case[24]("h24_pad", ctx, expect_fail=True)
    _run_head_pad_case[40]("h40_pad", ctx, expect_fail=True)


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        comptime if CG2_SLICE_BREAK:
            _run_pair_reachability(ctx)
            return
        comptime if CORR_BREAK:
            _run_head_cell_negative_control(ctx)
            return
        var split_only = arg_parse("split_only", 0)
        if split_only > 0:
            var seq: List[Int] = [12]
            var cache: List[Int] = [500]
            run_case[8, 512, True](
                "isolated_split",
                seq,
                cache,
                ctx,
                forced_np=split_only if split_only != 1 else 0,
            )
            return

        if arg_parse("padded_only", 0) != 0:
            var seq: List[Int] = [9, 0, 1]
            var cache: List[Int] = [0, 0, 32]
            run_case[8, 256, True](
                "padded_split", seq, cache, ctx, forced_np=2, dispatch_bound=64
            )
            run_case[16, 256, True](
                "padded_unsplit",
                seq,
                cache,
                ctx,
                forced_np=1,
                dispatch_bound=64,
            )
            var empty_seq: List[Int] = [0, 0, 1]
            run_case[8, 256, True](
                "empty_requests",
                empty_seq,
                cache,
                ctx,
                forced_np=2,
                dispatch_bound=64,
            )
            return

        # Ragged prompt with a mix of fresh and cached sequences. Early
        # tokens see far fewer keys than top_k, so most of their tiles skip.
        var seq_lens: List[Int] = [70, 33, 130]
        var cache_lens: List[Int] = [0, 200, 40]
        run_case[8, 256, False]("h8_pad", seq_lens, cache_lens, ctx)
        run_case[8, 256, True]("h8_lengths", seq_lens, cache_lens, ctx)
        run_case[64, 256, False]("h64_pad", seq_lens, cache_lens, ctx)
        run_case[64, 256, True]("h64_lengths", seq_lens, cache_lens, ctx)

        # A short single sequence leaves the grid small enough to split K,
        # so skipping empties whole partitions.
        var short_seq: List[Int] = [12]
        var short_cache: List[Int] = [500]
        run_case[8, 512, True]("h8_splitk", short_seq, short_cache, ctx)

        # On an empty cache every token sees fewer keys than top_k, so the
        # split is sized from cache + q rather than the top-k stride.
        var empty_cache_short: List[Int] = [0]
        run_case[8, 512, True](
            "h8_splitk_empty_cache", short_seq, empty_cache_short, ctx
        )
        run_case[64, 256, True](
            "h64_lengths_short", short_seq, empty_cache_short, ctx
        )

        # A bf16 cache runs the KV_BF16 kernel.
        run_case[8, 256, False, DType.bfloat16](
            "h8_pad_bf16", seq_lens, cache_lens, ctx
        )
        run_case[64, 256, True, DType.bfloat16](
            "h64_lengths_bf16", seq_lens, cache_lens, ctx
        )
        run_case[8, 512, True, DType.bfloat16](
            "h8_splitk_bf16", short_seq, short_cache, ctx
        )

        var ragged_seq: List[Int] = [9, 0, 1]
        var ragged_cache: List[Int] = [0, 0, 32]
        run_case[8, 256, True](
            "h8_padded",
            ragged_seq,
            ragged_cache,
            ctx,
            forced_np=2,
            dispatch_bound=64,
        )
        run_case[16, 256, True](
            "h16_padded", ragged_seq, ragged_cache, ctx, forced_np=1
        )
        run_case[32, 256, True](
            "h32_padded", ragged_seq, ragged_cache, ctx, forced_np=2
        )
        run_case[64, 256, True](
            "h64_padded", ragged_seq, ragged_cache, ctx, forced_np=1
        )
        var empty_seq: List[Int] = [0, 0, 1]
        var empty_cache: List[Int] = [0, 0, 32]
        run_case[8, 256, True](
            "h8_empty_requests",
            empty_seq,
            empty_cache,
            ctx,
            forced_np=2,
            dispatch_bound=64,
        )
        run_case[8, 256, True, DType.float8_e4m3fn, True](
            "h8_zero_lengths", ragged_seq, ragged_cache, ctx, forced_np=1
        )
        run_case[8, 256, True, DType.float8_e4m3fn, True](
            "h8_zero_lengths_split", ragged_seq, ragged_cache, ctx, forced_np=2
        )
        var one_seq: List[Int] = [1]
        var one_cache: List[Int] = [32]
        run_case[8, 256, True, DType.float8_e4m3fn, True](
            "h8_all_zero", one_seq, one_cache, ctx, forced_np=1
        )
        run_case[8, 256, True, DType.bfloat16, True](
            "h8_bf16_zero", ragged_seq, ragged_cache, ctx, forced_np=1
        )

        var lag_cache: List[Int] = [128, 128, 128]
        run_case[8, 256, True, logical=True, sink=True, selection_lag=2](
            "h8_logical_sink_lag", ragged_seq, lag_cache, ctx, forced_np=1
        )
        run_case[16, 256, True, logical=True, sink=True, selection_lag=2](
            "h16_logical_sink_lag_split",
            ragged_seq,
            lag_cache,
            ctx,
            forced_np=2,
        )
        run_case[64, 256, True, sink=True](
            "h64_sink_split", ragged_seq, lag_cache, ctx, forced_np=2
        )

        # Two head tiles per token.
        run_case[128, 256, False]("h128_pad", seq_lens, cache_lens, ctx)
        run_case[128, 512, True](
            "h128_splitk", short_seq, short_cache, ctx, forced_np=2
        )
        # The cells above walk at most 384 keys per CTA, so the default
        # routing sends them to the unpaired kernel. These walk more and keep
        # the FP8 CTA pair: -1 padding past a causal count of 401 to 403,
        # zero and shortened lengths over a padded ragged batch, and a split
        # of 512 keys per CTA with a sink.
        var long_pad_seq: List[Int] = [3, 4]
        var long_pad_cache: List[Int] = [400, 1000]
        run_case[128, 512, False](
            "h128_pair_pad", long_pad_seq, long_pad_cache, ctx, forced_np=1
        )
        var long_ragged_cache: List[Int] = [600, 0, 700]
        run_case[128, 512, True, DType.float8_e4m3fn, True](
            "h128_pair_zero_lengths",
            ragged_seq,
            long_ragged_cache,
            ctx,
            forced_np=1,
            dispatch_bound=64,
        )
        var long_split_cache: List[Int] = [1100, 0, 1200]
        run_case[128, 1024, True, sink=True](
            "h128_pair_split_sink",
            ragged_seq,
            long_split_cache,
            ctx,
            forced_np=2,
            dispatch_bound=64,
        )
        # The BF16 h=128 form (the CTA pair under `MLA_SPARSE_CG2_BF16=1`).
        run_case[128, 256, False, DType.bfloat16](
            "h128_pad_bf16", seq_lens, cache_lens, ctx
        )
        run_case[128, 512, True, DType.bfloat16](
            "h128_splitk_bf16", short_seq, short_cache, ctx, forced_np=2
        )
        run_case[128, 256, True, DType.bfloat16, sink=True](
            "h128_sink_split_bf16", ragged_seq, lag_cache, ctx, forced_np=2
        )

        # Tokens past 65535 are the ones a grid with a token axis in y could
        # not address. Both split-K partitions hold keys for every token, so
        # the combine reads them too.
        var long_seq: List[Int] = [65536 + 64, 5]
        var long_cache: List[Int] = [300, 1000]
        run_case[8, 256, True](
            "h8_long", long_seq, long_cache, ctx, edge_tokens=64, forced_np=2
        )

        # Head padding: head counts that leave rows of the kernel's head tile
        # (Layout G's BM=32, Layout E's BM=64) with no real head behind them.
        _run_head_pad_case[24]("h24_pad", ctx, expect_fail=False)
        _run_head_pad_case[40]("h40_pad", ctx, expect_fail=False)

        # Garbage-head canary: NaN in every head above a tile boundary must
        # not move head 0, across the head tile (24, 40) and across the two
        # head-group CTAs of an h=128 prompt.
        _run_head_canary[24, 8]("h24_head_canary", ctx)
        _run_head_canary[40, 16]("h40_head_canary", ctx)
        _run_head_canary[128, 64]("h128_head_canary", ctx)
