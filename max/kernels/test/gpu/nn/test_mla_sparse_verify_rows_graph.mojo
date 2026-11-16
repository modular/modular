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

"""A speculative verify's rows match one-token decode steps on the same inputs.

A target verify runs the sparse MLA graph op once over `Q_LEN` query tokens per
request. Greedy speculative decoding is lossless only if row `s` of that call
equals what a plain decode step at cache length `cache_len + s` computes. This
test drives `mla_prefill_decode_graph_bf16` (sparse, physical indices, causal)
both ways on identical inputs:

* verify: one call with `Q_LEN` tokens per request at `cache_len`,
* steps: `Q_LEN` calls with one token per request at `cache_len + s`, the
  paged FP8 cache carried from one call to the next.

Both arms write the new latent rows through the fused RoPE + RMSNorm, so the
comparison covers each row's rotary position and cache slot as well as its
attention. The pages are scattered and each token's index row is a random
permutation of the positions it can see (a random top-2048 past that), with
`-1` padding, as the DSA indexer produces.

Each case runs twice:

* production: the verify leaves its split-K count to the decode dispatcher,
  and the steps are pinned to the count the registered graph op forwards for
  every decode launch when its `split_rows_per_request` is `Q_LEN` (16 at
  batch 1-2, 9 at batch 3 on B200). Sparse decode sizes the split from the
  top-k stride, so the count does not depend on the context. A second verify
  pinned to that count must match the dispatcher's bit for bit, which checks
  that the op's count is the one the verify picks on its own. Every launch
  here passes the op's `row_invariant_verify` too, since the op carries its
  `split_rows_per_request` on all of them: the dispatcher only sizes a
  verify at the pin's count for a caller that asks for it, and
  `_check_unopted_split` pins the other side of that gate down -- without the
  option the fitted record sizes the same verify from the live cache, at a
  different count.
* no split: every call is pinned to one partition.

With the reduction order shared, every row must match its step bit for bit;
left to pick their own counts the two arms differ by about 1% at a 1000-key
context, which would hide a row that is one key off. A negative control swaps
one key in the last row's selection and must change the output.

What the test cannot see:

* With the default `SparseRowLayout` the per-row slot limit is `Int.MAX`, so
  the row-length bound the k-pool layouts rely on is not exercised; those
  layouts are out of scope.
* Both arms run the same kernels, so a bug that moves a verify row and its
  step alike passes by construction. The test pins row-dependence, not
  accuracy; `test_mla_sparse_nope512_graph.mojo` checks against a reference.
* Every request carries exactly `Q_LEN` tokens; ragged verify widths are not
  covered.
"""

from std.math import ceildiv, sqrt
from std.random import randn, random_ui64, seed

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.gpu.mla_graph import (
    SM100MLAGraph,
    mla_graph_takes_decode_branch,
    mla_prefill_decode_graph_bf16,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_key_runtime,
    sparse_decode_partitions_for_rows,
)
from std.utils import IndexList
from std.utils.numerics import isnan

comptime LATENT = 512
comptime ROPE = 64
comptime V_HEAD = 128
comptime PAGE_SIZE = 128
# Keys per split-K page in the decode dispatcher; a top-k of 2048 is above the
# 512-key threshold where it would halve.
comptime SPLIT_PAGE = 128
comptime TOPK = 2048
comptime Q_LEN = 4
comptime FP8 = DType.float8_e4m3fn
comptime EPSILON = Float32(1e-6)


@fieldwise_init
struct Diff(ImplicitlyCopyable, Writable):
    var max_abs: Float64
    var max_ref: Float64
    var bit_diffs: Int
    var nonfinite: Int
    var count: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "max_abs=",
            self.max_abs,
            " max_rel=",
            self.max_abs / max(self.max_ref, 1e-300),
            " bitdiff=",
            self.bit_diffs,
            "/",
            self.count,
            " nonfinite=",
            self.nonfinite,
        )


def _diff(
    a: HostBuffer[DType.bfloat16], b: HostBuffer[DType.bfloat16], n: Int
) -> Diff:
    var max_abs = Float64(0)
    var max_ref = Float64(0)
    var bit_diffs = 0
    var nonfinite = 0
    for i in range(n):
        var x = a[i].cast[DType.float64]()
        var y = b[i].cast[DType.float64]()
        if isnan(x) or isnan(y) or abs(x) > 1e30 or abs(y) > 1e30:
            nonfinite += 1
            continue
        if a[i].to_bits() != b[i].to_bits():
            bit_diffs += 1
        max_abs = max(max_abs, abs(x - y))
        max_ref = max(max_ref, abs(y))
    return Diff(max_abs, max_ref, bit_diffs, nonfinite, n)


struct Problem[num_heads: Int, row_depth: Int, qk_nope: Int]:
    """Host inputs shared by both arms."""

    comptime q_head_dim = Self.qk_nope + (Self.row_depth - LATENT)

    var batch_size: Int
    var cache_len: Int
    var pages_per_batch: Int
    var lut: List[Int]
    var cache_rows: HostBuffer[FP8]
    var kv_raw: HostBuffer[DType.bfloat16]
    var q: HostBuffer[DType.bfloat16]
    var indices: HostBuffer[DType.int32]
    var gamma: HostBuffer[DType.bfloat16]
    var w_uk: HostBuffer[DType.bfloat16]
    var w_uv: HostBuffer[DType.bfloat16]
    var freqs: HostBuffer[DType.bfloat16]

    def __init__(
        out self, ctx: DeviceContext, batch_size: Int, cache_len: Int
    ) raises:
        self.batch_size = batch_size
        self.cache_len = cache_len
        self.pages_per_batch = ceildiv(cache_len + Q_LEN, PAGE_SIZE)
        var total_pages = batch_size * self.pages_per_batch
        var rows = total_pages * PAGE_SIZE
        var num_tokens = batch_size * Q_LEN
        var max_pos = cache_len + Q_LEN

        # Scattered pages: logical page p of request b lives at a permuted
        # physical page, so a row's physical index is not its position.
        self.lut = List[Int]()
        for p in range(total_pages):
            self.lut.append(p)
        for i in range(total_pages):
            var j = i + Int(random_ui64(0, UInt64(total_pages - i - 1)))
            var t = self.lut[i]
            self.lut[i] = self.lut[j]
            self.lut[j] = t

        var src = ctx.enqueue_create_host_buffer[DType.bfloat16](
            rows * Self.row_depth
        )
        self.cache_rows = ctx.enqueue_create_host_buffer[FP8](
            rows * Self.row_depth
        )
        self.kv_raw = ctx.enqueue_create_host_buffer[DType.bfloat16](
            num_tokens * Self.row_depth
        )
        self.q = ctx.enqueue_create_host_buffer[DType.bfloat16](
            num_tokens * Self.num_heads * Self.q_head_dim
        )
        self.indices = ctx.enqueue_create_host_buffer[DType.int32](
            num_tokens * TOPK
        )
        self.gamma = ctx.enqueue_create_host_buffer[DType.bfloat16](LATENT)
        self.w_uk = ctx.enqueue_create_host_buffer[DType.bfloat16](
            Self.num_heads * LATENT * Self.qk_nope
        )
        self.w_uv = ctx.enqueue_create_host_buffer[DType.bfloat16](
            Self.num_heads * V_HEAD * LATENT
        )
        self.freqs = ctx.enqueue_create_host_buffer[DType.bfloat16](
            max_pos * ROPE
        )
        ctx.synchronize()

        randn(src.as_span(), mean=0.0, standard_deviation=1.0)
        randn(self.kv_raw.as_span(), mean=0.0, standard_deviation=2.0)
        randn(self.q.as_span(), mean=0.0, standard_deviation=1.0)
        randn(self.gamma.as_span(), mean=1.0, standard_deviation=0.25)
        # Sharp logits: a key the row should not see moves the output.
        randn(
            self.w_uk.as_span(),
            mean=0.0,
            standard_deviation=1.5 / sqrt(Float64(Self.qk_nope)),
        )
        randn(
            self.w_uv.as_span(),
            mean=0.0,
            standard_deviation=1.0 / sqrt(Float64(LATENT)),
        )
        randn(self.freqs.as_span(), mean=0.0, standard_deviation=1.0)

        for b in range(batch_size):
            for pos in range(self.pages_per_batch * PAGE_SIZE):
                var r = self.physical(b, pos)
                for d in range(Self.row_depth):
                    self.cache_rows[r * Self.row_depth + d] = src[
                        r * Self.row_depth + d
                    ].cast[FP8]() if pos < cache_len else Scalar[FP8](0)

        var pool = List[Int]()
        for b in range(batch_size):
            for s in range(Q_LEN):
                var g = b * Q_LEN + s
                var visible = cache_len + s + 1
                pool.clear()
                for p in range(visible):
                    pool.append(p)
                var keep = min(visible, TOPK)
                for i in range(keep):
                    var j = i + Int(random_ui64(0, UInt64(visible - i - 1)))
                    var t = pool[i]
                    pool[i] = pool[j]
                    pool[j] = t
                    self.indices[g * TOPK + i] = Int32(
                        self.physical(b, pool[i])
                    )
                for i in range(keep, TOPK):
                    self.indices[g * TOPK + i] = Int32(-1)

    def physical(self, b: Int, pos: Int) -> Int:
        return (
            self.lut[b * self.pages_per_batch + pos // PAGE_SIZE] * PAGE_SIZE
            + pos % PAGE_SIZE
        )


def _launch[
    num_heads: Int, row_depth: Int, qk_nope: Int
](
    ctx: DeviceContext,
    p: Problem[num_heads, row_depth, qk_nope],
    blocks_d: DeviceBuffer[FP8],
    cache_len: Int,
    q_len: Int,
    tokens: List[Int],
    indices: HostBuffer[DType.int32],
    out_host: HostBuffer[DType.bfloat16],
    num_partitions: Optional[Int],
    row_invariant_verify: Bool,
) raises:
    """Runs the graph op over `tokens`, `q_len` per request, at `cache_len`.

    `tokens` lists, request by request, which of the problem's tokens this
    call carries; their outputs land at the same token slots of `out_host`.
    `num_partitions` pins the split-K count; `None` leaves it to the decode
    dispatcher, which is what the registered sparse graph op does.
    `row_invariant_verify` is the graph op's `split_rows_per_request > 0`: a
    launch left to the dispatcher then takes the pin's shape-pure count.
    """
    comptime q_head_dim = Problem[num_heads, row_depth, qk_nope].q_head_dim
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=row_depth, is_mla=True
    )
    var batch_size = p.batch_size
    var n = len(tokens)
    var total_pages = batch_size * p.pages_per_batch
    var max_pos = p.cache_len + Q_LEN

    var q_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        n * num_heads * q_head_dim
    )
    var kv_h = ctx.enqueue_create_host_buffer[DType.bfloat16](n * row_depth)
    var idx_h = ctx.enqueue_create_host_buffer[DType.int32](n * TOPK)
    var lut_h = ctx.enqueue_create_host_buffer[DType.uint32](total_pages)
    var cl_h = ctx.enqueue_create_host_buffer[DType.uint32](batch_size)
    var ro_h = ctx.enqueue_create_host_buffer[DType.uint32](batch_size + 1)
    ctx.synchronize()
    for i in range(n):
        var g = tokens[i]
        for j in range(num_heads * q_head_dim):
            q_h[i * num_heads * q_head_dim + j] = p.q[
                g * num_heads * q_head_dim + j
            ]
        for d in range(row_depth):
            kv_h[i * row_depth + d] = p.kv_raw[g * row_depth + d]
        for k in range(TOPK):
            idx_h[i * TOPK + k] = indices[g * TOPK + k]
    for i in range(total_pages):
        lut_h[i] = UInt32(p.lut[i])
    for b in range(batch_size):
        cl_h[b] = UInt32(cache_len)
    for b in range(batch_size + 1):
        ro_h[b] = UInt32(b * q_len)

    var q_d = ctx.enqueue_create_buffer[DType.bfloat16](
        n * num_heads * q_head_dim
    )
    var kv_d = ctx.enqueue_create_buffer[DType.bfloat16](n * row_depth)
    var idx_d = ctx.enqueue_create_buffer[DType.int32](n * TOPK)
    var lut_d = ctx.enqueue_create_buffer[DType.uint32](total_pages)
    var cl_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var ro_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var gamma_d = ctx.enqueue_create_buffer[DType.bfloat16](LATENT)
    var w_uk_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * LATENT * qk_nope
    )
    var w_uv_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * V_HEAD * LATENT
    )
    var freqs_d = ctx.enqueue_create_buffer[DType.bfloat16](max_pos * ROPE)
    var out_d = ctx.enqueue_create_buffer[DType.bfloat16](
        n * num_heads * V_HEAD
    )
    # The dense-prefill operands; the sparse op never reads them.
    var w_k_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * qk_nope * LATENT
    )
    var buf_ro_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var cache_off_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_memset(w_k_d, 0)
    ctx.enqueue_memset(buf_ro_d, 0)
    ctx.enqueue_memset(cache_off_d, 0)
    ctx.enqueue_memset(out_d, 0)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(kv_d, kv_h)
    ctx.enqueue_copy(idx_d, idx_h)
    ctx.enqueue_copy(lut_d, lut_h)
    ctx.enqueue_copy(cl_d, cl_h)
    ctx.enqueue_copy(ro_d, ro_h)
    ctx.enqueue_copy(gamma_d, p.gamma)
    ctx.enqueue_copy(w_uk_d, p.w_uk)
    ctx.enqueue_copy(w_uv_d, p.w_uv)
    ctx.enqueue_copy(freqs_d, p.freqs)

    comptime Collection = PagedKVCacheCollection[
        FP8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                blocks_d,
                row_major(
                    Int64(total_pages),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE_SIZE],
                    Idx[kv_params.num_heads],
                    Idx[kv_params.head_size],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cl_d, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(
            lut_d, row_major(Int64(batch_size), Int64(p.pages_per_batch))
        )
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(cache_len),
    )

    var out_tt = TileTensor(out_d, row_major((n, Idx[num_heads], Idx[V_HEAD])))
    var q_tt = TileTensor(q_d, row_major((n, Idx[num_heads], Idx[q_head_dim])))
    var ro_tt = TileTensor(ro_d, row_major(batch_size + 1))
    var freqs_tt = TileTensor(freqs_d, row_major((max_pos, Idx[ROPE])))
    var gamma_tt = TileTensor(gamma_d, row_major(Idx[LATENT]))
    var buf_ro_tt = TileTensor(buf_ro_d, row_major(batch_size + 1)).as_imm()
    var cache_off_tt = TileTensor(cache_off_d, row_major(batch_size + 1))
    var w_k_tt = TileTensor(
        w_k_d, row_major((Idx[num_heads * qk_nope], Idx[LATENT]))
    )
    var w_uk_tt = TileTensor(
        w_uk_d, row_major((Idx[num_heads], Idx[LATENT], Idx[qk_nope]))
    )
    var w_uv_tt = TileTensor(
        w_uv_d, row_major((Idx[num_heads], Idx[V_HEAD], Idx[LATENT]))
    )
    var kv_in_tt = TileTensor(kv_d.unsafe_ptr(), row_major((n, Idx[row_depth])))
    var scalar_args = MLADispatchScalarArgs[
        num_heads=num_heads, is_fp8_kv=True
    ](batch_size, cache_len, q_len, ctx)
    var scale = Float32(1.0 / sqrt(Float64(q_head_dim)))
    var idx_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](idx_d.unsafe_ptr())

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[DType.bfloat16, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    mla_prefill_decode_graph_bf16[
        mask_str="causal",
        sparse_mla=True,
        sparse_indices_stride=TOPK,
        physical_indices=True,
        kernels=SM100MLAGraph,
    ](
        out_tt,
        q_tt,
        ro_tt,
        freqs_tt,
        gamma_tt,
        kv_collection,
        UInt32(0),
        scale,
        EPSILON,
        buf_ro_tt,
        cache_off_tt,
        0,
        q_len,
        w_k_tt,
        w_uk_tt,
        w_uv_tt,
        scalar_args.gpu_tile_tensor(),
        kv_input_fn,
        ctx,
        d_indices=idx_ptr,
        indices_stride=TOPK,
        num_partitions_in=num_partitions,
        row_invariant_verify=row_invariant_verify,
    )
    var call_out = ctx.enqueue_create_host_buffer[DType.bfloat16](
        n * num_heads * V_HEAD
    )
    ctx.enqueue_copy(call_out, out_d)
    ctx.synchronize()
    for i in range(n):
        var g = tokens[i]
        for j in range(num_heads * V_HEAD):
            out_host[g * num_heads * V_HEAD + j] = call_out[
                i * num_heads * V_HEAD + j
            ]

    _ = scalar_args
    _ = q_d
    _ = kv_d
    _ = idx_d
    _ = lut_d
    _ = cl_d
    _ = ro_d
    _ = gamma_d
    _ = w_uk_d
    _ = w_uv_d
    _ = freqs_d
    _ = out_d
    _ = w_k_d
    _ = buf_ro_d
    _ = cache_off_d


def _production_partitions[
    num_heads: Int
](ctx: DeviceContext, batch_size: Int) -> Int:
    """Returns the split-K count the graph op pins every decode launch to.

    This is the count `mo.mla.graph.prefill.decode.paged.sparse` forwards when
    its `split_rows_per_request` is `Q_LEN`. The pinned-verify check in
    `_run_split` proves it is also the count the dispatcher picks for the
    verify on its own.
    """
    comptime sm_count = ctx.default_device_info.sm_count
    return sparse_decode_partitions_for_rows[num_heads, Q_LEN, sm_count // 2](
        batch_size, TOPK, sm_count
    )


# The cache length of the cross-page cases: short enough that the fitted
# record sizes the verify from the live cache instead of the top-k stride.
comptime SHORT_CACHE = 126


def _check_unopted_split[num_heads: Int](ctx: DeviceContext) raises:
    """Checks the pin's reroute is the caller's to ask for.

    A caller that does not ask keeps main's routing: at a short cache the
    Layout-G record sizes an unfolded sparse FP8 verify from
    `min(topk, cache + q)`, which is far below the count the pin holds. That
    is the routing every other model's verify of this form gets, so the
    option has to travel with ML4's graph op rather than fire for the shape.
    """
    comptime sm_count = ctx.default_device_info.sm_count
    var pin = _production_partitions[num_heads](ctx, 1)
    var key = compute_mla_dispatch_key_runtime(
        1,
        SHORT_CACHE,
        Q_LEN,
        num_heads,
        True,
        TOPK,
        sm_count,
        page_size=PAGE_SIZE,
        fixed_topk=True,
        physical_indices=True,
    )
    var opted_key = compute_mla_dispatch_key_runtime(
        1,
        SHORT_CACHE,
        Q_LEN,
        num_heads,
        True,
        TOPK,
        sm_count,
        page_size=PAGE_SIZE,
        fixed_topk=True,
        physical_indices=True,
        row_invariant_verify=True,
    )
    print(
        t"== un-opted verify heads={num_heads} cache_len={SHORT_CACHE}"
        t" pin={pin} main={key[2]} opted={opted_key[2]}"
    )
    if opted_key[2] != pin:
        raise Error(
            t"an opted-in verify keys {opted_key[2]} splits, not the pin's"
            t" {pin}"
        )
    if key[2] == pin:
        raise Error(
            t"the un-opted verify keys the pin's {pin} splits, so the gate"
            t" this check exists for cannot fail"
        )


def _row_tokens(batch_size: Int, s: Int) -> List[Int]:
    var tokens = List[Int]()
    for b in range(batch_size):
        tokens.append(b * Q_LEN + s)
    return tokens^


def _rows_of(
    ctx: DeviceContext,
    outputs: HostBuffer[DType.bfloat16],
    batch_size: Int,
    s: Int,
    row_elems: Int,
) raises -> HostBuffer[DType.bfloat16]:
    var rows = ctx.enqueue_create_host_buffer[DType.bfloat16](
        batch_size * row_elems
    )
    ctx.synchronize()
    for r in range(batch_size):
        var g = r * Q_LEN + s
        for j in range(row_elems):
            rows[r * row_elems + j] = outputs[g * row_elems + j]
    return rows^


def _run_split[
    num_heads: Int, row_depth: Int, qk_nope: Int
](
    ctx: DeviceContext,
    name: StaticString,
    p: Problem[num_heads, row_depth, qk_nope],
    production: Bool,
) raises:
    """Runs the verify and the steps at one split-K configuration."""
    var batch_size = p.batch_size
    var cache_len = p.cache_len
    var num_tokens = batch_size * Q_LEN
    var rows = batch_size * p.pages_per_batch * PAGE_SIZE
    var out_elems = num_tokens * num_heads * V_HEAD
    var row_elems = num_heads * V_HEAD
    var split = _production_partitions[num_heads](
        ctx, batch_size
    ) if production else 1
    var label = String(name) + ("/production" if production else "/no_split")

    var verify_cache = ctx.enqueue_create_buffer[FP8](rows * row_depth)
    var pinned_cache = ctx.enqueue_create_buffer[FP8](rows * row_depth)
    var steps_cache = ctx.enqueue_create_buffer[FP8](rows * row_depth)
    ctx.enqueue_copy(verify_cache, p.cache_rows)
    ctx.enqueue_copy(pinned_cache, p.cache_rows)
    ctx.enqueue_copy(steps_cache, p.cache_rows)
    var out_verify = ctx.enqueue_create_host_buffer[DType.bfloat16](out_elems)
    var out_pinned = ctx.enqueue_create_host_buffer[DType.bfloat16](out_elems)
    var out_steps = ctx.enqueue_create_host_buffer[DType.bfloat16](out_elems)
    var out_ctl = ctx.enqueue_create_host_buffer[DType.bfloat16](out_elems)
    ctx.synchronize()

    var all_tokens = List[Int]()
    for g in range(num_tokens):
        all_tokens.append(g)
    var verify_split: Optional[Int] = None
    if not production:
        verify_split = split
    _launch[num_heads, row_depth, qk_nope](
        ctx,
        p,
        verify_cache,
        cache_len,
        Q_LEN,
        all_tokens,
        p.indices,
        out_verify,
        verify_split,
        production,
    )
    _launch[num_heads, row_depth, qk_nope](
        ctx,
        p,
        pinned_cache,
        cache_len,
        Q_LEN,
        all_tokens,
        p.indices,
        out_pinned,
        Optional(split),
        production,
    )
    for s in range(Q_LEN):
        _launch[num_heads, row_depth, qk_nope](
            ctx,
            p,
            steps_cache,
            cache_len + s,
            1,
            _row_tokens(batch_size, s),
            p.indices,
            out_steps,
            Optional(split),
            production,
        )

    # The pinned verify must reproduce the dispatcher's, or the steps are
    # pinned to a count the production verify did not run.
    var dp = _diff(out_pinned, out_verify, out_elems)
    print(t"  {label}: split_k={split} pinned vs dispatcher verify {dp}")

    # Both arms must have written the same cache rows.
    var cache_v = ctx.enqueue_create_host_buffer[FP8](rows * row_depth)
    var cache_s = ctx.enqueue_create_host_buffer[FP8](rows * row_depth)
    ctx.enqueue_copy(cache_v, verify_cache)
    ctx.enqueue_copy(cache_s, steps_cache)
    ctx.synchronize()
    var cache_diffs = 0
    for i in range(rows * row_depth):
        if cache_v[i].to_bits() != cache_s[i].to_bits():
            cache_diffs += 1
    print(t"  {label}: cache verify vs steps differing bytes={cache_diffs}")

    var worst = Float64(0)
    var row_bit_diffs = 0
    for s in range(Q_LEN):
        var d = _diff(
            _rows_of(ctx, out_verify, batch_size, s, row_elems),
            _rows_of(ctx, out_steps, batch_size, s, row_elems),
            batch_size * row_elems,
        )
        print(t"  {label}: row {s} verify vs step {d}")
        worst = max(worst, d.max_abs / max(d.max_ref, 1e-300))
        row_bit_diffs += d.bit_diffs
        if d.nonfinite != 0:
            raise Error(t"{label}: row {s} is not finite")

    # Negative control: the last row's step with one selected key swapped for
    # another visible key. The gate must reject a row that is one key off.
    var swapped = ctx.enqueue_create_host_buffer[DType.int32](num_tokens * TOPK)
    ctx.synchronize()
    for i in range(num_tokens * TOPK):
        swapped[i] = p.indices[i]
    for r in range(batch_size):
        var g = r * Q_LEN + Q_LEN - 1
        var own = Int32(p.physical(r, cache_len + Q_LEN - 1))
        var at = -1
        for k in range(TOPK):
            if swapped[g * TOPK + k] == own:
                at = k
        if at < 0:
            swapped[g * TOPK] = own
        else:
            swapped[g * TOPK + at] = swapped[g * TOPK + (1 if at == 0 else 0)]
    _launch[num_heads, row_depth, qk_nope](
        ctx,
        p,
        steps_cache,
        cache_len + Q_LEN - 1,
        1,
        _row_tokens(batch_size, Q_LEN - 1),
        swapped,
        out_ctl,
        Optional(split),
        production,
    )
    var dc = _diff(
        _rows_of(ctx, out_ctl, batch_size, Q_LEN - 1, row_elems),
        _rows_of(ctx, out_steps, batch_size, Q_LEN - 1, row_elems),
        batch_size * row_elems,
    )
    var ctl_rel = dc.max_abs / max(dc.max_ref, 1e-300)
    print(t"  {label}: control (one key swapped) {dc}")
    print(
        t"RESULT {label} row={row_depth} batch={batch_size}"
        t" cache_len={cache_len} split_k={split}"
        t" pinned_bitdiffs={dp.bit_diffs} worst_row_rel={worst}"
        t" row_bitdiffs={row_bit_diffs} control_rel={ctl_rel}"
        t" cache_diffs={cache_diffs}"
    )
    # Row identity first: it is what DSpark's acceptance needs, and a row can
    # diverge from its step even when both arms run the pinned count (the pin
    # fixes the split count, not the split boundaries, which still follow the
    # launch's own effective length). The dispatcher comparison below is a
    # second, weaker property: it says the pin is also the count an unpinned
    # verify picks.
    if cache_diffs != 0:
        raise Error(t"{label}: verify and steps wrote different cache rows")
    if row_bit_diffs != 0:
        raise Error(
            t"{label}: {row_bit_diffs} verify outputs differ from their decode"
            t" steps (worst relative {worst}; one swapped key moves {ctl_rel})"
        )
    if dc.bit_diffs == 0:
        raise Error(
            t"{label}: the control's swapped key left the row unchanged"
        )
    if dp.bit_diffs != 0:
        raise Error(
            t"{label}: the verify pinned to split_k={split} differs from the"
            t" dispatcher's ({dp})"
        )


def run_case[
    num_heads: Int, row_depth: Int, qk_nope: Int
](
    ctx: DeviceContext,
    name: StaticString,
    batch_size: Int,
    cache_len: Int,
) raises:
    print(
        t"== {name} heads={num_heads} row={row_depth} qk_nope={qk_nope}"
        t" batch={batch_size} cache_len={cache_len} q_len={Q_LEN}"
    )
    seed(batch_size * 1_000_003 + cache_len * 1009 + row_depth * 31 + num_heads)
    var p = Problem[num_heads, row_depth, qk_nope](ctx, batch_size, cache_len)
    _run_split(ctx, name, p, production=True)
    _run_split(ctx, name, p, production=False)


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # A multi-row verify is only this test's subject if the graph op
            # sends it to the decode arm, as a model's verify is.
            if not mla_graph_takes_decode_branch[FP8, 8, True](Q_LEN):
                raise Error(
                    t"a {Q_LEN}-row sparse verify takes the prefill arm"
                )
            # 8 heads is 64-head MLA under TP8; 576 rows carry the rope tail,
            # 512 rows are the NoPE layers'. Batch 3 runs the dispatcher's
            # 9-way split, batches 1-2 its 16-way one.
            #
            # A step at `cache_len + s` splits min(TOPK, cache_len + s + 1)
            # keys into SPLIT_PAGE-key pages, the verify min(TOPK, cache_len +
            # Q_LEN). Unless each split holds at most one page, bit-exactness
            # needs both to cover the same pages, so only the solo cases let
            # [cache_len + 1, cache_len + Q_LEN] cross a SPLIT_PAGE boundary
            # under TOPK; past TOPK every call reads TOPK.
            _check_unopted_split[8](ctx)
            run_case[8, 576, 64](ctx, "rope_short", 1, 21)
            run_case[8, 576, 64](ctx, "rope_short_b3", 3, 37)
            run_case[8, 576, 64](ctx, "rope_mid", 2, 1000)
            run_case[8, 576, 64](ctx, "rope_subset", 2, 3361)
            # A solo request's 16-way split holds at most one page, so its
            # rows match their steps even where the steps cross a page.
            run_case[8, 576, 64](ctx, "rope_solo_cross_page", 1, 126)
            run_case[8, 512, 128](ctx, "nope_short", 1, 21)
            run_case[8, 512, 128](ctx, "nope_short_b3", 3, 37)
            run_case[8, 512, 128](ctx, "nope_mid", 2, 1000)
            run_case[8, 512, 128](ctx, "nope_subset", 2, 3361)
            run_case[8, 512, 128](ctx, "nope_solo_cross_page", 1, 126)
        else:
            print("Skipping: needs an SM100 GPU")
