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

"""Sparse MLA graph op over a native 512-wide NoPE latent row.

A NoPE MLA layer stores a 512-wide FP8 latent with no rotary tail, and its
query carries 128 content dims per head that `w_uk` absorbs into the latent.
The kernel tests pin the attention kernels at depth 512; this test drives the
graph op a model calls (`mla_prefill_decode_graph_bf16`, sparse, physical
indices, causal) end to end: the `w_uk` absorption into an FP8 Q staging
buffer, the fused RMSNorm that writes the new latent rows into the paged FP8
cache, the sparse decode or prefill kernel, and the `w_uv` up-projection.

Every case runs twice on identical inputs. The native arm keeps the 512-wide
cache row and a 128-wide query; the padded arm stores a 576-wide row and a
192-wide query whose 64-wide tails are zero. Both use the explicit softmax
scale 1/sqrt(128). Each case checks:

* native against padded on every output element (bit-exactness is reported,
  the gate is the aggregate tolerance below),
* the cache rows both arms wrote: identical over [0, 512), zero tail at 576,
* both arms against a host reference for a sample of query tokens. The
  reference rounds where the op stores (FP8 Q latent, FP8 cache row, BF16
  attention output) and otherwise computes in float64,
* a negative control: the same reference at the padded geometry's default
  scale 1/sqrt(192) must fail the gate, so the gate can see a scale error.

Native against padded uses the tolerance `test_mla_decode_sparse_qkv_fp8.mojo`
applies to this kernel family: cosine >= 0.99, mean |err| <= 0.01, fewer than
0.15% of elements with |err| > 1e-2, max |err| <= 5e-2. Those bounds are
absolute, so the host-reference comparison gates relative error instead:
cosine >= 0.999, ||err|| / ||ref|| <= 3%, max |err| / max |ref| <= 5%.

The decode cases include the 4-row speculative-verify shape. Prefill gathers
top-2048 rows per token; a short prompt keeps every visible row and ends each
index row with -1.
"""

from std.math import ceildiv, exp, sqrt
from std.random import randn, random_ui64, seed

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.gpu.mla_graph import (
    SM100MLAGraph,
    mla_prefill_decode_graph_bf16,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils import IndexList
from std.utils.numerics import isnan

comptime LATENT = 512
comptime ROPE = 64
comptime PADDED = LATENT + ROPE
# Query content dims per head: a NoPE layer folds the rotary slice into them.
comptime QK_CONTENT = 128
comptime V_HEAD = 128
comptime PAGE_SIZE = 128
comptime TOPK = 2048
comptime FP8 = DType.float8_e4m3fn
comptime EPSILON = Float32(1e-5)

comptime COS_MIN = 0.99
comptime MEAN_ERR_MAX = 0.01
comptime TAIL_MAX = 0.0015
comptime ATOL = 5e-2
# The absolute bounds above were set on outputs averaged over many keys. A
# token that sees one or two keys returns roughly one latent row, an order of
# magnitude larger, so the host-reference comparison gates relative error.
comptime REF_COS_MIN = 0.999
comptime REF_L2_REL_MAX = 0.03
comptime REF_MAX_REL_MAX = 0.05


@fieldwise_init
struct Metrics(ImplicitlyCopyable, Writable):
    var cosine: Float64
    var max_abs: Float64
    var mean_abs: Float64
    var tail: Float64
    var max_rel: Float64
    var l2_rel: Float64
    var nonfinite: Int
    var nonzero: Int
    var count: Int

    def passes(self) -> Bool:
        return (
            self.nonfinite == 0
            and self.nonzero > 0
            and self.cosine >= COS_MIN
            and self.mean_abs <= MEAN_ERR_MAX
            and self.tail < TAIL_MAX
            and self.max_abs <= ATOL
        )

    def passes_relative(self) -> Bool:
        return (
            self.nonfinite == 0
            and self.nonzero > 0
            and self.cosine >= REF_COS_MIN
            and self.l2_rel <= REF_L2_REL_MAX
            and self.max_rel <= REF_MAX_REL_MAX
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "cos=",
            self.cosine,
            " max_abs=",
            self.max_abs,
            " mean_abs=",
            self.mean_abs,
            " tail=",
            self.tail,
            " max_rel=",
            self.max_rel,
            " l2_rel=",
            self.l2_rel,
            " nonfinite=",
            self.nonfinite,
            " n=",
            self.count,
        )


def _metrics(expected: List[Float64], got: List[Float64]) -> Metrics:
    var max_abs = Float64(0)
    var max_ref = Float64(0)
    var sum_abs = Float64(0)
    var dot = Float64(0)
    var norm_e = Float64(0)
    var norm_g = Float64(0)
    var norm_err = Float64(0)
    var tail = 0
    var nonfinite = 0
    var nonzero = 0
    for i in range(len(expected)):
        var e = expected[i]
        var g = got[i]
        if isnan(g) or isnan(e) or abs(g) > 1e300 or abs(e) > 1e300:
            nonfinite += 1
            continue
        if abs(g) > 1e-6:
            nonzero += 1
        var err = abs(g - e)
        max_abs = max(max_abs, err)
        max_ref = max(max_ref, abs(e))
        sum_abs += err
        if err > 1e-2:
            tail += 1
        dot += e * g
        norm_e += e * e
        norm_g += g * g
        norm_err += err * err
    var n = max(len(expected) - nonfinite, 1)
    var cosine = Float64(1)
    if norm_e > 1e-24 or norm_g > 1e-24:
        cosine = dot / max(sqrt(norm_e) * sqrt(norm_g), 1e-300)
    return Metrics(
        cosine=cosine,
        max_abs=max_abs,
        mean_abs=sum_abs / Float64(n),
        tail=Float64(tail) / Float64(n),
        max_rel=max_abs / max(max_ref, 1e-300),
        l2_rel=sqrt(norm_err) / max(sqrt(norm_e), 1e-300),
        nonfinite=nonfinite,
        nonzero=nonzero,
        count=len(expected),
    )


def _gate(label: String, m: Metrics, relative: Bool) raises:
    print(t"  {label}: {m}")
    if not (m.passes_relative() if relative else m.passes()):
        raise Error(t"{label} is outside the MLA FP8 tolerance: {m}")


def _run_arm[
    num_heads: Int, row_depth: Int
](
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    pages_per_batch: Int,
    cache_rows: HostBuffer[FP8],
    kv_raw: HostBuffer[DType.bfloat16],
    gamma: HostBuffer[DType.bfloat16],
    q: HostBuffer[DType.bfloat16],
    w_uk: HostBuffer[DType.bfloat16],
    w_uv: HostBuffer[DType.bfloat16],
    freqs: HostBuffer[DType.bfloat16],
    indices: HostBuffer[DType.int32],
    out_host: HostBuffer[DType.bfloat16],
    cache_host: HostBuffer[FP8],
    iters: Int,
) raises -> Float64:
    """Runs the graph op once at `row_depth` and returns microseconds per call.

    The 512-wide inputs widen to `row_depth` with a zero tail. The output and
    the written cache land in `out_host` and `cache_host`.
    """
    comptime q_head_dim = QK_CONTENT + (row_depth - LATENT)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=row_depth, is_mla=True
    )
    var num_tokens = batch_size * q_len
    var total_pages = batch_size * pages_per_batch
    var rows = total_pages * PAGE_SIZE
    var max_pos = cache_len + q_len

    var blocks_h = ctx.enqueue_create_host_buffer[FP8](rows * row_depth)
    var q_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * num_heads * q_head_dim
    )
    var kv_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * row_depth
    )
    var lut_h = ctx.enqueue_create_host_buffer[DType.uint32](total_pages)
    var cl_h = ctx.enqueue_create_host_buffer[DType.uint32](batch_size)
    var ro_h = ctx.enqueue_create_host_buffer[DType.uint32](batch_size + 1)
    ctx.synchronize()

    for r in range(rows):
        for d in range(row_depth):
            blocks_h[r * row_depth + d] = cache_rows[
                r * LATENT + d
            ] if d < LATENT else Scalar[FP8](0)
    for row in range(num_tokens * num_heads):
        for d in range(q_head_dim):
            q_h[row * q_head_dim + d] = q[
                row * QK_CONTENT + d
            ] if d < QK_CONTENT else BFloat16(0)
    for t in range(num_tokens):
        for d in range(row_depth):
            kv_h[t * row_depth + d] = kv_raw[
                t * LATENT + d
            ] if d < LATENT else BFloat16(0)
    for p in range(total_pages):
        lut_h[p] = UInt32(p)
    for b in range(batch_size):
        cl_h[b] = UInt32(cache_len)
    for b in range(batch_size + 1):
        ro_h[b] = UInt32(b * q_len)

    var blocks_d = ctx.enqueue_create_buffer[FP8](rows * row_depth)
    var q_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_tokens * num_heads * q_head_dim
    )
    var kv_d = ctx.enqueue_create_buffer[DType.bfloat16](num_tokens * row_depth)
    var lut_d = ctx.enqueue_create_buffer[DType.uint32](total_pages)
    var cl_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var ro_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var gamma_d = ctx.enqueue_create_buffer[DType.bfloat16](LATENT)
    var w_uk_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * LATENT * QK_CONTENT
    )
    var w_uv_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * V_HEAD * LATENT
    )
    var freqs_d = ctx.enqueue_create_buffer[DType.bfloat16](max_pos * ROPE)
    var idx_d = ctx.enqueue_create_buffer[DType.int32](num_tokens * TOPK)
    var out_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_tokens * num_heads * V_HEAD
    )
    # The dense-prefill operands; the sparse op never reads them.
    var w_k_d = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * QK_CONTENT * LATENT
    )
    var buf_ro_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var cache_off_d = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_memset(w_k_d, 0)
    ctx.enqueue_memset(buf_ro_d, 0)
    ctx.enqueue_memset(cache_off_d, 0)
    ctx.enqueue_memset(out_d, 0)

    ctx.enqueue_copy(blocks_d, blocks_h)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(kv_d, kv_h)
    ctx.enqueue_copy(lut_d, lut_h)
    ctx.enqueue_copy(cl_d, cl_h)
    ctx.enqueue_copy(ro_d, ro_h)
    ctx.enqueue_copy(gamma_d, gamma)
    ctx.enqueue_copy(w_uk_d, w_uk)
    ctx.enqueue_copy(w_uv_d, w_uv)
    ctx.enqueue_copy(freqs_d, freqs)
    ctx.enqueue_copy(idx_d, indices)
    ctx.synchronize()

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
        TileTensor(lut_d, row_major(Int64(batch_size), Int64(pages_per_batch)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(cache_len),
    )

    var out_tt = TileTensor(
        out_d, row_major((num_tokens, Idx[num_heads], Idx[V_HEAD]))
    )
    var q_tt = TileTensor(
        q_d, row_major((num_tokens, Idx[num_heads], Idx[q_head_dim]))
    )
    var ro_tt = TileTensor(ro_d, row_major(batch_size + 1))
    var freqs_tt = TileTensor(freqs_d, row_major((max_pos, Idx[ROPE])))
    var gamma_tt = TileTensor(gamma_d, row_major(Idx[LATENT]))
    var buf_ro_tt = TileTensor(buf_ro_d, row_major(batch_size + 1)).as_imm()
    var cache_off_tt = TileTensor(cache_off_d, row_major(batch_size + 1))
    var w_k_tt = TileTensor(
        w_k_d, row_major((Idx[num_heads * QK_CONTENT], Idx[LATENT]))
    )
    var w_uk_tt = TileTensor(
        w_uk_d, row_major((Idx[num_heads], Idx[LATENT], Idx[QK_CONTENT]))
    )
    var w_uv_tt = TileTensor(
        w_uv_d, row_major((Idx[num_heads], Idx[V_HEAD], Idx[LATENT]))
    )
    var kv_in_tt = TileTensor(
        kv_d.unsafe_ptr(), row_major((num_tokens, Idx[row_depth]))
    )
    var scalar_args = MLADispatchScalarArgs[
        num_heads=num_heads, is_fp8_kv=True
    ](batch_size, cache_len, q_len, ctx)
    var scale = Float32(1.0 / sqrt(Float64(QK_CONTENT)))
    var idx_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](idx_d.unsafe_ptr())

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[DType.bfloat16, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    def launch() raises {imm}:
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
        )

    launch()
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_d)
    ctx.enqueue_copy(cache_host, blocks_d)
    ctx.synchronize()

    # Rerunning rewrites the same cache rows with the same values, so the
    # timed calls see the state the checked call saw.
    for _ in range(3):
        launch()
    ctx.synchronize()
    var best_us = Float64(1e30)
    for _ in range(3):
        var us = (
            Float64(ctx.execution_time(launch, iters)) / Float64(iters) / 1e3
        )
        best_us = min(best_us, us)

    _ = scalar_args
    _ = blocks_d
    _ = q_d
    _ = kv_d
    _ = lut_d
    _ = cl_d
    _ = ro_d
    _ = gamma_d
    _ = w_uk_d
    _ = w_uv_d
    _ = freqs_d
    _ = idx_d
    _ = out_d
    _ = w_k_d
    _ = buf_ro_d
    _ = cache_off_d
    return best_us


def _reference_token[
    num_heads: Int
](
    g: Int,
    k_rows: List[Float32],
    q: HostBuffer[DType.bfloat16],
    w_uk: HostBuffer[DType.bfloat16],
    w_uv: HostBuffer[DType.bfloat16],
    indices: HostBuffer[DType.int32],
    scale: Float64,
    quantized: Bool,
    mut out: List[Float64],
):
    """Appends query token `g`'s output, [num_heads, V_HEAD], to `out`.

    `quantized` rounds the Q latent to FP8 and the attention output to BF16,
    where the op stores them; `k_rows` already holds the matching cache rows.
    """
    var keys = List[Int]()
    for i in range(TOPK):
        var r = Int(indices[g * TOPK + i])
        if r < 0:
            break
        keys.append(r)
    var n = len(keys)
    var q_lat = List(length=LATENT, fill=Float64(0))
    var s = List(length=n, fill=Float64(0))
    var o = List(length=LATENT, fill=Float64(0))
    for h in range(num_heads):
        for c in range(LATENT):
            var acc = Float32(0)
            for d in range(QK_CONTENT):
                acc += (
                    q[(g * num_heads + h) * QK_CONTENT + d].cast[
                        DType.float32
                    ]()
                    * w_uk[(h * LATENT + c) * QK_CONTENT + d].cast[
                        DType.float32
                    ]()
                )
            q_lat[c] = acc.cast[FP8]().cast[
                DType.float64
            ]() if quantized else Float64(acc)
        var m = Float64(-1e300)
        for j in range(n):
            var base = keys[j] * LATENT
            var dot = Float64(0)
            for c in range(LATENT):
                dot += q_lat[c] * Float64(k_rows[base + c])
            s[j] = dot * scale
            m = max(m, s[j])
        var denom = Float64(0)
        for j in range(n):
            s[j] = exp(s[j] - m)
            denom += s[j]
        for c in range(LATENT):
            o[c] = 0
        for j in range(n):
            var p = s[j] / denom
            var base = keys[j] * LATENT
            for c in range(LATENT):
                o[c] += p * Float64(k_rows[base + c])
        if quantized:
            for c in range(LATENT):
                o[c] = o[c].cast[DType.bfloat16]().cast[DType.float64]()
        for v in range(V_HEAD):
            var acc = Float64(0)
            for c in range(LATENT):
                acc += (
                    o[c]
                    * w_uv[(h * V_HEAD + v) * LATENT + c].cast[DType.float64]()
                )
            out.append(acc)


def _gather_output[
    num_heads: Int
](tokens: List[Int], out_host: HostBuffer[DType.bfloat16]) -> List[Float64]:
    var got = List[Float64]()
    for g in tokens:
        for i in range(num_heads * V_HEAD):
            got.append(
                out_host[g * num_heads * V_HEAD + i].cast[DType.float64]()
            )
    return got^


def run_case[
    num_heads: Int
](
    ctx: DeviceContext,
    name: StaticString,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    ref_tokens: Int,
    iters: Int,
) raises:
    print(
        t"== {name}  heads={num_heads}  batch={batch_size}"
        t"  cache_len={cache_len}  q_len={q_len}"
    )
    seed(batch_size * 1_000_003 + cache_len * 1009 + q_len * 31 + num_heads)

    var num_tokens = batch_size * q_len
    var pages_per_batch = ceildiv(cache_len + q_len, PAGE_SIZE)
    var rows = batch_size * pages_per_batch * PAGE_SIZE

    var ctx_src = ctx.enqueue_create_host_buffer[DType.bfloat16](rows * LATENT)
    var cache_rows = ctx.enqueue_create_host_buffer[FP8](rows * LATENT)
    var kv_raw = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * LATENT
    )
    var gamma = ctx.enqueue_create_host_buffer[DType.bfloat16](LATENT)
    var q = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * num_heads * QK_CONTENT
    )
    var w_uk = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_heads * LATENT * QK_CONTENT
    )
    var w_uv = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_heads * V_HEAD * LATENT
    )
    var freqs = ctx.enqueue_create_host_buffer[DType.bfloat16](
        (cache_len + q_len) * ROPE
    )
    var indices = ctx.enqueue_create_host_buffer[DType.int32](num_tokens * TOPK)
    ctx.synchronize()

    # Context rows stand for latents earlier steps normalized and stored.
    randn(ctx_src.as_span(), mean=0.0, standard_deviation=1.0)
    randn(kv_raw.as_span(), mean=0.0, standard_deviation=2.0)
    randn(gamma.as_span(), mean=1.0, standard_deviation=0.25)
    randn(q.as_span(), mean=0.0, standard_deviation=1.0)
    # The Q latent comes out near 1.5 RMS, putting the scaled logits near 3
    # RMS: sharp enough that a wrong softmax scale moves the output.
    randn(
        w_uk.as_span(),
        mean=0.0,
        standard_deviation=1.5 / sqrt(Float64(QK_CONTENT)),
    )
    # Unit-norm rows keep the output at the latent output's magnitude.
    randn(
        w_uv.as_span(), mean=0.0, standard_deviation=1.0 / sqrt(Float64(LATENT))
    )
    randn(freqs.as_span(), mean=0.0, standard_deviation=1.0)

    for b in range(batch_size):
        for pos in range(pages_per_batch * PAGE_SIZE):
            var r = b * pages_per_batch * PAGE_SIZE + pos
            for d in range(LATENT):
                cache_rows[r * LATENT + d] = ctx_src[r * LATENT + d].cast[
                    FP8
                ]() if pos < cache_len else Scalar[FP8](0)

    # Each query token sees positions [0, cache_len + s]; it keeps a random
    # top-2048 of them, or every one in random order when fewer are visible.
    var pool = List[Int]()
    for b in range(batch_size):
        for s in range(q_len):
            var g = b * q_len + s
            var visible = cache_len + s + 1
            pool.clear()
            for p in range(visible):
                pool.append(p)
            var keep = min(visible, TOPK)
            for i in range(keep):
                var j = i + Int(random_ui64(0, UInt64(visible - i - 1)))
                var tmp = pool[i]
                pool[i] = pool[j]
                pool[j] = tmp
                indices[g * TOPK + i] = Int32(
                    b * pages_per_batch * PAGE_SIZE + pool[i]
                )
            for i in range(keep, TOPK):
                indices[g * TOPK + i] = Int32(-1)

    var out_native = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * num_heads * V_HEAD
    )
    var out_padded = ctx.enqueue_create_host_buffer[DType.bfloat16](
        num_tokens * num_heads * V_HEAD
    )
    var cache_native = ctx.enqueue_create_host_buffer[FP8](rows * LATENT)
    var cache_padded = ctx.enqueue_create_host_buffer[FP8](rows * PADDED)
    ctx.synchronize()

    var us_native = _run_arm[num_heads, LATENT](
        ctx,
        batch_size,
        cache_len,
        q_len,
        pages_per_batch,
        cache_rows,
        kv_raw,
        gamma,
        q,
        w_uk,
        w_uv,
        freqs,
        indices,
        out_native,
        cache_native,
        iters,
    )
    var us_padded = _run_arm[num_heads, PADDED](
        ctx,
        batch_size,
        cache_len,
        q_len,
        pages_per_batch,
        cache_rows,
        kv_raw,
        gamma,
        q,
        w_uk,
        w_uv,
        freqs,
        indices,
        out_padded,
        cache_padded,
        iters,
    )

    # Cache rows: both arms hold the same bytes over [0, 512), the padded tail
    # stays zero, and the context rows are untouched. Rotating the zero rope
    # tail can store -0, which is still zero.
    var row_diffs = 0
    var tail_nonzero = 0
    var tail_neg_zero = 0
    for r in range(rows):
        for d in range(LATENT):
            if (
                cache_native[r * LATENT + d].to_bits()
                != cache_padded[r * PADDED + d].to_bits()
            ):
                row_diffs += 1
        for d in range(LATENT, PADDED):
            var v = cache_padded[r * PADDED + d]
            if v.cast[DType.float32]() != 0:
                tail_nonzero += 1
            elif v.to_bits() != 0:
                tail_neg_zero += 1
    var ctx_changed = 0
    for b in range(batch_size):
        for pos in range(cache_len):
            var r = b * pages_per_batch * PAGE_SIZE + pos
            for d in range(LATENT):
                if (
                    cache_native[r * LATENT + d].to_bits()
                    != cache_rows[r * LATENT + d].to_bits()
                ):
                    ctx_changed += 1
    print(
        t"  cache: native/padded differing bytes={row_diffs}"
        t"  padded tail nonzero={tail_nonzero} (-0: {tail_neg_zero})"
        t"  context bytes changed={ctx_changed}"
    )
    if row_diffs != 0 or tail_nonzero != 0 or ctx_changed != 0:
        raise Error(t"{name}: the native and padded caches disagree")

    # Native against padded, every element.
    var all_tokens = List[Int]()
    for g in range(num_tokens):
        all_tokens.append(g)
    var got_native_all = _gather_output[num_heads](all_tokens, out_native)
    var got_padded_all = _gather_output[num_heads](all_tokens, out_padded)
    var bit_diffs = 0
    for i in range(num_tokens * num_heads * V_HEAD):
        if out_native[i].to_bits() != out_padded[i].to_bits():
            bit_diffs += 1
    print(t"  native vs padded: differing BF16 outputs={bit_diffs}")
    var m_np = _metrics(got_padded_all, got_native_all)
    _gate("native vs padded", m_np, relative=False)

    # Reference rows, indexed like the cache: FP8 context rows plus the new
    # rows as the fused RMSNorm stores them (BF16 normalize, BF16 gamma, FP8),
    # and an unrounded set for the information-only float64 comparison.
    var k_q = List(length=rows * LATENT, fill=Float32(0))
    var k_f = List(length=rows * LATENT, fill=Float32(0))
    for b in range(batch_size):
        for pos in range(cache_len):
            var r = b * pages_per_batch * PAGE_SIZE + pos
            for d in range(LATENT):
                k_q[r * LATENT + d] = cache_rows[r * LATENT + d].cast[
                    DType.float32
                ]()
                k_f[r * LATENT + d] = ctx_src[r * LATENT + d].cast[
                    DType.float32
                ]()
        for s in range(q_len):
            var g = b * q_len + s
            var r = b * pages_per_batch * PAGE_SIZE + cache_len + s
            var m2 = Float32(0)
            for d in range(LATENT):
                var x = kv_raw[g * LATENT + d].cast[DType.float32]()
                m2 += x * x
            var nf = 1.0 / sqrt(m2 / Float32(LATENT) + EPSILON)
            for d in range(LATENT):
                var x = kv_raw[g * LATENT + d].cast[DType.float32]()
                var normed = (x * nf).cast[DType.bfloat16]() * gamma[d]
                k_q[r * LATENT + d] = normed.cast[FP8]().cast[DType.float32]()
                k_f[r * LATENT + d] = x * nf * gamma[d].cast[DType.float32]()

    var ref_list = List[Int]()
    var n_ref = min(ref_tokens, num_tokens)
    for i in range(n_ref):
        ref_list.append(
            0 if n_ref == 1 else i * (num_tokens - 1) // (n_ref - 1)
        )
    var scale = 1.0 / sqrt(Float64(QK_CONTENT))
    var ref_q = List[Float64]()
    var ref_f = List[Float64]()
    for g in ref_list:
        _reference_token[num_heads](
            g, k_q, q, w_uk, w_uv, indices, scale, True, ref_q
        )
        _reference_token[num_heads](
            g, k_f, q, w_uk, w_uv, indices, scale, False, ref_f
        )
    var got_native = _gather_output[num_heads](ref_list, out_native)
    var got_padded = _gather_output[num_heads](ref_list, out_padded)
    var ref_rms = Float64(0)
    for v in ref_q:
        ref_rms += v * v
    ref_rms = sqrt(ref_rms / Float64(max(len(ref_q), 1)))
    print(t"  reference: {n_ref} query tokens, output RMS={ref_rms}")
    var m_nr = _metrics(ref_q, got_native)
    var m_pr = _metrics(ref_q, got_padded)
    _gate("native vs reference", m_nr, relative=True)
    _gate("padded vs reference", m_pr, relative=True)
    var m_nf = _metrics(ref_f, got_native)
    print(t"  native vs float64 reference (info): {m_nf}")

    # Negative control: the padded geometry's default scale, 1/sqrt(192).
    var ctl_tokens = List[Int]()
    ctl_tokens.append(ref_list[0])
    ctl_tokens.append(ref_list[len(ref_list) - 1])
    var ref_bad = List[Float64]()
    for g in ctl_tokens:
        _reference_token[num_heads](
            g,
            k_q,
            q,
            w_uk,
            w_uv,
            indices,
            1.0 / sqrt(Float64(QK_CONTENT + ROPE)),
            True,
            ref_bad,
        )
    var got_ctl = _gather_output[num_heads](ctl_tokens, out_native)
    var m_bad = _metrics(ref_bad, got_ctl)
    print(t"  control (scale 1/sqrt(192)): {m_bad}")
    if m_bad.passes_relative():
        raise Error(t"{name}: the gate cannot see a 1/sqrt(192) softmax scale")

    print(
        t"RESULT {name} heads={num_heads}"
        t" | native_vs_padded bitdiff={bit_diffs} max_abs={m_np.max_abs}"
        t" max_rel={m_np.max_rel}"
        t" | native_vs_ref cos={m_nr.cosine} max_abs={m_nr.max_abs}"
        t" max_rel={m_nr.max_rel}"
        t" | padded_vs_ref cos={m_pr.cosine} max_abs={m_pr.max_abs}"
        t" max_rel={m_pr.max_rel}"
        t" | us native={us_native} padded={us_padded}"
        t" ratio={us_native / us_padded}"
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # 8 heads per rank is TP8 attention; 64 is DP8 attention.
            run_case[8](ctx, "decode", 8, 4096, 1, 16, 50)
            run_case[8](ctx, "verify4", 8, 4096, 4, 16, 50)
            run_case[8](ctx, "verify4_short", 4, 1000, 4, 16, 50)
            run_case[8](ctx, "prefill_chunk", 1, 4096, 512, 8, 10)
            run_case[8](ctx, "prefill_short", 2, 0, 300, 8, 10)
            run_case[64](ctx, "decode", 8, 4096, 1, 8, 50)
            run_case[64](ctx, "verify4", 8, 4096, 4, 8, 50)
            run_case[64](ctx, "verify4_short", 4, 1000, 4, 8, 50)
            run_case[64](ctx, "prefill_chunk", 1, 4096, 512, 6, 10)
            run_case[64](ctx, "prefill_short", 2, 0, 300, 6, 10)
        else:
            print("Skipping: needs an SM100 GPU")
