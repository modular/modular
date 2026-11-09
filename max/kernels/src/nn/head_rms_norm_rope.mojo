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
"""Muse Spark's per-head attention norms, one launch each.

Both kernels treat each 64-channel (token, head) row with 8 lanes of 8
channels (16 bytes) and no shared memory:

- `qk_norm_rope_store`: of a `[q | k | v | ...]` projection,
  `q = rope(rms_norm(q) * q_scale)` to the output, `k = rope(rms_norm(k))`
  and `v` to the paged cache. RoPE's rotate-half partner is 4 lanes over.
- `head_norm_gate`: `rms_norm(x) * sigmoid(gate)` per head.

Every step rounds where the unfused ops round, and the norm adds in their
order (`head_norm`), so the outputs are theirs bit for bit. The schedule and
the arithmetic are the Moka component's
(`Kernels/test/gpu/Moka/kernels/norm/head_norm_rope_plan.mojo`), whose ops call
the lane functions below.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv, rsqrt

from internal_utils.fp8_utils import cast_saturating
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp

from nn.activations import sigmoid
from nn.fused_qk_rope import rope_value

comptime HEAD_DIM = 64
comptime LANES = 8
"""Lanes per head row."""
comptime VEC = HEAD_DIM // LANES
comptime TILE_ROWS = 32
"""Head rows per CTA."""
comptime THREADS = TILE_ROWS * LANES

comptime POISONS = 5
"""Planted faults, `poison` 1 through 5, each of which a test must catch:
1 drops Q's scale, 2 stores K one position late, 3 ropes with the lane's own
half for its partner, 4 drops the gate's sigmoid, 5 skips the last token."""

comptime In[T: DType] = Pointer[Scalar[T], ImmUntrackedOrigin]
comptime Out[T: DType] = Pointer[Scalar[T], MutUntrackedOrigin]
comptime Piece = SIMD[DType.bfloat16, VEC]


@fieldwise_init
struct QkArgs[cache_dtype: DType, freq_dtype: DType = DType.float32](
    DevicePassable, ImplicitlyCopyable
):
    """`qk_norm_rope_store`'s operands, one kernel parameter."""

    var qkv: In[DType.bfloat16]
    """The projection, `qkv_stride` elements a token."""
    var q_out: Out[DType.bfloat16]
    """The roped Q `[tokens, n_q * HEAD_DIM]`."""
    var freqs: In[Self.freq_dtype]
    """Interleaved (cos, sin) pairs `[positions, HEAD_DIM]`."""
    var offsets: In[DType.uint32]
    """Ragged row offsets `[batch + 1]`."""
    var cache_lengths: In[DType.uint32]
    var lut: In[DType.uint32]
    """The page table, `lut_cols` pages a sequence."""
    var k_rows: Out[Self.cache_dtype]
    """This layer's key plane: rows of `n_kv * HEAD_DIM`, `page_rows` rows
    from one page to the next."""
    var v_rows: Out[Self.cache_dtype]
    var qkv_stride: Int64
    var tokens: Int64
    var batch: Int64
    var lut_cols: Int64
    var page_rows: Int64
    var eps: Float32
    var q_scale: Float32
    """The Q factor, already rounded to bf16."""

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "QkArgs"


@fieldwise_init
struct GateArgs(DevicePassable, ImplicitlyCopyable):
    """`head_norm_gate`'s operands, one kernel parameter."""

    var x: In[DType.bfloat16]
    """The attention output `[tokens, n_heads * HEAD_DIM]`."""
    var gate: In[DType.bfloat16]
    """The gate's first column, `gate_stride` elements a token."""
    var out: Out[DType.bfloat16]
    var gate_stride: Int64
    var tokens: Int64
    var eps: Float32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GateArgs"


@always_inline
def head_norm(x: Piece, eps: Float32) -> Piece:
    """The lane's piece of `rms_norm` over its 8-lane head row.

    Adds as `rowwise`'s warp tier does at 64 channels: lane `l` of 32 sums
    channels `2l, 2l + 1`, then strides 16, 8, 4, 2, 1. This lane holds that
    warp's lanes `4j .. 4j + 3`: three strides are shuffles, two are local.
    """
    var xf = x.cast[DType.float32]()
    var sq = xf * xf
    var p = sq.deinterleave()[0] + sq.deinterleave()[1]
    p += warp.shuffle_xor(p, 4)
    p += warp.shuffle_xor(p, 2)
    p += warp.shuffle_xor(p, 1)
    var ssq = (p[0] + p[2]) + (p[1] + p[3])
    return (xf * rsqrt(ssq / Float32(HEAD_DIM) + eps)).cast[DType.bfloat16]()


@always_inline
def _batch_of(offsets: In[DType.uint32], batch: Int, token: Int) -> Int:
    """The sequence holding `token`: the last offset at or below it."""
    var lo = 0
    var hi = batch
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if Int(offsets[unsafe_offset=mid]) <= token:
            lo = mid
        else:
            hi = mid
    return lo


@always_inline
def qk_lanes[n_q: Int, n_kv: Int](tokens: Int) -> Int:
    """Lanes over `tokens` tokens: whole warps, since a token's rows are."""
    comptime assert ((n_q + 2 * n_kv) * LANES) % WARP_SIZE == 0
    return tokens * (n_q + 2 * n_kv) * LANES


@always_inline
def qk_lane[
    cache_dtype: DType,
    freq_dtype: DType,
    //,
    n_q: Int,
    n_kv: Int,
    page_size: Int,
    poison: Int,
](a: QkArgs[cache_dtype, freq_dtype], i: Int):
    """Lane `i`'s 8 channels of its (token, head) row. Every lane of a
    token's warps must call it: the norm and RoPE shuffle across them."""
    comptime heads = n_q + 2 * n_kv
    comptime half = HEAD_DIM // 2
    var row = i // LANES
    var d = (i % LANES) * VEC
    var token = row // heads
    var head = row % heads
    var bi = _batch_of(a.offsets, Int(a.batch), token)
    var pos = (
        Int(a.cache_lengths[unsafe_offset=bi])
        + token
        - Int(a.offsets[unsafe_offset=bi])
    )
    var x = a.qkv.unsafe_offset(
        token * Int(a.qkv_stride) + head * HEAD_DIM + d
    ).unsafe_load[width=VEC, alignment=16]()
    var n = head_norm(x, a.eps)
    comptime if poison != 1:
        if head < n_q:
            n = (n.cast[DType.float32]() * a.q_scale).cast[DType.bfloat16]()
    # Every lane shuffles: a warp mixes K and V rows.
    var partner = n if poison == 3 else warp.shuffle_xor(n, LANES // 2)
    var at = pos + Int(poison == 2 and head >= n_q)
    var page = Int(a.lut[unsafe_offset=bi * Int(a.lut_cols) + at // page_size])
    var kv = (
        (page * Int(a.page_rows) + at % page_size) * n_kv * HEAD_DIM
        + (head - n_q) % n_kv * HEAD_DIM
        + d
    )
    if head >= n_q + n_kv:
        a.v_rows.unsafe_offset(kv).unsafe_store(cast_saturating[cache_dtype](x))
        return
    var lo = d < half
    var re = n if lo else partner
    var im = partner if lo else n
    var freq = a.freqs.unsafe_offset(
        pos * HEAD_DIM + 2 * (d % half)
    ).unsafe_load[width=2 * VEC]()
    var r_re, r_im = rope_value(re.interleave(im), freq).deinterleave()
    var roped = r_re if lo else r_im
    if head < n_q:
        a.q_out.unsafe_offset(
            token * n_q * HEAD_DIM + head * HEAD_DIM + d
        ).unsafe_store[alignment=16](roped)
    else:
        a.k_rows.unsafe_offset(kv).unsafe_store(
            cast_saturating[cache_dtype](roped)
        )


@always_inline
def gate_lane[n_heads: Int, poison: Int](a: GateArgs, i: Int):
    """Lane `i`'s 8 channels of `rms_norm(x) * sigmoid(gate)`."""
    comptime width = n_heads * HEAD_DIM
    var token = i // (n_heads * LANES)
    var c = (i % (n_heads * LANES)) * VEC
    var n = head_norm(
        a.x.unsafe_offset(token * width + c).unsafe_load[
            width=VEC, alignment=16
        ](),
        a.eps,
    )
    var g = a.gate.unsafe_offset(token * Int(a.gate_stride) + c).unsafe_load[
        width=VEC, alignment=16
    ]()
    comptime if poison == 4:
        a.out.unsafe_offset(token * width + c).unsafe_store[alignment=16](n * g)
    else:
        a.out.unsafe_offset(token * width + c).unsafe_store[alignment=16](
            n * sigmoid(g)
        )


@__name(t"qk_norm_rope_store_q{n_q}_kv{n_kv}_p{poison}")
def _qk_kernel[
    cache_dtype: DType,
    freq_dtype: DType,
    //,
    n_q: Int,
    n_kv: Int,
    page_size: Int,
    poison: Int,
](a: QkArgs[cache_dtype, freq_dtype]):
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    # Whole warps fall past the end, so the shuffles see full warps.
    if i < qk_lanes[n_q, n_kv](Int(a.tokens) - Int(poison == 5)):
        qk_lane[n_q, n_kv, page_size, poison](a, i)


@__name(t"head_norm_gate_h{n_heads}_p{poison}")
def _gate_kernel[n_heads: Int, poison: Int](a: GateArgs):
    comptime assert (n_heads * LANES) % WARP_SIZE == 0
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    if i < (Int(a.tokens) - Int(poison == 5)) * n_heads * LANES:
        gate_lane[n_heads, poison](a, i)


def qk_norm_rope_store[
    cache_dtype: DType,
    freq_dtype: DType,
    //,
    n_q: Int,
    n_kv: Int,
    page_size: Int,
    poison: Int = 0,
](a: QkArgs[cache_dtype, freq_dtype], ctx: DeviceContext) raises:
    """Writes the roped Q and stores the roped K and the V.

    The projection's rows and the Q output are 16-byte aligned. `poison`
    plants a fault for the tests.
    """
    var lanes = qk_lanes[n_q, n_kv](Int(a.tokens))
    if lanes == 0:
        return
    ctx.enqueue_function[
        _qk_kernel[
            cache_dtype=cache_dtype,
            freq_dtype=freq_dtype,
            n_q=n_q,
            n_kv=n_kv,
            page_size=page_size,
            poison=poison,
        ]
    ](
        a,
        grid_dim=ceildiv(lanes, THREADS),
        block_dim=THREADS,
    )


def head_norm_gate[
    n_heads: Int, poison: Int = 0
](a: GateArgs, ctx: DeviceContext) raises:
    """Writes `rms_norm(x) * sigmoid(gate)` per head.

    Rows are 16-byte aligned. `poison` plants a fault for the tests.
    """
    var lanes = Int(a.tokens) * n_heads * LANES
    if lanes == 0:
        return
    ctx.enqueue_function[_gate_kernel[n_heads=n_heads, poison=poison]](
        a,
        grid_dim=ceildiv(lanes, THREADS),
        block_dim=THREADS,
    )
