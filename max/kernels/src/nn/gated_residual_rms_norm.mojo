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
"""A residual boundary with a learned gate, in one persistent launch.

Muse Spark joins each branch to the residual stream as

    x = branch * (1 + post_weight)                    # bf16, when scaled
    n = cast(x * rsqrt(mean(x^2) + post_eps))         # weightless norm
    stream = cast(alpha * residual + beta * n)        # fp32 mix
    normed = cast(stream * rsqrt(mean(stream^2) + norm_eps) * (1 + norm_weight))

with `beta = sigmoid(gate / T)` and
`alpha = sqrt(max(sigmoid(-gate / T) * (1 + beta), gate_eps))`. Persistent
CTAs take rows cta, cta + G, ...: a load lane bulk-copies each row's branch
and residual into a two-deep ring, and eight work warps, a thread per four
16-byte pieces, compute their channels' gate coefficients once and keep them
and the weights in registers for the whole launch. Rows are summed in the
order of `rms_norm`'s rowwise block tier, so the outputs match the unfused
ops bit for bit at any row count. The schedule and the
arithmetic are the Moka component's
(`Kernels/test/gpu/Moka/kernels/norm/gated_residual_norm_plan.mojo`), whose
ops call the helpers below.

Under SM-count rows `gated_residual_rms_norm_split` spreads each row over a
cluster of `SPLIT` CTAs instead, a thread per 16-byte piece; it is the port
of `Kernels/test/gpu/Moka/kernels/norm/gated_split_plan.mojo`.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections.optional import OptionalReg
from std.math import ceildiv, fma, max, rsqrt, sqrt
from std.memory import bitcast, unsafe_stack_allocation
from std.sys import inlined_assembly
from std.utils import StaticTuple

from layout.tma_async import SharedMemBarrier
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, Dim, FuncAttribute
from max.gpu.intrinsics import Scope
from max.gpu.memory import (
    cp_async_bulk_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives import warp
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier, named_barrier

from nn.activations import sigmoid

comptime HIDDEN = 8192
"""Muse Spark's hidden size, the only width the kernel takes."""
comptime VEC = 8
"""The bf16 channels per 16-byte access."""
comptime PIECES = 4
"""16-byte pieces of a row per work thread: `rms_norm`'s 256-thread block
tier's tiles, so each work thread is one of its threads."""
comptime WORK_WARPS = HIDDEN // VEC // PIECES // WARP_SIZE
comptime WORKERS = WORK_WARPS * WARP_SIZE
comptime WIDE_WARPS = PIECES * WORK_WARPS
"""Warps of the 1024-thread block tier `rms_norm` takes under SM-count rows;
piece `p` of work warp `w` is its warp `p * WORK_WARPS + w`."""
comptime SLOTS = 2
comptime ROW_BYTES = HIDDEN * 2
comptime _COEF = SLOTS * 2 * ROW_BYTES
comptime COEF_BYTES = HIDDEN * (4 + 4 + 2 + 2)
"""Per channel: alpha, beta (fp32), `1 + post_weight`, `1 + norm_weight`."""
comptime _RED = _COEF + COEF_BYTES
comptime _BARS = _RED + 2 * 2 * WIDE_WARPS * 4
comptime SMEM = _BARS + 2 * SLOTS * 8
"""The ring slots (branch then residual rows), the channel coefficients,
the warps' two sums double buffered by row parity, then the full and empty
barriers. Each thread writes and reads only its own channels' coefficients:
in registers they spilled every row."""

comptime POISONS = 5
"""Planted faults, `poison` 1 through 5, each of which a test must catch:
1 swaps alpha and beta, 2 skips the branch scale, 3 drops the norm weight's
offset, 4 normalizes the stream by the branch's sum, 5 skips the last row."""

comptime In[T: DType] = Pointer[Scalar[T], ImmUntrackedOrigin]
comptime Out[T: DType] = Pointer[Scalar[T], MutUntrackedOrigin]
comptime Piece = SIMD[DType.bfloat16, VEC]
comptime Wide = SIMD[DType.float32, VEC]


@always_inline
def to_bf16(v: Wide) -> Piece:
    """Rounds to bf16 two channels per `cvt.rn.bf16x2.f32`: a plain cast
    takes one XU conversion per channel, the kernel's busiest pipe."""
    var out = SIMD[DType.uint32, VEC // 2]()
    comptime for i in range(VEC // 2):
        out[i] = inlined_assembly[
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            UInt32,
            constraints="=r,f,f",
            has_side_effect=False,
        ](v[2 * i + 1], v[2 * i])
    return bitcast[DType.bfloat16, VEC](out)


@always_inline
def chan(p: Int, rt: Int) -> Int:
    """The first channel of work thread `rt`'s piece `p`: a warp's pieces
    are contiguous, so its 16-byte accesses coalesce."""
    return (p * WORKERS + rt) * VEC


@always_inline
def scaled_branch[
    scale: Bool, poison: Int = 0
](branch: Piece, scale_weight: Piece) -> Piece:
    """The branch before its norm; `scale_weight` is `1 + post_weight`,
    rounded to bf16 as the graph does."""
    comptime if scale and poison != 2:
        return branch * scale_weight
    return branch


@always_inline
def _square_sum[n: Int](x: Array[Piece, PIECES], first: Int) -> Float32:
    # rowwise `ReduceSum`: lane-wise `acc += t * t` per tile in order, then
    # `reduce_add`
    var acc = Wide(0)
    comptime for p in range(n):
        var f = x[first + p].cast[DType.float32]()
        acc += f * f
    return acc.reduce_add()


@always_inline
def warp_sums(
    x: Array[Piece, PIECES], wide: Bool
) -> SIMD[DType.float32, PIECES]:
    """The row's per-warp sums of squares as `rms_norm`'s block tier forms
    them: under SM-count rows (`wide`, 1024 threads) one per piece, else one
    over the thread's pieces in element 0. Every lane holds them."""
    var out = SIMD[DType.float32, PIECES](0)
    if wide:
        comptime for p in range(PIECES):
            out[p] = warp.sum(_square_sum[1](x, p))
    else:
        out[0] = warp.sum(_square_sum[PIECES](x, 0))
    return out


@always_inline
def red_index(p: Int, wid: Int, wide: Bool) -> Int:
    """Where work warp `wid` publishes its sum `p` of `warp_sums`."""
    return p * WORK_WARPS + wid if wide else wid


@always_inline
def red_count(wide: Bool) -> Int:
    """The published warp sums: the block tier's warp count."""
    return WIDE_WARPS if wide else WORK_WARPS


@always_inline
def block_sum(lane_sum: Float32) -> Float32:
    """The row's total from lane `l` holding warp `l`'s published sum (0 past
    `red_count`), summed as `rms_norm`'s block reduce sums its warps."""
    return warp.sum(lane_sum)


@always_inline
def gate_coefficients(
    gate: Piece, inv_temperature: Float32, gate_eps: Float32
) -> Tuple[Wide, Wide]:
    """Returns `(alpha, beta)` of the gate's logits, computed as the unfused
    graph computes them (which folds `gate / T` into a reciprocal multiply)."""
    var w = gate.cast[DType.float32]() * inv_temperature
    var beta = sigmoid(w)
    return sqrt(max(sigmoid(-w) * (1 + beta), Wide(gate_eps))), beta


@always_inline
def mix[
    poison: Int = 0
](x: Piece, residual: Piece, alpha: Wide, beta: Wide, inv: Float32) -> Piece:
    """The new residual stream: the branch's norm mixed into the residual."""
    var n = to_bf16(x.cast[DType.float32]() * Wide(inv))
    var a = beta if poison == 1 else alpha
    var b = alpha if poison == 1 else beta
    # the contraction the unfused graph's mix compiles to, so the bits match
    return to_bf16(
        fma(b, n.cast[DType.float32](), a * residual.cast[DType.float32]())
    )


@always_inline
def next_norm(stream: Piece, inv: Float32, gamma: Piece) -> Piece:
    """The next pre-norm; `gamma` is `1 + norm_weight`, rounded to bf16."""
    return to_bf16(
        stream.cast[DType.float32]() * Wide(inv) * gamma.cast[DType.float32]()
    )


@always_inline
def inv_rms(sum: Float32, eps: Float32) -> Float32:
    return rsqrt(sum / Float32(HIDDEN) + eps)


@fieldwise_init
struct GatedResidualArgs(DevicePassable, ImplicitlyCopyable):
    """The boundary's operands, one kernel parameter. Rows are dense."""

    var branch: In[DType.bfloat16]
    var residual: In[DType.bfloat16]
    var post_weight: In[DType.bfloat16]
    var gate: In[DType.bfloat16]
    var norm_weight: In[DType.bfloat16]
    var stream: Out[DType.bfloat16]
    var normed: Out[DType.bfloat16]
    var rows: Int64
    var post_eps: Float32
    var norm_eps: Float32
    var temperature: Float32
    var gate_eps: Float32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GatedResidualArgs"


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(WORKERS + WARP_SIZE)
    )
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
def _gated_residual_ring[
    scale: Bool, poison: Int
](a: GatedResidualArgs, wide_tier: Int32):
    var smem = external_memory[UInt8, address_space=.SHARED, alignment=128]()
    var bars = smem.unsafe_offset(_BARS).unsafe_bitcast[SharedMemBarrier]()
    if thread_idx.x == 0:
        comptime for s in range(SLOTS):
            bars[unsafe_offset=s].init(1)
            bars[unsafe_offset=SLOTS + s].init(Int32(WORKERS))
        fence_mbarrier_init()
    barrier()
    var wide = wide_tier != 0
    var rows = Int(a.rows) - Int(poison == 5)
    var grid = grid_dim.x
    var cta = block_idx.x
    var trips = ceildiv(rows - cta, grid)

    if warp_id() == WORK_WARPS:
        if lane_id() != 0:
            return
        for i in range(trips):
            var slot = i % SLOTS
            var full = bars.unsafe_offset(slot)
            bars[unsafe_offset=SLOTS + slot].wait(UInt32((i // SLOTS) & 1) ^ 1)
            # the work warps read the slot through the generic proxy, and
            # the copies write it through the async one
            fence_async_view_proxy()
            full[].expect_bytes(Int32(2 * ROW_BYTES))
            var t = cta + i * grid
            var dst = smem.unsafe_offset(slot * 2 * ROW_BYTES)
            cp_async_bulk_shared_cluster_global(
                dst, a.branch.unsafe_offset(t * HIDDEN), Int32(ROW_BYTES), full
            )
            cp_async_bulk_shared_cluster_global(
                dst.unsafe_offset(ROW_BYTES),
                a.residual.unsafe_offset(t * HIDDEN),
                Int32(ROW_BYTES),
                full,
            )
        return

    var rt = thread_idx.x
    var wid = warp_id()
    var inv_t = 1 / a.temperature
    var alpha = smem.unsafe_offset(_COEF).unsafe_bitcast[Float32]()
    var beta = alpha.unsafe_offset(HIDDEN)
    var scale_w = beta.unsafe_offset(HIDDEN).unsafe_bitcast[BFloat16]()
    var gamma = scale_w.unsafe_offset(HIDDEN)
    comptime for p in range(PIECES):
        var c = chan(p, rt)
        comptime if scale:
            scale_w.unsafe_store[alignment=16](
                c, 1 + a.post_weight.unsafe_load[width=VEC, alignment=16](c)
            )
        var nw = a.norm_weight.unsafe_load[width=VEC, alignment=16](c)
        gamma.unsafe_store[alignment=16](c, nw if poison == 3 else 1 + nw)
        var ab = gate_coefficients(
            a.gate.unsafe_load[width=VEC, alignment=16](c), inv_t, a.gate_eps
        )
        alpha.unsafe_store[alignment=32](c, ab[0])
        beta.unsafe_store[alignment=32](c, ab[1])
    var red = smem.unsafe_offset(_RED).unsafe_bitcast[Float32]()
    for i in range(trips):
        var slot = i % SLOTS
        bars[unsafe_offset=slot].wait(UInt32((i // SLOTS) & 1))
        var t = cta + i * grid
        var bs = smem.unsafe_offset(slot * 2 * ROW_BYTES).unsafe_bitcast[
            BFloat16
        ]()
        var rs = bs.unsafe_offset(HIDDEN)
        var x = Array[Piece, PIECES](fill=Piece(0))
        var r = Array[Piece, PIECES](fill=Piece(0))
        comptime for p in range(PIECES):
            var c = chan(p, rt)
            x[p] = scaled_branch[scale, poison](
                bs.unsafe_load[width=VEC, alignment=16](c),
                scale_w.unsafe_load[width=VEC, alignment=16](c),
            )
            r[p] = rs.unsafe_load[width=VEC, alignment=16](c)
        var sums = red.unsafe_offset((i & 1) * 2 * WIDE_WARPS)
        var lane = lane_id()
        var n = red_count(wide)
        var s1 = warp_sums(x, wide)
        if lane == 0:
            comptime for p in range(PIECES):
                if p == 0 or wide:
                    sums[unsafe_offset=red_index(p, wid, wide)] = s1[p]
        # each reader releases its own reads: an arrive for the whole warp
        # does not order the other lanes' reads before the async refill
        _ = bars[unsafe_offset=SLOTS + slot].arrive()
        named_barrier[Int32(WORKERS)](1)

        var inv1 = inv_rms(
            block_sum(sums[unsafe_offset=lane] if lane < n else 0), a.post_eps
        )
        var h = Array[Piece, PIECES](fill=Piece(0))
        var sp = a.stream.unsafe_offset(t * HIDDEN)
        comptime for p in range(PIECES):
            h[p] = mix[poison](
                x[p],
                r[p],
                alpha.unsafe_load[width=VEC, alignment=32](chan(p, rt)),
                beta.unsafe_load[width=VEC, alignment=32](chan(p, rt)),
                inv1,
            )
            sp.unsafe_store[alignment=16](chan(p, rt), h[p])
        var s2 = warp_sums(h, wide)
        if lane == 0:
            comptime for p in range(PIECES):
                if p == 0 or wide:
                    sums[
                        unsafe_offset=WIDE_WARPS + red_index(p, wid, wide)
                    ] = s2[p]
        named_barrier[Int32(WORKERS)](2)

        var at = (0 if poison == 4 else WIDE_WARPS) + lane
        var inv2 = inv_rms(
            block_sum(sums[unsafe_offset=at] if lane < n else 0), a.norm_eps
        )
        var np = a.normed.unsafe_offset(t * HIDDEN)
        comptime for p in range(PIECES):
            np.unsafe_store[alignment=16](
                chan(p, rt),
                next_norm(
                    h[p],
                    inv2,
                    gamma.unsafe_load[width=VEC, alignment=16](chan(p, rt)),
                ),
            )


def gated_residual_rms_norm[
    scale: Bool, poison: Int = 0
](a: GatedResidualArgs, ctx: DeviceContext) raises:
    """Writes the gated residual stream and its next pre-norm.

    `a` holds dense `[rows, 8192]` bf16 rows and `[8192]` weights, 16-byte
    aligned. The outputs must not overlap the
    inputs. `scale` multiplies the branch by `1 + post_weight` before its
    norm; otherwise `post_weight` is not read. `poison` plants a fault for
    the tests.
    """
    var rows = Int(a.rows) - Int(poison == 5)
    if rows <= 0:
        return
    var sms = ctx.default_device_info.sm_count
    ctx.enqueue_function[_gated_residual_ring[scale, poison]](
        a,
        # `rms_norm`'s block tier: 1024 threads under SM-count rows
        Int32(Int(a.rows) < sms),
        grid_dim=min(Int(a.rows), sms),
        block_dim=WORKERS + WARP_SIZE,
        shared_mem_bytes=SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM)
        ),
    )


comptime SPLIT = 8
"""CTAs of a cluster that share one row under SM-count rows."""
comptime SPLIT_THREADS = HIDDEN // VEC // SPLIT
"""Threads of a split CTA: one 16-byte piece each, so CTA `c`'s warp `w` is
the 1024-thread block tier's warp `c * 4 + w`."""


@always_inline
def _st_async(
    buf: Pointer[mut=True, Float32, _, address_space=.SHARED],
    i: Int,
    peer: Int,
    v: Float32,
    bar: Pointer[mut=True, SharedMemBarrier, _, address_space=.SHARED],
):
    """Stores `v` at `buf[i]` in cluster CTA `peer`, completing 4 bytes on
    that CTA's copy of `bar`."""
    inlined_assembly[
        """{
        .reg .b32 ra, rb;
        mapa.shared::cluster.u32 ra, $0, $2;
        mapa.shared::cluster.u32 rb, $3, $2;
        st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [ra], $1, [rb];
    }""",
        NoneType,
        constraints="r,f,r,r",
        has_side_effect=True,
    ](
        UInt32(Int(buf)) + UInt32(4 * i),
        v,
        UInt32(peer),
        UInt32(Int(bar)),
    )


@always_inline
def _cluster_total(
    s: Float32,
    at: Int,
    rank: Int,
    mail: Pointer[mut=True, Float32, _, address_space=.SHARED],
    tot: Pointer[mut=True, Float32, _, address_space=.SHARED],
    gather: Pointer[mut=True, SharedMemBarrier, _, address_space=.SHARED],
    bcast: Pointer[mut=True, SharedMemBarrier, _, address_space=.SHARED],
) -> Float32:
    """The row's sum from each warp's `s`: every CTA's warp sums into rank
    0's mailbox, rank 0 sums them in the block tier's order and sends the
    total to every CTA."""
    var lane = lane_id()
    if thread_idx.x == 0:
        gather[].arrive_and_expect_bytes(
            Int32(4 * (WIDE_WARPS // SPLIT)), UInt32(0), UInt32(1)
        )
    if lane == 0:
        _st_async(mail, at, 0, s, gather)
    # the component's rank-0 reduce warp
    if rank == 0 and warp_id() == 0:
        gather[].wait_acquire[Scope.CLUSTER](0)
        var total = block_sum(mail[unsafe_offset=lane])
        if lane < SPLIT:
            bcast[].arrive_and_expect_bytes(4, UInt32(lane), UInt32(1))
        if lane == 0:
            comptime for peer in range(SPLIT):
                _st_async(tot, 0, peer, total, bcast)
    bcast[].wait_acquire[Scope.CLUSTER](0)
    return tot[unsafe_offset=0]


@__name(t"gated_residual_split_{scale}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SPLIT_THREADS))
)
def _gated_residual_split[scale: Bool](a: GatedResidualArgs):
    var rank = Int(block_rank_in_cluster())
    var t = block_idx.x // SPLIT
    var c = (rank * SPLIT_THREADS + thread_idx.x) * VEC
    var at = rank * (WIDE_WARPS // SPLIT) + warp_id()
    var bars = unsafe_stack_allocation[
        4, SharedMemBarrier, address_space=.SHARED
    ]()
    var mail1 = unsafe_stack_allocation[
        WIDE_WARPS, Float32, address_space=.SHARED
    ]()
    var mail2 = unsafe_stack_allocation[
        WIDE_WARPS, Float32, address_space=.SHARED
    ]()
    var tot1 = unsafe_stack_allocation[1, Float32, address_space=.SHARED]()
    var tot2 = unsafe_stack_allocation[1, Float32, address_space=.SHARED]()
    if thread_idx.x == 0:
        bars[unsafe_offset=0].init(SPLIT)
        bars[unsafe_offset=1].init(1)
        bars[unsafe_offset=2].init(SPLIT)
        bars[unsafe_offset=3].init(1)
        fence_mbarrier_init()
    var scale_w = Piece(1)
    comptime if scale:
        scale_w = 1 + a.post_weight.unsafe_load[width=VEC, alignment=16](c)
    var gamma = 1 + a.norm_weight.unsafe_load[width=VEC, alignment=16](c)
    var ab = gate_coefficients(
        a.gate.unsafe_load[width=VEC, alignment=16](c),
        1 / a.temperature,
        a.gate_eps,
    )
    wait_on_dependent_grids()
    # every CTA's barriers are initialized before any CTA arrives on a peer's
    cluster_sync()

    var x = scaled_branch[scale](
        a.branch.unsafe_load[width=VEC, alignment=16](t * HIDDEN + c), scale_w
    )
    var r = a.residual.unsafe_load[width=VEC, alignment=16](t * HIDDEN + c)
    var f = x.cast[DType.float32]()
    var s1 = warp.sum((f * f).reduce_add())
    var inv1 = inv_rms(
        _cluster_total(
            s1,
            at,
            rank,
            mail1,
            tot1,
            bars.unsafe_offset(0),
            bars.unsafe_offset(1),
        ),
        a.post_eps,
    )
    var h = mix(x, r, ab[0], ab[1], inv1)
    a.stream.unsafe_store[alignment=16](t * HIDDEN + c, h)
    var g = h.cast[DType.float32]()
    var s2 = warp.sum((g * g).reduce_add())
    var inv2 = inv_rms(
        _cluster_total(
            s2,
            at,
            rank,
            mail2,
            tot2,
            bars.unsafe_offset(2),
            bars.unsafe_offset(3),
        ),
        a.norm_eps,
    )
    a.normed.unsafe_store[alignment=16](
        t * HIDDEN + c, next_norm(h, inv2, gamma)
    )
    launch_dependent_grids()


def gated_residual_rms_norm_split[
    scale: Bool
](a: GatedResidualArgs, ctx: DeviceContext) raises:
    """Writes the gated residual stream and its next pre-norm for fewer
    rows than SMs: a cluster of `SPLIT` CTAs per row, each CTA's warp sums
    gathered into rank 0 and its total sent back, summed in `rms_norm`'s
    1024-thread block tier's order so the bits match the unfused ops."""
    var rows = Int(a.rows)
    if rows <= 0:
        return
    ctx.enqueue_function[_gated_residual_split[scale]](
        a,
        grid_dim=rows * SPLIT,
        block_dim=SPLIT_THREADS,
        cluster_dim=OptionalReg(Dim(SPLIT, 1, 1)),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
