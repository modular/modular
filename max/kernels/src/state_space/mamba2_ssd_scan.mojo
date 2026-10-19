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

"""Mamba-2 SSD (state-space duality) varlen scan with an in-place state pool.

This is the varlen (ragged, `query_start_loc`) Mamba-2 SSD forward op, matching
`mamba_chunk_scan_combined` semantics for the `NemotronHMamba2Mixer`
(Nemotron-H). It serves both prefill and decode. It differs from the Mamba-1
ops in `varlen_selective_scan.mojo` in three ways that the math requires:

  - `A` is a per-head SCALAR `(nheads,)` (shared across all head_dim channels
    and all dstate), not a per-channel `(dim, dstate)` diagonal.
  - `B`/`C` are GROUPED `(total_len, ngroups, dstate)`; `nheads/ngroups` heads
    share each group (`group_id = h // (nheads // ngroups)`).
  - `dt` is per-head `(total_len, nheads)` + per-head `dt_bias (nheads,)`,
    broadcast across head_dim; softplus is applied to `dt + dt_bias`.

The SSD chunked scan is a parallelism reformulation of the linear recurrence
below; in fp32 they are numerically equivalent. The kernels carry the state
sequentially per `(head, head_dim)` channel.

Per-token recurrence (per head `h`, head_dim channel `p`, group `g`):

    dt_t     = softplus(dt[t, h] + dt_bias[h])            # scalar per (t, h)
    dA_t     = exp(A[h] * dt_t)                           # scalar per (t, h)
    state_n  = state_n * dA_t + dt_t * B[t, g, n] * x[t, h, p]   # vector over n
    y[t,h,p] = sum_n C[t, g, n] * state_n  +  D[h] * x[t, h, p]

Each sequence starts from zero, or from its pool slot when
`has_initial_state[b]`, and its final state is written back to the same slot.
`D`, `dt_bias` and `has_initial_state` may be empty to drop that term. The
dstate axis of `ssm_pool` must be contiguous.

`update_state[b]` (empty means all true) says whether sequence `b`'s final
state is written back. With it clear the scan only reads the pool, which lets a
speculative verify run on the live state and leave it as it was.

`x`, `dt`, `B`, `C` and the slot indices are read through operand functions,
so a graph op can fold a slice of a larger tensor into its loads instead of
copying it first.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.sync import barrier
from max.algorithm import sync_parallelize
from std.math import ceildiv, exp2, fma
from std.memory import unsafe_stack_allocation
from std.sys.info import align_of
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple
from layout import TensorLayout, TensorEngine, TileTensor
from state_space.selective_scan import softplus

# exp(x) == exp2(x * LOG2E), and exp2 is the faster GPU instruction.
comptime LOG2E = 1.4426950408889634


@inline(.always)
def _strides[rank: Int](t: TileTensor[...]) -> IndexList[rank]:
    """Returns the strides of `t`, read from its layout."""
    var result = IndexList[rank]()
    comptime for i in range(rank):
        result[i] = Int(t.dynamic_stride(i))
    return result


@fieldwise_init
struct _HeadScalars(TrivialRegisterPassable):
    """The per-head scalars of the recurrence, widened to fp32.

    An empty `D` or `dt_bias` loads as zero, which drops that term.
    """

    var A: Float32
    """`A[h]` pre-scaled by `LOG2E` so the decay is a single `exp2`."""
    var dt_bias: Float32
    var D: Float32

    @staticmethod
    @inline(.always)
    def load[
        dtype: DType
    ](
        A: TileTensor[dtype, ...],
        D: TileTensor[dtype, ...],
        dt_bias: TileTensor[dtype, ...],
        h: Int,
    ) -> Self:
        var dt_bias_val = Float32(0)
        if Int(dt_bias.dim[0]()) > 0:
            dt_bias_val = dt_bias.load[width=1]((h,)).cast[.float32]()
        var D_val = Float32(0)
        if Int(D.dim[0]()) > 0:
            D_val = D.load[width=1]((h,)).cast[.float32]()
        return Self(
            A.load[width=1]((h,)).cast[.float32]() * LOG2E, dt_bias_val, D_val
        )

    @inline(.always)
    def delta[dt_softplus: Bool](self, dt: Float32) -> Float32:
        """Returns the step size for one token's raw `dt`."""
        var delta = dt + self.dt_bias
        comptime if dt_softplus:
            delta = softplus(delta)
        return delta

    @inline(.always)
    def decay(self, delta: Float32) -> Float32:
        """Returns `exp(A * delta)`."""
        return exp2(self.A * delta)


@inline(.always)
def _group_id(y: TileTensor[...], ngroups: Int32, h: Int) -> Int:
    """Returns the `B`/`C` group that head `h` reads."""
    # 64-bit division is a long software sequence on GPUs, and the decode
    # kernels run this once per thread for a single token.
    var heads_per_group = UInt32(y.dim[1]()) // UInt32(ngroups)
    return Int(UInt32(h) // heads_per_group)


@inline(.always)
def _use_initial_state(
    has_initial_state: TileTensor[DType.bool, ...], b: Int
) -> Bool:
    return Int(has_initial_state.dim[0]()) > 0 and Bool(has_initial_state[b])


@inline(.always)
def _write_final_state(
    update_state: TileTensor[DType.bool, ...], b: Int
) -> Bool:
    """Whether sequence `b` writes its final state back to the pool.

    An empty `update_state` means every sequence does. Unlike an empty
    `has_initial_state`, which means none loads a state.
    """
    return Int(update_state.dim[0]()) == 0 or Bool(update_state[b])


def _scatter_width(n: Int, lanes: Int) -> Int:
    """Returns how many of `n` sums each of `lanes` lanes keeps."""
    return n // lanes if n >= lanes else 1


@inline(.always)
def _scatter_sum[
    n: Int, lanes: Int
](v: SIMD[.float32, n], lane: Int) -> SIMD[.float32, _scatter_width(n, lanes)]:
    """Sums `v` across a `lanes`-lane group and splits the `n` sums over it.

    With `W = _scatter_width(n, lanes)`, lane `l` ends with elements
    `[(l % lanes) * W, (l % lanes + 1) * W)` when `n >= lanes`, and with
    element `l % n` otherwise. A butterfly reduce-scatter: it needs `n - 1`
    shuffles for `n` sums over `n` lanes where `n` separate
    `lane_group_sum`s need `n * log2(n)`.
    """
    comptime W = _scatter_width(n, lanes)
    comptime if lanes == 1:
        return rebind[SIMD[.float32, W]](v)
    elif n < lanes:
        # More lanes than sums: fold the top lane bit with a full butterfly
        # step, so every pair of lanes holds the same partial sums.
        comptime half_lanes = lanes // 2
        var s = v
        comptime for i in range(n):
            s[i] += shuffle_xor(v[i], UInt32(half_lanes))
        return rebind[SIMD[.float32, W]](_scatter_sum[n, half_lanes](s, lane))
    else:
        comptime half = n // 2
        comptime bit = lanes // 2
        var lo = v.slice[half, offset=0]()
        var hi = v.slice[half, offset=half]()
        var upper = (lane & bit) != 0
        var keep = SIMD[.float32, half](0)
        comptime for i in range(half):
            var send = lo[i] if upper else hi[i]
            keep[i] = (hi[i] if upper else lo[i]) + shuffle_xor(
                send, UInt32(bit)
            )
        return rebind[SIMD[.float32, W]](_scatter_sum[half, bit](keep, lane))


# NVIDIA B200 (sm_100) launch bounds. This kernel is only ever launched with a
# 128-thread block (production kernels.mojo and the unit test both use
# `block_dim=(DSTATE_SPLIT, 128 // DSTATE_SPLIT, 1)`), so `.maxntid 128` is
# exact.
#
# The kernel is latency-bound, so resident CTAs matter. The floor depends on
# the `L * CH_PER_THREAD` state elements a thread holds. At 16 (the split-8,
# one-channel tile) `.minnctapersm 7` caps it at 72 registers, which it meets
# without spills; a floor of 8 (64 registers) made the staged prefill path
# slower on single long sequences and 16 x 512 batches. At 32 (the split-16,
# four-channel dstate-128 tile) a floor of 4 (128 registers) was fastest on
# B200: 3 and 5 were 10-20% slower at batch-64 verify. Larger tiles keep
# `.minnctapersm 1`. This is a register allocation hint only; the output is
# bit-identical to the un-annotated kernel.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(128))
)
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(7) if (DSTATE // DSTATE_SPLIT) * CH_PER_THREAD
    <= 16 else (
        SIMDLength(4) if (DSTATE // DSTATE_SPLIT) * CH_PER_THREAD
        <= 32 else SIMDLength(1)
    )
)
def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
    kernel_dtype: DType,
    state_dtype: DType,
    DSTATE: Int,
    DSTATE_SPLIT: Int,
    A_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    update_state_LT: TensorLayout,
    # All operands come from one source (graph tensors in production, device
    # buffers in the tests), so a single engine binds every tile argument.
    Engine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    DtFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[kernel_dtype, width],
    BFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    CFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    dt_softplus: Bool = True,
    prefill: Bool = True,
    CH_PER_THREAD: Int = 1,
](
    ngroups: Int32,
    A: TileTensor[kernel_dtype, A_LT, ImmUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, ImmUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    ssm_pool: TileTensor[
        state_dtype, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    update_state: TileTensor[
        .bool, update_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    x_fn: XFn,
    dt_fn: DtFn,
    b_fn: BFn,
    c_fn: CFn,
    slot_fn: SlotFn,
):
    """GPU kernel: Mamba-2 SSD varlen in-place scan, cooperative DSTATE-split.

    The production kernel on CUDA and HIP GPUs. `DSTATE_SPLIT` threads
    cooperate on each `(h, p)` channel: thread `tx` owns the contiguous dstate
    sub-tile `[tx*L, (tx+1)*L)` with `L = DSTATE // DSTATE_SPLIT`. A single
    thread per channel left decode at batch 1 at ~4% achieved occupancy on
    B200. Each thread carries `CH_PER_THREAD` channels `p0 + i * ROWS` with
    `ROWS = 128 // DSTATE_SPLIT`: `B` and `C` depend only on the token, the
    group and the dstate slice, so every B/C vector a thread reads (from
    shared memory on the staged path) feeds that many channels.

    The recurrence is data-parallel over dstate, so each thread runs it on its
    own `L` lanes. The output `y = sum_n state[n]*C[n]` is a per-thread partial
    `reduce_add` followed by a butterfly reduce-scatter over the
    `DSTATE_SPLIT`-lane group. `DSTATE_SPLIT` divides 32 and `tx` is the fastest-varying thread
    index, so each group stays inside one warp on both warp-32 and
    wavefront-64 devices. Each thread writes back only its own lanes of the
    pool.

    Grid: `(ceildiv(head_dim, ROWS * CH_PER_THREAD), nheads, batch)`, block:
    `(DSTATE_SPLIT, ROWS, 1)`.

    Parameters:
        kernel_dtype: Element type of `x`, `dt`, `A`, `B`, `C`, `D`,
            `dt_bias` and `y`.
        state_dtype: Storage dtype of `ssm_pool`; the scan accumulates in
            fp32 and rounds once on the final write-back.
        DSTATE: State dimension per head.
        DSTATE_SPLIT: Threads cooperating on one channel's dstate recurrence.
        A_LT: Tensor layout of `A`.
        D_LT: Tensor layout of `D`.
        dt_bias_LT: Tensor layout of `dt_bias`.
        y_LT: Tensor layout of `y`.
        ssm_pool_LT: Tensor layout of `ssm_pool`.
        query_start_loc_LT: Tensor layout of `query_start_loc`.
        has_initial_state_LT: Tensor layout of `has_initial_state`.
        update_state_LT: Tensor layout of `update_state`.
        Engine: Engine shared by all tile operands.
        XFn: Reads `x`. Each operand function loads `width` logically
            consecutive elements of the last index, claiming `alignment`
            elements of alignment.
        DtFn: Reads `dt`.
        BFn: Reads `B`.
        CFn: Reads `C`.
        SlotFn: Reads the pool slot of each sequence.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.
        prefill: Whether to enable the staged path, which every sequence
            of the launch takes, a short last chunk masked. Single-token
            decode launches turn it off because its shared memory and
            barriers would slow them down.
        CH_PER_THREAD: Number of head_dim channels each thread carries. The B/C
            vectors of a token depend only on the group and dstate slice, so
            each load feeds this many channels.

    Args:
        ngroups: Number of `B`/`C` groups; `nheads // ngroups` heads share
            each group.
        A: Per-head scalar decay of shape `(nheads,)`.
        D: Per-head skip connection of shape `(nheads,)`, or empty.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`, or empty.
        y: Output of shape `(total_len, nheads, head_dim)`.
        ssm_pool: State pool of shape `(max_slots, nheads, head_dim,
            dstate)`, read and written in place at each sequence's slot.
        query_start_loc: Cumulative sequence offsets of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flag of shape `(batch,)` selecting
            whether to load the initial state from the pool, or empty.
        update_state: Per-sequence flag of shape `(batch,)` selecting whether
            the final state is written back to the pool, or empty to write
            every sequence's. A sequence with the flag clear leaves its slot
            untouched, so the scan reads the pool without advancing it.
        x_fn: Reads `x` of shape `(total_len, nheads, head_dim)`.
        dt_fn: Reads `dt` of shape `(total_len, nheads)`.
        b_fn: Reads `B` of shape `(total_len, ngroups, dstate)`.
        c_fn: Reads `C` of shape `(total_len, ngroups, dstate)`.
        slot_fn: Reads the pool slot of each sequence, `(batch,)` in
            `uint32`.
    """
    comptime L = DSTATE // DSTATE_SPLIT
    comptime P = CH_PER_THREAD
    # Channel rows of the 128-thread block, and channels it covers.
    comptime NY = 128 // DSTATE_SPLIT
    comptime CH = NY * P
    comptime assert DSTATE_SPLIT * NY == 128, "the block has 128 threads"
    # The staged path is sized for DSTATE == 128: a chunk of T tokens gives
    # each of the 128 threads one 8-wide B vector and one C vector, and each
    # 8-wide vector lands inside a single thread's L-wide sub-tile.
    comptime STAGED = prefill and DSTATE == 128 and L % 8 == 0
    comptime T = 8
    # `n_base` and DSTATE are multiples of L, so each sub-tile is L-aligned.
    comptime pool_align = align_of[SIMD[state_dtype, L]]()
    # Each lane of a DSTATE_SPLIT group keeps W of the T*P (staged) or P
    # (per-token) channel sums after the reduce-scatter.
    comptime W_STAGED = _scatter_width(T * P, DSTATE_SPLIT)
    comptime W_TOKEN = _scatter_width(P, DSTATE_SPLIT)
    var head_dim = Int(y.dim[2]())

    var tx = thread_idx.x
    var ty = thread_idx.y
    var h = block_idx.y
    var b = block_idx.z
    # Thread (tx, ty) owns channels `p0 + i * NY` for i in [0, P), so every
    # B/C vector it loads feeds P channels.
    var p0 = block_idx.x * CH + ty

    # Out-of-range channel threads must not return early: the lane shuffles
    # need every lane of the warp. Only their loads and stores are guarded.
    var n_base = tx * L
    var group_id = _group_id(y, ngroups, h)

    var seq_start = Int(query_start_loc[b])
    var seq_end = Int(query_start_loc[b + 1])
    if seq_end <= seq_start:
        return

    var head = _HeadScalars.load(A, D, dt_bias, h)
    var slot = Int(slot_fn[1, 1](b))
    # Read the flag outside the `active` guard so its load issues alongside
    # the others; behind the guard it serialized and slowed decode by 14%.
    var use_initial_state = _use_initial_state(has_initial_state, b)
    var write_state = _write_final_state(update_state, b)
    # The state loads issue first so their HBM latency overlaps the B/C/x
    # loads and staging below.
    var state = Array[SIMD[.float32, L], P](fill=0)
    if use_initial_state:
        comptime for i in range(P):
            if p0 + i * NY < head_dim:
                state[i] = ssm_pool.load[width=L, alignment=pool_align](
                    (slot, h, p0 + i * NY, n_base)
                ).cast[.float32]()

    comptime if STAGED:
        # Every thread of a block would re-read and re-convert the same B/C
        # rows and recompute the same dt, so each chunk of T tokens is staged
        # once per block, double buffered so the next chunk's global loads
        # overlap this chunk's scan. A short last chunk is masked, so every
        # multi-token sequence (a speculative verify or replay) takes this
        # path. B/C are stored as fp32 float4 quads grouped by quad index so
        # the per-thread float4 reads do not conflict.
        comptime B_QUADS = DSTATE_SPLIT * 4
        comptime X_PER_THREAD = ceildiv(T * CH, 128)
        # One shared buffer, carved into the double-buffered B, C, x, dA
        # and delta tiles.
        comptime BC_SIZE = 2 * T * DSTATE
        var smem = unsafe_stack_allocation[
            2 * BC_SIZE + 2 * T * CH + 4 * T,
            DType.float32,
            address_space=.SHARED,
            alignment=16,
        ]()
        var B_s = smem
        var C_s = smem.unsafe_offset(BC_SIZE)
        var x_s = smem.unsafe_offset(2 * BC_SIZE)
        var dA_s = x_s.unsafe_offset(2 * T * CH)
        var delta_s = dA_s.unsafe_offset(2 * T)
        var tid = ty * DSTATE_SPLIT + tx
        var t_w = tid // (DSTATE // 8)
        var n_w = (tid % (DSTATE // 8)) * 8
        var quad_w = (n_w % L) // 4 * B_QUADS + (n_w // L) * 4
        var B_w = SIMD[kernel_dtype, 8](0)
        var C_w = SIMD[kernel_dtype, 8](0)
        var x_w = SIMD[kernel_dtype, X_PER_THREAD](0)
        var dt_w = Scalar[kernel_dtype](0)
        var num_chunks = ceildiv(seq_end - seq_start, T)
        for c in range(num_chunks + 1):
            var next = seq_start + c * T
            if c < num_chunks:
                B_w = 0
                C_w = 0
                x_w = 0
                if next + t_w < seq_end:
                    B_w = b_fn[8, 8](next + t_w, group_id, n_w)
                    C_w = c_fn[8, 8](next + t_w, group_id, n_w)
                comptime for k in range(X_PER_THREAD):
                    var idx = tid + k * 128
                    var p_x = block_idx.x * CH + idx % CH
                    if (
                        idx < T * CH
                        and next + idx // CH < seq_end
                        and p_x < head_dim
                    ):
                        x_w[k] = x_fn[1, 1](next + idx // CH, h, p_x)
                if tid < T and next + tid < seq_end:
                    dt_w = dt_fn[1, 1](next + tid, h)
            if c > 0:
                var cur = seq_start + (c - 1) * T
                var n_tok = min(T, seq_end - cur)
                var buf = ((c - 1) % 2) * T
                # Rows past the end of the sequence are staged as no-ops
                # (dA = 1, B = x = 0), so the chunk runs branch free.
                # part[i * T + t] is channel i's partial y of token t.
                var part = SIMD[.float32, T * P](0)
                comptime for t in range(T):
                    var B_vals = SIMD[.float32, L](0)
                    var C_vals = SIMD[.float32, L](0)
                    comptime for q in range(L // 4):
                        var off = (buf + t) * DSTATE + q * B_QUADS + tx * 4
                        B_vals = B_vals.insert[offset=q * 4](
                            B_s.unsafe_load[width=4, alignment=16](off)
                        )
                        C_vals = C_vals.insert[offset=q * 4](
                            C_s.unsafe_load[width=4, alignment=16](off)
                        )
                    var delta = delta_s[buf + t]
                    var dA = dA_s[buf + t]
                    comptime for i in range(P):
                        var dt_x = delta * x_s[(buf + t) * CH + i * NY + ty]
                        state[i] = fma(
                            B_vals, SIMD[.float32, L](dt_x), state[i] * dA
                        )
                        part[i * T + t] = (state[i] * C_vals).reduce_add()
                var y_vals = _scatter_sum[T * P, DSTATE_SPLIT](part, tx)
                comptime for j in range(W_STAGED):
                    var k: Int
                    comptime if T * P >= DSTATE_SPLIT:
                        k = tx * W_STAGED + j
                    else:
                        k = tx % (T * P)
                    var i = k // T
                    var t = k % T
                    var p = p0 + i * NY
                    if t < n_tok and p < head_dim and tx * W_STAGED < T * P:
                        y.store(
                            (cur + t, h, p),
                            (
                                y_vals[j]
                                + head.D * x_s[(buf + t) * CH + i * NY + ty]
                            ).cast[kernel_dtype](),
                        )
            if c < num_chunks:
                var buf = (c % 2) * T
                comptime for i in range(2):
                    var off = (buf + t_w) * DSTATE + quad_w + i * B_QUADS
                    B_s.unsafe_store[alignment=16](
                        off, B_w.slice[4, offset=i * 4]().cast[.float32]()
                    )
                    C_s.unsafe_store[alignment=16](
                        off, C_w.slice[4, offset=i * 4]().cast[.float32]()
                    )
                comptime for k in range(X_PER_THREAD):
                    var idx = tid + k * 128
                    if idx < T * CH:
                        x_s[buf * CH + idx] = x_w[k].cast[.float32]()
                if tid < T:
                    var delta = Float32(0)
                    var dA = Float32(1)
                    if next + tid < seq_end:
                        delta = head.delta[dt_softplus](dt_w.cast[.float32]())
                        dA = head.decay(delta)
                    delta_s[buf + tid] = delta
                    dA_s[buf + tid] = dA
            barrier()
    else:
        for gt in range(seq_start, seq_end):
            var delta = head.delta[dt_softplus](
                dt_fn[1, 1](gt, h).cast[.float32]()
            )
            var dA = head.decay(delta)
            # Each load claims the alignment of its width, which a fused
            # input clamps to what its own view proves.
            var B_vals = b_fn[L, L](gt, group_id, n_base).cast[.float32]()
            var C_vals = c_fn[L, L](gt, group_id, n_base).cast[.float32]()
            var x_vals = SIMD[.float32, P](0)
            var part = SIMD[.float32, P](0)
            comptime for i in range(P):
                if p0 + i * NY < head_dim:
                    x_vals[i] = x_fn[1, 1](gt, h, p0 + i * NY).cast[.float32]()
                state[i] = fma(
                    B_vals, SIMD[.float32, L](delta * x_vals[i]), state[i] * dA
                )
                part[i] = (state[i] * C_vals).reduce_add()
            var y_vals = _scatter_sum[P, DSTATE_SPLIT](part, tx)
            comptime for j in range(W_TOKEN):
                var i: Int
                comptime if P >= DSTATE_SPLIT:
                    i = tx * W_TOKEN + j
                else:
                    i = tx % P
                var p = p0 + i * NY
                if tx * W_TOKEN < P and p < head_dim:
                    y.store(
                        (gt, h, p),
                        (y_vals[j] + head.D * x_vals[i]).cast[kernel_dtype](),
                    )

    if write_state:
        comptime for i in range(P):
            if p0 + i * NY < head_dim:
                ssm_pool.store[alignment=pool_align](
                    (slot, h, p0 + i * NY, n_base), state[i].cast[state_dtype]()
                )


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
    kernel_dtype: DType,
    state_dtype: DType,
    DSTATE: Int,
    A_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    update_state_LT: TensorLayout,
    Engine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    DtFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[kernel_dtype, width],
    BFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    CFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    dt_softplus: Bool = True,
    # 16 bytes of fp32 is the M5 vector-load cap. A width-8 bf16 load
    # scalarizes on M5, so 4 is the width that is safe for both state dtypes.
    VEC: Int = 4,
](
    ngroups: Int32,
    A: TileTensor[kernel_dtype, A_LT, ImmUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, ImmUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    ssm_pool: TileTensor[
        state_dtype, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    update_state: TileTensor[
        .bool, update_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    x_fn: XFn,
    dt_fn: DtFn,
    b_fn: BFn,
    c_fn: CFn,
    slot_fn: SlotFn,
):
    """GPU kernel: Mamba-2 SSD varlen in-place scan, one thread per channel.

    The production kernel on Apple silicon GPUs. Each thread owns one
    `(b, h, p)` channel and moves its dstate run in `VEC`-wide loads and
    stores, keeping the state as `NCHUNK = DSTATE // VEC` fp32 chunks. Scalar
    dstate loads left the scan latency bound on M5, and the cooperative split
    of the CUDA kernel buys nothing here because decode already fills the GPU.

    `state_dtype` is the pool storage dtype only. bf16 halves the pool
    traffic that dominates a decode step; loads widen to fp32 and the state is
    rounded once, at the final write-back. The recurrence is contractive
    (`dA < 1`), so the per-step rounding across decode steps does not compound.

    Grid: `(ceildiv(head_dim, BLOCK), nheads, batch)`, block: `(BLOCK, 1, 1)`.

    Parameters:
        kernel_dtype: Element type of `x`, `dt`, `A`, `B`, `C`, `D`,
            `dt_bias` and `y`.
        state_dtype: Storage dtype of `ssm_pool` (fp32 or bf16).
        DSTATE: State dimension per head.
        A_LT: Tensor layout of `A`.
        D_LT: Tensor layout of `D`.
        dt_bias_LT: Tensor layout of `dt_bias`.
        y_LT: Tensor layout of `y`.
        ssm_pool_LT: Tensor layout of `ssm_pool`.
        query_start_loc_LT: Tensor layout of `query_start_loc`.
        has_initial_state_LT: Tensor layout of `has_initial_state`.
        update_state_LT: Tensor layout of `update_state`.
        Engine: Engine shared by all tile operands.
        XFn: Reads `x`. Each operand function loads `width` logically
            consecutive elements of the last index, claiming `alignment`
            elements of alignment.
        DtFn: Reads `dt`.
        BFn: Reads `B`.
        CFn: Reads `C`.
        SlotFn: Reads the pool slot of each sequence.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.
        VEC: SIMD width of the dstate loads and stores.

    Args:
        ngroups: Number of `B`/`C` groups; `nheads // ngroups` heads share
            each group.
        A: Per-head scalar decay of shape `(nheads,)`.
        D: Per-head skip connection of shape `(nheads,)`, or empty.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`, or empty.
        y: Output of shape `(total_len, nheads, head_dim)`.
        ssm_pool: State pool of shape `(max_slots, nheads, head_dim,
            dstate)`, read and written in place at each sequence's slot.
        query_start_loc: Cumulative sequence offsets of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flag of shape `(batch,)` selecting
            whether to load the initial state from the pool, or empty.
        update_state: Per-sequence flag of shape `(batch,)` selecting whether
            the final state is written back to the pool, or empty to write
            every sequence's. A sequence with the flag clear leaves its slot
            untouched, so the scan reads the pool without advancing it.
        x_fn: Reads `x` of shape `(total_len, nheads, head_dim)`.
        dt_fn: Reads `dt` of shape `(total_len, nheads)`.
        b_fn: Reads `B` of shape `(total_len, ngroups, dstate)`.
        c_fn: Reads `C` of shape `(total_len, ngroups, dstate)`.
        slot_fn: Reads the pool slot of each sequence, `(batch,)` in
            `uint32`.
    """
    comptime assert (
        DSTATE % VEC == 0
    ), "DSTATE must be a multiple of the SIMD I/O width VEC"
    comptime NCHUNK = DSTATE // VEC
    comptime pool_align = align_of[SIMD[state_dtype, VEC]]()

    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y
    var b = block_idx.z
    if p >= Int(y.dim[2]()):
        return
    var group_id = _group_id(y, ngroups, h)

    var seq_start = Int(query_start_loc[b])
    var seq_end = Int(query_start_loc[b + 1])
    if seq_end <= seq_start:
        return

    var head = _HeadScalars.load(A, D, dt_bias, h)
    var slot = Int(slot_fn[1, 1](b))
    var state = Array[SIMD[.float32, VEC], NCHUNK](fill=0)
    if _use_initial_state(has_initial_state, b):
        comptime for c in range(NCHUNK):
            state[c] = ssm_pool.load[width=VEC, alignment=pool_align](
                (slot, h, p, c * VEC)
            ).cast[.float32]()

    for gt in range(seq_start, seq_end):
        var x_val = x_fn[1, 1](gt, h, p).cast[.float32]()
        var delta = head.delta[dt_softplus](dt_fn[1, 1](gt, h).cast[.float32]())
        var dA = head.decay(delta)
        var dt_x = delta * x_val

        var y_acc = SIMD[.float32, VEC](0)
        comptime for c in range(NCHUNK):
            var B_c = b_fn[VEC, VEC](gt, group_id, c * VEC).cast[.float32]()
            var C_c = c_fn[VEC, VEC](gt, group_id, c * VEC).cast[.float32]()
            state[c] = state[c] * dA + B_c * dt_x
            y_acc += state[c] * C_c

        y.store(
            (gt, h, p),
            (y_acc.reduce_add() + head.D * x_val).cast[kernel_dtype](),
        )

    if _write_final_state(update_state, b):
        comptime for c in range(NCHUNK):
            ssm_pool.store[alignment=pool_align](
                (slot, h, p, c * VEC), state[c].cast[state_dtype]()
            )


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[
    kernel_dtype: DType,
    state_dtype: DType,
    //,
    DSTATE: Int,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    DtFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[kernel_dtype, width],
    BFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    CFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    dt_softplus: Bool = True,
](
    ngroups: Int,
    A: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    y: TileTensor[mut=True, kernel_dtype, ...],
    ssm_pool: TileTensor[mut=True, state_dtype, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    update_state: TileTensor[mut=False, .bool, ...],
    x_fn: XFn,
    dt_fn: DtFn,
    b_fn: BFn,
    c_fn: CFn,
    slot_fn: SlotFn,
    ctx: Optional[DeviceContext.CPU] = None,
):
    """CPU reference: Mamba-2 SSD varlen scan with an in-place state pool.

    Computes the same per-token recurrence as the GPU kernels in fp32,
    parallelized over `(b, h, p)`.

    Parameters:
        kernel_dtype: Element type of `x`, `dt`, `A`, `B`, `C`, `D`,
            `dt_bias` and `y`.
        state_dtype: Storage dtype of `ssm_pool`.
        DSTATE: State dimension per head.
        XFn: Reads `x`. Each operand function loads `width` logically
            consecutive elements of the last index, claiming `alignment`
            elements of alignment.
        DtFn: Reads `dt`.
        BFn: Reads `B`.
        CFn: Reads `C`.
        SlotFn: Reads the pool slot of each sequence.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.

    Args:
        ngroups: Number of `B`/`C` groups; `nheads // ngroups` heads share
            each group.
        A: Per-head scalar decay of shape `(nheads,)`.
        D: Per-head skip connection of shape `(nheads,)`, or empty.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`, or empty.
        y: Output of shape `(total_len, nheads, head_dim)`.
        ssm_pool: State pool of shape `(max_slots, nheads, head_dim,
            dstate)`, read and written in place at each sequence's slot.
        query_start_loc: Cumulative sequence offsets of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flag of shape `(batch,)` selecting
            whether to load the initial state from the pool, or empty.
        update_state: Per-sequence flag of shape `(batch,)` selecting whether
            the final state is written back to the pool, or empty to write
            every sequence's. A sequence with the flag clear leaves its slot
            untouched, so the scan reads the pool without advancing it.
        x_fn: Reads `x` of shape `(total_len, nheads, head_dim)`.
        dt_fn: Reads `dt` of shape `(total_len, nheads)`.
        b_fn: Reads `B` of shape `(total_len, ngroups, dstate)`.
        c_fn: Reads `C` of shape `(total_len, ngroups, dstate)`.
        slot_fn: Reads the pool slot of each sequence, `(batch,)` in
            `uint32`.
        ctx: Device context for the parallel worker pool.
    """
    var nheads = Int(y.dim[1]())
    var head_dim = Int(y.dim[2]())
    var batch = Int(query_start_loc.dim[0]()) - 1
    var nheads_per_group = nheads // ngroups

    def worker(idx: Int) {imm}:
        var b, remaining = divmod(idx, nheads * head_dim)
        var h, p = divmod(remaining, head_dim)
        var group_id = h // nheads_per_group

        var seq_start = Int(query_start_loc[b])
        var seq_end = Int(query_start_loc[b + 1])
        if seq_end <= seq_start:
            return

        var head = _HeadScalars.load(A, D, dt_bias, h)
        var slot = Int(slot_fn[1, 1](b))
        var state = SIMD[.float32, DSTATE](0)
        if _use_initial_state(has_initial_state, b):
            state = ssm_pool.load[width=DSTATE]((slot, h, p, 0)).cast[
                .float32
            ]()

        for gt in range(seq_start, seq_end):
            var x_val = x_fn[1, 1](gt, h, p).cast[.float32]()
            var delta = head.delta[dt_softplus](
                dt_fn[1, 1](gt, h).cast[.float32]()
            )
            var B_vals = b_fn[DSTATE, 1](gt, group_id, 0).cast[.float32]()
            var C_vals = c_fn[DSTATE, 1](gt, group_id, 0).cast[.float32]()
            state = state * head.decay(delta) + B_vals * (delta * x_val)
            var y_val = (state * C_vals).reduce_add() + head.D * x_val
            y.store((gt, h, p), y_val.cast[kernel_dtype]())

        if _write_final_state(update_state, b):
            ssm_pool.store((slot, h, p, 0), state.cast[state_dtype]())

    sync_parallelize(worker, batch * nheads * head_dim, ctx)
