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
from max.gpu.primitives.warp import lane_group_sum, shuffle_xor
from max.gpu.sync import barrier
from max.algorithm import sync_parallelize
from std.math import exp2, fma
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
        result[i] = Int(t.layout.stride[i]().value())
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
    return Int(has_initial_state.dim[0]()) > 0 and Bool(
        has_initial_state.load[width=1]((b,))
    )


@inline(.always)
def _scatter_sum[n: Int](v: SIMD[.float32, n], lane: Int) -> Float32:
    """Sums `v` across an `n`-lane group so that lane `i` ends with element `i`.

    A butterfly reduce-scatter: it needs `n - 1` shuffles for `n` sums where
    `n` separate `lane_group_sum`s need `n * log2(n)`.
    """
    comptime if n == 1:
        return v[0]
    else:
        comptime half = n // 2
        var lo = v.slice[half, offset=0]()
        var hi = v.slice[half, offset=half]()
        var upper = (lane & half) != 0
        var keep = hi if upper else lo
        var send = lo if upper else hi
        comptime for i in range(half):
            keep[i] += shuffle_xor(send[i], UInt32(half))
        return _scatter_sum[half](keep, lane)


# NVIDIA B200 (sm_100) launch bounds. This kernel is only ever launched with a
# 128-thread block (production kernels.mojo and the unit test both use
# `block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1)` with
# DSTATE_SPLIT*CH_PER_BLOCK == 128), so `.maxntid 128` is exact.
#
# The kernel is latency-bound, so resident CTAs matter. `.minnctapersm 7` caps
# it at 72 registers per thread (7 CTAs/SM), which the L == 16 production tile
# meets without spills. A floor of 8 (64 registers) made the staged prefill
# path slower on single long sequences and 16 x 512 batches.
#
# The floor is gated on `L <= 16`: for larger tiles (DSTATE == 256 at split 8,
# L == 32) the state/B/C vectors spill under that cap, so they keep
# `.minnctapersm 1`. This is a register allocation hint only; the output is
# bit-identical to the un-annotated kernel.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(128))
)
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(7) if (DSTATE // DSTATE_SPLIT)
    <= 16 else SIMDLength(1)
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
    B200.

    The recurrence is data-parallel over dstate, so each thread runs it on its
    own `L` lanes. The output `y = sum_n state[n]*C[n]` is a per-thread partial
    `reduce_add` followed by a `lane_group_sum` over the `DSTATE_SPLIT`-lane
    group. `DSTATE_SPLIT` divides 32 and `tx` is the fastest-varying thread
    index, so each group stays inside one warp on both warp-32 and
    wavefront-64 devices. Each thread writes back only its own lanes of the
    pool.

    Grid: `(ceildiv(head_dim, CH_PER_BLOCK), nheads, batch)`, block:
    `(DSTATE_SPLIT, CH_PER_BLOCK, 1)`.

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
        Engine: Engine shared by all tile operands.
        XFn: Reads `x`. Each operand function loads `width` logically
            consecutive elements of the last index, claiming `alignment`
            elements of alignment.
        DtFn: Reads `dt`.
        BFn: Reads `B`.
        CFn: Reads `C`.
        SlotFn: Reads the pool slot of each sequence.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.
        prefill: Whether to enable the staged long-sequence path. Decode
            launches turn it off because its shared memory and registers
            would slow them down.

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
        x_fn: Reads `x` of shape `(total_len, nheads, head_dim)`.
        dt_fn: Reads `dt` of shape `(total_len, nheads)`.
        b_fn: Reads `B` of shape `(total_len, ngroups, dstate)`.
        c_fn: Reads `C` of shape `(total_len, ngroups, dstate)`.
        slot_fn: Reads the pool slot of each sequence, `(batch,)` in
            `uint32`.
    """
    comptime L = DSTATE // DSTATE_SPLIT
    # The staged path is sized for the production tiling: a chunk of T tokens
    # gives each of the 128 threads one 8-wide B vector, one C vector and one x.
    comptime STAGED = prefill and DSTATE == 128 and DSTATE_SPLIT == 8
    comptime T = 8
    comptime CH = 16
    # `n_base` and DSTATE are multiples of L, so each sub-tile is L-aligned.
    comptime pool_align = align_of[SIMD[state_dtype, L]]()
    var y_strides = _strides[3](y)

    var tx = thread_idx.x
    var p = block_idx.x * block_dim.y + thread_idx.y
    var h = block_idx.y
    var b = block_idx.z

    # Out-of-range channel threads must not return early: `lane_group_sum`
    # needs every lane of the warp. Only their loads and stores are guarded.
    var active = p < Int(y.dim[2]())
    var n_base = tx * L
    var group_id = _group_id(y, ngroups, h)

    var seq_start = Int(query_start_loc.load[width=1]((b,)))
    var seq_end = Int(query_start_loc.load[width=1]((b + 1,)))
    if seq_end <= seq_start:
        return

    var head = _HeadScalars.load(A, D, dt_bias, h)
    var slot = Int(slot_fn[1, 1](b))
    # Read the flag outside the `active` guard so its load issues alongside
    # the others; behind the guard it serialized and slowed decode by 14%.
    var use_initial_state = _use_initial_state(has_initial_state, b)
    var state = SIMD[.float32, L](0)
    if active and use_initial_state:
        state = ssm_pool.load[width=L, alignment=pool_align](
            (slot, h, p, n_base)
        ).cast[.float32]()

    var gt = seq_start
    comptime if STAGED:
        # Every thread of a block would re-read and re-convert the same B/C
        # rows and recompute the same dt, so each chunk of T tokens is staged
        # once per block, double buffered so the next chunk's global loads
        # overlap this chunk's scan. B/C are stored as fp32 float4 quads
        # grouped by quad index so the per-thread float4 reads do not conflict.
        if seq_end - seq_start >= T:
            comptime B_QUADS = DSTATE_SPLIT * 4
            var B_s = unsafe_stack_allocation[
                2 * T * DSTATE, DType.float32, address_space=.SHARED
            ]()
            var C_s = unsafe_stack_allocation[
                2 * T * DSTATE, DType.float32, address_space=.SHARED
            ]()
            var x_s = unsafe_stack_allocation[
                2 * T * CH, DType.float32, address_space=.SHARED
            ]()
            var dA_s = unsafe_stack_allocation[
                2 * T, DType.float32, address_space=.SHARED
            ]()
            var delta_s = unsafe_stack_allocation[
                2 * T, DType.float32, address_space=.SHARED
            ]()
            var ty = thread_idx.y
            var tid = ty * DSTATE_SPLIT + tx
            var t_w = tid // (DSTATE // 8)
            var n_w = (tid % (DSTATE // 8)) * 8
            var quad_w = (n_w % L) // 4 * B_QUADS + (n_w // L) * 4
            var B_w = SIMD[kernel_dtype, 8](0)
            var C_w = SIMD[kernel_dtype, 8](0)
            var x_w = Scalar[kernel_dtype](0)
            var dt_w = Scalar[kernel_dtype](0)
            var p_w = block_idx.x * CH + tid % CH
            var num_chunks = (seq_end - seq_start) // T
            for c in range(num_chunks + 1):
                var next = seq_start + c * T
                if c < num_chunks:
                    B_w = b_fn[8, 8](next + t_w, group_id, n_w)
                    C_w = c_fn[8, 8](next + t_w, group_id, n_w)
                    if p_w < Int(y.dim[2]()):
                        x_w = x_fn[1, 1](next + tid // CH, h, p_w)
                    if tid < T:
                        dt_w = dt_fn[1, 1](next + tid, h)
                if c > 0:
                    var cur = seq_start + (c - 1) * T
                    var buf = ((c - 1) % 2) * T
                    var part = SIMD[.float32, T](0)
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
                        var dt_x = delta_s[buf + t] * x_s[(buf + t) * CH + ty]
                        state = fma(
                            B_vals,
                            SIMD[.float32, L](dt_x),
                            state * dA_s[buf + t],
                        )
                        part[t] = (state * C_vals).reduce_add()
                    # Lane `tx` finishes token `tx` of this chunk.
                    var y_val = _scatter_sum[T](part, tx)
                    if active:
                        y.raw_store(
                            UInt32(
                                (cur + tx) * y_strides[0]
                                + h * y_strides[1]
                                + p * y_strides[2]
                            ),
                            (y_val + head.D * x_s[(buf + tx) * CH + ty]).cast[
                                kernel_dtype
                            ](),
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
                    x_s[buf * CH + tid] = x_w.cast[.float32]()
                    if tid < T:
                        var delta = head.delta[dt_softplus](
                            dt_w.cast[.float32]()
                        )
                        delta_s[buf + tid] = delta
                        dA_s[buf + tid] = head.decay(delta)
                barrier()
            gt = seq_start + num_chunks * T

    while gt < seq_end:
        var x_val = Float32(0)
        var dt_x = Float32(0)
        var dA = Float32(0)
        var B_vals = SIMD[.float32, L](0)
        var C_vals = SIMD[.float32, L](0)
        if active:
            x_val = x_fn[1, 1](gt, h, p).cast[.float32]()
            var delta = head.delta[dt_softplus](
                dt_fn[1, 1](gt, h).cast[.float32]()
            )
            dA = head.decay(delta)
            dt_x = delta * x_val
            # Each load claims the alignment of its width, which a fused
            # input clamps to what its own view proves.
            B_vals = b_fn[L, L](gt, group_id, n_base).cast[.float32]()
            C_vals = c_fn[L, L](gt, group_id, n_base).cast[.float32]()

        state = state * dA + B_vals * dt_x

        var y_val = (state * C_vals).reduce_add()
        comptime if DSTATE_SPLIT > 1:
            y_val = lane_group_sum[num_lanes=DSTATE_SPLIT, stride=1](y_val)

        if active and tx == 0:
            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                (y_val + head.D * x_val).cast[kernel_dtype](),
            )
        gt += 1

    if active:
        ssm_pool.store[width=L, alignment=pool_align](
            (slot, h, p, n_base), state.cast[state_dtype]()
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

    var seq_start = Int(query_start_loc.load[width=1]((b,)))
    var seq_end = Int(query_start_loc.load[width=1]((b + 1,)))
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

        y.store[width=1](
            (gt, h, p),
            (y_acc.reduce_add() + head.D * x_val).cast[kernel_dtype](),
        )

    comptime for c in range(NCHUNK):
        ssm_pool.store[width=VEC, alignment=pool_align](
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
    x_fn: XFn,
    dt_fn: DtFn,
    b_fn: BFn,
    c_fn: CFn,
    slot_fn: SlotFn,
    ctx: Optional[DeviceContext] = None,
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

        var seq_start = Int(query_start_loc.load[width=1]((b,)))
        var seq_end = Int(query_start_loc.load[width=1]((b + 1,)))
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
            y.store[width=1]((gt, h, p), y_val.cast[kernel_dtype]())

        ssm_pool.store[width=DSTATE]((slot, h, p, 0), state.cast[state_dtype]())

    sync_parallelize(worker, batch * nheads * head_dim, ctx)
