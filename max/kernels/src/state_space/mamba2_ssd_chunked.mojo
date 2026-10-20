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

"""Chunked Mamba-2 SSD prefill scan (varlen, in-place state pool).

The serial kernel in `mamba2_ssd_scan.mojo` walks every token of a sequence
one after another, which leaves a long prefill latency bound. This module
computes the same recurrence with the chunked state-space-duality algorithm of
the Mamba-2 paper (the same decomposition as `mamba_chunk_scan_combined`):

Each sequence is cut into chunks of `CHUNK = 256` tokens. For a chunk, with
`a_t = A * dt_t` and `cs_i = sum_{j <= i} a_j` (the within-chunk cumulative
log decay, in log2 units):

  1. `ssd_chunk_cumsum`: `dt_t` (softplus of `dt + dt_bias`) and `cs`.
  2. `ssd_chunk_state`: the state each chunk adds on its own,
     `S_c[p, n] = sum_t exp(cs_last - cs_t) * dt_t * x[t, p] * B[t, n]`.
  3. `ssd_state_passing`: a short serial scan over the chunks of a sequence,
     `S'_c = exp(cs_last_c) * S'_{c-1} + S_c`, seeded from the pool slot when
     the sequence has an initial state. It stores the state entering each
     chunk and writes the final state back to the pool.
  4. `ssd_chunk_output`: `y = exp(cs_i) * C_i S'_{c-1}`
     `+ sum_{j <= i} (C_i . B_j) exp(cs_i - cs_j) dt_j x_j + D x_i`.
     The first term is the carry from earlier chunks and the second is the
     masked-decay attention inside the chunk.

The GEMMs run on tensor cores (`m16n8k16`, bf16 operands, fp32 accumulation).
`x`, `B` and `C` are already bf16; the decayed score matrix and the chunk
states are rounded to bf16 for their GEMMs, as in the Triton reference
kernels. The decays, `dt`, the cumulative sums and the state-passing scan stay
in fp32, and the pool is rounded once at the final write-back like the serial
kernel. The serial kernel stays in charge of decode, verify and short
launches.
"""

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.compute.mma import ld_matrix, mma
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import ceildiv, divmod, exp2, fma
from std.memory import unsafe_stack_allocation
from std.sys.info import align_of
from layout import Coord, TensorLayout, TensorEngine, TileTensor, row_major
from state_space.mamba2_ssd_scan import (
    _HeadScalars,
    _use_initial_state,
    _write_final_state,
)

comptime CHUNK = 256
"""Tokens per chunk."""
comptime TILE = 64
"""Head-dim and row tile of a thread block."""
comptime NUM_THREADS = 256
"""Threads of the state-passing kernel."""
comptime OUT_THREADS = 128
"""Threads of the tensor-core kernels (4 warps)."""
comptime NUM_WARPS = OUT_THREADS // WARP_SIZE
comptime MMA_M = 16
"""Rows of an `m16n8k16` MMA (and the rows of the tile each warp owns)."""
comptime MMA_N = 8
"""Columns of an `m16n8k16` MMA."""
comptime MMA_K = 16
"""Reduction depth of an `m16n8k16` MMA."""
comptime HALF = 8
"""Offset of the second half (rows `g + 8`, columns `+ 8`) of a fragment."""
comptime QUAD = 4
"""Lanes that share a fragment row; lane `l` holds row `l // QUAD`."""
comptime LDM_ROWS = 8
"""Rows of one 8x8 `ldmatrix` tile (lanes `l % 8` address its rows)."""
comptime VEC = 8
"""Elements of one 16-byte bf16 vector load or store."""
comptime F32_VEC = 4
"""Elements of one 16-byte fp32 vector load or store."""
comptime FRAG_C = 4
"""Accumulator values per lane of an `m16n8k16` MMA."""
comptime SMEM_PAD = 8
"""Row padding (bf16 elements) of the shared tiles, against bank conflicts."""
comptime MIN_AVG_SEQ_LEN = 128
"""Average sequence length from which the chunked scan beats the serial one."""
comptime GROUPED_CB_AUTO = 0
"""Selects grouped score reuse only for the validated amortization regime."""
comptime GROUPED_CB_OFF = 1
"""Disables grouped score precomputation."""
comptime GROUPED_CB_ON = 2
"""Forces grouped score precomputation for explicit comparisons."""


def chunked_slots(total_len: Int, batch: Int) -> Int:
    """Returns an upper bound on the number of chunks of a launch.

    Each sequence adds at most one partial chunk to `total_len / CHUNK`.
    """
    return ceildiv(total_len, CHUNK) + batch


def use_chunked_ssd(
    total_len: Int, batch: Int, head_dim: Int, min_tokens: Int
) -> Bool:
    """Returns whether a launch should take the chunked scan over the serial one.

    The chunk setup costs a fixed ~0.2 ms, so the launch needs at least
    `min_tokens` tokens in total and an average sequence of at least `MIN_AVG_SEQ_LEN` tokens:
    a batch of short verify or decode rows stays on the serial kernel.
    `head_dim` must be a multiple of `TILE`.
    """
    return head_dim % TILE == 0 and total_len >= max(
        min_tokens, MIN_AVG_SEQ_LEN * batch
    )


def use_grouped_cb_ssd(
    total_len: Int, batch: Int, nheads: Int, head_dim: Int, ngroups: Int
) -> Bool:
    """Returns shape eligibility for amortized grouped score precomputation.

    The caller separately gates the architecture, device name and dtypes.
    With head and state dimensions 128, eight heads per group bound the extra
    fp32 score scratch to half of the existing chunk-state scratch. Long
    average rows and sufficient head-token work amortize the additional
    allocation/launch.
    """
    if batch <= 0 or nheads <= 0 or ngroups <= 0 or head_dim != 2 * TILE:
        return False
    var heads_per_group, remainder = divmod(nheads, ngroups)
    return (
        remainder == 0
        and heads_per_group >= 8
        and total_len // batch >= 32 * CHUNK
        and total_len >= ceildiv(1024 * CHUNK, nheads)
    )


def ssd_chunk_map[
    SLT: TensorLayout,
    SEngine: TensorEngine,
    QLT: TensorLayout,
    QEngine: TensorEngine,
](
    slots_i: Int32,
    batch_i: Int32,
    info: TileTensor[.int32, SLT, MutUntrackedOrigin, Engine=SEngine],
    query_start_loc: TileTensor[
        .int32, QLT, ImmUntrackedOrigin, Engine=QEngine
    ],
):
    """Builds the chunk table of the launch (one thread).

    `info` holds, in order: `slots` chunk start tokens, `slots` chunk
    lengths (0 for an unused slot), `slots` owning sequences, then `batch`
    first-chunk slots and `batch` chunk counts.
    """
    var slots = Int(slots_i)
    var batch = Int(batch_i)
    if thread_idx.x != 0:
        return
    var next = 0
    for b in range(batch):
        var s = Int(query_start_loc[b])
        var e = Int(query_start_loc[b + 1])
        var n = ceildiv(max(e - s, 0), CHUNK)
        info.store((3 * slots + b,), Int32(next))
        info.store((3 * slots + batch + b,), Int32(n))
        for c in range(n):
            var start = s + c * CHUNK
            info.store((next,), Int32(start))
            info.store((slots + next,), Int32(min(CHUNK, e - start)))
            info.store((2 * slots + next,), Int32(b))
            next += 1
    for k in range(next, slots):
        info.store((slots + k,), Int32(0))


def ssd_chunk_cumsum[
    kernel_dtype: DType,
    A_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    Engine: TensorEngine,
    SLT: TensorLayout,
    SEngine: TensorEngine,
    DtFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[kernel_dtype, width],
    dt_softplus: Bool,
](
    slots_i: Int32,
    A: TileTensor[kernel_dtype, A_LT, ImmUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, ImmUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    info: TileTensor[.int32, SLT, MutUntrackedOrigin, Engine=SEngine],
    cs: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    delta_out: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    dt_fn: DtFn,
):
    """Writes `dt` and the inclusive cumulative log2 decay of every chunk.

    Grid `(slots, nheads)`, block `CHUNK` threads, one token each.
    """
    var slots = Int(slots_i)
    var slot = Int(block_idx.x)
    var h = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var length = Int(info[slots + slot])
    if length == 0:
        return
    var start = Int(info[slot])
    var head = _HeadScalars.load(A, D, dt_bias, h)

    var scan = unsafe_stack_allocation[
        CHUNK, DType.float32, address_space=.SHARED, alignment=16
    ]()
    var delta = Float32(0)
    var a = Float32(0)
    if i < length:
        delta = head.delta[dt_softplus](
            dt_fn[1, 1](start + i, h).cast[.float32]()
        )
        a = head.A * delta
    scan[i] = a
    barrier()
    # Hillis-Steele inclusive scan over the chunk.
    var off = 1
    while off < CHUNK:
        var v = scan[i]
        if i >= off:
            v += scan[i - off]
        barrier()
        scan[i] = v
        barrier()
        off *= 2
    var base = (h * slots + slot) * CHUNK + i
    cs.store((base,), scan[i])
    delta_out.store((base,), delta)


def ssd_chunk_state[
    kernel_dtype: DType,
    SLT: TensorLayout,
    SEngine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    BFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
    DSTATE: Int,
](
    slots_i: Int32,
    nheads_i: Int32,
    head_dim_i: Int32,
    ngroups_i: Int32,
    info: TileTensor[.int32, SLT, MutUntrackedOrigin, Engine=SEngine],
    cs: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    delta_in: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    states: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    x_fn: XFn,
    b_fn: BFn,
):
    """Computes the state each chunk contributes by itself, on tensor cores.

    `states[slot, h, p, n] = sum_t exp2(cs_last - cs_t) dt_t x[t, p] B[t, n]`.
    Grid `(slots, nheads, head_dim / TILE)`, block `OUT_THREADS` (4 warps, 16
    rows of `p` each). The reduction runs over `t` in tiles of 64 tokens that
    stay `[t][p]` and `[t][n]` in shared memory: `ldmatrix.trans` turns them
    into the MMA fragments.
    """
    comptime assert kernel_dtype == .bfloat16, "tensor-core path is bf16"
    comptime LDK = DSTATE + SMEM_PAD
    comptime LDX = TILE + SMEM_PAD
    comptime TT = 64
    comptime NT = DSTATE // MMA_N
    var slots = Int(slots_i)
    var nheads = Int(nheads_i)
    var head_dim = Int(head_dim_i)
    var ngroups = Int(ngroups_i)
    var slot = Int(block_idx.x)
    var h = Int(block_idx.y)
    var p_base = Int(block_idx.z) * TILE
    var tid = Int(thread_idx.x)
    var warp = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var g = lane // QUAD
    var t4 = lane % QUAD
    var length = Int(info[slots + slot])
    if length == 0:
        return
    var start = Int(info[slot])
    var group_id = h // (nheads // ngroups)
    var sbase = (h * slots + slot) * CHUNK
    var cs_last = cs.load[width=1]((sbase + length - 1,))

    var Xs = unsafe_stack_allocation[
        TT * LDX,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var Bs = unsafe_stack_allocation[
        TT * LDK,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var scale_s = unsafe_stack_allocation[
        CHUNK, DType.float32, address_space=.SHARED, alignment=16
    ]()
    comptime for r in range(CHUNK // OUT_THREADS):
        var k = tid + r * OUT_THREADS
        scale_s[k] = exp2(cs_last - cs.load[width=1]((sbase + k,))) * (
            delta_in.load[width=1]((sbase + k,))
        )
    barrier()

    var acc = Array[SIMD[.float32, FRAG_C], NT](fill=0)
    var m = lane // LDM_ROWS
    for t0 in range(0, length, TT):
        comptime for r in range(TT * TILE // VEC // OUT_THREADS):
            var e = tid + r * OUT_THREADS
            var t = e // (TILE // VEC)
            var p8 = (e % (TILE // VEC)) * VEC
            var v = SIMD[.float32, VEC](0)
            if t0 + t < length:
                v = (
                    x_fn[VEC, 1](start + t0 + t, h, p_base + p8).cast[
                        .float32
                    ]()
                    * scale_s[t0 + t]
                )
            Xs.unsafe_store[alignment=16](t * LDX + p8, v.cast[.bfloat16]())
        comptime for r in range(TT * DSTATE // VEC // OUT_THREADS):
            var e = tid + r * OUT_THREADS
            var t = e // (DSTATE // VEC)
            var n8 = (e % (DSTATE // VEC)) * VEC
            var v = SIMD[kernel_dtype, VEC](0)
            if t0 + t < length:
                v = b_fn[VEC, VEC](start + t0 + t, group_id, n8)
            Bs.unsafe_store[alignment=16](t * LDK + n8, v.cast[.bfloat16]())
        barrier()
        comptime for ks in range(TT // MMA_K):
            # A = X'^T (p x t) from the [t][p] tile.
            var a = ld_matrix[8, transpose=True](
                Xs.unsafe_offset(
                    (ks * MMA_K + lane % LDM_ROWS + (m // 2) * HALF) * LDX
                )
                .unsafe_offset(warp * MMA_M)
                .unsafe_offset((m % 2) * HALF)
            )
            comptime for np in range(NT // 2):
                # Two n8 tiles of B (t x n) per x4 load.
                var bq = ld_matrix[8, transpose=True](
                    Bs.unsafe_offset(
                        (ks * MMA_K + lane % LDM_ROWS + (m % 2) * HALF) * LDK
                    )
                    .unsafe_offset(np * MMA_K)
                    .unsafe_offset((m // 2) * HALF)
                )
                var b0 = SIMD[.bfloat16, 4](bq[0], bq[1], bq[2], bq[3])
                var b1 = SIMD[.bfloat16, 4](bq[4], bq[5], bq[6], bq[7])
                mma(acc[np * 2], a, b0, acc[np * 2])
                mma(acc[np * 2 + 1], a, b1, acc[np * 2 + 1])
        barrier()

    comptime for nt in range(NT):
        comptime for half in range(2):
            var p = p_base + warp * MMA_M + g + half * HALF
            states.store[alignment=8](
                (
                    ((slot * nheads + h) * head_dim + p) * DSTATE
                    + nt * MMA_N
                    + t4 * 2,
                ),
                SIMD[.float32, 2](acc[nt][half * 2], acc[nt][half * 2 + 1]),
            )


def ssd_state_passing[
    kernel_dtype: DType,
    state_dtype: DType,
    SLT: TensorLayout,
    SEngine: TensorEngine,
    pool_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    update_state_LT: TensorLayout,
    Engine: TensorEngine,
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    DSTATE: Int,
](
    slots_i: Int32,
    batch_i: Int32,
    nheads_i: Int32,
    head_dim_i: Int32,
    info: TileTensor[.int32, SLT, MutUntrackedOrigin, Engine=SEngine],
    cs: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    states: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    ssm_pool: TileTensor[
        state_dtype, pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    update_state: TileTensor[
        .bool, update_state_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    slot_fn: SlotFn,
):
    """Passes the state across the chunks of each sequence.

    Replaces `states[c]` with the state entering chunk `c` and writes the final
    state to the pool slot. Grid `(ceil(head_dim * DSTATE / F32_VEC / NUM_THREADS),
    nheads, batch)`, block `NUM_THREADS`; a thread owns four contiguous `n`.
    """
    var slots = Int(slots_i)
    var batch = Int(batch_i)
    var nheads = Int(nheads_i)
    var head_dim = Int(head_dim_i)
    comptime pool_align = align_of[SIMD[state_dtype, F32_VEC]]()
    comptime U = 8
    var h = Int(block_idx.y)
    var b = Int(block_idx.z)
    var tid = Int(thread_idx.x)
    var e = (Int(block_idx.x) * NUM_THREADS + tid) * F32_VEC
    var active = e < head_dim * DSTATE
    var nch = Int(info[3 * slots + batch + b])
    if nch == 0:
        return
    var first = Int(info[3 * slots + b])
    var p = e // DSTATE
    var n = e % DSTATE
    var pool_slot = Int(slot_fn[1, 1](b))
    # The decay across each chunk, shared by every thread of the block.
    var decay_s = unsafe_stack_allocation[
        NUM_THREADS, DType.float32, address_space=.SHARED, alignment=16
    ]()

    var state = SIMD[.float32, F32_VEC](0)
    if active and _use_initial_state(has_initial_state, b):
        state = ssm_pool.load[width=4, alignment=pool_align](
            (pool_slot, h, p, n)
        ).cast[.float32]()
    var base = h * head_dim * DSTATE + e
    var slot_stride = nheads * head_dim * DSTATE
    for c0 in range(0, nch, NUM_THREADS):
        var n_here = min(NUM_THREADS, nch - c0)
        if tid < n_here:
            var slot = first + c0 + tid
            var length = Int(info[slots + slot])
            decay_s[tid] = exp2(
                cs.load[width=1](((h * slots + slot) * CHUNK + length - 1,))
            )
        barrier()
        for c in range(0, n_here, U):
            # Issue the loads of U chunks before the dependent chain.
            var s_c = Array[SIMD[.float32, F32_VEC], U](fill=0)
            comptime for u in range(U):
                if active and c + u < n_here:
                    s_c[u] = states.load[width=4, alignment=16](
                        ((first + c0 + c + u) * slot_stride + base,)
                    )
            comptime for u in range(U):
                if c + u < n_here:
                    if active:
                        states.store[alignment=16](
                            ((first + c0 + c + u) * slot_stride + base,), state
                        )
                    state = fma(
                        state, SIMD[.float32, F32_VEC](decay_s[c + u]), s_c[u]
                    )
        barrier()
    if active and _write_final_state(update_state, b):
        ssm_pool.store[alignment=pool_align](
            (pool_slot, h, p, n), state.cast[state_dtype]()
        )


@always_inline
def _ld2(ptr: Pointer[Scalar[.bfloat16], ...], off: Int) -> SIMD[.bfloat16, 2]:
    return ptr.unsafe_load[width=2, alignment=4](off)


@always_inline
def _frag_a(
    ptr: Pointer[Scalar[.bfloat16], ...], row: Int, col: Int, ld: Int
) -> SIMD[.bfloat16, 8]:
    """Loads this lane's A fragment (16 x 16, row major) of an m16n8k16 MMA.

    `row` is the lane's group row (`lane // QUAD`) and `col` its first k column
    (`lane % QUAD * 2`), both already offset to the fragment's origin.
    """
    var a = SIMD[.bfloat16, 8](0)
    a = a.insert[offset=0](_ld2(ptr, row * ld + col))
    a = a.insert[offset=2](_ld2(ptr, (row + HALF) * ld + col))
    a = a.insert[offset=4](_ld2(ptr, row * ld + col + HALF))
    a = a.insert[offset=6](_ld2(ptr, (row + HALF) * ld + col + HALF))
    return a


@always_inline
def _frag_b(
    ptr: Pointer[Scalar[.bfloat16], ...], row: Int, col: Int, ld: Int
) -> SIMD[.bfloat16, 4]:
    """Loads this lane's B fragment (16 x 8) of an m16n8k16 MMA.

    The tile is stored `[n][k]` (k contiguous): `row` is the lane's n
    (`lane // QUAD`) and `col` its first k (`lane % QUAD * 2`).
    """
    var b = SIMD[.bfloat16, 4](0)
    b = b.insert[offset=0](_ld2(ptr, row * ld + col))
    b = b.insert[offset=2](_ld2(ptr, row * ld + col + HALF))
    return b


def ssd_chunk_cb[
    kernel_dtype: DType,
    SLT: TensorLayout,
    SEngine: TensorEngine,
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
    DSTATE: Int,
](
    slots_i: Int32,
    ngroups_i: Int32,
    info: TileTensor[.int32, SLT, ImmUntrackedOrigin, Engine=SEngine],
    cb_out: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    b_fn: BFn,
    c_fn: CFn,
):
    """Computes undecayed fp32 `C B^T` once per chunk and B/C group.

    Grid `(slots, ngroups, CHUNK / TILE)`, block `OUT_THREADS`. The fragment
    traversal and increasing-k MMA order match `ssd_chunk_output` exactly;
    every fragment the consumer loads, including its masked lanes, is stored.
    """
    comptime assert kernel_dtype == .bfloat16, "tensor-core path is bf16"
    comptime LDK = DSTATE + SMEM_PAD
    comptime KSTEPS = DSTATE // MMA_K
    var slots = Int(slots_i)
    var ngroups = Int(ngroups_i)
    var slot = block_idx.x
    var group_id = block_idx.y
    var rt = block_idx.z
    var tid = thread_idx.x
    var warp, lane = divmod(tid, WARP_SIZE)
    var g, t4 = divmod(lane, QUAD)
    var length = Int(info.load[width=1]((slots + slot,)))
    var i0 = rt * TILE
    if i0 >= length:
        return
    var start = Int(info.load[width=1]((slot,)))
    var base = (slot * ngroups + group_id) * CHUNK * CHUNK
    var Cs = unsafe_stack_allocation[
        TILE * LDK,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var Bs = unsafe_stack_allocation[
        TILE * LDK,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    comptime for r in range(TILE * DSTATE // VEC // OUT_THREADS):
        var e = tid + r * OUT_THREADS
        var i, n8 = divmod(e, DSTATE // VEC)
        n8 *= VEC
        var v = SIMD[kernel_dtype, VEC](0)
        if i0 + i < length:
            v = c_fn[VEC, VEC](start + i0 + i, group_id, n8)
        Cs.unsafe_store[alignment=16](i * LDK + n8, v.cast[.bfloat16]())
    barrier()

    var r0 = warp * MMA_M
    var a_frag = Array[SIMD[.bfloat16, 8], KSTEPS](fill=0)
    comptime for ks in range(KSTEPS):
        a_frag[ks] = _frag_a(Cs, r0 + g, ks * MMA_K + t4 * 2, LDK)
    var n_jt = min(rt + 1, ceildiv(length, TILE))
    for jt in range(n_jt):
        var j0 = jt * TILE
        comptime for r in range(TILE * DSTATE // VEC // OUT_THREADS):
            var e = tid + r * OUT_THREADS
            var j, n8 = divmod(e, DSTATE // VEC)
            n8 *= VEC
            var v = SIMD[kernel_dtype, VEC](0)
            if j0 + j < length:
                v = b_fn[VEC, VEC](start + j0 + j, group_id, n8)
            Bs.unsafe_store[alignment=16](j * LDK + n8, v.cast[.bfloat16]())
        barrier()
        comptime for sub in range(TILE // MMA_K):
            if jt < rt or sub * MMA_K <= r0 + MMA_M - 1:
                var cb = Array[SIMD[.float32, FRAG_C], 2](fill=0)
                comptime for ks in range(KSTEPS):
                    comptime for n8 in range(MMA_K // MMA_N):
                        var bf = _frag_b(
                            Bs,
                            sub * MMA_K + n8 * MMA_N + g,
                            ks * MMA_K + t4 * 2,
                            LDK,
                        )
                        mma(cb[n8], a_frag[ks], bf, cb[n8])
                comptime for n8 in range(MMA_K // MMA_N):
                    comptime for half in range(2):
                        var i = i0 + r0 + g + half * HALF
                        var j = j0 + sub * MMA_K + n8 * MMA_N + t4 * 2
                        cb_out.store[width=2, alignment=8](
                            (base + i * CHUNK + j,),
                            SIMD[.float32, 2](
                                cb[n8][half * 2], cb[n8][half * 2 + 1]
                            ),
                        )
        barrier()


def ssd_chunk_output[
    kernel_dtype: DType,
    SLT: TensorLayout,
    SEngine: TensorEngine,
    A_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    Engine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int, Int) -> SIMD[
        kernel_dtype, width
    ],
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
    DSTATE: Int,
    use_grouped_cb: Bool = False,
](
    slots_i: Int32,
    nheads_i: Int32,
    head_dim_i: Int32,
    ngroups_i: Int32,
    D: TileTensor[kernel_dtype, D_LT, ImmUntrackedOrigin, Engine=Engine],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    info: TileTensor[.int32, SLT, MutUntrackedOrigin, Engine=SEngine],
    cs: TileTensor[.float32, SLT, ImmUntrackedOrigin, Engine=SEngine],
    delta_in: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    states: TileTensor[.float32, SLT, MutUntrackedOrigin, Engine=SEngine],
    cb_in: TileTensor[.float32, SLT, ImmUntrackedOrigin, Engine=SEngine],
    x_fn: XFn,
    b_fn: BFn,
    c_fn: CFn,
):
    """Computes `y` of one 64-row tile of a chunk for one head on tensor cores.

    Grid `(slots, nheads, (CHUNK / TILE) * head_dim / TILE)`, block
    `OUT_THREADS` (4 warps, 16 rows of the tile each). Operands are bf16 and
    accumulation is fp32 (`m16n8k16`). Each warp keeps its `C` fragments in
    registers: the carry-in term `C S'^T` and, per 16 columns, `C B^T` are
    MMAs, the decay mask is applied to the accumulator fragment, which is
    already the A-operand layout of the next MMA `M X`.
    """
    comptime assert kernel_dtype == .bfloat16, "tensor-core path is bf16"
    comptime LDK = DSTATE + SMEM_PAD
    comptime LDX = TILE + SMEM_PAD
    comptime KSTEPS = DSTATE // MMA_K
    comptime ROW_TILES = CHUNK // TILE
    var slots = Int(slots_i)
    var nheads = Int(nheads_i)
    var head_dim = Int(head_dim_i)
    var ngroups = Int(ngroups_i)
    var slot = Int(block_idx.x)
    var h = Int(block_idx.y)
    var rt = Int(block_idx.z) % ROW_TILES
    var p_base = (Int(block_idx.z) // ROW_TILES) * TILE
    var tid = Int(thread_idx.x)
    var warp = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var g = lane // QUAD
    var t4 = lane % QUAD
    var length = Int(info[slots + slot])
    var i0 = rt * TILE
    if i0 >= length:
        return
    var start = Int(info[slot])
    var group_id = h // (nheads // ngroups)
    var sbase = (h * slots + slot) * CHUNK
    var D_val = Float32(0)
    if Int(D.dim[0]()) > 0:
        D_val = D.load[width=1]((h,)).cast[.float32]()

    var Cs = unsafe_stack_allocation[
        TILE * LDK,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var SBs = unsafe_stack_allocation[
        TILE * LDK,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var Xt = unsafe_stack_allocation[
        TILE * LDX,
        DType.bfloat16,
        address_space=.SHARED,
        alignment=16,
    ]()
    var cs_s = unsafe_stack_allocation[
        CHUNK, DType.float32, address_space=.SHARED, alignment=16
    ]()
    var dl_s = unsafe_stack_allocation[
        CHUNK, DType.float32, address_space=.SHARED, alignment=16
    ]()

    # Per-chunk decay vectors, the C tile and the carry-in state tile.
    comptime for r in range(CHUNK // OUT_THREADS):
        var k = tid + r * OUT_THREADS
        cs_s[k] = cs.load[width=1]((sbase + k,))
        dl_s[k] = delta_in.load[width=1]((sbase + k,))
    comptime for r in range(TILE * DSTATE // VEC // OUT_THREADS):
        var e = tid + r * OUT_THREADS
        var i = e // (DSTATE // VEC)
        var n8 = (e % (DSTATE // VEC)) * VEC
        var v = SIMD[kernel_dtype, VEC](0)
        if i0 + i < length:
            v = c_fn[VEC, VEC](start + i0 + i, group_id, n8)
        Cs.unsafe_store[alignment=16](i * LDK + n8, v.cast[.bfloat16]())
    comptime for r in range(TILE * DSTATE // F32_VEC // OUT_THREADS):
        var e = tid + r * OUT_THREADS
        var p = e // (DSTATE // F32_VEC)
        var n4 = (e % (DSTATE // F32_VEC)) * F32_VEC
        var s4 = states.load[width=4, alignment=16](
            (((slot * nheads + h) * head_dim + p_base + p) * DSTATE + n4,)
        )
        SBs.unsafe_store[alignment=8](p * LDK + n4, s4.cast[.bfloat16]())
    barrier()

    var r0 = warp * MMA_M
    var acc = Array[SIMD[.float32, FRAG_C], TILE // MMA_N](fill=0)
    var a_frag = Array[SIMD[.bfloat16, 8], KSTEPS](fill=0)
    comptime for ks in range(KSTEPS):
        a_frag[ks] = _frag_a(Cs, r0 + g, ks * MMA_K + t4 * 2, LDK)
        comptime for nt in range(TILE // MMA_N):
            var bf = _frag_b(SBs, nt * MMA_N + g, ks * MMA_K + t4 * 2, LDK)
            mma(acc[nt], a_frag[ks], bf, acc[nt])
    var cs_lo = cs_s[i0 + r0 + g]
    var cs_hi = cs_s[i0 + r0 + g + HALF]
    var sc_lo = exp2(cs_lo)
    var sc_hi = exp2(cs_hi)
    comptime for nt in range(TILE // MMA_N):
        acc[nt][0] *= sc_lo
        acc[nt][1] *= sc_lo
        acc[nt][2] *= sc_hi
        acc[nt][3] *= sc_hi
    barrier()

    # Inside the chunk: 64-column tiles of B up to the row tile's diagonal.
    var n_jt = min(rt + 1, ceildiv(length, TILE))
    for jt in range(n_jt):
        var j0 = jt * TILE
        comptime if not use_grouped_cb:
            comptime for r in range(TILE * DSTATE // VEC // OUT_THREADS):
                var e = tid + r * OUT_THREADS
                var j = e // (DSTATE // VEC)
                var n8 = (e % (DSTATE // VEC)) * VEC
                var v = SIMD[kernel_dtype, VEC](0)
                if j0 + j < length:
                    v = b_fn[VEC, VEC](start + j0 + j, group_id, n8)
                SBs.unsafe_store[alignment=16](
                    j * LDK + n8, v.cast[.bfloat16]()
                )
        comptime for r in range(TILE * TILE // VEC // OUT_THREADS):
            var e = tid + r * OUT_THREADS
            var j = e // (TILE // VEC)
            var p8 = (e % (TILE // VEC)) * VEC
            var v = SIMD[kernel_dtype, VEC](0)
            if j0 + j < length:
                v = x_fn[VEC, 1](start + j0 + j, h, p_base + p8)
            comptime for q in range(VEC):
                Xt[(p8 + q) * LDX + j] = v[q].cast[.bfloat16]()
        barrier()
        comptime for sub in range(TILE // MMA_K):
            # Columns past this warp's last row are fully masked.
            if jt < rt or sub * MMA_K <= r0 + MMA_M - 1:
                var cb = Array[SIMD[.float32, FRAG_C], 2](fill=0)
                comptime if use_grouped_cb:
                    comptime for n8 in range(MMA_K // MMA_N):
                        comptime for half in range(2):
                            var i = i0 + r0 + g + half * HALF
                            var j = j0 + sub * MMA_K + n8 * MMA_N + t4 * 2
                            var pair = cb_in.load[width=2, alignment=8](
                                (
                                    ((slot * ngroups + group_id) * CHUNK + i)
                                    * CHUNK
                                    + j,
                                )
                            )
                            cb[n8][half * 2] = pair[0]
                            cb[n8][half * 2 + 1] = pair[1]
                else:
                    comptime for ks in range(KSTEPS):
                        comptime for n8 in range(MMA_K // MMA_N):
                            var bf = _frag_b(
                                SBs,
                                sub * MMA_K + n8 * MMA_N + g,
                                ks * MMA_K + t4 * 2,
                                LDK,
                            )
                            mma(cb[n8], a_frag[ks], bf, cb[n8])
                var m = SIMD[.bfloat16, 8](0)
                comptime for n8 in range(MMA_K // MMA_N):
                    comptime for e in range(4):
                        var i_loc = i0 + r0 + g + (e // 2) * HALF
                        var j_loc = (
                            j0 + sub * MMA_K + n8 * MMA_N + t4 * 2 + (e % 2)
                        )
                        var v = Float32(0)
                        if j_loc <= i_loc:
                            v = (
                                cb[n8][e]
                                * exp2(cs_s[i_loc] - cs_s[j_loc])
                                * dl_s[j_loc]
                            )
                        m[n8 * 4 + e] = v.cast[.bfloat16]()
                comptime for nt in range(TILE // MMA_N):
                    var bf = _frag_b(
                        Xt, nt * MMA_N + g, sub * MMA_K + t4 * 2, LDX
                    )
                    mma(acc[nt], m, bf, acc[nt])
        barrier()

    comptime for half in range(2):
        var i = i0 + r0 + g + half * HALF
        if i < length:
            comptime for nt in range(TILE // MMA_N):
                var p = p_base + nt * MMA_N + t4 * 2
                var xv = x_fn[2, 1](start + i, h, p).cast[.float32]()
                var out = SIMD[.float32, 2](
                    acc[nt][half * 2], acc[nt][half * 2 + 1]
                )
                out = fma(SIMD[.float32, 2](D_val), xv, out)
                y.store[alignment=align_of[kernel_dtype]()](
                    Coord(start + i, h, p), out.cast[kernel_dtype]()
                )


def mamba2_ssd_chunked_fwd_inplace_gpu[
    kernel_dtype: DType,
    state_dtype: DType,
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
    //,
    DSTATE: Int,
    dt_softplus: Bool,
    grouped_cb_mode: Int = GROUPED_CB_AUTO,
](
    ctx: DeviceContext,
    total_len: Int,
    batch: Int,
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
) raises:
    """Runs the chunked SSD scan over a varlen batch.

    Same operands and semantics as the serial scan kernel, including
    `has_initial_state` and `update_state`. `head_dim` must be a multiple of
    `TILE` and `DSTATE` a multiple of 64.

    Parameters:
        kernel_dtype: Element type of `x`, `dt`, `A`, `B`, `C`, `D`,
            `dt_bias` and `y`.
        state_dtype: Storage dtype of `ssm_pool`.
        XFn: Reads `x`.
        DtFn: Reads `dt`.
        BFn: Reads `B`.
        CFn: Reads `C`.
        SlotFn: Reads the pool slot of each sequence.
        DSTATE: State dimension per head.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.
        grouped_cb_mode: `GROUPED_CB_OFF`, `GROUPED_CB_ON`, or
            `GROUPED_CB_AUTO`. AUTO selects only the validated B200 bf16
            kernel, fp32 state, dimension-128 amortization regime.

    Args:
        ctx: Device context.
        total_len: Total tokens of the batch.
        batch: Number of sequences.
        ngroups: Number of `B`/`C` groups.
        A: Per-head scalar decay.
        D: Per-head skip connection, or empty.
        dt_bias: Per-head bias added to `dt`, or empty.
        y: Output of shape `(total_len, nheads, head_dim)`.
        ssm_pool: State pool, read and written in place.
        query_start_loc: Cumulative sequence offsets of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flag, or empty.
        update_state: Per-sequence write-back flag, or empty.
        x_fn: Reads `x`.
        dt_fn: Reads `dt`.
        b_fn: Reads `B`.
        c_fn: Reads `C`.
        slot_fn: Reads the pool slot of each sequence.
    """
    comptime assert DSTATE % 64 == 0, "the chunked scan needs DSTATE % 64 == 0"
    comptime assert grouped_cb_mode in (
        GROUPED_CB_AUTO,
        GROUPED_CB_OFF,
        GROUPED_CB_ON,
    ), "invalid grouped score precomputation mode"
    var nheads = Int(y.dim[1]())
    var head_dim = Int(y.dim[2]())
    var slots = chunked_slots(total_len, batch)

    var states_d = ctx.enqueue_create_buffer[.float32](
        slots * nheads * head_dim * DSTATE
    )
    var cs_d = ctx.enqueue_create_buffer[.float32](nheads * slots * CHUNK)
    var delta_d = ctx.enqueue_create_buffer[.float32](nheads * slots * CHUNK)
    var info_d = ctx.enqueue_create_buffer[.int32](3 * slots + 2 * batch)
    var states = TileTensor(
        states_d, row_major(slots * nheads * head_dim * DSTATE)
    )
    var cs = TileTensor(cs_d, row_major(nheads * slots * CHUNK))
    var delta = TileTensor(delta_d, row_major(nheads * slots * CHUNK))
    var info = TileTensor(info_d, row_major(3 * slots + 2 * batch))

    comptime map_kernel = ssd_chunk_map[
        info.LayoutType,
        info.Engine,
        query_start_loc.LayoutType,
        query_start_loc.Engine,
    ]
    ctx.enqueue_function[map_kernel](
        Int32(slots),
        Int32(batch),
        info,
        query_start_loc,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    comptime cumsum_kernel = ssd_chunk_cumsum[
        kernel_dtype,
        A.LayoutType,
        D.LayoutType,
        dt_bias.LayoutType,
        A.Engine,
        cs.LayoutType,
        cs.Engine,
        type_of(dt_fn),
        dt_softplus,
    ]
    ctx.enqueue_function[cumsum_kernel](
        Int32(slots),
        A,
        D,
        dt_bias,
        info,
        cs,
        delta,
        host_args=(dt_fn,),
        grid_dim=(slots, nheads, 1),
        block_dim=(CHUNK, 1, 1),
    )

    comptime state_kernel = ssd_chunk_state[
        kernel_dtype,
        cs.LayoutType,
        cs.Engine,
        type_of(x_fn),
        type_of(b_fn),
        DSTATE,
    ]
    ctx.enqueue_function[state_kernel](
        Int32(slots),
        Int32(nheads),
        Int32(head_dim),
        Int32(ngroups),
        info,
        cs,
        delta,
        states,
        host_args=(x_fn, b_fn),
        grid_dim=(slots, nheads, head_dim // TILE),
        block_dim=(OUT_THREADS, 1, 1),
    )

    comptime pass_kernel = ssd_state_passing[
        kernel_dtype,
        state_dtype,
        cs.LayoutType,
        cs.Engine,
        ssm_pool.LayoutType,
        has_initial_state.LayoutType,
        update_state.LayoutType,
        ssm_pool.Engine,
        type_of(slot_fn),
        DSTATE,
    ]
    ctx.enqueue_function[pass_kernel](
        Int32(slots),
        Int32(batch),
        Int32(nheads),
        Int32(head_dim),
        info,
        cs,
        states,
        ssm_pool,
        has_initial_state,
        update_state,
        host_args=(slot_fn,),
        grid_dim=(
            ceildiv(head_dim * DSTATE // F32_VEC, NUM_THREADS),
            nheads,
            batch,
        ),
        block_dim=(NUM_THREADS, 1, 1),
    )

    def launch_output[use_grouped_cb: Bool]() raises {imm}:
        comptime out_kernel = ssd_chunk_output[
            kernel_dtype,
            cs.LayoutType,
            cs.Engine,
            A.LayoutType,
            D.LayoutType,
            dt_bias.LayoutType,
            y.LayoutType,
            y.Engine,
            type_of(x_fn),
            type_of(b_fn),
            type_of(c_fn),
            DSTATE,
            use_grouped_cb,
        ]
        comptime if use_grouped_cb:
            var cb_d = ctx.enqueue_create_buffer[.float32](
                slots * ngroups * CHUNK * CHUNK
            )
            var cb = TileTensor(
                cb_d, row_major(slots * ngroups * CHUNK * CHUNK)
            )
            comptime cb_kernel = ssd_chunk_cb[
                kernel_dtype,
                cs.LayoutType,
                cs.Engine,
                type_of(b_fn),
                type_of(c_fn),
                DSTATE,
            ]
            ctx.enqueue_function[cb_kernel](
                Int32(slots),
                Int32(ngroups),
                info.as_imm(),
                cb,
                host_args=(b_fn, c_fn),
                grid_dim=(slots, ngroups, CHUNK // TILE),
                block_dim=(OUT_THREADS, 1, 1),
            )
            ctx.enqueue_function[out_kernel](
                Int32(slots),
                Int32(nheads),
                Int32(head_dim),
                Int32(ngroups),
                D,
                y,
                info,
                cs.as_imm(),
                delta,
                states,
                cb.as_imm(),
                host_args=(x_fn, b_fn, c_fn),
                grid_dim=(slots, nheads, (CHUNK // TILE) * (head_dim // TILE)),
                block_dim=(OUT_THREADS, 1, 1),
            )
            return
        ctx.enqueue_function[out_kernel](
            Int32(slots),
            Int32(nheads),
            Int32(head_dim),
            Int32(ngroups),
            D,
            y,
            info,
            cs.as_imm(),
            delta,
            states,
            cs.as_imm(),
            host_args=(x_fn, b_fn, c_fn),
            grid_dim=(slots, nheads, (CHUNK // TILE) * (head_dim // TILE)),
            block_dim=(OUT_THREADS, 1, 1),
        )

    comptime if grouped_cb_mode == GROUPED_CB_ON:
        launch_output[True]()
    elif grouped_cb_mode == GROUPED_CB_OFF:
        launch_output[False]()
    else:
        comptime if (
            kernel_dtype == .bfloat16
            and state_dtype == .float32
            and DSTATE == 128
            and (
                ctx.target.is_nvidia_gpu["sm_100"]()
                or ctx.target.is_nvidia_gpu["sm_100a"]()
            )
        ):
            if (
                use_grouped_cb_ssd(total_len, batch, nheads, head_dim, ngroups)
                and ctx.name() == "NVIDIA B200"
            ):
                launch_output[True]()
                return
        launch_output[False]()
