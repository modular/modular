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

"""Mamba-2 SSD (state-space duality) varlen prefill scan.

Target hardware family: B200 (sm_100), but the kernel is architecture-agnostic
(plain elementwise + SIMD-over-dstate recurrence, no MMA/TMA/swizzle). bf16
in/out, fp32 accumulate, states fp32.

This is the varlen (ragged, `query_start_loc`) Mamba-2 SSD forward prefill op,
matching `mamba_chunk_scan_combined` semantics for the `NemotronHMamba2Mixer`
(Nemotron-H). It differs from the Mamba-1 ops in
`varlen_selective_scan.mojo` in three ways that the math requires:

  - `A` is a per-head SCALAR `(nheads,)` (shared across all head_dim channels
    and all dstate), not a per-channel `(dim, dstate)` diagonal.
  - `B`/`C` are GROUPED `(total_len, ngroups, dstate)`; `nheads/ngroups` heads
    share each group (`group_id = h // (nheads // ngroups)`).
  - `dt` is per-head `(total_len, nheads)` + per-head `dt_bias (nheads,)`,
    broadcast across head_dim; softplus is applied to `dt + dt_bias`.

Reference math (the source of truth for parity) is the HF `torch_forward` SSD
path (`segment_sum` / `reshape_into_chunks` / chunk-state recurrence). The SSD
chunked scan is a parallelism reformulation of the linear recurrence below; in
fp32 they are numerically equivalent. This first implementation carries the
state sequentially per `(head, head_dim)` channel (one thread per channel, SIMD
over dstate), mirroring the tiling style of `varlen_selective_scan_fwd_gpu`. A
chunk-tiled rewrite (segment_sum + matmul) is a follow-up perf slice; this op
is gated on CORRECTNESS first.

Per-token recurrence (per head `h`, head_dim channel `p`, group `g`):

    dt_t     = softplus(dt[t, h] + dt_bias[h])            # scalar per (t, h)
    dA_t     = exp(A[h] * dt_t)                           # scalar per (t, h)
    state_n  = state_n * dA_t + dt_t * B[t, g, n] * x[t, h, p]   # vector over n
    y[t,h,p] = sum_n C[t, g, n] * state_n  +  D[h] * x[t, h, p]

State resets to zero (or to `initial_states`) at each `query_start_loc`
boundary -- no cross-sequence bleed. `final_states (batch, nheads, head_dim,
dstate)` is written at each sequence end.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.algorithm import sync_parallelize
from std.math import exp2
from std.utils.index import IndexList
from layout import TensorLayout, TileTensor
from state_space.selective_scan import softplus

# LOG2E: convert exp(x) -> exp2(x * LOG2E) (faster on GPU), matching the
# convention in varlen_selective_scan.mojo.
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 256  # Mamba-2 dstate <= 256

comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


def mamba2_ssd_chunk_scan_varlen_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    initial_states_LT: TensorLayout,
    y_LT: TensorLayout,
    final_states_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    # Tensors (varlen / ragged: time dim is the packed total_len)
    x: TileTensor[
        kernel_dtype, x_LT, MutUntrackedOrigin
    ],  # (total_len, nheads, head_dim)
    dt: TileTensor[
        kernel_dtype, dt_LT, MutUntrackedOrigin
    ],  # (total_len, nheads)
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin],  # (nheads,)
    B: TileTensor[
        kernel_dtype, B_LT, MutUntrackedOrigin
    ],  # (total_len, ngroups, dstate)
    C: TileTensor[
        kernel_dtype, C_LT, MutUntrackedOrigin
    ],  # (total_len, ngroups, dstate)
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin],  # (nheads,) optional
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin
    ],  # (nheads,) optional
    initial_states: TileTensor[
        DType.float32, initial_states_LT, MutUntrackedOrigin
    ],  # (batch, nheads, head_dim, dstate) optional
    y: TileTensor[
        kernel_dtype, y_LT, MutUntrackedOrigin
    ],  # (total_len, nheads, head_dim)
    final_states: TileTensor[
        DType.float32, final_states_LT, MutUntrackedOrigin
    ],  # (batch, nheads, head_dim, dstate)
    query_start_loc: TileTensor[
        DType.int32, query_start_loc_LT, MutUntrackedOrigin
    ],  # (batch + 1,)
    has_initial_state: TileTensor[
        DType.bool, has_initial_state_LT, MutUntrackedOrigin
    ],  # (batch,) optional
    x_strides: Strides3D,  # (total_len, nheads, head_dim)
    dt_strides: Strides2D,  # (total_len, nheads)
    A_strides: Strides1D,  # (nheads,)
    B_strides: Strides3D,  # (total_len, ngroups, dstate)
    C_strides: Strides3D,  # (total_len, ngroups, dstate)
    D_strides: Strides1D,  # (nheads,)
    dt_bias_strides: Strides1D,  # (nheads,)
    initial_states_strides: Strides4D,  # (batch, nheads, head_dim, dstate)
    y_strides: Strides3D,  # (total_len, nheads, head_dim)
    final_states_strides: Strides4D,  # (batch, nheads, head_dim, dstate)
):
    """GPU kernel: Mamba-2 SSD varlen prefill scan, one thread per (head, channel).

    Grid: (ceildiv(head_dim, BLOCK), nheads, batch). Each thread owns one
    `(b, h, p)` channel, carries the `dstate`-vector state in registers, and
    walks its sequence `[seq_start, seq_end)` sequentially.
    """
    # block_idx.x * block_dim.x + thread_idx.x -> head_dim channel p
    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y  # head
    var b = block_idx.z  # batch (sequence)

    if p >= head_dim or h >= nheads or b >= batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    # Sequence bounds for this batch element.
    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    # Per-head scalar A, pre-multiplied by LOG2E for exp2.
    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[DType.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    # State vector over dstate, fp32. Initialise from initial_states if present.
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if use_initial:
        comptime for n in range(DSTATE):
            var off = UInt32(
                b * initial_states_strides[0]
                + h * initial_states_strides[1]
                + p * initial_states_strides[2]
                + n * initial_states_strides[3]
            )
            state[n] = initial_states.raw_load(off)

    # Sequential recurrence over the sequence.
    for t in range(seq_len):
        var gt = seq_start + t  # global (packed) time index

        # x[gt, h, p]
        var x_val = Scalar[kernel_dtype](
            x.raw_load(
                UInt32(gt * x_strides[0] + h * x_strides[1] + p * x_strides[2])
            )
        ).cast[DType.float32]()

        # dt[gt, h] (+ dt_bias), softplus -> per-(t,h) scalar (broadcast over p).
        var dt_val = Scalar[kernel_dtype](
            dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
        ).cast[DType.float32]()
        if has_dt_bias:
            dt_val += dt_bias_val
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        # dA = exp(A * dt) (scalar), dt_x = dt * x (discretised input).
        var dA = exp2(A_val * dt_val)
        var dt_x = dt_val * x_val

        # Load B and C rows for this (gt, group_id).
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.raw_load(
                    UInt32(
                        gt * B_strides[0]
                        + group_id * B_strides[1]
                        + n * B_strides[2]
                    )
                )
            ).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.raw_load(
                    UInt32(
                        gt * C_strides[0]
                        + group_id * C_strides[1]
                        + n * C_strides[2]
                    )
                )
            ).cast[DType.float32]()

        # state_n = state_n * dA + (dt * x) * B_n   (vector over dstate)
        state = state * dA + B_vals * dt_x

        # y = sum_n C_n * state_n  + D * x
        var y_val = (state * C_vals).reduce_add()
        if has_D:
            y_val += D_val * x_val

        y.raw_store(
            UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
            Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
        )

    # Write final state (fp32) for chunked-prefill continuation / decode handoff.
    comptime for n in range(DSTATE):
        var off = UInt32(
            b * final_states_strides[0]
            + h * final_states_strides[1]
            + p * final_states_strides[2]
            + n * final_states_strides[3]
        )
        final_states.raw_store(off, state[n])


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    x: TileTensor[kernel_dtype, x_LT, MutUntrackedOrigin],
    dt: TileTensor[kernel_dtype, dt_LT, MutUntrackedOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin],
    C: TileTensor[kernel_dtype, C_LT, MutUntrackedOrigin],
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin],
    dt_bias: TileTensor[kernel_dtype, dt_bias_LT, MutUntrackedOrigin],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin],
    # ssm_pool: [max_slots, nheads, head_dim, dstate] fp32 — read for initial
    # state (when has_initial_state[b]) and written in-place at slot
    # cache_indices[b] (instead of a separate final_states output).
    ssm_pool: TileTensor[DType.float32, ssm_pool_LT, MutUntrackedOrigin],
    query_start_loc: TileTensor[
        DType.int32, query_start_loc_LT, MutUntrackedOrigin
    ],
    has_initial_state: TileTensor[
        DType.bool, has_initial_state_LT, MutUntrackedOrigin
    ],
    cache_indices: TileTensor[
        DType.uint32, cache_indices_LT, MutUntrackedOrigin
    ],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
):
    """GPU kernel: Mamba-2 SSD varlen prefill scan with in-place SSM-pool write.

    Identical to ``mamba2_ssd_chunk_scan_varlen_fwd_gpu`` except final states
    are written directly into ``ssm_pool[cache_indices[b], ...]`` (fp32,
    [max_slots, nheads, head_dim, dstate]) instead of a separate
    ``final_states`` output tensor.  This eliminates the graph-side
    gather/scatter_nd/buffer_store whole-pool round-trip.

    Grid: (ceildiv(head_dim, BLOCK), nheads, batch). Same launch shape as the
    non-inplace variant.
    """
    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y
    var b = block_idx.z

    if p >= head_dim or h >= nheads or b >= batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[DType.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    # Load initial state from ssm_pool at the slot for this sequence.
    var slot = Int(cache_indices.raw_load(b))
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if use_initial:
        # Read initial state from ssm_pool[slot, h, p, n].
        comptime for n in range(DSTATE):
            var off = UInt32(
                slot * ssm_pool_strides[0]
                + h * ssm_pool_strides[1]
                + p * ssm_pool_strides[2]
                + n * ssm_pool_strides[3]
            )
            state[n] = ssm_pool.raw_load(off)

    for t in range(seq_len):
        var gt = seq_start + t

        var x_val = Scalar[kernel_dtype](
            x.raw_load(
                UInt32(gt * x_strides[0] + h * x_strides[1] + p * x_strides[2])
            )
        ).cast[DType.float32]()

        var dt_val = Scalar[kernel_dtype](
            dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
        ).cast[DType.float32]()
        if has_dt_bias:
            dt_val += dt_bias_val
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var dA = exp2(A_val * dt_val)
        var dt_x = dt_val * x_val

        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.raw_load(
                    UInt32(
                        gt * B_strides[0]
                        + group_id * B_strides[1]
                        + n * B_strides[2]
                    )
                )
            ).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.raw_load(
                    UInt32(
                        gt * C_strides[0]
                        + group_id * C_strides[1]
                        + n * C_strides[2]
                    )
                )
            ).cast[DType.float32]()

        state = state * dA + B_vals * dt_x

        var y_val = (state * C_vals).reduce_add()
        if has_D:
            y_val += D_val * x_val

        y.raw_store(
            UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
            Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
        )

    # Write final state directly into ssm_pool at slot cache_indices[b].
    comptime for n in range(DSTATE):
        var off = UInt32(
            slot * ssm_pool_strides[0]
            + h * ssm_pool_strides[1]
            + p * ssm_pool_strides[2]
            + n * ssm_pool_strides[3]
        )
        ssm_pool.raw_store(off, state[n])


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    x: TileTensor[mut=False, kernel_dtype, ...],
    dt: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    y: TileTensor[mut=True, kernel_dtype, ...],
    # ssm_pool: [max_slots, nheads, head_dim, dstate] fp32 — read for initial
    # state (when has_initial_state[b]) and written in-place at slot
    # cache_indices[b] (instead of a separate final_states output).
    ssm_pool: TileTensor[mut=True, DType.float32, ...],
    query_start_loc: TileTensor[mut=False, DType.int32, ...],
    has_initial_state: TileTensor[mut=False, DType.bool, ...],
    cache_indices: TileTensor[mut=False, DType.uint32, ...],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU reference: Mamba-2 SSD varlen scan with in-place SSM-pool write.

    Mirrors ``mamba2_ssd_chunk_scan_varlen_fwd_cpu`` but writes final states
    into ``ssm_pool[cache_indices[b], ...]`` directly.
    """
    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    @parameter
    def worker(idx: Int):
        var b, remaining = divmod(idx, nheads * head_dim)
        var h, p = divmod(remaining, head_dim)

        var group_id = h // nheads_ngroups_ratio

        var seq_start = Int(query_start_loc.raw_load(b))
        var seq_end = Int(query_start_loc.raw_load(b + 1))
        var seq_len = seq_end - seq_start
        if seq_len <= 0:
            return

        var A_val = (
            Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
                DType.float32
            ]()
            * LOG2E
        )

        var dt_bias_val = Float32(0.0)
        if has_dt_bias:
            dt_bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
            ).cast[DType.float32]()

        var D_val = Float32(0.0)
        if has_D:
            D_val = Scalar[kernel_dtype](
                D.raw_load(UInt32(h * D_strides[0]))
            ).cast[DType.float32]()

        var slot = Int(cache_indices.raw_load(b))
        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var use_initial = False
        if has_init_tensor:
            use_initial = Bool(has_initial_state.raw_load(b))
        if use_initial:
            # Read initial state from ssm_pool[slot, h, p, n].
            comptime for n in range(DSTATE):
                var off = UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + n * ssm_pool_strides[3]
                )
                state[n] = ssm_pool.raw_load(off)

        for t in range(seq_len):
            var gt = seq_start + t

            var x_val = Scalar[kernel_dtype](
                x.raw_load(
                    UInt32(
                        gt * x_strides[0] + h * x_strides[1] + p * x_strides[2]
                    )
                )
            ).cast[DType.float32]()

            var dt_val = Scalar[kernel_dtype](
                dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
            ).cast[DType.float32]()
            if has_dt_bias:
                dt_val += dt_bias_val
            if dt_softplus_bool:
                dt_val = softplus(dt_val)

            var dA = exp2(A_val * dt_val)
            var dt_x = dt_val * x_val

            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.raw_load(
                        UInt32(
                            gt * B_strides[0]
                            + group_id * B_strides[1]
                            + n * B_strides[2]
                        )
                    )
                ).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.raw_load(
                        UInt32(
                            gt * C_strides[0]
                            + group_id * C_strides[1]
                            + n * C_strides[2]
                        )
                    )
                ).cast[DType.float32]()

            state = state * dA + B_vals * dt_x

            var y_val = (state * C_vals).reduce_add()
            if has_D:
                y_val += D_val * x_val

            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
            )

        # Write final state into ssm_pool at slot cache_indices[b].
        comptime for n in range(DSTATE):
            var off = UInt32(
                slot * ssm_pool_strides[0]
                + h * ssm_pool_strides[1]
                + p * ssm_pool_strides[2]
                + n * ssm_pool_strides[3]
            )
            ssm_pool.raw_store(off, state[n])

    sync_parallelize[worker](batch * nheads * head_dim, ctx)


def mamba2_ssd_chunk_scan_varlen_fwd_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    x: TileTensor[mut=False, kernel_dtype, ...],
    dt: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    initial_states: TileTensor[mut=False, DType.float32, ...],
    y: TileTensor[mut=True, kernel_dtype, ...],
    final_states: TileTensor[mut=True, DType.float32, ...],
    query_start_loc: TileTensor[mut=False, DType.int32, ...],
    has_initial_state: TileTensor[mut=False, DType.bool, ...],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    initial_states_strides: Strides4D,
    y_strides: Strides3D,
    final_states_strides: Strides4D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU reference for the Mamba-2 SSD varlen prefill scan.

    This is the trusted reference for numerical-equivalence testing: it computes
    the same per-token recurrence in fp32. Parallelised over `(b, h, p)`.
    """
    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    @parameter
    def worker(idx: Int):
        var b, remaining = divmod(idx, nheads * head_dim)
        var h, p = divmod(remaining, head_dim)

        var group_id = h // nheads_ngroups_ratio

        var seq_start = Int(query_start_loc.raw_load(b))
        var seq_end = Int(query_start_loc.raw_load(b + 1))
        var seq_len = seq_end - seq_start
        if seq_len <= 0:
            return

        var A_val = (
            Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
                DType.float32
            ]()
            * LOG2E
        )

        var dt_bias_val = Float32(0.0)
        if has_dt_bias:
            dt_bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
            ).cast[DType.float32]()

        var D_val = Float32(0.0)
        if has_D:
            D_val = Scalar[kernel_dtype](
                D.raw_load(UInt32(h * D_strides[0]))
            ).cast[DType.float32]()

        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var use_initial = False
        if has_init_tensor:
            use_initial = Bool(has_initial_state.raw_load(b))
        if use_initial:
            comptime for n in range(DSTATE):
                var off = UInt32(
                    b * initial_states_strides[0]
                    + h * initial_states_strides[1]
                    + p * initial_states_strides[2]
                    + n * initial_states_strides[3]
                )
                state[n] = initial_states.raw_load(off)

        for t in range(seq_len):
            var gt = seq_start + t

            var x_val = Scalar[kernel_dtype](
                x.raw_load(
                    UInt32(
                        gt * x_strides[0] + h * x_strides[1] + p * x_strides[2]
                    )
                )
            ).cast[DType.float32]()

            var dt_val = Scalar[kernel_dtype](
                dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
            ).cast[DType.float32]()
            if has_dt_bias:
                dt_val += dt_bias_val
            if dt_softplus_bool:
                dt_val = softplus(dt_val)

            var dA = exp2(A_val * dt_val)
            var dt_x = dt_val * x_val

            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.raw_load(
                        UInt32(
                            gt * B_strides[0]
                            + group_id * B_strides[1]
                            + n * B_strides[2]
                        )
                    )
                ).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.raw_load(
                        UInt32(
                            gt * C_strides[0]
                            + group_id * C_strides[1]
                            + n * C_strides[2]
                        )
                    )
                ).cast[DType.float32]()

            state = state * dA + B_vals * dt_x

            var y_val = (state * C_vals).reduce_add()
            if has_D:
                y_val += D_val * x_val

            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
            )

        comptime for n in range(DSTATE):
            var off = UInt32(
                b * final_states_strides[0]
                + h * final_states_strides[1]
                + p * final_states_strides[2]
                + n * final_states_strides[3]
            )
            final_states.raw_store(off, state[n])

    sync_parallelize[worker](batch * nheads * head_dim, ctx)
