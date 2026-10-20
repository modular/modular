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

"""Tests for the Mamba-2 SSD varlen scan with an in-place state pool.

Every kernel (CPU, cooperative-split GPU, Apple vectorized GPU) is checked
against a scalar host reference of the per-token recurrence. Covers Mamba-2
grouping (scalar A, grouped B/C, per-head dt+bias softplus), ragged batches,
optional D/dt_bias, seeded initial states, strided x/B/C, bf16 state storage,
and that a packed ragged batch equals independent per-sequence runs.
"""

from std.math import exp, log1p
from std.memory.alloc import unsafe_alloc
from std.random import rand, seed
from std.sys import size_of
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_false,
    assert_true,
)
from std.utils.index import Index, IndexList

from layout import MixedLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from ssd_scan_test_utils.ssd_scan import (
    check_chunked_parity,
    check_preserved_slots,
    run_ssd_gpu,
)
from state_space.mamba2_ssd_chunked import (
    GROUPED_CB_AUTO,
    GROUPED_CB_ON,
    use_chunked_ssd,
    use_grouped_cb_ssd,
)
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
)


def _scan_cpu[
    dtype: DType, state_dtype: DType, //, DSTATE: Int
](
    x: TileTensor[mut=False, dtype, ...],
    dt: TileTensor[mut=False, dtype, ...],
    A: TileTensor[mut=False, dtype, ...],
    B: TileTensor[mut=False, dtype, ...],
    C: TileTensor[mut=False, dtype, ...],
    D: TileTensor[mut=False, dtype, ...],
    dt_bias: TileTensor[mut=False, dtype, ...],
    y: TileTensor[mut=True, dtype, ...],
    ssm_pool: TileTensor[mut=True, state_dtype, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    update_state: TileTensor[mut=False, .bool, ...],
    cache_indices: TileTensor[mut=False, .uint32, ...],
):
    """Runs the CPU scan with operand functions that read these tensors."""
    comptime elt = size_of[dtype]()

    def x_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int, p: Int) {var x} -> SIMD[dtype, width]:
        return x.load[width=width, alignment=alignment * elt]((t, h, p))

    def dt_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int) {var dt} -> SIMD[dtype, width]:
        return dt.load[width=width, alignment=alignment * elt]((t, h))

    def b_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var B} -> SIMD[dtype, width]:
        return B.load[width=width, alignment=alignment * elt]((t, g, n))

    def c_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var C} -> SIMD[dtype, width]:
        return C.load[width=width, alignment=alignment * elt]((t, g, n))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var cache_indices} -> SIMD[.uint32, width]:
        return cache_indices.load[width=width]((b,))

    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[DSTATE](
        Int(B.dim[1]()),
        A,
        D,
        dt_bias,
        y,
        ssm_pool,
        query_start_loc,
        has_initial_state,
        update_state,
        x_fn,
        dt_fn,
        b_fn,
        c_fn,
        slot_fn,
    )


def _reference_scan[
    dtype: DType
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    dstate: Int,
    seq_lengths: IndexList,
    x: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    x_row: Int,
    dt: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    A: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    B: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    C: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    bc_row: Int,
    D: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    has_D: Bool,
    dt_bias: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    has_dt_bias: Bool,
    initial_states: MutPointer[Float32, MutUntrackedOrigin],
    init_state: Bool,
    y: MutPointer[Float32, MutUntrackedOrigin],
    final_states: MutPointer[Float32, MutUntrackedOrigin],
):
    """Scalar fp32 reference; `y` is packed `(total_len, nheads, head_dim)`."""
    var heads_per_group = nheads // ngroups
    var seq_start = 0
    for b in range(len(seq_lengths)):
        for h in range(nheads):
            var g = h // heads_per_group
            for p in range(head_dim):
                var base = ((b * nheads + h) * head_dim + p) * dstate
                for n in range(dstate):
                    final_states.unsafe_store(
                        base + n,
                        initial_states.unsafe_load(
                            base + n
                        ) if init_state else 0.0,
                    )
                for t in range(seq_start, seq_start + seq_lengths[b]):
                    var x_val = Float32(
                        x.unsafe_load(t * x_row + h * head_dim + p)
                    )
                    var delta = Float32(dt.unsafe_load(t * nheads + h))
                    if has_dt_bias:
                        delta += Float32(dt_bias.unsafe_load(h))
                    delta = log1p(exp(delta))
                    var dA = exp(Float32(A.unsafe_load(h)) * delta)
                    var y_val = Float32(0)
                    for n in range(dstate):
                        var bc = t * bc_row + g * dstate + n
                        var s = (
                            final_states.unsafe_load(base + n) * dA
                            + Float32(B.unsafe_load(bc)) * delta * x_val
                        )
                        final_states.unsafe_store(base + n, s)
                        y_val += s * Float32(C.unsafe_load(bc))
                    if has_D:
                        y_val += Float32(D.unsafe_load(h)) * x_val
                    y.unsafe_store((t * nheads + h) * head_dim + p, y_val)
        seq_start += seq_lengths[b]


def _check_outputs[
    out_dtype: DType, pool_dtype: DType
](
    y_ref: MutPointer[Float32, MutUntrackedOrigin],
    fs_ref: MutPointer[Float32, MutUntrackedOrigin],
    slots: MutPointer[UInt32, MutUntrackedOrigin],
    batch: Int,
    y_size: Int,
    state_size: Int,
    y: MutPointer[Scalar[out_dtype], MutUntrackedOrigin],
    pool: MutPointer[Scalar[pool_dtype], MutUntrackedOrigin],
    rtol: Float64,
    label: String,
    atol_frac: Float64 = 0.0,
) raises:
    # `atol_frac` adds an absolute tolerance of that fraction of the largest
    # reference magnitude, so signed data that cancels near zero stays
    # meaningful.
    var y_max = Float32(0)
    for i in range(y_size):
        y_max = max(y_max, abs(y_ref.unsafe_load(i)))
    var s_max = Float32(0)
    for i in range(batch * state_size):
        s_max = max(s_max, abs(fs_ref.unsafe_load(i)))
    var y_atol = atol_frac * Float64(y_max)
    var s_atol = atol_frac * Float64(s_max)
    for i in range(y_size):
        assert_almost_equal(
            y_ref.unsafe_load(i),
            Float32(y.unsafe_load(i)),
            rtol=rtol,
            atol=y_atol,
            msg=label + " y mismatch at index " + String(i),
        )
    for i in range(state_size):
        assert_almost_equal(
            Float32(0),
            Float32(pool.unsafe_load(i)),
            msg=label + " wrote unused slot 0",
        )
    for b in range(batch):
        var slot = Int(slots.unsafe_load(b))
        for i in range(state_size):
            assert_almost_equal(
                fs_ref.unsafe_load(b * state_size + i),
                Float32(pool.unsafe_load(slot * state_size + i)),
                rtol=rtol,
                atol=s_atol,
                msg=label
                + " final state mismatch at batch="
                + String(b)
                + " i="
                + String(i),
            )


def _bf16_ulp_distance(a: BFloat16, b: BFloat16) -> Int:
    """Number of representable bf16 steps between `a` and `b`."""

    def ordered(v: BFloat16) -> Int:
        var bits = Int(v.to_bits())
        return -(bits & 0x7FFF) if bits & 0x8000 else bits

    return abs(ordered(a) - ordered(b))


def _check_tile_parity[
    dtype: DType, state_dtype: DType
](
    y_base: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    y_new: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    pool_base: MutPointer[Scalar[state_dtype], MutUntrackedOrigin],
    pool_new: MutPointer[Scalar[state_dtype], MutUntrackedOrigin],
    y_size: Int,
    pool_size: Int,
) raises:
    """Two tiles on the same input: identical final-state bits, and outputs
    within one bf16 ulp (only the y reduction order differs)."""
    comptime assert dtype == .bfloat16, "y parity is stated in bf16 ulps"
    for i in range(pool_size):
        if (
            pool_base.unsafe_load(i).to_bits()
            != pool_new.unsafe_load(i).to_bits()
        ):
            raise Error(
                "tile parity: final state bits differ at pool index "
                + String(i)
                + ": "
                + String(pool_base.unsafe_load(i))
                + " vs "
                + String(pool_new.unsafe_load(i))
            )
    for i in range(y_size):
        var a = y_base.unsafe_load(i).cast[.bfloat16]()
        var b = y_new.unsafe_load(i).cast[.bfloat16]()
        if _bf16_ulp_distance(a, b) > 1:
            raise Error(
                "tile parity: y differs by more than one bf16 ulp at index "
                + String(i)
                + ": "
                + String(a)
                + " vs "
                + String(b)
            )


def run_mamba2_ssd[
    dtype: DType,
    DSTATE: Int,
    # 0 runs the Apple vectorized kernel; otherwise the cooperative split.
    DSTATE_SPLIT: Int = 0,
    # SSM-pool storage dtype of the GPU run; the CPU kernel stays fp32.
    state_dtype: DType = .float32,
    # Seeds the pool with state_dtype-representable initial states and sets
    # has_initial_state, so the state load path runs.
    init_state: Bool = False,
    has_D: Bool = True,
    has_dt_bias: Bool = True,
    # Clears update_state on every odd sequence: its slot must keep its initial
    # state while the outputs still match the reference.
    update_flags: Bool = False,
    # Channels per thread of the cooperative split, and whether its staged
    # (multi-token) path is enabled, as the production launch picks it.
    CH_PER_THREAD: Int = 1,
    prefill: Bool = True,
    # Also runs the previous tile (split 8, one channel per thread) on the
    # same inputs and initial pool, and requires bit-identical final states
    # and outputs within one bf16 ulp of it.
    check_parity: Bool = False,
    # Scale of the random decay rate: A = -(u * A_scale + 0.1), u in [0, 1).
    A_scale: Float32 = 1.0,
    # Draws x, B, C and the initial states from [-1, 1) instead of [0, 1).
    signed: Bool = False,
    # Offset added to the random dt before softplus: -9 gives dt ~ 1e-4, which
    # leaves most of the state alive across chunk boundaries.
    dt_offset: Float32 = -0.5,
    guard_output: Bool = False,
    grouped_pair: Bool = False,
    grouped_mode: Int = GROUPED_CB_ON,
    # Small forced-ON cases gate scalar math; large AUTO cases use exact GPU A/B.
    check_reference: Bool = True,
    use_grouped_default: Bool = False,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.02,
    row_pad: Int = 0,
    atol_frac: Float64 = 0.0,
    slot_stride: Int = 1,
) raises:
    """Run the CPU and one GPU kernel and compare both against the reference.

    `row_pad` widens the token stride of x, B and C, like the column slices of
    the conv output the model passes.
    """
    comptime dstate = DSTATE
    comptime if not check_reference:
        comptime assert (
            grouped_pair
        ), "GPU-only checks require strict grouped A/B"
    var batch = len(seq_lengths)
    var total_len = 0
    for i in range(batch):
        total_len += seq_lengths[i]

    var x_row = nheads * head_dim + row_pad
    var bc_row = ngroups * dstate + row_pad
    var D_size = nheads if has_D else 0
    var dt_bias_size = nheads if has_dt_bias else 0
    var y_size = total_len * nheads * head_dim
    var state_size = nheads * head_dim * dstate
    var pool_size = max_slots * state_size
    assert_true(slot_stride > 0, "slot stride must be positive")
    assert_true(
        max_slots > 1 + (batch - 1) * slot_stride,
        "pool must contain every scattered slot",
    )

    var x_h = unsafe_alloc[Scalar[dtype]](total_len * x_row)
    var dt_h = unsafe_alloc[Scalar[dtype]](total_len * nheads)
    var A_h = unsafe_alloc[Scalar[dtype]](nheads)
    var B_h = unsafe_alloc[Scalar[dtype]](total_len * bc_row)
    var C_h = unsafe_alloc[Scalar[dtype]](total_len * bc_row)
    var D_h = unsafe_alloc[Scalar[dtype]](nheads)
    var dt_bias_h = unsafe_alloc[Scalar[dtype]](nheads)
    var his_h = unsafe_alloc[Scalar[.bool]](batch)
    var us_h = unsafe_alloc[Scalar[.bool]](batch)
    var qsl_h = unsafe_alloc[Int32](batch + 1)
    var slot_h = unsafe_alloc[UInt32](batch)
    var is_h = unsafe_alloc[Float32](batch * state_size)

    comptime if not check_reference:
        seed(1234)
    rand(x_h, total_len * x_row)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * bc_row)
    rand(C_h, total_len * bc_row)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    rand(is_h, batch * state_size)
    comptime if signed:
        for i in range(total_len * x_row):
            x_h.unsafe_store(
                i, Scalar[dtype](Float32(x_h.unsafe_load(i)) * 2 - 1)
            )
        for i in range(total_len * bc_row):
            B_h.unsafe_store(
                i, Scalar[dtype](Float32(B_h.unsafe_load(i)) * 2 - 1)
            )
            C_h.unsafe_store(
                i, Scalar[dtype](Float32(C_h.unsafe_load(i)) * 2 - 1)
            )
        for i in range(batch * state_size):
            is_h.unsafe_store(i, is_h.unsafe_load(i) * 2 - 1)
    # A = -exp(A_log) is negative; keep it in a range where the recurrence
    # stays bounded over the sequence.
    for i in range(nheads):
        A_h.unsafe_store(
            i, Scalar[dtype](Float32(A_h.unsafe_load(i)) * -A_scale - 0.1)
        )
    for i in range(total_len * nheads):
        dt_h.unsafe_store(
            i, Scalar[dtype](Float32(dt_h.unsafe_load(i)) + dt_offset)
        )
    # Pre-round the initial states to state_dtype so the fp32 reference and a
    # bf16 pool start from identical states.
    for i in range(batch * state_size):
        is_h.unsafe_store(
            i, is_h.unsafe_load(i).cast[state_dtype]().cast[.float32]()
        )
    for i in range(batch):
        his_h.unsafe_store(i, Scalar[.bool](init_state))
        us_h.unsafe_store(i, Scalar[.bool](not update_flags or i % 2 == 0))
    # An empty update_state writes every sequence.
    var us_size = batch if update_flags else 0
    # Slot 0 and gaps between scattered slots must stay untouched.
    for i in range(batch):
        slot_h.unsafe_store(i, UInt32(1 + i * slot_stride))
    var cum = 0
    qsl_h.unsafe_store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.unsafe_store(i + 1, Int32(cum))

    var pool_init_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        pool_init_h.unsafe_store(i, Scalar[state_dtype](0))
    for b in range(batch):
        var slot = Int(slot_h.unsafe_load(b))
        for i in range(state_size):
            pool_init_h.unsafe_store(
                slot * state_size + i,
                is_h.unsafe_load(b * state_size + i).cast[state_dtype](),
            )

    comptime if not check_reference:
        var y_check_h = unsafe_alloc[Scalar[dtype]](
            y_size + (64 if guard_output else 0)
        )
        run_ssd_gpu[
            DSTATE,
            DSTATE_SPLIT,
            CH_PER_THREAD,
            prefill,
            guard_output=guard_output,
            grouped_pair=grouped_pair,
            grouped_mode=grouped_mode,
            use_grouped_default=use_grouped_default,
        ](
            ctx,
            nheads,
            head_dim,
            ngroups,
            max_slots,
            batch,
            total_len,
            x_row,
            bc_row,
            D_size,
            dt_bias_size,
            batch if init_state else 0,
            us_size,
            x_h,
            dt_h,
            A_h,
            B_h,
            C_h,
            D_h,
            dt_bias_h,
            qsl_h,
            his_h,
            us_h,
            slot_h,
            y_check_h,
            pool_init_h,
        )
        y_check_h.unsafe_free()
        x_h.unsafe_free()
        dt_h.unsafe_free()
        A_h.unsafe_free()
        B_h.unsafe_free()
        C_h.unsafe_free()
        D_h.unsafe_free()
        dt_bias_h.unsafe_free()
        his_h.unsafe_free()
        us_h.unsafe_free()
        qsl_h.unsafe_free()
        slot_h.unsafe_free()
        is_h.unsafe_free()
        pool_init_h.unsafe_free()
        return

    else:
        var y_ref_h = unsafe_alloc[Float32](y_size)
        var fs_ref_h = unsafe_alloc[Float32](batch * state_size)

        _reference_scan(
            nheads,
            head_dim,
            ngroups,
            dstate,
            seq_lengths,
            x_h,
            x_row,
            dt_h,
            A_h,
            B_h,
            C_h,
            bc_row,
            D_h,
            has_D,
            dt_bias_h,
            has_dt_bias,
            is_h,
            init_state,
            y_ref_h,
            fs_ref_h,
        )

        comptime if update_flags:
            # A sequence that does not update leaves its slot as it was seeded.
            for b in range(batch):
                if b % 2 == 1:
                    var slot = Int(slot_h.unsafe_load(b))
                    for i in range(state_size):
                        fs_ref_h.unsafe_store(
                            b * state_size + i,
                            pool_init_h.unsafe_load(slot * state_size + i).cast[
                                .float32
                            ](),
                        )

        var x_layout = MixedLayout(
            (total_len, nheads, head_dim), (x_row, head_dim, 1)
        )
        var bc_layout = MixedLayout(
            (total_len, ngroups, dstate), (bc_row, dstate, 1)
        )
        var his_size = batch if init_state else 0

        # ---- CPU kernel (fp32 pool) ----
        var y_cpu_h = unsafe_alloc[Scalar[dtype]](
            y_size + (64 if guard_output else 0)
        )
        var pool_cpu_h = unsafe_alloc[Float32](pool_size)
        for i in range(pool_size):
            pool_cpu_h.unsafe_store(
                i, pool_init_h.unsafe_load(i).cast[.float32]()
            )
        _scan_cpu[DSTATE](
            TileTensor(x_h, x_layout),
            TileTensor(dt_h, row_major(total_len, nheads)),
            TileTensor(A_h, row_major(nheads)),
            TileTensor(B_h, bc_layout),
            TileTensor(C_h, bc_layout),
            TileTensor(D_h, row_major(D_size)),
            TileTensor(dt_bias_h, row_major(dt_bias_size)),
            TileTensor(y_cpu_h, row_major(total_len, nheads, head_dim)),
            TileTensor(
                pool_cpu_h, row_major(max_slots, nheads, head_dim, dstate)
            ),
            TileTensor(qsl_h, row_major(batch + 1)),
            TileTensor(his_h, row_major(his_size)),
            TileTensor(us_h, row_major(us_size)),
            TileTensor(slot_h, row_major(batch)),
        )
        _check_outputs(
            y_ref_h,
            fs_ref_h,
            slot_h,
            batch,
            y_size,
            state_size,
            y_cpu_h,
            pool_cpu_h,
            rtol,
            "CPU",
            atol_frac,
        )

        # ---- GPU kernel ----
        var y_gpu_h = unsafe_alloc[Scalar[dtype]](
            y_size + (64 if guard_output else 0)
        )
        var pool_gpu_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
        for i in range(pool_size):
            pool_gpu_h.unsafe_store(i, pool_init_h.unsafe_load(i))
        run_ssd_gpu[
            DSTATE,
            DSTATE_SPLIT,
            CH_PER_THREAD,
            prefill,
            guard_output=guard_output,
            grouped_pair=grouped_pair,
            grouped_mode=grouped_mode,
            use_grouped_default=use_grouped_default,
        ](
            ctx,
            nheads,
            head_dim,
            ngroups,
            max_slots,
            batch,
            total_len,
            x_row,
            bc_row,
            D_size,
            dt_bias_size,
            his_size,
            us_size,
            x_h,
            dt_h,
            A_h,
            B_h,
            C_h,
            D_h,
            dt_bias_h,
            qsl_h,
            his_h,
            us_h,
            slot_h,
            y_gpu_h,
            pool_gpu_h,
        )
        # A bf16 pool adds one rounding at the final write-back (<= 2^-9
        # relative), well inside rtol.
        _check_outputs(
            y_ref_h,
            fs_ref_h,
            slot_h,
            batch,
            y_size,
            state_size,
            y_gpu_h,
            pool_gpu_h,
            rtol,
            "GPU",
            atol_frac,
        )

        if guard_output or slot_stride != 1 or update_flags:
            check_preserved_slots(
                pool_gpu_h,
                pool_init_h,
                slot_h,
                us_h,
                us_size,
                batch,
                state_size,
                max_slots,
            )

        comptime if DSTATE_SPLIT < 0:
            # The chunked scan rounds its matmul operands (decayed scores, chunk
            # states) to bf16, so it is compared with the serial split kernel
            # (split 16, 4 channels per thread, the production decode tile) at a
            # bf16-sized tolerance relative to the largest output and state.
            var y_ser_h = unsafe_alloc[Scalar[dtype]](
                y_size + (64 if guard_output else 0)
            )
            var pool_ser_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
            for i in range(pool_size):
                pool_ser_h.unsafe_store(i, pool_init_h.unsafe_load(i))
            run_ssd_gpu[DSTATE, 16, 4, prefill, guard_output=guard_output](
                ctx,
                nheads,
                head_dim,
                ngroups,
                max_slots,
                batch,
                total_len,
                x_row,
                bc_row,
                D_size,
                dt_bias_size,
                his_size,
                us_size,
                x_h,
                dt_h,
                A_h,
                B_h,
                C_h,
                D_h,
                dt_bias_h,
                qsl_h,
                his_h,
                us_h,
                slot_h,
                y_ser_h,
                pool_ser_h,
            )
            if guard_output or slot_stride != 1 or update_flags:
                check_preserved_slots(
                    pool_ser_h,
                    pool_init_h,
                    slot_h,
                    us_h,
                    us_size,
                    batch,
                    state_size,
                    max_slots,
                )
            check_chunked_parity(
                y_ser_h, y_gpu_h, pool_ser_h, pool_gpu_h, y_size, pool_size
            )
            # Determinism: no atomics, so a second launch on the same inputs and
            # initial pool must match the first bit for bit.
            var y_rep_h = unsafe_alloc[Scalar[dtype]](
                y_size + (64 if guard_output else 0)
            )
            var pool_rep_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
            for i in range(pool_size):
                pool_rep_h.unsafe_store(i, pool_init_h.unsafe_load(i))
            run_ssd_gpu[
                DSTATE,
                DSTATE_SPLIT,
                CH_PER_THREAD,
                prefill,
                guard_output=guard_output,
                grouped_pair=grouped_pair,
                grouped_mode=grouped_mode,
                use_grouped_default=use_grouped_default,
            ](
                ctx,
                nheads,
                head_dim,
                ngroups,
                max_slots,
                batch,
                total_len,
                x_row,
                bc_row,
                D_size,
                dt_bias_size,
                his_size,
                us_size,
                x_h,
                dt_h,
                A_h,
                B_h,
                C_h,
                D_h,
                dt_bias_h,
                qsl_h,
                his_h,
                us_h,
                slot_h,
                y_rep_h,
                pool_rep_h,
            )
            for i in range(y_size):
                if y_rep_h.unsafe_load(i) != y_gpu_h.unsafe_load(i):
                    raise Error(
                        "chunked y is not deterministic at index " + String(i)
                    )
            for i in range(pool_size):
                if pool_rep_h.unsafe_load(i) != pool_gpu_h.unsafe_load(i):
                    raise Error(
                        "chunked state is not deterministic at pool index "
                        + String(i)
                    )
            y_rep_h.unsafe_free()
            pool_rep_h.unsafe_free()
            y_ser_h.unsafe_free()
            pool_ser_h.unsafe_free()

        comptime if check_parity:
            comptime assert DSTATE_SPLIT > 0, "parity compares split tiles"
            var y_base_h = unsafe_alloc[Scalar[dtype]](
                y_size + (64 if guard_output else 0)
            )
            var pool_base_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
            for i in range(pool_size):
                pool_base_h.unsafe_store(i, pool_init_h.unsafe_load(i))
            run_ssd_gpu[DSTATE, 8, 1, prefill, guard_output=guard_output](
                ctx,
                nheads,
                head_dim,
                ngroups,
                max_slots,
                batch,
                total_len,
                x_row,
                bc_row,
                D_size,
                dt_bias_size,
                his_size,
                us_size,
                x_h,
                dt_h,
                A_h,
                B_h,
                C_h,
                D_h,
                dt_bias_h,
                qsl_h,
                his_h,
                us_h,
                slot_h,
                y_base_h,
                pool_base_h,
            )
            _check_tile_parity(
                y_base_h, y_gpu_h, pool_base_h, pool_gpu_h, y_size, pool_size
            )
            y_base_h.unsafe_free()
            pool_base_h.unsafe_free()

        x_h.unsafe_free()
        dt_h.unsafe_free()
        A_h.unsafe_free()
        B_h.unsafe_free()
        C_h.unsafe_free()
        D_h.unsafe_free()
        dt_bias_h.unsafe_free()
        his_h.unsafe_free()
        us_h.unsafe_free()
        qsl_h.unsafe_free()
        slot_h.unsafe_free()
        is_h.unsafe_free()
        y_ref_h.unsafe_free()
        fs_ref_h.unsafe_free()
        pool_init_h.unsafe_free()
        y_cpu_h.unsafe_free()
        pool_cpu_h.unsafe_free()
        y_gpu_h.unsafe_free()
        pool_gpu_h.unsafe_free()


def test_mamba2_ssd_varlen_no_cross_sequence_bleed() raises:
    """A packed ragged batch equals independent per-sequence runs."""
    comptime dtype = DType.float32
    comptime dstate = 16
    var nheads = 8
    var head_dim = 16
    var ngroups = 2
    var seq_lengths = Index(9, 4, 6)
    var batch = len(seq_lengths)
    var total_len = 19
    var state_size = nheads * head_dim * dstate

    var x_h = unsafe_alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var dt_h = unsafe_alloc[Scalar[dtype]](total_len * nheads)
    var A_h = unsafe_alloc[Scalar[dtype]](nheads)
    var B_h = unsafe_alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var C_h = unsafe_alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var D_h = unsafe_alloc[Scalar[dtype]](nheads)
    var dt_bias_h = unsafe_alloc[Scalar[dtype]](nheads)
    var his_h = unsafe_alloc[Scalar[.bool]](1)
    rand(x_h, total_len * nheads * head_dim)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * ngroups * dstate)
    rand(C_h, total_len * ngroups * dstate)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    for i in range(nheads):
        A_h.unsafe_store(i, Float32(A_h.unsafe_load(i)) * -1.0 - 0.1)
    for i in range(total_len * nheads):
        dt_h.unsafe_store(i, Float32(dt_h.unsafe_load(i)) - 0.5)

    var A_tt = TileTensor(A_h, row_major(nheads))
    var D_tt = TileTensor(D_h, row_major(nheads))
    var dt_bias_tt = TileTensor(dt_bias_h, row_major(nheads))
    var his_tt = TileTensor(his_h, row_major(0))
    var us_tt = TileTensor(his_h, row_major(0))

    # Packed run: sequence b writes pool slot b.
    var y_packed = unsafe_alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var pool_packed = unsafe_alloc[Float32](batch * state_size)
    var qsl_h = unsafe_alloc[Int32](batch + 1)
    var slot_h = unsafe_alloc[UInt32](batch)
    var cum = 0
    qsl_h.unsafe_store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.unsafe_store(i + 1, Int32(cum))
        slot_h.unsafe_store(i, UInt32(i))
    _scan_cpu[dstate](
        TileTensor(x_h, row_major(total_len, nheads, head_dim)),
        TileTensor(dt_h, row_major(total_len, nheads)),
        A_tt,
        TileTensor(B_h, row_major(total_len, ngroups, dstate)),
        TileTensor(C_h, row_major(total_len, ngroups, dstate)),
        D_tt,
        dt_bias_tt,
        TileTensor(y_packed, row_major(total_len, nheads, head_dim)),
        TileTensor(pool_packed, row_major(batch, nheads, head_dim, dstate)),
        TileTensor(qsl_h, row_major(batch + 1)),
        his_tt,
        us_tt,
        TileTensor(slot_h, row_major(batch)),
    )

    # Independent runs over views into the packed buffers.
    var y_indep = unsafe_alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var pool_indep = unsafe_alloc[Float32](batch * state_size)
    var qsl_s_h = unsafe_alloc[Int32](2)
    var slot_s_h = unsafe_alloc[UInt32](1)
    qsl_s_h.unsafe_store(0, Int32(0))
    slot_s_h.unsafe_store(0, UInt32(0))
    var off = 0
    for s in range(batch):
        var slen = seq_lengths[s]
        qsl_s_h.unsafe_store(1, Int32(slen))
        _scan_cpu[dstate](
            TileTensor(
                x_h.unsafe_offset(off * nheads * head_dim),
                row_major(slen, nheads, head_dim),
            ),
            TileTensor(
                dt_h.unsafe_offset(off * nheads), row_major(slen, nheads)
            ),
            A_tt,
            TileTensor(
                B_h.unsafe_offset(off * ngroups * dstate),
                row_major(slen, ngroups, dstate),
            ),
            TileTensor(
                C_h.unsafe_offset(off * ngroups * dstate),
                row_major(slen, ngroups, dstate),
            ),
            D_tt,
            dt_bias_tt,
            TileTensor(
                y_indep.unsafe_offset(off * nheads * head_dim),
                row_major(slen, nheads, head_dim),
            ),
            TileTensor(
                pool_indep.unsafe_offset(s * state_size),
                row_major(1, nheads, head_dim, dstate),
            ),
            TileTensor(qsl_s_h, row_major(2)),
            his_tt,
            us_tt,
            TileTensor(slot_s_h, row_major(1)),
        )
        off += slen

    # Same fp32 ops in the same order, so the runs agree to rounding.
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_packed.unsafe_load(i)),
            Float32(y_indep.unsafe_load(i)),
            rtol=1e-5,
        )
    for i in range(batch * state_size):
        assert_almost_equal(
            pool_packed.unsafe_load(i), pool_indep.unsafe_load(i), rtol=1e-5
        )

    x_h.unsafe_free()
    dt_h.unsafe_free()
    A_h.unsafe_free()
    B_h.unsafe_free()
    C_h.unsafe_free()
    D_h.unsafe_free()
    dt_bias_h.unsafe_free()
    his_h.unsafe_free()
    y_packed.unsafe_free()
    pool_packed.unsafe_free()
    qsl_h.unsafe_free()
    slot_h.unsafe_free()
    y_indep.unsafe_free()
    pool_indep.unsafe_free()
    qsl_s_h.unsafe_free()
    slot_s_h.unsafe_free()


def test_mamba2_ssd_dstate_split() raises:
    """Cooperative DSTATE-split kernel, the production CUDA/HIP path.

    Sweeps DSTATE_SPLIT in {2, 4, 8} (8 is the production choice) over the
    96h/8g/dstate128 Nemotron-H grouping, plus the seqlen-1 decode shape the
    split targets and a DSTATE=16 tiling.
    """
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            return
        run_mamba2_ssd[.bfloat16, 128, 2](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, 4](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, 8](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, 8](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=8,
            seq_lengths=Index(1, 1, 1, 1),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 16, 4](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=8,
            seq_lengths=Index(6, 4),
            ctx=ctx,
        )
        run_mamba2_ssd[.float32, 64, 8](
            nheads=12,
            head_dim=16,
            ngroups=4,
            max_slots=4,
            seq_lengths=Index(10, 6, 1),
            ctx=ctx,
            rtol=1e-4,
        )


def test_mamba2_ssd_dstate_split_strided_with_initial_state() raises:
    """Nemotron-3.5-Lightning shapes: a seeded initial state, and x/B/C as
    strided column slices, for batch-64 decode and a short ragged prefill."""
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            return
        run_mamba2_ssd[.bfloat16, 128, 8, init_state=True](
            nheads=64,
            head_dim=64,
            ngroups=8,
            max_slots=65,
            seq_lengths=IndexList[64](1),
            ctx=ctx,
            row_pad=32,
        )
        run_mamba2_ssd[.bfloat16, 128, 8, init_state=True](
            nheads=64,
            head_dim=64,
            ngroups=8,
            max_slots=5,
            seq_lengths=Index(5, 1, 17, 2),
            ctx=ctx,
            row_pad=32,
        )


def test_mamba2_ssd_dstate_split_chunk_boundaries() raises:
    """Sequences around the staged chunk size (8 tokens) that mix full chunks
    with a scalar tail, with and without an initial state."""
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            return
        run_mamba2_ssd[.bfloat16, 128, 8, init_state=True](
            nheads=64,
            head_dim=64,
            ngroups=8,
            max_slots=5,
            seq_lengths=Index(8, 16, 37, 130),
            ctx=ctx,
            row_pad=32,
        )
        run_mamba2_ssd[.bfloat16, 128, 8](
            nheads=64,
            head_dim=64,
            ngroups=8,
            max_slots=5,
            seq_lengths=Index(9, 7, 24, 100),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, 8](
            nheads=8,
            head_dim=72,
            ngroups=2,
            max_slots=4,
            seq_lengths=Index(17, 9),
            ctx=ctx,
        )


def _channel_blocked_decode_shapes[
    update_flags: Bool
](ctx: DeviceContext) raises:
    """Verify/replay rows 1 to 8 (one masked staged chunk) and a decode."""
    run_mamba2_ssd[
        .bfloat16,
        128,
        16,
        init_state=True,
        update_flags=update_flags,
        CH_PER_THREAD=4,
        check_parity=True,
    ](
        nheads=32,
        head_dim=64,
        ngroups=4,
        max_slots=9,
        seq_lengths=Index(1, 2, 3, 4, 5, 6, 7, 8),
        ctx=ctx,
        row_pad=32,
    )
    run_mamba2_ssd[
        .bfloat16,
        128,
        16,
        init_state=True,
        update_flags=update_flags,
        CH_PER_THREAD=4,
        prefill=False,
        check_parity=True,
    ](
        nheads=32,
        head_dim=64,
        ngroups=4,
        max_slots=5,
        seq_lengths=Index(1, 1, 1, 1),
        ctx=ctx,
    )


def test_mamba2_ssd_dstate_split_channel_blocked() raises:
    """Several channels per thread, the production tiling for dstate 128.

    Nemotron-3.5-Lightning per-device grouping (32 heads, 8 per group,
    head_dim 64) at the speculative decode shapes: every verify/replay length
    from 1 to 8 rows, which the staged path takes as one masked chunk, longer
    sequences that end in a partial chunk, and the single-token decode
    launch. Each runs with every slot updated and with alternate slots
    left untouched. A head_dim that is not a multiple of the block's channels
    covers the channel guards. The bf16 cases also run the previous split-8,
    one-channel tile on the same input: final states must match it bit for
    bit and outputs within one bf16 ulp.
    """
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            return
        _channel_blocked_decode_shapes[update_flags=False](ctx)
        _channel_blocked_decode_shapes[update_flags=True](ctx)
        run_mamba2_ssd[
            .bfloat16,
            128,
            16,
            init_state=True,
            CH_PER_THREAD=4,
            check_parity=True,
        ](
            nheads=32,
            head_dim=64,
            ngroups=4,
            max_slots=4,
            seq_lengths=Index(9, 21, 40),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, 16, CH_PER_THREAD=4, check_parity=True](
            nheads=8,
            head_dim=72,
            ngroups=2,
            max_slots=4,
            seq_lengths=Index(13, 1, 6),
            ctx=ctx,
        )
        run_mamba2_ssd[
            .bfloat16,
            128,
            16,
            CH_PER_THREAD=4,
            prefill=False,
            check_parity=True,
        ](
            nheads=8,
            head_dim=72,
            ngroups=2,
            max_slots=4,
            seq_lengths=Index(3, 1, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.float32, 64, 8, CH_PER_THREAD=2](
            nheads=12,
            head_dim=16,
            ngroups=4,
            max_slots=4,
            seq_lengths=Index(10, 6, 1),
            ctx=ctx,
            rtol=1e-4,
        )


def test_mamba2_ssd_no_D_no_bias() raises:
    """Empty D and dt_bias drop their terms."""
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            return
        run_mamba2_ssd[.bfloat16, 16, 8, has_D=False, has_dt_bias=False](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=4,
            seq_lengths=Index(7, 7),
            ctx=ctx,
        )


def test_mamba2_ssd_apple() raises:
    """Apple vectorized kernel. It is portable fp32 SIMD, so it runs on any
    GPU here for cross-hardware coverage."""
    with DeviceContext() as ctx:
        run_mamba2_ssd[.bfloat16, 128](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=8,
            seq_lengths=Index(1, 1, 1, 1),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 16](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=8,
            seq_lengths=Index(6, 4),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, init_state=True](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )


def test_mamba2_ssd_bf16_state() raises:
    """Checks bf16 SSM-state storage against the fp32 reference.

    `init_state=True` runs both bf16 boundaries: the initial-state widen and
    the final write-back round. The initial values are pre-rounded, so the
    only bf16 effect is that one final rounding; a kernel that carried bf16
    through the per-token recurrence would drift on the long sequences.
    """
    with DeviceContext() as ctx:
        run_mamba2_ssd[.bfloat16, 128, state_dtype=.bfloat16, init_state=True](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        run_mamba2_ssd[.bfloat16, 128, state_dtype=.bfloat16, init_state=True](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=4,
            seq_lengths=Index(64, 33),
            ctx=ctx,
        )
        comptime if ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu():
            run_mamba2_ssd[
                .bfloat16, 128, 8, state_dtype=.bfloat16, init_state=True
            ](
                nheads=8,
                head_dim=16,
                ngroups=2,
                max_slots=4,
                seq_lengths=Index(64, 1),
                ctx=ctx,
            )


def test_mamba2_ssd_update_state_flag() raises:
    """A cleared `update_state` flag leaves that sequence's slot untouched.

    Odd sequences keep their seeded state while even ones advance it, on the
    CPU kernel, the Apple kernel and (on CUDA/HIP) the cooperative split, for
    a decode batch and a prefill long enough to take the staged path.
    """
    with DeviceContext() as ctx:
        run_mamba2_ssd[.bfloat16, 128, init_state=True, update_flags=True](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=6,
            seq_lengths=Index(1, 1, 3, 1, 2),
            ctx=ctx,
        )
        comptime if ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu():
            run_mamba2_ssd[
                .bfloat16, 128, 8, init_state=True, update_flags=True
            ](
                nheads=8,
                head_dim=16,
                ngroups=2,
                max_slots=6,
                seq_lengths=Index(1, 1, 3, 1, 2),
                ctx=ctx,
            )
            run_mamba2_ssd[
                .bfloat16, 128, 8, init_state=True, update_flags=True
            ](
                nheads=8,
                head_dim=16,
                ngroups=2,
                max_slots=5,
                seq_lengths=Index(17, 9, 24, 1),
                ctx=ctx,
            )


def test_mamba2_ssd_chunked() raises:
    """The chunked prefill scan matches the fp32 reference.

    Lengths are not multiples of the 256-token chunk, a one-token sequence and
    a sub-chunk sequence share the batch with multi-chunk ones, and the pool
    seeds nonzero initial states.
    """
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        # One sequence, 2 full chunks + a 188-token tail, nonzero state.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            8, 64, 2, 2, Index(700), ctx
        )
        # Varlen batch, fresh state.
        run_mamba2_ssd[DType.bfloat16, 128, -1](
            8, 64, 2, 6, Index(300, 1, 513, 256, 77), ctx
        )
        # A long sequence: 16 full chunks plus a tail, so the state passes
        # through many chunk boundaries, next to a short one.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            4, 64, 2, 3, Index(4133, 300), ctx
        )
        # Varlen batch, initial states, strided inputs, no write-back for odd
        # sequences.
        run_mamba2_ssd[
            DType.bfloat16, 128, -1, init_state=True, update_flags=True
        ](8, 64, 4, 6, Index(257, 5, 1000, 64, 256), ctx, row_pad=8)


def test_mamba2_ssd_chunked_boundaries() raises:
    """Sequence lengths around the 256-token chunk size, one launch each.

    Tolerances: the chunked scan rounds the decayed score matrix and the carry
    state to bf16 before the matmuls (relative error 2^-9 per operand, fp32
    accumulation), so y is checked against the fp32 reference at rtol 0.02 and
    against the serial kernel at 1% of the largest output.
    """
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        # 1, 255, 256, 257 (chunk -1/0/+1), 511, 512, 513 (two chunks -1/0/+1),
        # fresh and seeded states.
        run_mamba2_ssd[DType.bfloat16, 128, -1](
            4, 64, 2, 8, Index(1, 255, 256, 257, 511, 512, 513), ctx
        )
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            4, 64, 2, 8, Index(1, 255, 256, 257, 511, 512, 513), ctx
        )
        # A single exact multiple of the chunk, and a sub-chunk sequence next
        # to a multi-chunk one.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            4, 64, 2, 2, Index(1024), ctx
        )
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            4, 64, 2, 3, Index(5, 1300), ctx
        )


def test_mamba2_ssd_chunked_many_sequences() raises:
    """Many sequences in one launch, mixing 1-token and multi-chunk rows."""
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            4,
            64,
            2,
            35,
            Index(
                1,
                300,
                2,
                257,
                64,
                1,
                512,
                3,
                128,
                255,
                1,
                600,
                7,
                256,
                90,
                1,
                33,
                1,
                513,
                20,
                1,
                1,
                400,
                2,
                17,
                255,
                256,
                1,
                100,
                1,
                129,
                3,
                2,
                70,
            ),
            ctx,
        )


def test_mamba2_ssd_chunked_shapes() raises:
    """Head layouts the Nemotron configs use, and optional inputs."""
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        # One group shared by every head; 8 heads per group; 2 heads per group
        # with head_dim 128 (two head tiles).
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            32, 64, 1, 3, Index(300, 513), ctx
        )
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            32, 64, 4, 3, Index(300, 513), ctx
        )
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True](
            8, 128, 4, 3, Index(300, 513), ctx
        )
        # No D skip and no dt_bias.
        run_mamba2_ssd[
            DType.bfloat16,
            128,
            -1,
            init_state=True,
            has_D=False,
            has_dt_bias=False,
        ](8, 64, 2, 3, Index(300, 513), ctx)
        # bf16 state pool: one extra rounding at the write-back.
        run_mamba2_ssd[
            DType.bfloat16, 128, -1, state_dtype=.bfloat16, init_state=True
        ](8, 64, 2, 3, Index(300, 513), ctx)
        # Seeded states with a cleared update_state on odd sequences, strided
        # inputs and non-contiguous pool slots (slot b + 1, slot 0 untouched).
        run_mamba2_ssd[
            DType.bfloat16, 128, -1, init_state=True, update_flags=True
        ](8, 64, 2, 7, Index(1, 256, 257, 300, 2, 700), ctx, row_pad=8)


def test_mamba2_ssd_chunked_decay_range() raises:
    """Very strong and very weak decay stay finite and match the reference."""
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        # Strong decay: exp2 of the cumulative log decay underflows to 0 within
        # a chunk; the carry and the masked scores must not produce NaN or inf.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True, A_scale=40.0](
            4, 64, 2, 3, Index(600, 300), ctx
        )
        # Weak decay: the state integrates over the whole sequence.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True, A_scale=0.005](
            4, 64, 2, 3, Index(1100, 300), ctx
        )


def test_mamba2_ssd_chunked_signed_slow_decay() raises:
    """Signed inputs, slow decay across many chunks, and a second head tile.

    Slow decay (dt ~ 1e-4, A in [-1.1, -0.1]) keeps most of the incoming state
    alive across each 256-token chunk, so the state-passing recurrence and the
    carry-in term are exercised over many boundaries, with a nonzero initial
    state. Signed x, B, C and states cancel inside the bf16-rounded score and
    state products; the comparison against the fp32 reference then adds an
    absolute tolerance of 2% of the largest value to the 2% relative one.
    """
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            return
        run_mamba2_ssd[
            DType.bfloat16, 128, -1, init_state=True, dt_offset=-9.0
        ](4, 64, 2, 3, Index(1800, 600), ctx)
        run_mamba2_ssd[
            DType.bfloat16,
            128,
            -1,
            init_state=True,
            dt_offset=-9.0,
            signed=True,
        ](4, 64, 2, 3, Index(1800, 600), ctx, atol_frac=0.02)
        # Signed, head_dim 128 (two channel tiles), varlen with tails.
        run_mamba2_ssd[DType.bfloat16, 128, -1, init_state=True, signed=True](
            8, 128, 4, 4, Index(300, 513, 1), ctx, atol_frac=0.02
        )
        # Signed with update flags and strided inputs.
        run_mamba2_ssd[
            DType.bfloat16,
            128,
            -1,
            init_state=True,
            update_flags=True,
            signed=True,
        ](8, 64, 2, 5, Index(257, 1, 700, 256), ctx, row_pad=8, atol_frac=0.02)


def test_mamba2_ssd_chunked_dispatch() raises:
    """The launch-shape rule behind the op: token and average-length edges."""
    comptime MIN = 512
    # Token threshold at batch 1 (average length equals the total).
    assert_false(use_chunked_ssd(511, 1, 64, MIN))
    assert_true(use_chunked_ssd(512, 1, 64, MIN))
    # Average sequence length 127 vs 128 with the token threshold met.
    assert_false(use_chunked_ssd(127 * 8, 8, 64, MIN))
    assert_true(use_chunked_ssd(128 * 8, 8, 64, MIN))
    assert_false(use_chunked_ssd(128 * 8 - 1, 8, 64, MIN))
    # A decode or verify batch (64 rows of 8 tokens) stays serial.
    assert_false(use_chunked_ssd(512, 64, 64, MIN))
    # head_dim must tile by 64.
    assert_false(use_chunked_ssd(4096, 1, 80, MIN))
    assert_true(use_chunked_ssd(4096, 1, 128, MIN))
    # A huge threshold pins the serial kernel.
    assert_false(use_chunked_ssd(70000, 1, 64, 1000000000))


def _check_ssd_grouped() raises:
    with DeviceContext() as ctx:
        comptime if not ctx.target._is_sm_8x_or_newer():
            raise Error("grouped SSD checks require NVIDIA SM80 or newer")
        seed(1234)
        run_mamba2_ssd[
            .bfloat16,
            128,
            -1,
            signed=True,
            dt_offset=-9.0,
            update_flags=True,
            guard_output=True,
            grouped_pair=True,
            grouped_mode=GROUPED_CB_ON,
        ](
            8,
            64,
            2,
            8,
            Index(257, 1, 513),
            ctx,
            row_pad=32,
            slot_stride=2,
            atol_frac=0.02,
        )
        run_mamba2_ssd[
            .bfloat16,
            128,
            -1,
            init_state=True,
            signed=True,
            dt_offset=-9.0,
            update_flags=True,
            guard_output=True,
            grouped_pair=True,
            grouped_mode=GROUPED_CB_ON,
        ](
            8,
            128,
            2,
            8,
            Index(255, 257, 1),
            ctx,
            row_pad=32,
            slot_stride=2,
            atol_frac=0.02,
        )
        run_mamba2_ssd[
            .bfloat16,
            128,
            -1,
            state_dtype=.bfloat16,
            init_state=True,
            signed=True,
            dt_offset=-9.0,
            update_flags=True,
            guard_output=True,
            grouped_pair=True,
            grouped_mode=GROUPED_CB_ON,
        ](
            8,
            64,
            2,
            8,
            Index(255, 257, 1),
            ctx,
            row_pad=32,
            slot_stride=2,
            atol_frac=0.02,
        )
        run_mamba2_ssd[
            .bfloat16,
            128,
            -1,
            init_state=True,
            signed=True,
            dt_offset=-9.0,
            update_flags=True,
            guard_output=True,
            grouped_pair=True,
            grouped_mode=GROUPED_CB_ON,
        ](
            8,
            64,
            8,
            8,
            Index(255, 257, 1),
            ctx,
            row_pad=32,
            slot_stride=2,
            atol_frac=0.02,
        )
        print("SSD_GROUPED_ORACLE_PASS,4")


def test_mamba2_ssd_grouped_policy() raises:
    assert_true(use_grouped_cb_ssd(8192, 1, 32, 128, 4))
    assert_true(use_grouped_cb_ssd(16384, 1, 16, 128, 2))
    assert_false(use_grouped_cb_ssd(16383, 1, 16, 128, 2))
    assert_true(use_grouped_cb_ssd(8192, 1, 64, 128, 8))
    assert_false(use_grouped_cb_ssd(8191, 1, 64, 128, 8))
    assert_true(use_grouped_cb_ssd(32768, 4, 32, 128, 4))
    assert_false(use_grouped_cb_ssd(32767, 4, 32, 128, 4))
    assert_true(use_grouped_cb_ssd(10923, 1, 24, 128, 3))
    assert_false(use_grouped_cb_ssd(10922, 1, 24, 128, 3))
    assert_false(use_grouped_cb_ssd(8192, 1, 32, 128, 8))
    assert_false(use_grouped_cb_ssd(8192, 1, 33, 128, 4))
    assert_false(use_grouped_cb_ssd(8192, 1, 32, 64, 4))
    assert_false(use_grouped_cb_ssd(8192, 1, 32, 80, 4))
    assert_false(use_grouped_cb_ssd(8192, 1, 32, 256, 4))
    assert_false(use_grouped_cb_ssd(8192, 0, 32, 128, 4))
    assert_false(use_grouped_cb_ssd(8192, -1, 32, 128, 4))
    assert_false(use_grouped_cb_ssd(8192, 1, 0, 128, 4))
    assert_false(use_grouped_cb_ssd(8192, 1, 32, 128, 0))
    assert_false(use_grouped_cb_ssd(0, 1, 32, 128, 4))
    assert_false(use_grouped_cb_ssd(-1, 1, 32, 128, 4))


def test_mamba2_ssd_grouped_oracle() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target._is_sm_8x_or_newer():
            _check_ssd_grouped()


def test_mamba2_ssd_grouped_auto() raises:
    with DeviceContext() as ctx:
        comptime if (
            ctx.target.is_nvidia_gpu["sm_100"]()
            or ctx.target.is_nvidia_gpu["sm_100a"]()
        ):
            assert_true(use_grouped_cb_ssd(8192, 1, 32, 128, 4))
            print("SSD_AUTO_DEVICE_NAME,", ctx.name(), sep="")
            if ctx.name() == "NVIDIA B200":
                run_mamba2_ssd[
                    .bfloat16,
                    128,
                    -1,
                    init_state=True,
                    guard_output=True,
                    grouped_pair=True,
                    grouped_mode=GROUPED_CB_AUTO,
                    check_reference=False,
                ](32, 128, 4, 3, Index(8192), ctx)
                print("SSD_AUTO_ENABLED_CHECK_PASS")
                run_mamba2_ssd[
                    .bfloat16,
                    128,
                    -1,
                    init_state=True,
                    guard_output=True,
                    grouped_pair=True,
                    grouped_mode=GROUPED_CB_AUTO,
                    check_reference=False,
                    use_grouped_default=True,
                ](32, 128, 4, 3, Index(8192), ctx)
                print("SSD_DEFAULT_ENABLED_CHECK_PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
