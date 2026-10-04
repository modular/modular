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

from max.gpu.host import DeviceContext
from layout import MixedLayout, TileTensor, row_major
from std.random import rand
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)
from std.sys import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


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
                    final_states.store(
                        base + n,
                        initial_states.load(base + n) if init_state else 0.0,
                    )
                for t in range(seq_start, seq_start + seq_lengths[b]):
                    var x_val = Float32(x.load(t * x_row + h * head_dim + p))
                    var delta = Float32(dt.load(t * nheads + h))
                    if has_dt_bias:
                        delta += Float32(dt_bias.load(h))
                    delta = log1p(exp(delta))
                    var dA = exp(Float32(A.load(h)) * delta)
                    var y_val = Float32(0)
                    for n in range(dstate):
                        var bc = t * bc_row + g * dstate + n
                        var s = (
                            final_states.load(base + n) * dA
                            + Float32(B.load(bc)) * delta * x_val
                        )
                        final_states.store(base + n, s)
                        y_val += s * Float32(C.load(bc))
                    if has_D:
                        y_val += Float32(D.load(h)) * x_val
                    y.store((t * nheads + h) * head_dim + p, y_val)
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
) raises:
    for i in range(y_size):
        assert_almost_equal(
            y_ref.load(i),
            Float32(y.load(i)),
            rtol=rtol,
            msg=label + " y mismatch at index " + String(i),
        )
    for i in range(state_size):
        assert_almost_equal(
            Float32(0),
            Float32(pool.load(i)),
            msg=label + " wrote unused slot 0",
        )
    for b in range(batch):
        var slot = Int(slots.load(b))
        for i in range(state_size):
            assert_almost_equal(
                fs_ref.load(b * state_size + i),
                Float32(pool.load(slot * state_size + i)),
                rtol=rtol,
                msg=label
                + " final state mismatch at batch="
                + String(b)
                + " i="
                + String(i),
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
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.02,
    row_pad: Int = 0,
) raises:
    """Run the CPU and one GPU kernel and compare both against the reference.

    `row_pad` widens the token stride of x, B and C, like the column slices of
    the conv output the model passes.
    """
    comptime dstate = DSTATE
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

    var x_h = alloc[Scalar[dtype]](total_len * x_row)
    var dt_h = alloc[Scalar[dtype]](total_len * nheads)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total_len * bc_row)
    var C_h = alloc[Scalar[dtype]](total_len * bc_row)
    var D_h = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = alloc[Scalar[dtype]](nheads)
    var his_h = alloc[Scalar[.bool]](batch)
    var qsl_h = alloc[Int32](batch + 1)
    var slot_h = alloc[UInt32](batch)
    var is_h = alloc[Float32](batch * state_size)
    var y_ref_h = alloc[Float32](y_size)
    var fs_ref_h = alloc[Float32](batch * state_size)

    rand(x_h, total_len * x_row)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * bc_row)
    rand(C_h, total_len * bc_row)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    rand(is_h, batch * state_size)
    # A = -exp(A_log) is negative; keep it in a range where the recurrence
    # stays bounded over the sequence.
    for i in range(nheads):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -1.0 - 0.1))
    for i in range(total_len * nheads):
        dt_h.store(i, Scalar[dtype](Float32(dt_h.load(i)) - 0.5))
    # Pre-round the initial states to state_dtype so the fp32 reference and a
    # bf16 pool start from identical states.
    for i in range(batch * state_size):
        is_h.store(i, is_h.load(i).cast[state_dtype]().cast[.float32]())
    for i in range(batch):
        his_h.store(i, Scalar[.bool](init_state))
    # Sequence b uses slot b + 1, so slot 0 must stay untouched.
    for i in range(batch):
        slot_h.store(i, UInt32(i + 1))
    var cum = 0
    qsl_h.store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.store(i + 1, Int32(cum))

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

    var pool_init_h = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        pool_init_h.store(i, Scalar[state_dtype](0))
    for b in range(batch):
        var slot = Int(slot_h.load(b))
        for i in range(state_size):
            pool_init_h.store(
                slot * state_size + i,
                is_h.load(b * state_size + i).cast[state_dtype](),
            )

    var x_layout = MixedLayout(
        (total_len, nheads, head_dim), (x_row, head_dim, 1)
    )
    var bc_layout = MixedLayout(
        (total_len, ngroups, dstate), (bc_row, dstate, 1)
    )
    var his_size = batch if init_state else 0

    # ---- CPU kernel (fp32 pool) ----
    var y_cpu_h = alloc[Scalar[dtype]](y_size)
    var pool_cpu_h = alloc[Float32](pool_size)
    for i in range(pool_size):
        pool_cpu_h.store(i, pool_init_h.load(i).cast[.float32]())
    _scan_cpu[DSTATE](
        TileTensor(x_h, x_layout),
        TileTensor(dt_h, row_major(total_len, nheads)),
        TileTensor(A_h, row_major(nheads)),
        TileTensor(B_h, bc_layout),
        TileTensor(C_h, bc_layout),
        TileTensor(D_h, row_major(D_size)),
        TileTensor(dt_bias_h, row_major(dt_bias_size)),
        TileTensor(y_cpu_h, row_major(total_len, nheads, head_dim)),
        TileTensor(pool_cpu_h, row_major(max_slots, nheads, head_dim, dstate)),
        TileTensor(qsl_h, row_major(batch + 1)),
        TileTensor(his_h, row_major(his_size)),
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
    )

    # ---- GPU kernel ----
    var x_d = ctx.enqueue_create_buffer[dtype](total_len * x_row)
    var dt_d = ctx.enqueue_create_buffer[dtype](total_len * nheads)
    var A_d = ctx.enqueue_create_buffer[dtype](nheads)
    var B_d = ctx.enqueue_create_buffer[dtype](total_len * bc_row)
    var C_d = ctx.enqueue_create_buffer[dtype](total_len * bc_row)
    var D_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](nheads)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var his_d = ctx.enqueue_create_buffer[.bool](batch)
    var slot_d = ctx.enqueue_create_buffer[.uint32](batch)
    var y_d = ctx.enqueue_create_buffer[dtype](y_size)
    var pool_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    ctx.enqueue_copy(D_d, D_h)
    ctx.enqueue_copy(dt_bias_d, dt_bias_h)
    ctx.enqueue_copy(qsl_d, qsl_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.enqueue_copy(pool_d, pool_init_h)

    var x_g = TileTensor(x_d, x_layout)
    var dt_g = TileTensor(dt_d, row_major(total_len, nheads))
    var A_g = TileTensor(A_d, row_major(nheads))
    var B_g = TileTensor(B_d, bc_layout)
    var C_g = TileTensor(C_d, bc_layout)
    var D_g = TileTensor(D_d, row_major(D_size))
    var dt_bias_g = TileTensor(dt_bias_d, row_major(dt_bias_size))
    var qsl_g = TileTensor(qsl_d, row_major(batch + 1))
    var his_g = TileTensor(his_d, row_major(his_size))
    var slot_g = TileTensor(slot_d, row_major(batch))
    var y_g = TileTensor(y_d, row_major(total_len, nheads, head_dim))
    var pool_g = TileTensor(
        pool_d, row_major(max_slots, nheads, head_dim, dstate)
    )

    comptime elt = size_of[dtype]()

    def x_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int, p: Int) {var x_g} -> SIMD[dtype, width]:
        return x_g.load[width=width, alignment=alignment * elt]((t, h, p))

    def dt_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int) {var dt_g} -> SIMD[dtype, width]:
        return dt_g.load[width=width, alignment=alignment * elt]((t, h))

    def b_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var B_g} -> SIMD[dtype, width]:
        return B_g.load[width=width, alignment=alignment * elt]((t, g, n))

    def c_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var C_g} -> SIMD[dtype, width]:
        return C_g.load[width=width, alignment=alignment * elt]((t, g, n))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var slot_g} -> SIMD[.uint32, width]:
        return slot_g.load[width=width]((b,))

    comptime if DSTATE_SPLIT > 0:
        # Mirrors the production launch in kernels.mojo.
        comptime CH_PER_BLOCK = 128 // DSTATE_SPLIT
        comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
            dtype,
            state_dtype,
            DSTATE,
            DSTATE_SPLIT,
            A_g.LayoutType,
            D_g.LayoutType,
            dt_bias_g.LayoutType,
            y_g.LayoutType,
            pool_g.LayoutType,
            qsl_g.LayoutType,
            his_g.LayoutType,
            y_g.Engine,
            type_of(x_fn),
            type_of(dt_fn),
            type_of(b_fn),
            type_of(c_fn),
            type_of(slot_fn),
        ]
        ctx.enqueue_function[kernel](
            Int32(ngroups),
            A_g,
            D_g,
            dt_bias_g,
            y_g,
            pool_g,
            qsl_g,
            his_g,
            host_arg=x_fn,
            host_arg2=dt_fn,
            host_arg3=b_fn,
            host_arg4=c_fn,
            host_arg5=slot_fn,
            grid_dim=(
                (head_dim + CH_PER_BLOCK - 1) // CH_PER_BLOCK,
                nheads,
                batch,
            ),
            block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1),
        )
    else:
        comptime BLOCK_SIZE = 64
        comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
            dtype,
            state_dtype,
            DSTATE,
            A_g.LayoutType,
            D_g.LayoutType,
            dt_bias_g.LayoutType,
            y_g.LayoutType,
            pool_g.LayoutType,
            qsl_g.LayoutType,
            his_g.LayoutType,
            y_g.Engine,
            type_of(x_fn),
            type_of(dt_fn),
            type_of(b_fn),
            type_of(c_fn),
            type_of(slot_fn),
        ]
        ctx.enqueue_function[kernel](
            Int32(ngroups),
            A_g,
            D_g,
            dt_bias_g,
            y_g,
            pool_g,
            qsl_g,
            his_g,
            host_arg=x_fn,
            host_arg2=dt_fn,
            host_arg3=b_fn,
            host_arg4=c_fn,
            host_arg5=slot_fn,
            grid_dim=((head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE, nheads, batch),
            block_dim=(BLOCK_SIZE, 1, 1),
        )

    var y_gpu_h = alloc[Scalar[dtype]](y_size)
    var pool_gpu_h = alloc[Scalar[state_dtype]](pool_size)
    ctx.enqueue_copy(y_gpu_h, y_d)
    ctx.enqueue_copy(pool_gpu_h, pool_d)
    ctx.synchronize()
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
    )

    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    his_h.free()
    qsl_h.free()
    slot_h.free()
    is_h.free()
    y_ref_h.free()
    fs_ref_h.free()
    pool_init_h.free()
    y_cpu_h.free()
    pool_cpu_h.free()
    y_gpu_h.free()
    pool_gpu_h.free()


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

    var x_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var dt_h = alloc[Scalar[dtype]](total_len * nheads)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var C_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var D_h = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = alloc[Scalar[dtype]](nheads)
    var his_h = alloc[Scalar[.bool]](1)
    rand(x_h, total_len * nheads * head_dim)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * ngroups * dstate)
    rand(C_h, total_len * ngroups * dstate)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    for i in range(nheads):
        A_h.store(i, Float32(A_h.load(i)) * -1.0 - 0.1)
    for i in range(total_len * nheads):
        dt_h.store(i, Float32(dt_h.load(i)) - 0.5)

    var A_tt = TileTensor(A_h, row_major(nheads))
    var D_tt = TileTensor(D_h, row_major(nheads))
    var dt_bias_tt = TileTensor(dt_bias_h, row_major(nheads))
    var his_tt = TileTensor(his_h, row_major(0))

    # Packed run: sequence b writes pool slot b.
    var y_packed = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var pool_packed = alloc[Float32](batch * state_size)
    var qsl_h = alloc[Int32](batch + 1)
    var slot_h = alloc[UInt32](batch)
    var cum = 0
    qsl_h.store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.store(i + 1, Int32(cum))
        slot_h.store(i, UInt32(i))
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
        TileTensor(slot_h, row_major(batch)),
    )

    # Independent runs over views into the packed buffers.
    var y_indep = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var pool_indep = alloc[Float32](batch * state_size)
    var qsl_s_h = alloc[Int32](2)
    var slot_s_h = alloc[UInt32](1)
    qsl_s_h.store(0, Int32(0))
    slot_s_h.store(0, UInt32(0))
    var off = 0
    for s in range(batch):
        var slen = seq_lengths[s]
        qsl_s_h.store(1, Int32(slen))
        _scan_cpu[dstate](
            TileTensor(
                x_h + off * nheads * head_dim,
                row_major(slen, nheads, head_dim),
            ),
            TileTensor(dt_h + off * nheads, row_major(slen, nheads)),
            A_tt,
            TileTensor(
                B_h + off * ngroups * dstate, row_major(slen, ngroups, dstate)
            ),
            TileTensor(
                C_h + off * ngroups * dstate, row_major(slen, ngroups, dstate)
            ),
            D_tt,
            dt_bias_tt,
            TileTensor(
                y_indep + off * nheads * head_dim,
                row_major(slen, nheads, head_dim),
            ),
            TileTensor(
                pool_indep + s * state_size,
                row_major(1, nheads, head_dim, dstate),
            ),
            TileTensor(qsl_s_h, row_major(2)),
            his_tt,
            TileTensor(slot_s_h, row_major(1)),
        )
        off += slen

    # Same fp32 ops in the same order, so the runs agree to rounding.
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_packed.load(i)), Float32(y_indep.load(i)), rtol=1e-5
        )
    for i in range(batch * state_size):
        assert_almost_equal(pool_packed.load(i), pool_indep.load(i), rtol=1e-5)

    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    his_h.free()
    y_packed.free()
    pool_packed.free()
    qsl_h.free()
    slot_h.free()
    y_indep.free()
    pool_indep.free()
    qsl_s_h.free()
    slot_s_h.free()


def test_mamba2_ssd_dstate_split() raises:
    """Cooperative DSTATE-split kernel, the production CUDA/HIP path.

    Sweeps DSTATE_SPLIT in {2, 4, 8} (8 is the production choice) over the
    96h/8g/dstate128 Nemotron-H grouping, plus the seqlen-1 decode shape the
    split targets and a DSTATE=16 tiling.
    """
    with DeviceContext() as ctx:
        comptime if not (
            has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        ):
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
        comptime if not (
            has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        ):
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
        comptime if not (
            has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        ):
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


def test_mamba2_ssd_no_D_no_bias() raises:
    """Empty D and dt_bias drop their terms."""
    with DeviceContext() as ctx:
        comptime if not (
            has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
        ):
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
        comptime if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
