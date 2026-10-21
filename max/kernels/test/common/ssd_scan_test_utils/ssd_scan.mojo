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

"""Shares GPU launches and parity checks between SSD tests and benchmarks."""

from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.testing import assert_almost_equal
from std.utils.numerics import isfinite, isnan, nan

from layout import MixedLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.mamba2_ssd_chunked import (
    GROUPED_CB_OFF,
    GROUPED_CB_ON,
    mamba2_ssd_chunked_fwd_inplace_gpu,
)
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)


comptime SSD_RESET = 0
comptime SSD_BASELINE = 1
comptime SSD_CANDIDATE = 2


trait SSDRunObserver:
    """Receives full-operation launch callbacks after correctness preflight."""

    comptime enabled: Bool = False

    @staticmethod
    def observe[
        Operation: def(DeviceContext, Int) raises -> None,
        //,
        grouped_pair: Bool,
        paired: Bool,
        grouped_mode: Int,
    ](ctx: DeviceContext, operation: Operation) raises:
        """Leaves correctness-only launches unobserved."""
        return


struct NoSSDObservation(SSDRunObserver):
    """Disables observation for correctness-only callers."""

    comptime enabled = False


def check_grouped_exact[
    dtype: DType, state_dtype: DType, //
](
    y_base: ImmPointer[Scalar[dtype], _],
    y_new: ImmPointer[Scalar[dtype], _],
    pool_base: ImmPointer[Scalar[state_dtype], _],
    pool_new: ImmPointer[Scalar[state_dtype], _],
    y_size: Int,
    pool_size: Int,
) raises:
    for i in range(y_size):
        var a = y_base.unsafe_load(i)
        var b = y_new.unsafe_load(i)
        if not isfinite(a) or not isfinite(b) or a.to_bits() != b.to_bits():
            raise Error(
                "grouped SSD output bits differ or are nonfinite at "
                + String(i)
            )
    for i in range(pool_size):
        var a = pool_base.unsafe_load(i)
        var b = pool_new.unsafe_load(i)
        if not isfinite(a) or not isfinite(b) or a.to_bits() != b.to_bits():
            raise Error(
                "grouped SSD state bits differ or are nonfinite at " + String(i)
            )


def check_preserved_slots[
    dtype: DType, //
](
    pool: ImmPointer[Scalar[dtype], _],
    initial_pool: ImmPointer[Scalar[dtype], _],
    slots: ImmPointer[UInt32, _],
    update_state: ImmPointer[Scalar[.bool], _],
    update_state_size: Int,
    batch: Int,
    state_size: Int,
    max_slots: Int,
) raises:
    for slot in range(max_slots):
        var writes_slot = False
        for b in range(batch):
            if slot == Int(slots.unsafe_load(b)):
                writes_slot = update_state_size == 0 or Bool(
                    update_state.unsafe_load(b)
                )
        if writes_slot:
            continue
        for i in range(slot * state_size, (slot + 1) * state_size):
            if (
                pool.unsafe_load(i).to_bits()
                != initial_pool.unsafe_load(i).to_bits()
            ):
                raise Error(
                    "SSD preserved pool slot overwritten at " + String(i)
                )


def check_chunked_parity[
    dtype: DType, state_dtype: DType, //
](
    y_serial: ImmPointer[Scalar[dtype], _],
    y_chunked: ImmPointer[Scalar[dtype], _],
    pool_serial: ImmPointer[Scalar[state_dtype], _],
    pool_chunked: ImmPointer[Scalar[state_dtype], _],
    y_size: Int,
    pool_size: Int,
) raises:
    """Checks chunked reductions against the serial kernel's bf16 tolerance."""
    var y_max = Float32(0)
    for i in range(y_size):
        y_max = max(y_max, abs(Float32(y_serial.unsafe_load(i))))
    for i in range(y_size):
        assert_almost_equal(
            Float32(y_serial.unsafe_load(i)),
            Float32(y_chunked.unsafe_load(i)),
            rtol=0.03,
            atol=Float64(0.01 * y_max),
            msg="chunked vs serial y mismatch at index " + String(i),
        )
    var s_max = Float32(0)
    for i in range(pool_size):
        s_max = max(s_max, abs(Float32(pool_serial.unsafe_load(i))))
    for i in range(pool_size):
        assert_almost_equal(
            Float32(pool_serial.unsafe_load(i)),
            Float32(pool_chunked.unsafe_load(i)),
            rtol=0.03,
            atol=Float64(0.01 * s_max),
            msg="chunked vs serial state mismatch at pool index " + String(i),
        )


def run_ssd_gpu[
    dtype: DType,
    state_dtype: DType,
    //,
    DSTATE: Int,
    DSTATE_SPLIT: Int,
    CH_PER_THREAD: Int,
    prefill: Bool,
    compare_chunked: Bool = False,
    guard_output: Bool = False,
    grouped_pair: Bool = False,
    grouped_mode: Int = GROUPED_CB_ON,
    use_grouped_default: Bool = False,
    Observer: SSDRunObserver = NoSSDObservation,
](
    ctx: DeviceContext,
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    max_slots: Int,
    batch: Int,
    total_len: Int,
    x_row: Int,
    bc_row: Int,
    D_size: Int,
    dt_bias_size: Int,
    his_size: Int,
    us_size: Int,
    x_h: ImmPointer[Scalar[dtype], _],
    dt_h: ImmPointer[Scalar[dtype], _],
    A_h: ImmPointer[Scalar[dtype], _],
    B_h: ImmPointer[Scalar[dtype], _],
    C_h: ImmPointer[Scalar[dtype], _],
    D_h: ImmPointer[Scalar[dtype], _],
    dt_bias_h: ImmPointer[Scalar[dtype], _],
    qsl_h: ImmPointer[Int32, _],
    his_h: ImmPointer[Scalar[.bool], _],
    us_h: ImmPointer[Scalar[.bool], _],
    slot_h: ImmPointer[UInt32, _],
    y_out: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    pool: MutPointer[Scalar[state_dtype], MutUntrackedOrigin],
) raises:
    """Runs a GPU scan, optionally checking a paired implementation first.

    `pool` holds the initial pool on entry and the final pool on return.
    """
    comptime dstate = DSTATE
    var y_size = total_len * nheads * head_dim
    var pool_size = max_slots * nheads * head_dim * dstate
    var x_layout = MixedLayout(
        (total_len, nheads, head_dim), (x_row, head_dim, 1)
    )
    var bc_layout = MixedLayout(
        (total_len, ngroups, dstate), (bc_row, dstate, 1)
    )

    var x_d = ctx.enqueue_create_buffer[dtype](total_len * x_row)
    var dt_d = ctx.enqueue_create_buffer[dtype](total_len * nheads)
    var A_d = ctx.enqueue_create_buffer[dtype](nheads)
    var B_d = ctx.enqueue_create_buffer[dtype](total_len * bc_row)
    var C_d = ctx.enqueue_create_buffer[dtype](total_len * bc_row)
    var D_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](nheads)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var his_d = ctx.enqueue_create_buffer[.bool](batch)
    var us_d = ctx.enqueue_create_buffer[.bool](batch)
    var slot_d = ctx.enqueue_create_buffer[.uint32](batch)
    var y_d = ctx.enqueue_create_buffer[dtype](
        y_size + (64 if guard_output else 0)
    )
    comptime if guard_output:
        ctx.enqueue_memset(y_d, nan[dtype]())
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
    ctx.enqueue_copy(us_d, us_h)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.enqueue_copy(pool_d, pool)

    var x_g = TileTensor(x_d, x_layout)
    var dt_g = TileTensor(dt_d, row_major(total_len, nheads))
    var A_g = TileTensor(A_d, row_major(nheads))
    var B_g = TileTensor(B_d, bc_layout)
    var C_g = TileTensor(C_d, bc_layout)
    var D_g = TileTensor(D_d, row_major(D_size))
    var dt_bias_g = TileTensor(dt_bias_d, row_major(dt_bias_size))
    var qsl_g = TileTensor(qsl_d, row_major(batch + 1))
    var his_g = TileTensor(his_d, row_major(his_size))
    var us_g = TileTensor(us_d, row_major(us_size))
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

    def launch_chunked(lctx: DeviceContext) raises {imm}:
        comptime if (
            (DSTATE_SPLIT < 0 or compare_chunked)
            and lctx.target._is_sm_8x_or_newer()
        ):
            mamba2_ssd_chunked_fwd_inplace_gpu[
                DSTATE, dt_softplus=True, grouped_cb_mode=GROUPED_CB_OFF
            ](
                lctx,
                total_len,
                batch,
                ngroups,
                A_g,
                D_g,
                dt_bias_g,
                y_g,
                pool_g,
                qsl_g,
                his_g,
                us_g,
                x_fn,
                dt_fn,
                b_fn,
                c_fn,
                slot_fn,
            )

    def launch_grouped(lctx: DeviceContext) raises {imm}:
        comptime if (grouped_pair and lctx.target._is_sm_8x_or_newer()):
            comptime if use_grouped_default:
                mamba2_ssd_chunked_fwd_inplace_gpu[DSTATE, dt_softplus=True](
                    lctx,
                    total_len,
                    batch,
                    ngroups,
                    A_g,
                    D_g,
                    dt_bias_g,
                    y_g,
                    pool_g,
                    qsl_g,
                    his_g,
                    us_g,
                    x_fn,
                    dt_fn,
                    b_fn,
                    c_fn,
                    slot_fn,
                )
            else:
                mamba2_ssd_chunked_fwd_inplace_gpu[
                    DSTATE, dt_softplus=True, grouped_cb_mode=grouped_mode
                ](
                    lctx,
                    total_len,
                    batch,
                    ngroups,
                    A_g,
                    D_g,
                    dt_bias_g,
                    y_g,
                    pool_g,
                    qsl_g,
                    his_g,
                    us_g,
                    x_fn,
                    dt_fn,
                    b_fn,
                    c_fn,
                    slot_fn,
                )

    def launch(lctx: DeviceContext) raises {imm}:
        comptime if DSTATE_SPLIT < 0:
            launch_chunked(lctx)
        elif DSTATE_SPLIT > 0:
            # Mirrors the production launch in kernels.mojo.
            comptime ROWS = 128 // DSTATE_SPLIT
            comptime CH_PER_BLOCK = ROWS * CH_PER_THREAD
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
                us_g.LayoutType,
                y_g.Engine,
                type_of(x_fn),
                type_of(dt_fn),
                type_of(b_fn),
                type_of(c_fn),
                type_of(slot_fn),
                prefill=prefill,
                CH_PER_THREAD=CH_PER_THREAD,
            ]
            lctx.enqueue_function[kernel](
                Int32(ngroups),
                A_g,
                D_g,
                dt_bias_g,
                y_g,
                pool_g,
                qsl_g,
                his_g,
                us_g,
                host_args=(x_fn, dt_fn, b_fn, c_fn, slot_fn),
                grid_dim=(
                    ceildiv(head_dim, CH_PER_BLOCK),
                    nheads,
                    batch,
                ),
                block_dim=(DSTATE_SPLIT, ROWS, 1),
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
                us_g.LayoutType,
                y_g.Engine,
                type_of(x_fn),
                type_of(dt_fn),
                type_of(b_fn),
                type_of(c_fn),
                type_of(slot_fn),
            ]
            lctx.enqueue_function[kernel](
                Int32(ngroups),
                A_g,
                D_g,
                dt_bias_g,
                y_g,
                pool_g,
                qsl_g,
                his_g,
                us_g,
                host_args=(x_fn, dt_fn, b_fn, c_fn, slot_fn),
                grid_dim=(ceildiv(head_dim, BLOCK_SIZE), nheads, batch),
                block_dim=(BLOCK_SIZE, 1, 1),
            )

    comptime if grouped_pair or compare_chunked:
        comptime if grouped_pair:
            comptime assert (
                DSTATE_SPLIT < 0
            ), "grouped A/B compares two chunked arms"
        else:
            comptime assert DSTATE_SPLIT > 0, "paired baseline must be serial"
        comptime assert (
            guard_output
        ), "grouped A/B requires poisoned output guards"
        var pool_seed_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
        ctx.enqueue_copy(pool_seed_d, pool)
        var y_base_h = unsafe_alloc[Scalar[dtype]](y_size + 64)
        var pool_base_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
        var pool_initial_h = unsafe_alloc[Scalar[state_dtype]](pool_size)
        for i in range(pool_size):
            pool_initial_h.unsafe_store(i, pool.unsafe_load(i))

        ctx.enqueue_memset(y_d, nan[dtype]())
        comptime if grouped_pair:
            launch_chunked(ctx)
        else:
            launch(ctx)
        ctx.enqueue_copy(y_base_h, y_d)
        ctx.enqueue_copy(pool_base_h, pool_d)
        ctx.synchronize()
        ctx.enqueue_copy(pool_d, pool_seed_d)
        ctx.enqueue_memset(y_d, nan[dtype]())
        comptime if grouped_pair:
            launch_grouped(ctx)
        else:
            launch_chunked(ctx)
        ctx.enqueue_copy(y_out, y_d)
        ctx.enqueue_copy(pool, pool_d)
        ctx.synchronize()
        comptime if grouped_pair:
            check_grouped_exact(
                y_base_h, y_out, pool_base_h, pool, y_size, pool_size
            )
        else:
            check_chunked_parity(
                y_base_h, y_out, pool_base_h, pool, y_size, pool_size
            )
        check_preserved_slots(
            pool_base_h,
            pool_initial_h,
            slot_h,
            us_h,
            us_size,
            batch,
            nheads * head_dim * dstate,
            max_slots,
        )
        check_preserved_slots(
            pool,
            pool_initial_h,
            slot_h,
            us_h,
            us_size,
            batch,
            nheads * head_dim * dstate,
            max_slots,
        )
        for i in range(y_size, y_size + 64):
            if not isnan(y_base_h.unsafe_load(i)) or not isnan(
                y_out.unsafe_load(i)
            ):
                raise Error(
                    "grouped SSD output guard overwritten at " + String(i)
                )
        y_base_h.unsafe_free()
        pool_base_h.unsafe_free()
        pool_initial_h.unsafe_free()
        comptime if grouped_pair:
            print("SSD_GROUPED_EXACT_PASS")
            print("SSD_GROUPED_CANDIDATE_MODE,", grouped_mode, sep="")
        else:
            print("SSD_PAIR_CHECK_PASS")

        comptime if Observer.enabled:

            def pair_operation(
                lctx: DeviceContext, action: Int
            ) raises {
                imm pool_d,
                imm pool_seed_d,
                imm launch,
                imm launch_chunked,
                imm launch_grouped,
            }:
                if action == SSD_RESET:
                    lctx.enqueue_copy(pool_d, pool_seed_d)
                elif action == SSD_BASELINE:
                    comptime if grouped_pair:
                        launch_chunked(lctx)
                    else:
                        launch(lctx)
                else:
                    comptime if grouped_pair:
                        launch_grouped(lctx)
                    else:
                        launch_chunked(lctx)

            Observer.observe[grouped_pair, compare_chunked, grouped_mode](
                ctx, pair_operation
            )
    elif Observer.enabled:
        var pool_seed_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
        ctx.enqueue_copy(pool_seed_d, pool)
        ctx.synchronize()

        def single_operation(
            lctx: DeviceContext, action: Int
        ) raises {imm pool_d, imm pool_seed_d, imm launch}:
            if action == SSD_RESET:
                lctx.enqueue_copy(pool_d, pool_seed_d)
            else:
                launch(lctx)

        Observer.observe[grouped_pair, compare_chunked, grouped_mode](
            ctx, single_operation
        )
    else:
        launch(ctx)

    ctx.enqueue_copy(y_out, y_d)
    ctx.enqueue_copy(pool, pool_d)
    ctx.synchronize()
    comptime if guard_output:
        for i in range(y_size, y_size + 64):
            if not isnan(y_out.unsafe_load(i)):
                raise Error("SSD output guard overwritten at " + String(i))
