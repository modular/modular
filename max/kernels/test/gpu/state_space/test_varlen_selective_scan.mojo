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

from max.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from std.math import ceildiv
from std.random import rand
from state_space.varlen_selective_scan import (
    varlen_selective_scan_fwd_cpu,
    varlen_selective_scan_fwd_gpu,
    varlen_selective_state_update_cpu,
    varlen_selective_state_update_gpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index, IndexList


def run_varlen_selective_scan_fwd_gpu[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    ngroups: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen selective scan forward GPU kernel against CPU reference."""
    comptime dstate = DSTATE
    if dstate > 256:
        return  # Skip if dstate exceeds kernel limit

    # Calculate total_length
    var total_length = 0
    for i in range(batch):
        total_length += seq_lengths[i]

    # Allocate host memory

    var ssm_states_cpu_h = alloc[Scalar[dtype]](batch * dim * dstate)
    var ssm_states_gpu_h = alloc[Scalar[dtype]](batch * dim * dstate)
    var output_cpu_h = alloc[Scalar[dtype]](dim * total_length)
    var output_gpu_h = alloc[Scalar[dtype]](dim * total_length)
    var u_h = alloc[Scalar[dtype]](dim * total_length)
    var delta_h = alloc[Scalar[dtype]](dim * total_length)
    var A_h = alloc[Scalar[dtype]](dim * dstate)
    var B_h = alloc[Scalar[dtype]](ngroups * dstate * total_length)
    var C_h = alloc[Scalar[dtype]](ngroups * dstate * total_length)
    var D_size = dim if has_D else 0
    var D_h = alloc[Scalar[dtype]](max(D_size, 1))
    var z_size = dim * total_length if has_z else 0
    var z_cpu_h = alloc[Scalar[dtype]](max(z_size, 1))
    var z_gpu_h = alloc[Scalar[dtype]](max(z_size, 1))
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_h = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var query_start_loc_h = alloc[Int32](batch + 1)
    var cache_indices_h = alloc[Int32](batch)
    var has_initial_state_h = alloc[Scalar[.bool]](batch)

    # Initialize input data
    rand(u_h, dim * total_length)
    rand(delta_h, dim * total_length)
    rand(A_h, dim * dstate)
    rand(B_h, ngroups * dstate * total_length)
    rand(C_h, ngroups * dstate * total_length)
    if has_D:
        rand(D_h, D_size)
    if has_z:
        rand(z_cpu_h, z_size)
    if has_delta_bias:
        rand(delta_bias_h, delta_bias_size)

    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.load(i)
        A_h.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Scale delta to be positive
    for i in range(dim * total_length):
        var val = delta_h.load(i)
        delta_h.store(i, Scalar[dtype](abs(Float32(val)) * 0.5))

    # Initialize query_start_loc (cumulative lengths)
    var cumsum = 0
    query_start_loc_h.store(0, Int32(0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_h.store(i + 1, Int32(cumsum))

    # Initialize cache_indices (identity mapping)
    for i in range(batch):
        cache_indices_h.store(i, Int32(i))

    # Initialize has_initial_state (all False)
    for i in range(batch):
        has_initial_state_h.store(i, Scalar[.bool](False))

    # Copy z for GPU
    if has_z:
        for i in range(dim * total_length):
            z_gpu_h.store(i, z_cpu_h.load(i))

    # Copy ssm_states for GPU
    for i in range(batch * dim * dstate):
        ssm_states_gpu_h.store(i, ssm_states_cpu_h.load(i))

    # Create TileTensors for CPU kernel
    var u_cpu_tt = TileTensor(u_h, row_major(dim, total_length))
    var delta_cpu_tt = TileTensor(delta_h, row_major(dim, total_length))
    var A_cpu_tt = TileTensor(A_h, row_major(dim, dstate))
    var B_cpu_tt = TileTensor(B_h, row_major(ngroups, dstate, total_length))
    var C_cpu_tt = TileTensor(C_h, row_major(ngroups, dstate, total_length))
    var D_cpu_tt = TileTensor(
        D_h,
        row_major(
            D_size,
        ),
    )
    var z_cpu_tt = TileTensor(
        z_cpu_h,
        row_major(
            (
                dim if has_z else 0,
                total_length if has_z else 0,
            )
        ),
    )
    var delta_bias_cpu_tt = TileTensor(
        delta_bias_h,
        row_major(
            delta_bias_size,
        ),
    )
    var ssm_states_cpu_tt = TileTensor(
        ssm_states_cpu_h, row_major(batch, dim, dstate)
    )
    var output_cpu_tt = TileTensor(output_cpu_h, row_major(dim, total_length))
    var query_start_loc_cpu_tt = TileTensor(
        query_start_loc_h,
        row_major(
            batch + 1,
        ),
    )
    var cache_indices_cpu_tt = TileTensor(
        cache_indices_h,
        row_major(
            batch,
        ),
    )
    var has_initial_state_cpu_tt = TileTensor(
        has_initial_state_h,
        row_major(
            batch,
        ),
    )

    # Run CPU kernel
    varlen_selective_scan_fwd_cpu[
        dtype,
        DSTATE,
    ](
        dim,
        ngroups,
        batch,
        Int32(-1),  # pad_slot_id
        Int8(1) if delta_softplus else Int8(0),
        u_cpu_tt,
        delta_cpu_tt,
        A_cpu_tt,
        B_cpu_tt,
        C_cpu_tt,
        D_cpu_tt,
        z_cpu_tt,
        delta_bias_cpu_tt,
        ssm_states_cpu_tt,
        output_cpu_tt,
        query_start_loc_cpu_tt,
        cache_indices_cpu_tt,
        has_initial_state_cpu_tt,
    )

    # Allocate device memory
    var ssm_states_gpu_d = ctx.enqueue_create_buffer[dtype](
        batch * dim * dstate
    )
    var output_gpu_d = ctx.enqueue_create_buffer[dtype](dim * total_length)
    var u_d = ctx.enqueue_create_buffer[dtype](dim * total_length)
    var delta_d = ctx.enqueue_create_buffer[dtype](dim * total_length)
    var A_d = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_d = ctx.enqueue_create_buffer[dtype](ngroups * dstate * total_length)
    var C_d = ctx.enqueue_create_buffer[dtype](ngroups * dstate * total_length)
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_d = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var delta_bias_d = ctx.enqueue_create_buffer[dtype](max(delta_bias_size, 1))
    var query_start_loc_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var cache_indices_d = ctx.enqueue_create_buffer[.int32](batch)
    var has_initial_state_d = ctx.enqueue_create_buffer[.bool](batch)

    # Copy to device
    ctx.enqueue_copy(u_d, u_h)
    ctx.enqueue_copy(delta_d, delta_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    if has_D:
        ctx.enqueue_copy(D_d, D_h)
    if has_z:
        ctx.enqueue_copy(z_d, z_gpu_h)
    if has_delta_bias:
        ctx.enqueue_copy(delta_bias_d, delta_bias_h)
    ctx.enqueue_copy(query_start_loc_d, query_start_loc_h)
    ctx.enqueue_copy(cache_indices_d, cache_indices_h)
    ctx.enqueue_copy(has_initial_state_d, has_initial_state_h)
    ctx.enqueue_copy(ssm_states_gpu_d, ssm_states_gpu_h)

    # Create TileTensors for GPU kernel
    var u_gpu_tt = TileTensor(
        u_d,
        row_major(dim, total_length),
    )
    var delta_gpu_tt = TileTensor(
        delta_d,
        row_major(dim, total_length),
    )
    var A_gpu_tt = TileTensor(
        A_d,
        row_major(dim, dstate),
    )
    var B_gpu_tt = TileTensor(
        B_d,
        row_major(ngroups, dstate, total_length),
    )
    var C_gpu_tt = TileTensor(
        C_d,
        row_major(ngroups, dstate, total_length),
    )
    var D_gpu_tt = TileTensor(
        D_d,
        row_major(
            D_size,
        ),
    )
    var z_gpu_tt = TileTensor(
        z_d,
        row_major(
            (
                dim if has_z else 0,
                total_length if has_z else 0,
            )
        ),
    )
    var delta_bias_gpu_tt = TileTensor(
        delta_bias_d,
        row_major(
            delta_bias_size,
        ),
    )
    var ssm_states_gpu_tt = TileTensor(
        ssm_states_gpu_d,
        row_major(batch, dim, dstate),
    )
    var output_gpu_tt = TileTensor(
        output_gpu_d,
        row_major(dim, total_length),
    )
    var query_start_loc_gpu_tt = TileTensor(
        query_start_loc_d,
        row_major(
            batch + 1,
        ),
    )
    var cache_indices_gpu_tt = TileTensor(
        cache_indices_d,
        row_major(
            batch,
        ),
    )
    var has_initial_state_gpu_tt = TileTensor(
        has_initial_state_d,
        row_major(
            batch,
        ),
    )

    # Launch GPU kernel
    comptime BLOCK_SIZE = 128
    var num_dim_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    var compiled_kernel = ctx.compile_function[
        varlen_selective_scan_fwd_gpu[
            dtype,
            DSTATE,
            u_gpu_tt.LayoutType,
            delta_gpu_tt.LayoutType,
            A_gpu_tt.LayoutType,
            B_gpu_tt.LayoutType,
            C_gpu_tt.LayoutType,
            D_gpu_tt.LayoutType,
            z_gpu_tt.LayoutType,
            delta_bias_gpu_tt.LayoutType,
            ssm_states_gpu_tt.LayoutType,
            output_gpu_tt.LayoutType,
            query_start_loc_gpu_tt.LayoutType,
            cache_indices_gpu_tt.LayoutType,
            has_initial_state_gpu_tt.LayoutType,
        ]
    ]()

    ctx.enqueue_function(
        compiled_kernel,
        Int32(dim),
        Int32(ngroups),
        Int32(batch),
        Int32(-1),  # pad_slot_id
        Int8(1) if delta_softplus else Int8(0),
        u_gpu_tt,
        delta_gpu_tt,
        A_gpu_tt,
        B_gpu_tt,
        C_gpu_tt,
        D_gpu_tt,
        z_gpu_tt,
        delta_bias_gpu_tt,
        ssm_states_gpu_tt,
        output_gpu_tt,
        query_start_loc_gpu_tt,
        cache_indices_gpu_tt,
        has_initial_state_gpu_tt,
        grid_dim=(num_dim_blocks, batch, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )

    # Copy results back
    var output_to_check = z_d if has_z else output_gpu_d
    var output_to_check_host = z_gpu_h if has_z else output_gpu_h
    ctx.enqueue_copy(output_to_check_host, output_to_check)
    ctx.synchronize()

    # Compare outputs
    var output_to_check_cpu = z_cpu_h if has_z else output_cpu_h
    var output_size = dim * total_length

    for i in range(output_size):
        var cpu_val = Float32(output_to_check_cpu.load(i))
        var gpu_val = Float32(output_to_check_host.load(i))
        assert_almost_equal(cpu_val, gpu_val, rtol=rtol)

    # Cleanup
    u_h.free()
    delta_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_cpu_h.free()
    z_gpu_h.free()
    delta_bias_h.free()
    ssm_states_cpu_h.free()
    ssm_states_gpu_h.free()
    output_cpu_h.free()
    output_gpu_h.free()
    query_start_loc_h.free()
    cache_indices_h.free()
    has_initial_state_h.free()


# =============================================================================
# Test functions for varlen selective scan forward on GPU
# =============================================================================


def test_varlen_selective_scan_fwd_gpu_equal_lengths() raises:
    """Test varlen selective scan forward GPU with equal-length sequences."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_scan_fwd_gpu[
            DType.float32,
            4,
            has_D=True,
            has_z=True,
            has_delta_bias=True,
            delta_softplus=False,
        ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8), ctx=ctx)


def test_varlen_selective_scan_fwd_gpu_variable_lengths() raises:
    """Test varlen selective scan forward GPU with variable-length sequences."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_scan_fwd_gpu[
            DType.float32,
            4,
            has_D=True,
            has_z=True,
            has_delta_bias=True,
            delta_softplus=False,
        ](
            batch=3,
            dim=4,
            ngroups=1,
            seq_lengths=Index(10, 6, 1),
            ctx=ctx,
        )


def test_varlen_selective_scan_fwd_gpu_without_D() raises:
    """Test varlen selective scan forward GPU without D tensor."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_scan_fwd_gpu[
            DType.float32,
            4,
            has_D=False,
            has_z=True,
            has_delta_bias=True,
            delta_softplus=False,
        ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8), ctx=ctx)


def test_varlen_selective_scan_fwd_gpu_without_z() raises:
    """Test varlen selective scan forward GPU without z tensor."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_scan_fwd_gpu[
            DType.float32,
            4,
            has_D=True,
            has_z=False,
            has_delta_bias=True,
            delta_softplus=False,
        ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8), ctx=ctx)


def test_varlen_selective_scan_fwd_gpu_with_delta_softplus() raises:
    """Test varlen selective scan forward GPU with delta softplus activation."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_scan_fwd_gpu[
            DType.float32,
            4,
            has_D=True,
            has_z=True,
            has_delta_bias=True,
            delta_softplus=True,
        ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8), ctx=ctx)


def run_varlen_selective_state_update_gpu[
    dtype: DType,
    DSTATE: Int,
    dt_softplus: Bool = False,
](
    batch: Int,
    nheads: Int,
    dim: Int,
    ngroups: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Checks the updated state and the output of state update, GPU vs CPU."""
    comptime dstate = DSTATE
    var ratio = nheads // ngroups
    var n_state = batch * nheads * dim * dstate
    var n_x = batch * nheads * dim
    var n_A = nheads * dim * dstate
    var n_B = batch * ngroups * dstate
    var n_D = nheads * dim

    var state_cpu_h = alloc[Scalar[dtype]](n_state)
    var state_gpu_h = alloc[Scalar[dtype]](n_state)
    var out_cpu_h = alloc[Scalar[dtype]](n_x)
    var out_gpu_h = alloc[Scalar[dtype]](n_x)
    var x_h = alloc[Scalar[dtype]](n_x)
    var dt_h = alloc[Scalar[dtype]](n_x)
    var A_h = alloc[Scalar[dtype]](n_A)
    var B_h = alloc[Scalar[dtype]](n_B)
    var C_h = alloc[Scalar[dtype]](n_B)
    var D_h = alloc[Scalar[dtype]](n_D)
    var z_h = alloc[Scalar[dtype]](n_x)
    var dt_bias_h = alloc[Scalar[dtype]](n_D)
    var sbi_h = alloc[Int32](batch)

    rand(state_cpu_h, n_state)
    rand(x_h, n_x)
    rand(dt_h, n_x)
    rand(A_h, n_A)
    rand(B_h, n_B)
    rand(C_h, n_B)
    rand(D_h, n_D)
    rand(z_h, n_x)
    rand(dt_bias_h, n_D)
    for i in range(n_state):
        state_gpu_h.store(i, state_cpu_h.load(i))
    for i in range(n_A):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -0.5))
    for i in range(n_x):
        dt_h.store(i, Scalar[dtype](abs(Float32(dt_h.load(i))) * 0.5))
    # Reversed mapping exercises the state_batch_indices indirection.
    for i in range(batch):
        sbi_h.store(i, Int32(batch - 1 - i))

    var state_cpu_tt = TileTensor(
        state_cpu_h, row_major(batch, nheads, dim, dstate)
    )
    var out_cpu_tt = TileTensor(out_cpu_h, row_major(batch, nheads, dim))
    var x_cpu_tt = TileTensor(x_h, row_major(batch, nheads, dim))
    var dt_cpu_tt = TileTensor(dt_h, row_major(batch, nheads, dim))
    var A_cpu_tt = TileTensor(A_h, row_major(nheads, dim, dstate))
    var B_cpu_tt = TileTensor(B_h, row_major(batch, ngroups, dstate))
    var C_cpu_tt = TileTensor(C_h, row_major(batch, ngroups, dstate))
    var D_cpu_tt = TileTensor(D_h, row_major(nheads, dim))
    var z_cpu_tt = TileTensor(z_h, row_major(batch, nheads, dim))
    var dt_bias_cpu_tt = TileTensor(dt_bias_h, row_major(nheads, dim))
    var sbi_cpu_tt = TileTensor(sbi_h, row_major(batch))

    varlen_selective_state_update_cpu[dtype, DSTATE](
        batch,
        nheads,
        dim,
        ratio,
        Int32(-1),
        Int8(1) if dt_softplus else Int8(0),
        Int8(1),
        state_cpu_tt,
        x_cpu_tt,
        dt_cpu_tt,
        A_cpu_tt,
        B_cpu_tt,
        C_cpu_tt,
        D_cpu_tt,
        z_cpu_tt,
        out_cpu_tt,
        dt_bias_cpu_tt,
        sbi_cpu_tt,
    )

    var state_d = ctx.enqueue_create_buffer[dtype](n_state)
    var out_d = ctx.enqueue_create_buffer[dtype](n_x)
    var x_d = ctx.enqueue_create_buffer[dtype](n_x)
    var dt_d = ctx.enqueue_create_buffer[dtype](n_x)
    var A_d = ctx.enqueue_create_buffer[dtype](n_A)
    var B_d = ctx.enqueue_create_buffer[dtype](n_B)
    var C_d = ctx.enqueue_create_buffer[dtype](n_B)
    var D_d = ctx.enqueue_create_buffer[dtype](n_D)
    var z_d = ctx.enqueue_create_buffer[dtype](n_x)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](n_D)
    var sbi_d = ctx.enqueue_create_buffer[.int32](batch)
    ctx.enqueue_copy(state_d, state_gpu_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    ctx.enqueue_copy(D_d, D_h)
    ctx.enqueue_copy(z_d, z_h)
    ctx.enqueue_copy(dt_bias_d, dt_bias_h)
    ctx.enqueue_copy(sbi_d, sbi_h)

    var state_tt = TileTensor(state_d, row_major(batch, nheads, dim, dstate))
    var out_tt = TileTensor(out_d, row_major(batch, nheads, dim))
    var x_tt = TileTensor(x_d, row_major(batch, nheads, dim))
    var dt_tt = TileTensor(dt_d, row_major(batch, nheads, dim))
    var A_tt = TileTensor(A_d, row_major(nheads, dim, dstate))
    var B_tt = TileTensor(B_d, row_major(batch, ngroups, dstate))
    var C_tt = TileTensor(C_d, row_major(batch, ngroups, dstate))
    var D_tt = TileTensor(D_d, row_major(nheads, dim))
    var z_tt = TileTensor(z_d, row_major(batch, nheads, dim))
    var dt_bias_tt = TileTensor(dt_bias_d, row_major(nheads, dim))
    var sbi_tt = TileTensor(sbi_d, row_major(batch))

    comptime BLOCK_M = 4  # dims handled per thread in the kernel
    var num_dim_blocks = ceildiv(dim, BLOCK_M)

    var compiled_kernel = ctx.compile_function[
        varlen_selective_state_update_gpu[
            dtype,
            DSTATE,
            state_tt.LayoutType,
            x_tt.LayoutType,
            dt_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            z_tt.LayoutType,
            out_tt.LayoutType,
            dt_bias_tt.LayoutType,
            sbi_tt.LayoutType,
        ]
    ]()
    ctx.enqueue_function(
        compiled_kernel,
        Int32(batch * nheads * num_dim_blocks),
        Int32(batch),
        Int32(nheads),
        Int32(dim),
        Int32(ratio),
        Int32(-1),
        Int8(1) if dt_softplus else Int8(0),
        Int8(1),
        state_tt,
        x_tt,
        dt_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        z_tt,
        out_tt,
        dt_bias_tt,
        sbi_tt,
        grid_dim=(num_dim_blocks, batch, nheads),
        block_dim=(1,),
    )
    ctx.enqueue_copy(state_gpu_h, state_d)
    ctx.enqueue_copy(out_gpu_h, out_d)
    ctx.synchronize()

    for i in range(n_state):
        assert_almost_equal(
            Float32(state_cpu_h.load(i)),
            Float32(state_gpu_h.load(i)),
            rtol=rtol,
        )
    for i in range(n_x):
        assert_almost_equal(
            Float32(out_cpu_h.load(i)),
            Float32(out_gpu_h.load(i)),
            rtol=rtol,
        )

    state_cpu_h.free()
    state_gpu_h.free()
    out_cpu_h.free()
    out_gpu_h.free()
    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    dt_bias_h.free()
    sbi_h.free()


def test_varlen_selective_state_update_gpu() raises:
    """Test varlen selective state update GPU against the CPU kernel."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_state_update_gpu[DType.float32, 4](
            batch=3, nheads=4, dim=10, ngroups=2, ctx=ctx
        )


def test_varlen_selective_state_update_gpu_softplus() raises:
    """Test varlen selective state update GPU with dt softplus."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_varlen_selective_state_update_gpu[
            DType.float32, 8, dt_softplus=True
        ](batch=2, nheads=2, dim=7, ngroups=1, ctx=ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
