# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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

from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]
from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from random import rand
from nn.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_selective_scan_gpu[
    dtype: DType,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    dstate: Int,
    n_groups: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test selective scan GPU kernel against CPU reference."""
    if dstate > 16:
        return  # Skip if dstate exceeds kernel limit
    
    var group_size = dim // n_groups
    var chunk_size = 2048
    var n_chunks = (seqlen + chunk_size - 1) // chunk_size
    
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    var output_cpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    var output_gpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    var x_cpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * n_chunks * 2 * dstate)
    var x_gpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * n_chunks * 2 * dstate)
    var out_z_cpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    var out_z_gpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    
    # Initialize output buffers to zero
    for i in range(batch * dim * seqlen):
        output_cpu_h[i] = Scalar[dtype](0)
        output_gpu_h[i] = Scalar[dtype](0)
        out_z_cpu_h[i] = Scalar[dtype](0)
        out_z_gpu_h[i] = Scalar[dtype](0)
    for i in range(batch * dim * n_chunks * 2 * dstate):
        x_cpu_h[i] = Scalar[dtype](0)
        x_gpu_h[i] = Scalar[dtype](0)
    var u_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    var delta_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
    var A_h = UnsafePointer[Scalar[dtype]].alloc(dim * dstate)
    var B_h = UnsafePointer[Scalar[dtype]].alloc(batch * n_groups * dstate * seqlen)
    var C_h = UnsafePointer[Scalar[dtype]].alloc(batch * n_groups * dstate * seqlen)
    var D_size = dim if has_D else 0
    var D_h = UnsafePointer[Scalar[dtype]].alloc(max(D_size, 1))
    var z_size = batch * dim * seqlen if has_z else 0
    var z_h = UnsafePointer[Scalar[dtype]].alloc(max(z_size, 1))
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_h = UnsafePointer[Scalar[dtype]].alloc(max(delta_bias_size, 1))
    
    # Create LayoutTensors for initialization
    var u_init = LayoutTensor[dtype, layout_3d](
        u_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_init = LayoutTensor[dtype, layout_3d](
        delta_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var A_init = LayoutTensor[dtype, layout_2d](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_init = LayoutTensor[dtype, layout_4d](
        B_h, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var C_init = LayoutTensor[dtype, layout_4d](
        C_h, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var D_init = LayoutTensor[dtype, layout_1d](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_init = LayoutTensor[dtype, layout_3d](
        z_h, RuntimeLayout[layout_3d].row_major(Index(batch if has_z else 0, dim if has_z else 0, seqlen if has_z else 0))
    )
    var delta_bias_init = LayoutTensor[dtype, layout_1d](
        delta_bias_h, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    
    # Initialize input data
    rand[dtype](u_init.ptr, u_init.size())
    rand[dtype](delta_init.ptr, delta_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](delta_bias_init.ptr, delta_bias_init.size())
    
    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.offset(i).load()
        A_h.offset(i).store(Scalar[dtype](Float32(val) * -0.5))
    
    # Scale delta to be positive
    for i in range(batch * dim * seqlen):
        var val = delta_h.offset(i).load()
        delta_h.offset(i).store(Scalar[dtype](abs(Float32(val)) * 0.5))
    
    # Allocate device memory
    var output_cpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var output_gpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var x_cpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * n_chunks * 2 * dstate)
    var x_gpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * n_chunks * 2 * dstate)
    var out_z_cpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var out_z_gpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var u_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var delta_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var A_d = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_d = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate * seqlen)
    var C_d = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate * seqlen)
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_d = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var delta_bias_d = ctx.enqueue_create_buffer[dtype](max(delta_bias_size, 1))
    
    # Copy to device
    ctx.enqueue_copy(u_d, u_h)
    ctx.enqueue_copy(delta_d, delta_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    if has_D:
        ctx.enqueue_copy(D_d, D_h)
    if has_z:
        ctx.enqueue_copy(z_d, z_h)
    if has_delta_bias:
        ctx.enqueue_copy(delta_bias_d, delta_bias_h)
    
    # Create LayoutTensors for CPU
    # Create CPU LayoutTensors with MutAnyOrigin for CPU function (using host memory)
    var output_cpu_buf = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_cpu_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var x_cpu_buf = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_cpu_h, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var out_z_cpu_buf = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_cpu_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var u_cpu_buf = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        u_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_cpu_buf = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        delta_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var A_cpu_buf = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_cpu_buf = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_h, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var C_cpu_buf = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_h, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var D_cpu_buf = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_cpu_buf = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_h, RuntimeLayout[layout_3d].row_major(Index(batch if has_z else 0, dim if has_z else 0, seqlen if has_z else 0))
    )
    var delta_bias_cpu_buf = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_h, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    
    
    # Create LayoutTensors for GPU
    var output_gpu_buf = LayoutTensor[dtype, layout_3d](
        output_gpu_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var x_gpu_buf = LayoutTensor[dtype, layout_4d](
        x_gpu_d, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var out_z_gpu_buf = LayoutTensor[dtype, layout_3d](
        out_z_gpu_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var u_gpu_buf = LayoutTensor[dtype, layout_3d](
        u_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_gpu_buf = LayoutTensor[dtype, layout_3d](
        delta_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var A_gpu_buf = LayoutTensor[dtype, layout_2d](
        A_d, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_gpu_buf = LayoutTensor[dtype, layout_4d](
        B_d, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var C_gpu_buf = LayoutTensor[dtype, layout_4d](
        C_d, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var D_gpu_buf = LayoutTensor[dtype, layout_1d](
        D_d, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_gpu_buf = LayoutTensor[dtype, layout_3d](
        z_d, RuntimeLayout[layout_3d].row_major(Index(batch if has_z else 0, dim if has_z else 0, seqlen if has_z else 0))
    )
    var delta_bias_gpu_buf = LayoutTensor[dtype, layout_1d](
        delta_bias_d, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    
    # Strides for row-major layout
    var output_b_stride: UInt32 = dim * seqlen
    var output_d_stride: UInt32 = seqlen
    var output_t_stride: UInt32 = 1
    var x_b_stride: UInt32 = dim * n_chunks * 2 * dstate
    var x_d_stride: UInt32 = n_chunks * 2 * dstate
    var x_chunk_stride: UInt32 = 2 * dstate
    var x_n_stride: UInt32 = 1
    var out_z_b_stride: UInt32 = dim * seqlen
    var out_z_d_stride: UInt32 = seqlen
    var out_z_t_stride: UInt32 = 1
    var u_b_stride: UInt32 = dim * seqlen
    var u_d_stride: UInt32 = seqlen
    var u_t_stride: UInt32 = 1
    var delta_b_stride: UInt32 = dim * seqlen
    var delta_d_stride: UInt32 = seqlen
    var delta_t_stride: UInt32 = 1
    var A_d_stride: UInt32 = dstate
    var A_n_stride: UInt32 = 1
    var B_b_stride: UInt32 = n_groups * dstate * seqlen
    var B_g_stride: UInt32 = dstate * seqlen
    var B_n_stride: UInt32 = seqlen
    var B_t_stride: UInt32 = 1
    var C_b_stride: UInt32 = n_groups * dstate * seqlen
    var C_g_stride: UInt32 = dstate * seqlen
    var C_n_stride: UInt32 = seqlen
    var C_t_stride: UInt32 = 1
    var D_stride: UInt32 = 1
    var z_b_stride: UInt32 = dim * seqlen
    var z_d_stride: UInt32 = seqlen
    var z_t_stride: UInt32 = 1
    var delta_bias_stride: UInt32 = 1
    
    comptime delta_softplus_int8: Int8 = Int8(1) if delta_softplus else Int8(0)
    
    # Run CPU kernel
    selective_scan_fwd_cpu[
        dtype,
        output_cpu_buf.layout,
        x_cpu_buf.layout,
        out_z_cpu_buf.layout,
        u_cpu_buf.layout,
        delta_cpu_buf.layout,
        A_cpu_buf.layout,
        B_cpu_buf.layout,
        C_cpu_buf.layout,
        D_cpu_buf.layout,
        z_cpu_buf.layout,
        delta_bias_cpu_buf.layout,
    ](
        batch,
        dim,
        seqlen,
        dstate,
        group_size,
        delta_softplus_int8,
        output_cpu_buf,
        x_cpu_buf,
        out_z_cpu_buf,
        u_cpu_buf,
        delta_cpu_buf,
        A_cpu_buf,
        B_cpu_buf,
        C_cpu_buf,
        D_cpu_buf,
        z_cpu_buf,
        delta_bias_cpu_buf,
        output_b_stride,
        output_d_stride,
        output_t_stride,
        x_b_stride,
        x_d_stride,
        x_chunk_stride,
        x_n_stride,
        out_z_b_stride,
        out_z_d_stride,
        out_z_t_stride,
        u_b_stride,
        u_d_stride,
        u_t_stride,
        delta_b_stride,
        delta_d_stride,
        delta_t_stride,
        A_d_stride,
        A_n_stride,
        B_b_stride,
        B_g_stride,
        B_n_stride,
        B_t_stride,
        C_b_stride,
        C_g_stride,
        C_n_stride,
        C_t_stride,
        D_stride,
        z_b_stride,
        z_d_stride,
        z_t_stride,
        delta_bias_stride,
    )
    
    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    from math import ceildiv
    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
    
    var compiled_kernel = ctx.compile_function_checked[
        selective_scan_fwd_gpu[
            dtype,
            output_gpu_buf.layout,
            x_gpu_buf.layout,
            out_z_gpu_buf.layout,
            u_gpu_buf.layout,
            delta_gpu_buf.layout,
            A_gpu_buf.layout,
            B_gpu_buf.layout,
            C_gpu_buf.layout,
            D_gpu_buf.layout,
            z_gpu_buf.layout,
            delta_bias_gpu_buf.layout,
        ],
        selective_scan_fwd_gpu[
            dtype,
            output_gpu_buf.layout,
            x_gpu_buf.layout,
            out_z_gpu_buf.layout,
            u_gpu_buf.layout,
            delta_gpu_buf.layout,
            A_gpu_buf.layout,
            B_gpu_buf.layout,
            C_gpu_buf.layout,
            D_gpu_buf.layout,
            z_gpu_buf.layout,
            delta_bias_gpu_buf.layout,
        ]
    ]()
    
    ctx.enqueue_function_checked(
        compiled_kernel,
        total_batch_dim,
        batch,
        dim,
        seqlen,
        dstate,
        group_size,
        delta_softplus_int8,
        output_gpu_buf,
        x_gpu_buf,
        out_z_gpu_buf,
        u_gpu_buf,
        delta_gpu_buf,
        A_gpu_buf,
        B_gpu_buf,
        C_gpu_buf,
        D_gpu_buf,
        z_gpu_buf,
        delta_bias_gpu_buf,
        output_b_stride,
        output_d_stride,
        output_t_stride,
        x_b_stride,
        x_d_stride,
        x_chunk_stride,
        x_n_stride,
        out_z_b_stride,
        out_z_d_stride,
        out_z_t_stride,
        u_b_stride,
        u_d_stride,
        u_t_stride,
        delta_b_stride,
        delta_d_stride,
        delta_t_stride,
        A_d_stride,
        A_n_stride,
        B_b_stride,
        B_g_stride,
        B_n_stride,
        B_t_stride,
        C_b_stride,
        C_g_stride,
        C_n_stride,
        C_t_stride,
        D_stride,
        z_b_stride,
        z_d_stride,
        z_t_stride,
        delta_bias_stride,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    
    # Copy GPU results back (CPU results are already in output_cpu_h)
    ctx.enqueue_copy(output_gpu_h, output_gpu_d)
    ctx.synchronize()
    
    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h.offset(i).load(),
            output_gpu_h.offset(i).load(),
            rtol=rtol,
        )
    
    # Cleanup
    output_cpu_h.free()
    output_gpu_h.free()
    x_cpu_h.free()
    x_gpu_h.free()
    out_z_cpu_h.free()
    out_z_gpu_h.free()
    u_h.free()
    delta_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    delta_bias_h.free()


fn run_selective_scan_update_gpu[
    dtype: DType,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    dstate: Int,
    n_groups: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test selective scan update GPU kernel against CPU reference."""
    if dstate > 16:
        return  # Skip if dstate exceeds kernel limit
    
    var group_size = dim // n_groups
    
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    var state_in_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * dstate)
    var state_out_gpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * dstate)
    var state_out_cpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * dstate)
    var output_gpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim)
    var output_cpu_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim)
    var x_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim)
    var dt_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim)
    var A_h = UnsafePointer[Scalar[dtype]].alloc(dim * dstate)
    var B_h = UnsafePointer[Scalar[dtype]].alloc(batch * n_groups * dstate)
    var C_h = UnsafePointer[Scalar[dtype]].alloc(batch * n_groups * dstate)
    var D_size = dim if has_D else 0
    var D_h = UnsafePointer[Scalar[dtype]].alloc(max(D_size, 1))
    var z_size = batch * dim if has_z else 0
    var z_h = UnsafePointer[Scalar[dtype]].alloc(max(z_size, 1))
    var dt_bias_size = dim if has_delta_bias else 0
    var dt_bias_h = UnsafePointer[Scalar[dtype]].alloc(max(dt_bias_size, 1))
    
    # Initialize output buffers to zero
    for i in range(batch * dim * dstate):
        state_out_gpu_h[i] = Scalar[dtype](0)
        state_out_cpu_h[i] = Scalar[dtype](0)
    for i in range(batch * dim):
        output_gpu_h[i] = Scalar[dtype](0)
        output_cpu_h[i] = Scalar[dtype](0)
    
    # Create LayoutTensors for initialization
    var state_in_init = LayoutTensor[dtype, layout_3d](
        state_in_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate))
    )
    var x_init = LayoutTensor[dtype, layout_2d](
        x_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var dt_init = LayoutTensor[dtype, layout_2d](
        dt_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var A_init = LayoutTensor[dtype, layout_2d](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_init = LayoutTensor[dtype, layout_3d](
        B_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var C_init = LayoutTensor[dtype, layout_3d](
        C_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var D_init = LayoutTensor[dtype, layout_1d](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_init = LayoutTensor[dtype, layout_2d](
        z_h, RuntimeLayout[layout_2d].row_major(Index(batch if has_z else 0, dim if has_z else 0))
    )
    var dt_bias_init = LayoutTensor[dtype, layout_1d](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    
    # Initialize input data
    rand[dtype](state_in_init.ptr, state_in_init.size())
    rand[dtype](x_init.ptr, x_init.size())
    rand[dtype](dt_init.ptr, dt_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](dt_bias_init.ptr, dt_bias_init.size())
    
    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.offset(i).load()
        A_h.offset(i).store(Scalar[dtype](Float32(val) * -0.5))
    
    # Copy state_in for CPU and GPU
    for i in range(batch * dim * dstate):
        state_out_cpu_h[i] = state_in_h[i]
    
    # Allocate device buffers
    var state_in_device = ctx.enqueue_create_buffer[dtype](batch * dim * dstate)
    var state_out_device = ctx.enqueue_create_buffer[dtype](batch * dim * dstate)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var x_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var dt_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var A_device = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_device = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate)
    var C_device = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate)
    var D_device = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_device = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var dt_bias_device = ctx.enqueue_create_buffer[dtype](max(dt_bias_size, 1))
    
    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(state_in_device, state_in_h)
        ctx.enqueue_copy(x_device, x_h)
        ctx.enqueue_copy(dt_device, dt_h)
        ctx.enqueue_copy(A_device, A_h)
        ctx.enqueue_copy(B_device, B_h)
        ctx.enqueue_copy(C_device, C_h)
        if has_D:
            ctx.enqueue_copy(D_device, D_h)
        if has_z:
            ctx.enqueue_copy(z_device, z_h)
        if has_delta_bias:
            ctx.enqueue_copy(dt_bias_device, dt_bias_h)
    
    # Create device tensors
    var state_in_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        state_in_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )
    var state_out_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        state_out_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )
    var output_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(batch, dim)),
    )
    var x_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        x_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(batch, dim)),
    )
    var dt_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        dt_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(batch, dim)),
    )
    var A_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(dim, dstate)),
    )
    var B_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        B_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate)),
    )
    var C_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate)),
    )
    var D_device_tensor = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(D_size)),
    )
    var z_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        z_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(batch if has_z else 0, dim if has_z else 0)),
    )
    var dt_bias_device_tensor = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        dt_bias_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(dt_bias_size)),
    )
    
    # Strides for row-major layout
    var state_out_b_stride: UInt32 = dim * dstate
    var state_out_d_stride: UInt32 = dstate
    var state_out_n_stride: UInt32 = 1
    
    var output_b_stride: UInt32 = dim
    var output_d_stride: UInt32 = 1
    
    var state_in_b_stride: UInt32 = dim * dstate
    var state_in_d_stride: UInt32 = dstate
    var state_in_n_stride: UInt32 = 1
    
    var x_b_stride: UInt32 = dim
    var x_d_stride: UInt32 = 1
    
    var dt_b_stride: UInt32 = dim
    var dt_d_stride: UInt32 = 1
    
    var A_d_stride: UInt32 = dstate
    var A_n_stride: UInt32 = 1
    
    var B_b_stride: UInt32 = n_groups * dstate
    var B_g_stride: UInt32 = dstate
    var B_n_stride: UInt32 = 1
    
    var C_b_stride: UInt32 = n_groups * dstate
    var C_g_stride: UInt32 = dstate
    var C_n_stride: UInt32 = 1
    
    var D_stride: UInt32 = 1
    
    var z_b_stride: UInt32 = dim
    var z_d_stride: UInt32 = 1
    
    var dt_bias_stride: UInt32 = 1
    
    # Run GPU kernel
    var total_batch_dim = batch * dim
    with ctx.push_context():
        var compiled_func = ctx.compile_function_checked[
            selective_scan_update_gpu[
                dtype,
                state_out_device_tensor.layout,
                output_device_tensor.layout,
                state_in_device_tensor.layout,
                x_device_tensor.layout,
                dt_device_tensor.layout,
                A_device_tensor.layout,
                B_device_tensor.layout,
                C_device_tensor.layout,
                D_device_tensor.layout,
                z_device_tensor.layout,
                dt_bias_device_tensor.layout,
            ],
            selective_scan_update_gpu[
                dtype,
                state_out_device_tensor.layout,
                output_device_tensor.layout,
                state_in_device_tensor.layout,
                x_device_tensor.layout,
                dt_device_tensor.layout,
                A_device_tensor.layout,
                B_device_tensor.layout,
                C_device_tensor.layout,
                D_device_tensor.layout,
                z_device_tensor.layout,
                dt_bias_device_tensor.layout,
            ],
        ]()
        ctx.enqueue_function_checked(
            compiled_func,
            total_batch_dim,
            batch,
            dim,
            dstate,
            group_size,
            Int8(1) if delta_softplus else Int8(0),
            state_out_device_tensor,
            output_device_tensor,
            state_in_device_tensor,
            x_device_tensor,
            dt_device_tensor,
            A_device_tensor,
            B_device_tensor,
            C_device_tensor,
            D_device_tensor,
            z_device_tensor,
            dt_bias_device_tensor,
            state_out_b_stride,
            state_out_d_stride,
            state_out_n_stride,
            output_b_stride,
            output_d_stride,
            state_in_b_stride,
            state_in_d_stride,
            state_in_n_stride,
            x_b_stride,
            x_d_stride,
            dt_b_stride,
            dt_d_stride,
            A_d_stride,
            A_n_stride,
            B_b_stride,
            B_g_stride,
            B_n_stride,
            C_b_stride,
            C_g_stride,
            C_n_stride,
            D_stride,
            z_b_stride,
            z_d_stride,
            dt_bias_stride,
            grid_dim=(ceildiv(total_batch_dim, 256),),
            block_dim=(256,),
        )
    
    # Copy results back from device
    with ctx.push_context():
        ctx.enqueue_copy(state_out_gpu_h, state_out_device)
        ctx.enqueue_copy(output_gpu_h, output_device)
    
    # Create CPU tensors for reference
    var state_in_cpu = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        state_in_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate))
    )
    var state_out_cpu = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        state_out_cpu_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate))
    )
    var output_cpu = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_cpu_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var x_cpu = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        x_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var dt_cpu = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        dt_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var A_cpu = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_cpu = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        B_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var C_cpu = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var D_cpu = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_cpu = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        z_h, RuntimeLayout[layout_2d].row_major(Index(batch if has_z else 0, dim if has_z else 0))
    )
    var dt_bias_cpu = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    
    # Run CPU reference
    selective_scan_update_cpu[
        dtype,
        state_out_cpu.layout,
        output_cpu.layout,
        state_in_cpu.layout,
        x_cpu.layout,
        dt_cpu.layout,
        A_cpu.layout,
        B_cpu.layout,
        C_cpu.layout,
        D_cpu.layout,
        z_cpu.layout,
        dt_bias_cpu.layout,
    ](
        batch,
        dim,
        dstate,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        state_out_cpu,
        output_cpu,
        state_in_cpu,
        x_cpu,
        dt_cpu,
        A_cpu,
        B_cpu,
        C_cpu,
        D_cpu,
        z_cpu,
        dt_bias_cpu,
        state_out_b_stride,
        state_out_d_stride,
        state_out_n_stride,
        output_b_stride,
        output_d_stride,
        state_in_b_stride,
        state_in_d_stride,
        state_in_n_stride,
        x_b_stride,
        x_d_stride,
        dt_b_stride,
        dt_d_stride,
        A_d_stride,
        A_n_stride,
        B_b_stride,
        B_g_stride,
        B_n_stride,
        C_b_stride,
        C_g_stride,
        C_n_stride,
        D_stride,
        z_b_stride,
        z_d_stride,
        dt_bias_stride,
    )
    
    # Compare results
    var state_size = batch * dim * dstate
    for i in range(state_size):
        assert_almost_equal(
            state_out_gpu_h[i],
            state_out_cpu_h[i],
            rtol=rtol,
        )
    
    var output_size = batch * dim
    for i in range(output_size):
        assert_almost_equal(
            output_gpu_h[i],
            output_cpu_h[i],
            rtol=rtol,
        )
    
    # Cleanup
    state_in_h.free()
    state_out_gpu_h.free()
    state_out_cpu_h.free()
    output_gpu_h.free()
    output_cpu_h.free()
    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    dt_bias_h.free()


def main():
    with DeviceContext() as ctx:
        # Test basic selective scan
        run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Basic selective scan GPU test passed")
        
        # Test without D
        run_selective_scan_gpu[DType.float32, has_D=False, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan GPU without D test passed")
        
        # Test without z
        run_selective_scan_gpu[DType.float32, has_D=True, has_z=False, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan GPU without z test passed")
        
        # Test with delta_softplus
        run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
            batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan GPU with delta_softplus test passed")
        
        # Test longer sequence
        run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=8, seqlen=32, dstate=8, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan GPU longer sequence test passed")
        
        # Test specific sequence lengths that might expose CPU tiling bugs
        # CPU uses TILE_SIZE=4, so test edge cases around multiples of 4
        for seqlen in [5, 6, 7, 9, 10, 11]:
            run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
                batch=1, dim=4, seqlen=seqlen, dstate=4, n_groups=1, ctx=ctx
            )
        
        # Test with mamba-130m-hf realistic dimensions
        # dim=1536, dstate=16, n_groups=1
        for seqlen in [5, 6, 7]:
            run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
                batch=1, dim=1536, seqlen=seqlen, dstate=16, n_groups=1, ctx=ctx
            )
        
        # Strict tolerance test - check if CPU/GPU have tiny differences that could compound
        run_selective_scan_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
            batch=1, dim=1536, seqlen=7, dstate=16, n_groups=1, ctx=ctx, rtol=0.0001  # 0.01% tolerance
        )
        
        # Test selective scan update
        run_selective_scan_update_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Basic selective scan update GPU test passed")
        
        # Test update without D
        run_selective_scan_update_gpu[DType.float32, has_D=False, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan update GPU without D test passed")
        
        # Test update without z
        run_selective_scan_update_gpu[DType.float32, has_D=True, has_z=False, has_delta_bias=True, delta_softplus=False](
            batch=2, dim=4, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan update GPU without z test passed")
        
        # Test update with delta_softplus
        run_selective_scan_update_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
            batch=2, dim=4, dstate=4, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan update GPU with delta_softplus test passed")
        
        # Test update with larger dimensions
        run_selective_scan_update_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
            batch=4, dim=8, dstate=8, n_groups=1, ctx=ctx
        )
        print("✓ Selective scan update GPU larger dimensions test passed")

