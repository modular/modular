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
    ssd_combined_cpu,
    ssd_combined_gpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_ssd_combined_gpu[
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
    """Test SSD combined GPU kernel against CPU reference."""
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
    var residual_h = UnsafePointer[Scalar[dtype]].alloc(batch * dim * seqlen)
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
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(dim)
    
    # Create LayoutTensors for initialization
    var u_init = LayoutTensor[dtype, layout_3d](
        u_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_init = LayoutTensor[dtype, layout_3d](
        delta_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var residual_init = LayoutTensor[dtype, layout_3d](
        residual_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
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
    var gamma_init = LayoutTensor[dtype, layout_1d](
        gamma_h, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    
    # Initialize with random data
    rand[dtype](u_init.ptr, u_init.size())
    rand[dtype](delta_init.ptr, delta_init.size())
    rand[dtype](residual_init.ptr, residual_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](delta_bias_init.ptr, delta_bias_init.size())
    rand[dtype](gamma_init.ptr, gamma_init.size())
    
    # Initialize gamma to positive values
    for i in range(dim):
        gamma_h[i] = abs(gamma_h[i]) + Scalar[dtype](0.1)
    
    # Allocate GPU memory
    var output_cpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var output_gpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var x_cpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * n_chunks * 2 * dstate)
    var x_gpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * n_chunks * 2 * dstate)
    var out_z_cpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var out_z_gpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var residual_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var u_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var delta_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var A_gpu = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_gpu = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate * seqlen)
    var C_gpu = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate * seqlen)
    var D_gpu = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_gpu = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var delta_bias_gpu = ctx.enqueue_create_buffer[dtype](max(delta_bias_size, 1))
    var gamma_gpu = ctx.enqueue_create_buffer[dtype](dim)
    
    # Copy to GPU
    with ctx.push_context():
        ctx.enqueue_copy[dtype](output_cpu_gpu, output_cpu_h)
        ctx.enqueue_copy[dtype](output_gpu_gpu, output_gpu_h)
        ctx.enqueue_copy[dtype](x_cpu_gpu, x_cpu_h)
        ctx.enqueue_copy[dtype](x_gpu_gpu, x_gpu_h)
        ctx.enqueue_copy[dtype](out_z_cpu_gpu, out_z_cpu_h)
        ctx.enqueue_copy[dtype](out_z_gpu_gpu, out_z_gpu_h)
        ctx.enqueue_copy[dtype](residual_gpu, residual_h)
        ctx.enqueue_copy[dtype](u_gpu, u_h)
        ctx.enqueue_copy[dtype](delta_gpu, delta_h)
        ctx.enqueue_copy[dtype](A_gpu, A_h)
        ctx.enqueue_copy[dtype](B_gpu, B_h)
        ctx.enqueue_copy[dtype](C_gpu, C_h)
        if has_D:
            ctx.enqueue_copy[dtype](D_gpu, D_h)
        if has_z:
            ctx.enqueue_copy[dtype](z_gpu, z_h)
        if has_delta_bias:
            ctx.enqueue_copy[dtype](delta_bias_gpu, delta_bias_h)
        ctx.enqueue_copy[dtype](gamma_gpu, gamma_h)
    
    # Create GPU LayoutTensors
    var output_cpu_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_cpu_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var output_gpu_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_gpu_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var x_cpu_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_cpu_gpu, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var x_gpu_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_gpu_gpu, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var out_z_cpu_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_cpu_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var out_z_gpu_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_gpu_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var residual_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        residual_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var u_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        u_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        delta_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var A_lt = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_gpu, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_gpu, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var C_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_gpu, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    var D_lt = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_gpu, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_gpu, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen if has_z else 0))
    )
    var delta_bias_lt = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_gpu, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    var gamma_lt = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_gpu, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)
    
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
    var residual_b_stride: UInt32 = dim * seqlen
    var residual_d_stride: UInt32 = seqlen
    var residual_t_stride: UInt32 = 1
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
    var gamma_stride: UInt32 = 1
    
    # Run CPU kernel
    ssd_combined_cpu[
        dtype,
        output_cpu_lt.layout,
        x_cpu_lt.layout,
        out_z_cpu_lt.layout,
        residual_lt.layout,
        u_lt.layout,
        delta_lt.layout,
        A_lt.layout,
        B_lt.layout,
        C_lt.layout,
        D_lt.layout,
        z_lt.layout,
        delta_bias_lt.layout,
        gamma_lt.layout,
    ](
        batch,
        dim,
        seqlen,
        dstate,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        output_cpu_lt,
        x_cpu_lt,
        out_z_cpu_lt,
        residual_lt,
        u_lt,
        delta_lt,
        A_lt,
        B_lt,
        C_lt,
        D_lt,
        z_lt,
        delta_bias_lt,
        gamma_lt,
        epsilon,
        weight_offset,
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
        residual_b_stride,
        residual_d_stride,
        residual_t_stride,
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
        gamma_stride,
    )
    
    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    from math import ceildiv
    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
    
    var compiled_kernel = ctx.compile_function_checked[
        ssd_combined_gpu[
            dtype,
            output_gpu_lt.layout,
            x_gpu_lt.layout,
            out_z_gpu_lt.layout,
            residual_lt.layout,
            u_lt.layout,
            delta_lt.layout,
            A_lt.layout,
            B_lt.layout,
            C_lt.layout,
            D_lt.layout,
            z_lt.layout,
            delta_bias_lt.layout,
            gamma_lt.layout,
        ],
        ssd_combined_gpu[
            dtype,
            output_gpu_lt.layout,
            x_gpu_lt.layout,
            out_z_gpu_lt.layout,
            residual_lt.layout,
            u_lt.layout,
            delta_lt.layout,
            A_lt.layout,
            B_lt.layout,
            C_lt.layout,
            D_lt.layout,
            z_lt.layout,
            delta_bias_lt.layout,
            gamma_lt.layout,
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
        Int8(1) if delta_softplus else Int8(0),
        output_gpu_lt,
        x_gpu_lt,
        out_z_gpu_lt,
        residual_lt,
        u_lt,
        delta_lt,
        A_lt,
        B_lt,
        C_lt,
        D_lt,
        z_lt,
        delta_bias_lt,
        gamma_lt,
        epsilon,
        weight_offset,
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
        residual_b_stride,
        residual_d_stride,
        residual_t_stride,
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
        gamma_stride,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    
    ctx.synchronize()
    
    # Copy results back
    with ctx.push_context():
        ctx.enqueue_copy[dtype](output_cpu_h, output_cpu_gpu)
        ctx.enqueue_copy[dtype](output_gpu_h, output_gpu_gpu)
    
    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_gpu_h[i],
            rtol=rtol,
        )
    
    # Cleanup
    output_cpu_h.free()
    output_gpu_h.free()
    x_cpu_h.free()
    x_gpu_h.free()
    out_z_cpu_h.free()
    out_z_gpu_h.free()
    residual_h.free()
    u_h.free()
    delta_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    delta_bias_h.free()
    gamma_h.free()
    # Device buffers are automatically freed when they go out of scope
    _ = output_cpu_gpu^
    _ = output_gpu_gpu^
    _ = x_cpu_gpu^
    _ = x_gpu_gpu^
    _ = out_z_cpu_gpu^
    _ = out_z_gpu_gpu^
    _ = residual_gpu^
    _ = u_gpu^
    _ = delta_gpu^
    _ = A_gpu^
    _ = B_gpu^
    _ = C_gpu^
    _ = D_gpu^
    _ = z_gpu^
    _ = delta_bias_gpu^
    _ = gamma_gpu^


def main():
    var ctx = DeviceContext()
    
    # Test basic ssd_combined
    run_ssd_combined_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
    )
    print("✓ Basic SSD combined GPU test passed")
    
    # Test without D
    run_ssd_combined_gpu[DType.float32, has_D=False, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
    )
    print("✓ SSD combined GPU without D test passed")
    
    # Test without z
    run_ssd_combined_gpu[DType.float32, has_D=True, has_z=False, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
    )
    print("✓ SSD combined GPU without z test passed")
    
    # Test without delta_bias
    run_ssd_combined_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=False, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
    )
    print("✓ SSD combined GPU without delta_bias test passed")
    
    # Test with delta_softplus
    run_ssd_combined_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1, ctx=ctx
    )
    print("✓ SSD combined GPU with delta_softplus test passed")
    
    # Test larger shapes
    run_ssd_combined_gpu[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=4, dim=8, seqlen=16, dstate=8, n_groups=1, ctx=ctx
    )
    print("✓ SSD combined GPU larger shapes test passed")
    
    print("All SSD combined GPU tests passed!")

