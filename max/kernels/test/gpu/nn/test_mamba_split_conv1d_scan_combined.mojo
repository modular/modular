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

from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from layout._fillers import random
from memory import alloc
from nn.selective_scan import (
    mamba_split_conv1d_scan_combined_cpu,
    mamba_split_conv1d_scan_combined_gpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_mamba_split_conv1d_scan_combined_gpu[
    dtype: DType,
    has_D: Bool,
    has_rmsnorm: Bool,
    has_outproj: Bool,
    norm_before_gate: Bool,
    delta_softplus: Bool,
](
    batch: Int,
    seqlen: Int,
    dim: Int,
    nheads: Int,
    headdim: Int,
    dstate: Int,
    ngroups: Int,
    width: Int,
    chunk_size: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    var group_size = dim // nheads
    var n_chunks = ceildiv(seqlen, chunk_size)
    
    # Allocate host memory
    var zxbcdt_channels = 2 * dim + 2 * ngroups * dstate + nheads
    var zxbcdt_size = batch * seqlen * zxbcdt_channels
    var zxbcdt_h = alloc[Scalar[dtype]](zxbcdt_size)
    var conv_weight_channels = dim + 2 * ngroups * dstate
    var conv_weight_size = conv_weight_channels * width
    var conv_weight_h = alloc[Scalar[dtype]](conv_weight_size)
    var conv_bias_size = conv_weight_channels
    var conv_bias_h = alloc[Scalar[dtype]](conv_bias_size)
    var dt_bias_size = nheads
    var dt_bias_h = alloc[Scalar[dtype]](dt_bias_size)
    var A_size = nheads
    var A_h = alloc[Scalar[dtype]](A_size)
    var D_size = nheads * headdim if has_D else 0
    var D_h = alloc[Scalar[dtype]](max(D_size, 1))
    var x_size = batch * dim * n_chunks * 2 * dstate
    var x_h = alloc[Scalar[dtype]](x_size)
    var out_z_size = batch * dim * seqlen
    var out_z_h = alloc[Scalar[dtype]](out_z_size)
    var dt_size = batch * nheads * seqlen
    var dt_h = alloc[Scalar[dtype]](dt_size)
    var B_size = batch * ngroups * dstate * seqlen
    var B_h = alloc[Scalar[dtype]](B_size)
    var C_size = batch * ngroups * dstate * seqlen
    var C_h = alloc[Scalar[dtype]](C_size)
    var z_size = batch * dim * seqlen
    var z_h = alloc[Scalar[dtype]](z_size)
    var rmsnorm_weight_size = dim if has_rmsnorm else 0
    var rmsnorm_weight_h = alloc[Scalar[dtype]](max(rmsnorm_weight_size, 1))
    var out_dim = dim
    var outproj_weight_size = out_dim * dim if has_outproj else 0
    var outproj_weight_h = alloc[Scalar[dtype]](max(outproj_weight_size, 1))
    var outproj_bias_size = out_dim if has_outproj else 0
    var outproj_bias_h = alloc[Scalar[dtype]](max(outproj_bias_size, 1))
    var output_size = batch * seqlen * (out_dim if has_outproj else dim)
    var output_cpu_h = alloc[Scalar[dtype]](output_size)
    var output_gpu_h = alloc[Scalar[dtype]](output_size)
    
    # Create LayoutTensors for initialization
    comptime layout_3d = Layout.row_major(UNKNOWN_VALUE)
    comptime layout_4d = Layout.row_major(UNKNOWN_VALUE)
    comptime layout_2d = Layout.row_major(UNKNOWN_VALUE)
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    var zxbcdt_init = LayoutTensor[dtype, layout_3d](
        zxbcdt_h, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, zxbcdt_channels))
    )
    var conv_weight_init = LayoutTensor[dtype, layout_2d](
        conv_weight_h, RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width))
    )
    var conv_bias_init = LayoutTensor[dtype, layout_1d](
        conv_bias_h, RuntimeLayout[layout_1d].row_major(Index(conv_bias_size))
    )
    var dt_bias_init = LayoutTensor[dtype, layout_1d](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    var A_init = LayoutTensor[dtype, layout_1d](
        A_h, RuntimeLayout[layout_1d].row_major(Index(A_size))
    )
    var D_init = LayoutTensor[dtype, layout_2d](
        D_h, RuntimeLayout[layout_2d].row_major(Index(nheads if has_D else 0, headdim if has_D and D_size > nheads else 0))
    )
    var rmsnorm_weight_init = LayoutTensor[dtype, layout_1d](
        rmsnorm_weight_h, RuntimeLayout[layout_1d].row_major(Index(rmsnorm_weight_size))
    )
    var outproj_weight_init = LayoutTensor[dtype, layout_2d](
        outproj_weight_h, RuntimeLayout[layout_2d].row_major(Index(out_dim if has_outproj else 0, dim if has_outproj else 0))
    )
    var outproj_bias_init = LayoutTensor[dtype, layout_1d](
        outproj_bias_h, RuntimeLayout[layout_1d].row_major(Index(outproj_bias_size))
    )
    
    # Initialize with random data
    random(zxbcdt_init)
    random(conv_weight_init)
    random(conv_bias_init)
    random(dt_bias_init)
    random(A_init)
    if has_D:
        random(D_init)
    if has_rmsnorm:
        random(rmsnorm_weight_init)
        for i in range(dim):
            rmsnorm_weight_h[i] = abs(rmsnorm_weight_h[i]) + Scalar[dtype](0.1)
    if has_outproj:
        random(outproj_weight_init)
        random(outproj_bias_init)
    
    # Allocate GPU memory
    var zxbcdt_d = ctx.enqueue_create_buffer[dtype](zxbcdt_size)
    var conv_weight_d = ctx.enqueue_create_buffer[dtype](conv_weight_size)
    var conv_bias_d = ctx.enqueue_create_buffer[dtype](conv_bias_size)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](dt_bias_size)
    var A_d = ctx.enqueue_create_buffer[dtype](A_size)
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var x_d = ctx.enqueue_create_buffer[dtype](x_size)
    var out_z_d = ctx.enqueue_create_buffer[dtype](out_z_size)
    var dt_d = ctx.enqueue_create_buffer[dtype](dt_size)
    var B_d = ctx.enqueue_create_buffer[dtype](B_size)
    var C_d = ctx.enqueue_create_buffer[dtype](C_size)
    var z_d = ctx.enqueue_create_buffer[dtype](z_size)
    var rmsnorm_weight_d = ctx.enqueue_create_buffer[dtype](max(rmsnorm_weight_size, 1))
    var outproj_weight_d = ctx.enqueue_create_buffer[dtype](max(outproj_weight_size, 1))
    var outproj_bias_d = ctx.enqueue_create_buffer[dtype](max(outproj_bias_size, 1))
    var output_cpu_d = ctx.enqueue_create_buffer[dtype](output_size)
    var output_gpu_d = ctx.enqueue_create_buffer[dtype](output_size)
    
    # Copy to GPU
    with ctx.push_context():
        ctx.enqueue_copy(zxbcdt_d, zxbcdt_h)
        ctx.enqueue_copy(conv_weight_d, conv_weight_h)
        ctx.enqueue_copy(conv_bias_d, conv_bias_h)
        ctx.enqueue_copy(dt_bias_d, dt_bias_h)
        ctx.enqueue_copy(A_d, A_h)
        if has_D:
            ctx.enqueue_copy(D_d, D_h)
        if has_rmsnorm:
            ctx.enqueue_copy(rmsnorm_weight_d, rmsnorm_weight_h)
        if has_outproj:
            ctx.enqueue_copy(outproj_weight_d, outproj_weight_h)
            ctx.enqueue_copy(outproj_bias_d, outproj_bias_h)
    
    # Create LayoutTensors for GPU
    var zxbcdt_gpu_lt = LayoutTensor[dtype, layout_3d](
        zxbcdt_d, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, zxbcdt_channels))
    )
    var conv_weight_gpu_lt = LayoutTensor[dtype, layout_2d](
        conv_weight_d, RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width))
    )
    var conv_bias_gpu_lt = LayoutTensor[dtype, layout_1d](
        conv_bias_d, RuntimeLayout[layout_1d].row_major(Index(conv_bias_size))
    )
    var dt_bias_gpu_lt = LayoutTensor[dtype, layout_1d](
        dt_bias_d, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    var A_gpu_lt = LayoutTensor[dtype, layout_1d](
        A_d, RuntimeLayout[layout_1d].row_major(Index(A_size))
    )
    var D_gpu_lt = LayoutTensor[dtype, layout_2d](
        D_d, RuntimeLayout[layout_2d].row_major(Index(nheads if has_D else 0, headdim if has_D and D_size > nheads else 0))
    )
    var x_gpu_lt = LayoutTensor[dtype, layout_4d](
        x_d, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var out_z_gpu_lt = LayoutTensor[dtype, layout_3d](
        out_z_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var dt_gpu_lt = LayoutTensor[dtype, layout_3d](
        dt_d, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, seqlen))
    )
    var B_gpu_lt = LayoutTensor[dtype, layout_4d](
        B_d, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    var C_gpu_lt = LayoutTensor[dtype, layout_4d](
        C_d, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    var z_gpu_lt = LayoutTensor[dtype, layout_3d](
        z_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var rmsnorm_weight_gpu_lt = LayoutTensor[dtype, layout_1d](
        rmsnorm_weight_d, RuntimeLayout[layout_1d].row_major(Index(rmsnorm_weight_size))
    )
    var outproj_weight_gpu_lt = LayoutTensor[dtype, layout_2d](
        outproj_weight_d, RuntimeLayout[layout_2d].row_major(Index(out_dim if has_outproj else 0, dim if has_outproj else 0))
    )
    var outproj_bias_gpu_lt = LayoutTensor[dtype, layout_1d](
        outproj_bias_d, RuntimeLayout[layout_1d].row_major(Index(outproj_bias_size))
    )
    var output_cpu_gpu_lt = LayoutTensor[dtype, layout_3d](
        output_cpu_d, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, out_dim if has_outproj else dim))
    )
    var output_gpu_gpu_lt = LayoutTensor[dtype, layout_3d](
        output_gpu_d, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, out_dim if has_outproj else dim))
    )
    
    # Create CPU LayoutTensors for reference
    var zxbcdt_cpu_lt = LayoutTensor[dtype, layout_3d](
        zxbcdt_h, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, zxbcdt_channels))
    )
    var conv_weight_cpu_lt = LayoutTensor[dtype, layout_2d](
        conv_weight_h, RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width))
    )
    var conv_bias_cpu_lt = LayoutTensor[dtype, layout_1d](
        conv_bias_h, RuntimeLayout[layout_1d].row_major(Index(conv_bias_size))
    )
    var dt_bias_cpu_lt = LayoutTensor[dtype, layout_1d](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    var A_cpu_lt = LayoutTensor[dtype, layout_1d](
        A_h, RuntimeLayout[layout_1d].row_major(Index(A_size))
    )
    var D_cpu_lt = LayoutTensor[dtype, layout_2d](
        D_h, RuntimeLayout[layout_2d].row_major(Index(nheads if has_D else 0, headdim if has_D and D_size > nheads else 0))
    )
    var x_cpu_lt = LayoutTensor[dtype, layout_4d](
        x_h, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    var out_z_cpu_lt = LayoutTensor[dtype, layout_3d](
        out_z_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var dt_cpu_lt = LayoutTensor[dtype, layout_3d](
        dt_h, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, seqlen))
    )
    var B_cpu_lt = LayoutTensor[dtype, layout_4d](
        B_h, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    var C_cpu_lt = LayoutTensor[dtype, layout_4d](
        C_h, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    var z_cpu_lt = LayoutTensor[dtype, layout_3d](
        z_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var rmsnorm_weight_cpu_lt = LayoutTensor[dtype, layout_1d](
        rmsnorm_weight_h, RuntimeLayout[layout_1d].row_major(Index(rmsnorm_weight_size))
    )
    var outproj_weight_cpu_lt = LayoutTensor[dtype, layout_2d](
        outproj_weight_h, RuntimeLayout[layout_2d].row_major(Index(out_dim if has_outproj else 0, dim if has_outproj else 0))
    )
    var outproj_bias_cpu_lt = LayoutTensor[dtype, layout_1d](
        outproj_bias_h, RuntimeLayout[layout_1d].row_major(Index(outproj_bias_size))
    )
    var output_cpu_cpu_lt = LayoutTensor[dtype, layout_3d](
        output_cpu_h, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, out_dim if has_outproj else dim))
    )
    
    var epsilon = Scalar[dtype](0.001)
    
    # Strides for row-major layout
    var zxbcdt_b_stride: UInt32 = seqlen * zxbcdt_channels
    var zxbcdt_s_stride: UInt32 = zxbcdt_channels
    var zxbcdt_c_stride: UInt32 = 1
    var conv_weight_c_stride: UInt32 = width
    var conv_weight_w_stride: UInt32 = 1
    var conv_bias_stride: UInt32 = 1
    var output_b_stride: UInt32 = seqlen * (out_dim if has_outproj else dim)
    var output_s_stride: UInt32 = (out_dim if has_outproj else dim)
    var output_c_stride: UInt32 = 1
    var x_b_stride: UInt32 = dim * n_chunks * 2 * dstate
    var x_d_stride: UInt32 = n_chunks * 2 * dstate
    var x_chunk_stride: UInt32 = 2 * dstate
    var x_n_stride: UInt32 = 1
    var out_z_b_stride: UInt32 = dim * seqlen
    var out_z_d_stride: UInt32 = seqlen
    var out_z_t_stride: UInt32 = 1
    var dt_b_stride: UInt32 = nheads * seqlen
    var dt_h_stride: UInt32 = seqlen
    var dt_s_stride: UInt32 = 1
    var A_stride: UInt32 = 1
    var B_b_stride: UInt32 = ngroups * dstate * seqlen
    var B_g_stride: UInt32 = dstate * seqlen
    var B_n_stride: UInt32 = seqlen
    var B_t_stride: UInt32 = 1
    var C_b_stride: UInt32 = ngroups * dstate * seqlen
    var C_g_stride: UInt32 = dstate * seqlen
    var C_n_stride: UInt32 = seqlen
    var C_t_stride: UInt32 = 1
    var D_h_stride: UInt32 = headdim if has_D and D_size > nheads else 1
    var D_p_stride: UInt32 = 1
    var z_b_stride: UInt32 = dim * seqlen
    var z_d_stride: UInt32 = seqlen
    var z_t_stride: UInt32 = 1
    var dt_bias_stride: UInt32 = 1
    var rmsnorm_weight_stride: UInt32 = 1
    var outproj_weight_out_stride: UInt32 = dim
    var outproj_weight_in_stride: UInt32 = 1
    var outproj_bias_stride: UInt32 = 1
    
    # Run CPU kernel
    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        zxbcdt_cpu_lt.layout,
        conv_weight_cpu_lt.layout,
        conv_bias_cpu_lt.layout,
        output_cpu_cpu_lt.layout,
        x_cpu_lt.layout,
        out_z_cpu_lt.layout,
        dt_cpu_lt.layout,
        A_cpu_lt.layout,
        B_cpu_lt.layout,
        C_cpu_lt.layout,
        D_cpu_lt.layout,
        z_cpu_lt.layout,
        dt_bias_cpu_lt.layout,
        rmsnorm_weight_cpu_lt.layout,
        outproj_weight_cpu_lt.layout,
        outproj_bias_cpu_lt.layout,
    ](
        batch,
        seqlen,
        dim,
        nheads,
        headdim,
        dstate,
        ngroups,
        width,
        chunk_size,
        Int8(1) if delta_softplus else Int8(0),
        Int8(1) if norm_before_gate else Int8(0),
        Int8(1) if has_rmsnorm else Int8(0),
        Int8(1) if has_outproj else Int8(0),
        zxbcdt_cpu_lt,
        conv_weight_cpu_lt,
        conv_bias_cpu_lt,
        dt_bias_cpu_lt,
        A_cpu_lt,
        D_cpu_lt,
        x_cpu_lt,
        out_z_cpu_lt,
        dt_cpu_lt,
        B_cpu_lt,
        C_cpu_lt,
        z_cpu_lt,
        rmsnorm_weight_cpu_lt,
        outproj_weight_cpu_lt,
        outproj_bias_cpu_lt,
        output_cpu_cpu_lt,
        epsilon,
        zxbcdt_b_stride,
        zxbcdt_s_stride,
        zxbcdt_c_stride,
        conv_weight_c_stride,
        conv_weight_w_stride,
        conv_bias_stride,
        output_b_stride,
        output_s_stride,
        output_c_stride,
        x_b_stride,
        x_d_stride,
        x_chunk_stride,
        x_n_stride,
        out_z_b_stride,
        out_z_d_stride,
        out_z_t_stride,
        dt_b_stride,
        dt_h_stride,
        dt_s_stride,
        A_stride,
        B_b_stride,
        B_g_stride,
        B_n_stride,
        B_t_stride,
        C_b_stride,
        C_g_stride,
        C_n_stride,
        C_t_stride,
        D_h_stride,
        D_p_stride,
        z_b_stride,
        z_d_stride,
        z_t_stride,
        dt_bias_stride,
        rmsnorm_weight_stride,
        outproj_weight_out_stride,
        outproj_weight_in_stride,
        outproj_bias_stride,
    )
    
    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
    
    var compiled_kernel = ctx.compile_function_checked[
        mamba_split_conv1d_scan_combined_gpu[
            dtype,
            zxbcdt_gpu_lt.layout,
            conv_weight_gpu_lt.layout,
            conv_bias_gpu_lt.layout,
            output_gpu_gpu_lt.layout,
            x_gpu_lt.layout,
            out_z_gpu_lt.layout,
            dt_gpu_lt.layout,
            A_gpu_lt.layout,
            B_gpu_lt.layout,
            C_gpu_lt.layout,
            D_gpu_lt.layout,
            z_gpu_lt.layout,
            dt_bias_gpu_lt.layout,
            rmsnorm_weight_gpu_lt.layout,
            outproj_weight_gpu_lt.layout,
            outproj_bias_gpu_lt.layout,
        ],
        mamba_split_conv1d_scan_combined_gpu[
            dtype,
            zxbcdt_gpu_lt.layout,
            conv_weight_gpu_lt.layout,
            conv_bias_gpu_lt.layout,
            output_gpu_gpu_lt.layout,
            x_gpu_lt.layout,
            out_z_gpu_lt.layout,
            dt_gpu_lt.layout,
            A_gpu_lt.layout,
            B_gpu_lt.layout,
            C_gpu_lt.layout,
            D_gpu_lt.layout,
            z_gpu_lt.layout,
            dt_bias_gpu_lt.layout,
            rmsnorm_weight_gpu_lt.layout,
            outproj_weight_gpu_lt.layout,
            outproj_bias_gpu_lt.layout,
        ]
    ]()
    
    ctx.enqueue_function_checked(
        compiled_kernel,
        total_batch_dim,
        batch,
        seqlen,
        dim,
        nheads,
        headdim,
        dstate,
        ngroups,
        width,
        chunk_size,
        Int8(1) if delta_softplus else Int8(0),
        Int8(1) if norm_before_gate else Int8(0),
        Int8(1) if has_rmsnorm else Int8(0),
        Int8(1) if has_outproj else Int8(0),
        zxbcdt_gpu_lt,
        conv_weight_gpu_lt,
        conv_bias_gpu_lt,
        dt_bias_gpu_lt,
        A_gpu_lt,
        D_gpu_lt,
        x_gpu_lt,
        out_z_gpu_lt,
        dt_gpu_lt,
        B_gpu_lt,
        C_gpu_lt,
        z_gpu_lt,
        rmsnorm_weight_gpu_lt,
        outproj_weight_gpu_lt,
        outproj_bias_gpu_lt,
        output_gpu_gpu_lt,
        epsilon,
        zxbcdt_b_stride,
        zxbcdt_s_stride,
        zxbcdt_c_stride,
        conv_weight_c_stride,
        conv_weight_w_stride,
        conv_bias_stride,
        output_b_stride,
        output_s_stride,
        output_c_stride,
        x_b_stride,
        x_d_stride,
        x_chunk_stride,
        x_n_stride,
        out_z_b_stride,
        out_z_d_stride,
        out_z_t_stride,
        dt_b_stride,
        dt_h_stride,
        dt_s_stride,
        A_stride,
        B_b_stride,
        B_g_stride,
        B_n_stride,
        B_t_stride,
        C_b_stride,
        C_g_stride,
        C_n_stride,
        C_t_stride,
        D_h_stride,
        D_p_stride,
        z_b_stride,
        z_d_stride,
        z_t_stride,
        dt_bias_stride,
        rmsnorm_weight_stride,
        outproj_weight_out_stride,
        outproj_weight_in_stride,
        outproj_bias_stride,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    
    # Copy results back
    with ctx.push_context():
        ctx.enqueue_copy(output_cpu_h, output_cpu_d)
        ctx.enqueue_copy(output_gpu_h, output_gpu_d)
    
    # Compare results
    for i in range(output_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_gpu_h[i],
            rtol=rtol,
        )
    
    # Cleanup
    zxbcdt_h.free()
    conv_weight_h.free()
    conv_bias_h.free()
    dt_bias_h.free()
    A_h.free()
    D_h.free()
    x_h.free()
    out_z_h.free()
    dt_h.free()
    B_h.free()
    C_h.free()
    z_h.free()
    rmsnorm_weight_h.free()
    outproj_weight_h.free()
    outproj_bias_h.free()
    output_cpu_h.free()
    output_gpu_h.free()
    # Device buffers are automatically freed when they go out of scope
    _ = zxbcdt_d^
    _ = conv_weight_d^
    _ = conv_bias_d^
    _ = dt_bias_d^
    _ = A_d^
    _ = D_d^
    _ = x_d^
    _ = out_z_d^
    _ = dt_d^
    _ = B_d^
    _ = C_d^
    _ = z_d^
    _ = rmsnorm_weight_d^
    _ = outproj_weight_d^
    _ = outproj_bias_d^
    _ = output_cpu_d^
    _ = output_gpu_d^


def main():
    var ctx = DeviceContext()
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4, ctx=ctx
    )
    print("✓ Basic mamba_split_conv1d_scan_combined GPU test passed")
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=False, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4, ctx=ctx
    )
    print("✓ mamba_split_conv1d_scan_combined GPU without D test passed")
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=True, has_rmsnorm=True, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4, ctx=ctx
    )
    print("✓ mamba_split_conv1d_scan_combined GPU with RMSNorm test passed")
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=False, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4, ctx=ctx
    )
    print("✓ mamba_split_conv1d_scan_combined GPU with norm_after_gate test passed")
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=False](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4, ctx=ctx
    )
    print("✓ mamba_split_conv1d_scan_combined GPU without delta_softplus test passed")
    
    run_mamba_split_conv1d_scan_combined_gpu[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=1, seqlen=32, dim=16, nheads=4, headdim=4, dstate=8, ngroups=2, width=4, chunk_size=8, ctx=ctx
    )
    print("✓ Larger mamba_split_conv1d_scan_combined GPU test passed")
    
    print("All mamba_split_conv1d_scan_combined GPU tests passed!")

