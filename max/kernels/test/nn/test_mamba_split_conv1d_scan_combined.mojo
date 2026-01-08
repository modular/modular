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
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_mamba_split_conv1d_scan_combined[
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
) raises:
    var group_size = dim // nheads
    var n_chunks = ceildiv(seqlen, chunk_size)
    
    # zxbcdt: (batch, seqlen, 2*dim + 2*ngroups*dstate + nheads)
    var zxbcdt_channels = 2 * dim + 2 * ngroups * dstate + nheads
    var zxbcdt_size = batch * seqlen * zxbcdt_channels
    var zxbcdt_heap = alloc[Scalar[dtype]](zxbcdt_size)
    comptime layout_3d = Layout.row_major[3]()
    var zxbcdt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        zxbcdt_heap, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, zxbcdt_channels))
    )
    
    # conv_weight: (dim + 2*ngroups*dstate, width)
    var conv_weight_channels = dim + 2 * ngroups * dstate
    var conv_weight_size = conv_weight_channels * width
    var conv_weight_heap = alloc[Scalar[dtype]](conv_weight_size)
    comptime layout_2d = Layout.row_major[2]()
    var conv_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_weight_heap, RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width))
    )
    
    # conv_bias: (dim + 2*ngroups*dstate,)
    var conv_bias_size = conv_weight_channels
    var conv_bias_heap = alloc[Scalar[dtype]](conv_bias_size)
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var conv_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        conv_bias_heap, RuntimeLayout[layout_1d].row_major(Index(conv_bias_size))
    )
    
    # dt_bias: (nheads,)
    var dt_bias_size = nheads
    var dt_bias_heap = alloc[Scalar[dtype]](dt_bias_size)
    var dt_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        dt_bias_heap, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    
    # A: (nheads,)
    var A_size = nheads
    var A_heap = alloc[Scalar[dtype]](A_size)
    var A_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_1d].row_major(Index(A_size))
    )
    
    # D: (nheads, headdim) or (nheads,)
    var D_size = nheads * headdim if has_D else 0
    var D_heap = alloc[Scalar[dtype]](max(D_size, 1))
    var D_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        D_heap, RuntimeLayout[layout_2d].row_major(Index(nheads if has_D else 0, headdim if has_D and D_size > nheads else 0))
    )
    
    # x: (batch, dim, num_chunks, 2*dstate)
    var x_size = batch * dim * n_chunks * 2 * dstate
    var x_heap = alloc[Scalar[dtype]](x_size)
    comptime layout_4d = Layout.row_major[4]()
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    )
    
    # out_z: (batch, dim, seqlen)
    var out_z_size = batch * dim * seqlen
    var out_z_heap = alloc[Scalar[dtype]](out_z_size)
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # dt: (batch, nheads, seqlen)
    var dt_size = batch * nheads * seqlen
    var dt_heap = alloc[Scalar[dtype]](dt_size)
    var dt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        dt_heap, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, seqlen))
    )
    
    # B: (batch, ngroups, dstate, seqlen)
    var B_size = batch * ngroups * dstate * seqlen
    var B_heap = alloc[Scalar[dtype]](B_size)
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    
    # C: (batch, ngroups, dstate, seqlen)
    var C_size = batch * ngroups * dstate * seqlen
    var C_heap = alloc[Scalar[dtype]](C_size)
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap, RuntimeLayout[layout_4d].row_major(Index(batch, ngroups, dstate, seqlen))
    )
    
    # z: (batch, dim, seqlen)
    var z_size = batch * dim * seqlen
    var z_heap = alloc[Scalar[dtype]](z_size)
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # rmsnorm_weight: (dim,)
    var rmsnorm_weight_size = dim if has_rmsnorm else 0
    var rmsnorm_weight_heap = alloc[Scalar[dtype]](max(rmsnorm_weight_size, 1))
    var rmsnorm_weight_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        rmsnorm_weight_heap, RuntimeLayout[layout_1d].row_major(Index(rmsnorm_weight_size))
    )
    
    # outproj_weight: (out_dim, dim)
    var out_dim = dim  # For simplicity, use same as input dim
    var outproj_weight_size = out_dim * dim if has_outproj else 0
    var outproj_weight_heap = alloc[Scalar[dtype]](max(outproj_weight_size, 1))
    var outproj_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        outproj_weight_heap, RuntimeLayout[layout_2d].row_major(Index(out_dim if has_outproj else 0, dim if has_outproj else 0))
    )
    
    # outproj_bias: (out_dim,)
    var outproj_bias_size = out_dim if has_outproj else 0
    var outproj_bias_heap = alloc[Scalar[dtype]](max(outproj_bias_size, 1))
    var outproj_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        outproj_bias_heap, RuntimeLayout[layout_1d].row_major(Index(outproj_bias_size))
    )
    
    # output: (batch, seqlen, dim) or (batch, seqlen, out_dim)
    var output_size = batch * seqlen * (out_dim if has_outproj else dim)
    var output_heap = alloc[Scalar[dtype]](output_size)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap, RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, out_dim if has_outproj else dim))
    )
    
    # Initialize data
    random(zxbcdt_h)
    random(conv_weight_h)
    random(conv_bias_h)
    random(dt_bias_h)
    random(A_h)
    if has_D:
        random(D_h)
    if has_rmsnorm:
        random(rmsnorm_weight_h)
        # Make positive
        for i in range(dim):
            rmsnorm_weight_h.ptr[i] = abs(rmsnorm_weight_h.ptr[i]) + Scalar[dtype](0.1)
    if has_outproj:
        random(outproj_weight_h)
        random(outproj_bias_h)
    
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
    
    # Call kernel
    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        zxbcdt_h.layout,
        conv_weight_h.layout,
        conv_bias_h.layout,
        output_h.layout,
        x_h.layout,
        out_z_h.layout,
        dt_h.layout,
        A_h.layout,
        B_h.layout,
        C_h.layout,
        D_h.layout,
        z_h.layout,
        dt_bias_h.layout,
        rmsnorm_weight_h.layout,
        outproj_weight_h.layout,
        outproj_bias_h.layout,
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
        zxbcdt_h,
        conv_weight_h,
        conv_bias_h,
        dt_bias_h,
        A_h,
        D_h,
        x_h,
        out_z_h,
        dt_h,
        B_h,
        C_h,
        z_h,
        rmsnorm_weight_h,
        outproj_weight_h,
        outproj_bias_h,
        output_h,
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
    
    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    var sample_size = min(10, batch * seqlen * (out_dim if has_outproj else dim))
    for i in range(sample_size):
        if abs(Float32(output_h.ptr[i])) > 1e-8:
            has_nonzero = True
            break
    
    if not has_nonzero:
        raise Error("Output is all zeros - kernel may not be executing correctly")
    
    # Cleanup
    zxbcdt_heap.free()
    conv_weight_heap.free()
    conv_bias_heap.free()
    dt_bias_heap.free()
    A_heap.free()
    D_heap.free()
    x_heap.free()
    out_z_heap.free()
    dt_heap.free()
    B_heap.free()
    C_heap.free()
    z_heap.free()
    rmsnorm_weight_heap.free()
    outproj_weight_heap.free()
    outproj_bias_heap.free()
    output_heap.free()


def main():
    # Test basic mamba_split_conv1d_scan_combined
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4
    )
    print("✓ Basic mamba_split_conv1d_scan_combined test passed")
    
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=False, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4
    )
    print("✓ mamba_split_conv1d_scan_combined without D test passed")
    
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=True, has_rmsnorm=True, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4
    )
    print("✓ mamba_split_conv1d_scan_combined with RMSNorm test passed")
    
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=False, delta_softplus=True](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4
    )
    print("✓ mamba_split_conv1d_scan_combined with norm_after_gate test passed")
    
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=False](
        batch=2, seqlen=8, dim=4, nheads=2, headdim=2, dstate=4, ngroups=1, width=4, chunk_size=4
    )
    print("✓ mamba_split_conv1d_scan_combined without delta_softplus test passed")
    
    run_mamba_split_conv1d_scan_combined[DType.float32, has_D=True, has_rmsnorm=False, has_outproj=False, norm_before_gate=True, delta_softplus=True](
        batch=1, seqlen=32, dim=16, nheads=4, headdim=4, dstate=8, ngroups=2, width=4, chunk_size=8
    )
    print("✓ Larger mamba_split_conv1d_scan_combined test passed")
    
    print("All mamba_split_conv1d_scan_combined tests passed!")

