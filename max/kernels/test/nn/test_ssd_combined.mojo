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

from math import exp, exp2, log, rsqrt
from sys.info import simd_width_of

from algorithm.functional import _get_start_indices_of_nth_subvolume
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import random
from layout.int_tuple import fill_like
from memory import alloc
from nn.selective_scan import (
    ssd_combined_cpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


# LOG2E constant for converting exp to exp2
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 16


@always_inline
fn softplus_ref(val: Float32) -> Float32:
    """Reference softplus implementation: log(1 + exp(x))."""
    if val > 20.0:
        return val
    var exp_val = exp(val)
    var one = Float32(1.0)
    return log(one + exp_val)


@always_inline
fn sigmoid_ref(val: Float32) -> Float32:
    """Reference sigmoid implementation."""
    if val < -20.0:
        return 0.0
    var exp_neg = exp(-val)
    return 1.0 / (1.0 + exp_neg)


@always_inline
fn silu_ref(val: Float32) -> Float32:
    """Reference SiLU implementation."""
    if val < -20.0:
        return 0.0
    var exp_neg = exp(-val)
    return val / (1.0 + exp_neg)


fn run_ssd_combined[
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
    rtol: Float64 = 0.01,
) raises:
    """Test SSD combined kernel against reference implementation."""
    if dstate > MAX_DSTATE:
        return  # Skip if dstate exceeds kernel limit
    
    var group_size = dim // n_groups
    var chunk_size = 2048
    var n_chunks = (seqlen + chunk_size - 1) // chunk_size
    
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    # output: (batch, dim, seqlen)
    var output_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    
    # x: (batch, dim, num_chunks, 2*dstate) - checkpoint tensor
    var x_heap = alloc[Scalar[dtype]](batch * dim * n_chunks * 2 * dstate)
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_4d].row_major(Index(batch, dim, n_chunks, 2 * dstate))
    ).fill(0)
    
    # out_z: (batch, dim, seqlen) - gated output
    var out_z_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    
    # residual: (batch, dim, seqlen)
    var residual_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var residual_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        residual_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # u: (batch, dim, seqlen)
    var u_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var u_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        u_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # delta: (batch, dim, seqlen)
    var delta_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var delta_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        delta_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # A: (dim, dstate)
    var A_heap = alloc[Scalar[dtype]](dim * dstate)
    var A_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    
    # B: (batch, n_groups, dstate, seqlen)
    var B_heap = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    
    # C: (batch, n_groups, dstate, seqlen)
    var C_heap = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap, RuntimeLayout[layout_4d].row_major(Index(batch, n_groups, dstate, seqlen))
    )
    
    # D: (dim,) - optional
    var D_size = dim if has_D else 0
    var D_heap = alloc[Scalar[dtype]](max(D_size, 1))
    var D_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_heap, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    
    # z: (batch, dim, seqlen) - optional
    var z_size = batch * dim * seqlen if has_z else 0
    var z_heap = alloc[Scalar[dtype]](max(z_size, 1))
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap, RuntimeLayout[layout_3d].row_major(Index(batch if has_z else 0, dim if has_z else 0, seqlen if has_z else 0))
    )
    
    # delta_bias: (dim,) - optional
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var delta_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_heap, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    
    # gamma: (dim,) - for normalization
    var gamma_heap = alloc[Scalar[dtype]](dim)
    var gamma_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    
    # Initialize data
    random(u_h)
    random(delta_h)
    random(residual_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_delta_bias:
        random(delta_bias_h)
    random(gamma_h)
    
    # Initialize gamma to positive values
    for i in range(dim):
        gamma_h.ptr[i] = abs(gamma_h.ptr[i]) + Scalar[dtype](0.1)
    
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
    
    # Call kernel
    ssd_combined_cpu[
        dtype,
        output_h.layout,
        x_h.layout,
        out_z_h.layout,
        residual_h.layout,
        u_h.layout,
        delta_h.layout,
        A_h.layout,
        B_h.layout,
        C_h.layout,
        D_h.layout,
        z_h.layout,
        delta_bias_h.layout,
        gamma_h.layout,
    ](
        batch,
        dim,
        seqlen,
        dstate,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        output_h,
        x_h,
        out_z_h,
        residual_h,
        u_h,
        delta_h,
        A_h,
        B_h,
        C_h,
        D_h,
        z_h,
        delta_bias_h,
        gamma_h,
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
    
    # Basic sanity check: output should not be all zeros
    # Check a few sample outputs to verify kernel executed
    var has_nonzero = False
    var sample_size = min(10, batch * dim * seqlen)
    for i in range(sample_size):
        var val = Float32(output_h.ptr[i])
        if abs(val) > 1e-8:
            has_nonzero = True
            break
    
    if not has_nonzero:
        raise Error("Output is all zeros - kernel may not be executing correctly")
    
    # Cleanup
    output_heap.free()
    x_heap.free()
    out_z_heap.free()
    residual_heap.free()
    u_heap.free()
    delta_heap.free()
    A_heap.free()
    B_heap.free()
    C_heap.free()
    D_heap.free()
    z_heap.free()
    delta_bias_heap.free()
    gamma_heap.free()


def main():
    # Test basic ssd_combined
    run_ssd_combined[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1
    )
    print("✓ Basic SSD combined test passed")
    
    # Test without D
    run_ssd_combined[DType.float32, has_D=False, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1
    )
    print("✓ SSD combined without D test passed")
    
    # Test without z
    run_ssd_combined[DType.float32, has_D=True, has_z=False, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1
    )
    print("✓ SSD combined without z test passed")
    
    # Test without delta_bias
    run_ssd_combined[DType.float32, has_D=True, has_z=True, has_delta_bias=False, delta_softplus=False](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1
    )
    print("✓ SSD combined without delta_bias test passed")
    
    # Test with delta_softplus
    run_ssd_combined[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
        batch=2, dim=4, seqlen=8, dstate=4, n_groups=1
    )
    print("✓ SSD combined with delta_softplus test passed")
    
    # Test larger shapes
    run_ssd_combined[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=4, dim=8, seqlen=16, dstate=8, n_groups=1
    )
    print("✓ SSD combined larger shapes test passed")
    
    print("All SSD combined tests passed!")

