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

from math import exp, exp2, log
from sys.info import simd_width_of

from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from layout._fillers import random
from memory import alloc
from nn.varlen_selective_scan import (
    varlen_selective_scan_fwd_cpu,
    varlen_selective_state_update_cpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


# LOG2E constant for converting exp to exp2
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 256


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


fn run_varlen_selective_scan_fwd[
    dtype: DType,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    dstate: Int,
    ngroups: Int,
    seq_lengths: IndexList,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen selective scan forward kernel.
    
    Args:
        batch: Number of sequences
        dim: Hidden dimension
        dstate: State dimension
        ngroups: Number of groups
        seq_lengths: List of sequence lengths for each batch item
    """
    if dstate > MAX_DSTATE:
        return  # Skip if dstate exceeds kernel limit
    
    # Calculate total_length (sum of all sequence lengths)
    var total_length = 0
    for i in range(batch):
        total_length += seq_lengths[i]
    
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    # u: (dim, total_length)
    var u_heap = alloc[Scalar[dtype]](dim * total_length)
    var u_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        u_heap, RuntimeLayout[layout_2d].row_major(Index(dim, total_length))
    )
    
    # delta: (dim, total_length) - also used as output if no z
    var delta_heap = alloc[Scalar[dtype]](dim * total_length)
    var delta_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        delta_heap, RuntimeLayout[layout_2d].row_major(Index(dim, total_length))
    )
    
    # A: (dim, dstate)
    var A_heap = alloc[Scalar[dtype]](dim * dstate)
    var A_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    
    # B: (ngroups, dstate, total_length)
    var B_heap = alloc[Scalar[dtype]](ngroups * dstate * total_length)
    var B_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        B_heap, RuntimeLayout[layout_3d].row_major(Index(ngroups, dstate, total_length))
    )
    
    # C: (ngroups, dstate, total_length)
    var C_heap = alloc[Scalar[dtype]](ngroups * dstate * total_length)
    var C_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_heap, RuntimeLayout[layout_3d].row_major(Index(ngroups, dstate, total_length))
    )
    
    # D: (dim,) or empty
    var D_size = dim if has_D else 0
    var D_heap = alloc[Scalar[dtype]](max(D_size, 1))
    var D_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_heap, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    
    # z: (dim, total_length) or empty
    var z_size = dim * total_length if has_z else 0
    var z_heap = alloc[Scalar[dtype]](max(z_size, 1))
    var z_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        z_heap, RuntimeLayout[layout_2d].row_major(Index(dim if has_z else 0, total_length if has_z else 0))
    )
    
    # delta_bias: (dim,) or empty
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var delta_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_heap, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    
    # ssm_states: (batch, dim, dstate) - in/out
    var ssm_states_heap = alloc[Scalar[dtype]](batch * dim * dstate)
    var ssm_states_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        ssm_states_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate))
    ).fill(0)
    
    # output: (dim, total_length) - same as delta
    var output_heap = alloc[Scalar[dtype]](dim * total_length)
    var output_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_heap, RuntimeLayout[layout_2d].row_major(Index(dim, total_length))
    ).fill(0)
    
    # query_start_loc: (batch + 1,) - cumulative sequence lengths
    var query_start_loc_heap = alloc[Scalar[DType.int32]](batch + 1)
    var query_start_loc_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        query_start_loc_heap, RuntimeLayout[layout_1d].row_major(Index(batch + 1))
    )
    var cumsum = 0
    query_start_loc_h.ptr.offset(0).store(Scalar[DType.int32](0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_h.ptr.offset(i + 1).store(Scalar[DType.int32](cumsum))
    
    # cache_indices: (batch,) - can be empty or identity mapping
    var cache_indices_heap = alloc[Scalar[DType.int32]](batch)
    var cache_indices_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        cache_indices_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        cache_indices_h.ptr.offset(i).store(Scalar[DType.int32](i))
    
    # has_initial_state: (batch,) - can be empty or all False
    var has_initial_state_heap = alloc[Scalar[DType.bool]](batch)
    var has_initial_state_h = LayoutTensor[DType.bool, layout_1d, MutAnyOrigin](
        has_initial_state_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        has_initial_state_h.ptr.offset(i).store(Scalar[DType.bool](False))
    
    # Initialize input data
    random(u_h)
    random(delta_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_delta_bias:
        random(delta_bias_h)
    
    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.ptr.offset(i).load()
        A_h.ptr.offset(i).store(Scalar[dtype](Float32(val) * -0.5))
    
    # Scale delta to be positive
    for i in range(dim * total_length):
        var val = delta_h.ptr.offset(i).load()
        delta_h.ptr.offset(i).store(Scalar[dtype](abs(Float32(val)) * 0.5))
    
    var u_buf = u_h
    var delta_buf = delta_h
    var A_buf = A_h
    var B_buf = B_h
    var C_buf = C_h
    var D_buf = D_h
    var z_buf = z_h
    var delta_bias_buf = delta_bias_h
    var ssm_states_buf = ssm_states_h
    var output_buf = output_h
    var query_start_loc_buf = query_start_loc_h
    var cache_indices_buf = cache_indices_h
    var has_initial_state_buf = has_initial_state_h
    
    # Strides for row-major layout
    var u_dim_stride: UInt32 = total_length
    var u_len_stride: UInt32 = 1
    var delta_dim_stride: UInt32 = total_length
    var delta_len_stride: UInt32 = 1
    var A_dim_stride: UInt32 = dstate
    var A_dstate_stride: UInt32 = 1
    var B_group_stride: UInt32 = dstate * total_length
    var B_dstate_stride: UInt32 = total_length
    var B_len_stride: UInt32 = 1
    var C_group_stride: UInt32 = dstate * total_length
    var C_dstate_stride: UInt32 = total_length
    var C_len_stride: UInt32 = 1
    var D_dim_stride: UInt32 = 1
    var z_dim_stride: UInt32 = total_length
    var z_len_stride: UInt32 = 1
    var delta_bias_dim_stride: UInt32 = 1
    var ssm_batch_stride: UInt32 = dim * dstate
    var ssm_dim_stride: UInt32 = dstate
    var ssm_dstate_stride: UInt32 = 1
    var out_dim_stride: UInt32 = total_length
    var out_len_stride: UInt32 = 1
    
    # Call kernel
    varlen_selective_scan_fwd_cpu[
        dtype,
        u_buf.layout,
        delta_buf.layout,
        A_buf.layout,
        B_buf.layout,
        C_buf.layout,
        D_buf.layout,
        z_buf.layout,
        delta_bias_buf.layout,
        ssm_states_buf.layout,
        output_buf.layout,
        query_start_loc_buf.layout,
        cache_indices_buf.layout,
        has_initial_state_buf.layout,
    ](
        dim,
        dstate,
        ngroups,
        batch,
        Int32(-1),  # pad_slot_id
        Int8(1) if delta_softplus else Int8(0),
        u_buf,
        delta_buf,
        A_buf,
        B_buf,
        C_buf,
        D_buf,
        z_buf,
        delta_bias_buf,
        ssm_states_buf,
        output_buf,
        query_start_loc_buf,
        cache_indices_buf,
        has_initial_state_buf,
        u_dim_stride,
        u_len_stride,
        delta_dim_stride,
        delta_len_stride,
        A_dim_stride,
        A_dstate_stride,
        B_group_stride,
        B_dstate_stride,
        B_len_stride,
        C_group_stride,
        C_dstate_stride,
        C_len_stride,
        D_dim_stride,
        z_dim_stride,
        z_len_stride,
        delta_bias_dim_stride,
        ssm_batch_stride,
        ssm_dim_stride,
        ssm_dstate_stride,
        out_dim_stride,
        out_len_stride,
    )
    
    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    var output_to_check = z_buf if has_z else output_buf
    var output_size = dim * total_length
    for i in range(output_size):
        if abs(Float32(output_to_check.ptr.offset(i).load())) > 1e-6:
            has_nonzero = True
            break
    
    if not has_nonzero:
        raise Error("Output is all zeros - kernel may not be executing correctly")
    
    # Cleanup
    u_heap.free()
    delta_heap.free()
    A_heap.free()
    B_heap.free()
    C_heap.free()
    D_heap.free()
    z_heap.free()
    delta_bias_heap.free()
    ssm_states_heap.free()
    output_heap.free()
    query_start_loc_heap.free()
    cache_indices_heap.free()
    has_initial_state_heap.free()


fn run_varlen_selective_state_update[
    dtype: DType,
    has_D: Bool = True,
    has_z: Bool = True,
    has_dt_bias: Bool = True,
    dt_softplus: Bool = False,
](
    batch: Int,
    nheads: Int,
    dim: Int,
    dstate: Int,
    ngroups: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen selective state update kernel (single-step, multi-head SSM)."""
    if dstate > MAX_DSTATE:
        return  # Skip if dstate exceeds kernel limit
    
    var nheads_ngroups_ratio = nheads // ngroups
    
    # Allocate host memory
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    # state: (batch, nheads, dim, dstate) - in/out
    var state_heap = alloc[Scalar[dtype]](batch * nheads * dim * dstate)
    var state_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        state_heap, RuntimeLayout[layout_4d].row_major(Index(batch, nheads, dim, dstate))
    ).fill(0)
    
    # output: (batch, nheads, dim)
    var output_heap = alloc[Scalar[dtype]](batch * nheads * dim)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, dim))
    ).fill(0)
    
    # x: (batch, nheads, dim)
    var x_heap = alloc[Scalar[dtype]](batch * nheads * dim)
    var x_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, dim))
    )
    
    # dt: (batch, nheads, dim)
    var dt_heap = alloc[Scalar[dtype]](batch * nheads * dim)
    var dt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        dt_heap, RuntimeLayout[layout_3d].row_major(Index(batch, nheads, dim))
    )
    
    # A: (nheads, dim, dstate)
    var A_heap = alloc[Scalar[dtype]](nheads * dim * dstate)
    var A_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_3d].row_major(Index(nheads, dim, dstate))
    )
    
    # B: (batch, ngroups, dstate)
    var B_heap = alloc[Scalar[dtype]](batch * ngroups * dstate)
    var B_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        B_heap, RuntimeLayout[layout_3d].row_major(Index(batch, ngroups, dstate))
    )
    
    # C: (batch, ngroups, dstate)
    var C_heap = alloc[Scalar[dtype]](batch * ngroups * dstate)
    var C_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_heap, RuntimeLayout[layout_3d].row_major(Index(batch, ngroups, dstate))
    )
    
    # D: (nheads, dim) or empty
    var D_size = nheads * dim if has_D else 0
    var D_heap = alloc[Scalar[dtype]](max(D_size, 1))
    var D_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        D_heap, RuntimeLayout[layout_2d].row_major(Index(nheads if has_D else 0, dim if has_D else 0))
    )
    
    # z: (batch, nheads, dim) or empty
    var z_size = batch * nheads * dim if has_z else 0
    var z_heap = alloc[Scalar[dtype]](max(z_size, 1))
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap, RuntimeLayout[layout_3d].row_major(Index(batch if has_z else 0, nheads if has_z else 0, dim if has_z else 0))
    )
    
    # dt_bias: (nheads, dim) or empty
    var dt_bias_size = nheads * dim if has_dt_bias else 0
    var dt_bias_heap = alloc[Scalar[dtype]](max(dt_bias_size, 1))
    var dt_bias_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        dt_bias_heap, RuntimeLayout[layout_2d].row_major(Index(nheads if has_dt_bias else 0, dim if has_dt_bias else 0))
    )
    
    # state_batch_indices: (batch,) - can be empty or identity
    var state_batch_indices_heap = alloc[Scalar[DType.int32]](batch)
    var state_batch_indices_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        state_batch_indices_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        state_batch_indices_h.ptr.offset(i).store(Scalar[DType.int32](i))
    
    # Initialize input data
    random(x_h)
    random(dt_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_dt_bias:
        random(dt_bias_h)
    
    # Scale A to be negative for stability
    for i in range(nheads * dim * dstate):
        var val = A_h.ptr.offset(i).load()
        A_h.ptr.offset(i).store(Scalar[dtype](Float32(val) * -0.5))
    
    # Scale dt to be positive
    for i in range(batch * nheads * dim):
        var val = dt_h.ptr.offset(i).load()
        dt_h.ptr.offset(i).store(Scalar[dtype](abs(Float32(val)) * 0.5))
    
    var state_buf = state_h
    var output_buf = output_h
    var x_buf = x_h
    var dt_buf = dt_h
    var A_buf = A_h
    var B_buf = B_h
    var C_buf = C_h
    var D_buf = D_h
    var z_buf = z_h
    var dt_bias_buf = dt_bias_h
    var state_batch_indices_buf = state_batch_indices_h
    
    # Strides for row-major layout
    var state_batch_stride: UInt32 = nheads * dim * dstate
    var state_head_stride: UInt32 = dim * dstate
    var state_dim_stride: UInt32 = dstate
    var state_dstate_stride: UInt32 = 1
    var x_batch_stride: UInt32 = nheads * dim
    var x_head_stride: UInt32 = dim
    var x_dim_stride: UInt32 = 1
    var dt_batch_stride: UInt32 = nheads * dim
    var dt_head_stride: UInt32 = dim
    var dt_dim_stride: UInt32 = 1
    var dt_bias_head_stride: UInt32 = dim
    var dt_bias_dim_stride: UInt32 = 1
    var A_head_stride: UInt32 = dim * dstate
    var A_dim_stride: UInt32 = dstate
    var A_dstate_stride: UInt32 = 1
    var B_batch_stride: UInt32 = ngroups * dstate
    var B_group_stride: UInt32 = dstate
    var B_dstate_stride: UInt32 = 1
    var C_batch_stride: UInt32 = ngroups * dstate
    var C_group_stride: UInt32 = dstate
    var C_dstate_stride: UInt32 = 1
    var D_head_stride: UInt32 = dim
    var D_dim_stride: UInt32 = 1
    var z_batch_stride: UInt32 = nheads * dim
    var z_head_stride: UInt32 = dim
    var z_dim_stride: UInt32 = 1
    var out_batch_stride: UInt32 = nheads * dim
    var out_head_stride: UInt32 = dim
    var out_dim_stride: UInt32 = 1
    
    # Call kernel
    varlen_selective_state_update_cpu[
        dtype,
        state_buf.layout,
        x_buf.layout,
        dt_buf.layout,
        A_buf.layout,
        B_buf.layout,
        C_buf.layout,
        D_buf.layout,
        z_buf.layout,
        output_buf.layout,
        dt_bias_buf.layout,
        state_batch_indices_buf.layout,
    ](
        batch,
        nheads,
        dim,
        dstate,
        nheads_ngroups_ratio,
        Int32(-1),  # pad_slot_id
        Int8(1) if dt_softplus else Int8(0),
        Int8(1),  # has_state_batch_indices
        state_buf,
        x_buf,
        dt_buf,
        A_buf,
        B_buf,
        C_buf,
        D_buf,
        z_buf,
        output_buf,
        dt_bias_buf,
        state_batch_indices_buf,
        state_batch_stride,
        state_head_stride,
        state_dim_stride,
        state_dstate_stride,
        x_batch_stride,
        x_head_stride,
        x_dim_stride,
        dt_batch_stride,
        dt_head_stride,
        dt_dim_stride,
        dt_bias_head_stride,
        dt_bias_dim_stride,
        A_head_stride,
        A_dim_stride,
        A_dstate_stride,
        B_batch_stride,
        B_group_stride,
        B_dstate_stride,
        C_batch_stride,
        C_group_stride,
        C_dstate_stride,
        D_head_stride,
        D_dim_stride,
        z_batch_stride,
        z_head_stride,
        z_dim_stride,
        out_batch_stride,
        out_head_stride,
        out_dim_stride,
    )
    
    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    for i in range(batch * nheads * dim):
        if abs(Float32(output_h.ptr.offset(i).load())) > 1e-6:
            has_nonzero = True
            break
    
    if not has_nonzero:
        raise Error("Output is all zeros - kernel may not be executing correctly")
    
    # Cleanup
    state_heap.free()
    output_heap.free()
    x_heap.free()
    dt_heap.free()
    A_heap.free()
    B_heap.free()
    C_heap.free()
    D_heap.free()
    z_heap.free()
    dt_bias_heap.free()
    state_batch_indices_heap.free()


def main():
    # Test varlen_selective_scan_fwd with equal-length sequences
    run_varlen_selective_scan_fwd[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, dstate=4, ngroups=1, seq_lengths=Index(8, 8)
    )
    print("✓ Varlen selective scan fwd (equal lengths) test passed")
    
    # Test varlen_selective_scan_fwd with variable-length sequences
    run_varlen_selective_scan_fwd[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=3, dim=4, dstate=4, ngroups=1, seq_lengths=Index(10, 6, 1)
    )
    print("✓ Varlen selective scan fwd (variable lengths) test passed")
    
    # Test without D
    run_varlen_selective_scan_fwd[DType.float32, has_D=False, has_z=True, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, dstate=4, ngroups=1, seq_lengths=Index(8, 8)
    )
    print("✓ Varlen selective scan fwd without D test passed")
    
    # Test without z
    run_varlen_selective_scan_fwd[DType.float32, has_D=True, has_z=False, has_delta_bias=True, delta_softplus=False](
        batch=2, dim=4, dstate=4, ngroups=1, seq_lengths=Index(8, 8)
    )
    print("✓ Varlen selective scan fwd without z test passed")
    
    # Test with delta_softplus
    run_varlen_selective_scan_fwd[DType.float32, has_D=True, has_z=True, has_delta_bias=True, delta_softplus=True](
        batch=2, dim=4, dstate=4, ngroups=1, seq_lengths=Index(8, 8)
    )
    print("✓ Varlen selective scan fwd with delta_softplus test passed")
    
    # Test varlen_selective_state_update
    run_varlen_selective_state_update[DType.float32, has_D=True, has_z=True, has_dt_bias=True, dt_softplus=False](
        batch=2, nheads=2, dim=4, dstate=4, ngroups=1
    )
    print("✓ Varlen selective state update test passed")
    
    # Test state update without D
    run_varlen_selective_state_update[DType.float32, has_D=False, has_z=True, has_dt_bias=True, dt_softplus=False](
        batch=2, nheads=2, dim=4, dstate=4, ngroups=1
    )
    print("✓ Varlen selective state update without D test passed")
    
    # Test state update without z
    run_varlen_selective_state_update[DType.float32, has_D=True, has_z=False, has_dt_bias=True, dt_softplus=False](
        batch=2, nheads=2, dim=4, dstate=4, ngroups=1
    )
    print("✓ Varlen selective state update without z test passed")
    
    # Test state update with dt_softplus
    run_varlen_selective_state_update[DType.float32, has_D=True, has_z=True, has_dt_bias=True, dt_softplus=True](
        batch=2, nheads=2, dim=4, dstate=4, ngroups=1
    )
    print("✓ Varlen selective state update with dt_softplus test passed")

