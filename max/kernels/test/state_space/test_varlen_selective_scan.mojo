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
from state_space.varlen_selective_scan import (
    varlen_selective_scan_fwd_cpu,
    varlen_selective_state_update_cpu,
)
from testing import TestSuite, assert_almost_equal

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
    rtol: Float64 = 0.01,
) raises:
    """Test varlen selective scan forward kernel.

    Args:
        batch: Number of sequences.
        dim: Hidden dimension.
        ngroups: Number of groups.
        seq_lengths: List of sequence lengths for each batch item.
        rtol: Relative tolerance for numerical comparisons.
    """
    comptime dstate = DSTATE
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
        B_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(ngroups, dstate, total_length)
        ),
    )

    # C: (ngroups, dstate, total_length)
    var C_heap = alloc[Scalar[dtype]](ngroups * dstate * total_length)
    var C_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(ngroups, dstate, total_length)
        ),
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
        z_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(dim if has_z else 0, total_length if has_z else 0)
        ),
    )

    # delta_bias: (dim,) or empty
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var delta_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(delta_bias_size)),
    )

    # ssm_states: (batch, dim, dstate) - in/out
    var ssm_states_heap = alloc[Scalar[dtype]](batch * dim * dstate)
    var ssm_states_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        ssm_states_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    ).fill(0)

    # output: (dim, total_length) - same as delta
    var output_heap = alloc[Scalar[dtype]](dim * total_length)
    var output_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_length)),
    ).fill(0)

    # query_start_loc: (batch + 1,) - cumulative sequence lengths
    var query_start_loc_heap = alloc[Scalar[DType.int32]](batch + 1)
    var query_start_loc_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        query_start_loc_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch + 1)),
    )
    var cumsum = 0
    query_start_loc_h.ptr.store(0, Scalar[DType.int32](0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_h.ptr.store(i + 1, Scalar[DType.int32](cumsum))

    # cache_indices: (batch,) - can be empty or identity mapping
    var cache_indices_heap = alloc[Scalar[DType.int32]](batch)
    var cache_indices_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        cache_indices_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        cache_indices_h.ptr.store(i, Scalar[DType.int32](i))

    # has_initial_state: (batch,) - can be empty or all False
    var has_initial_state_heap = alloc[Scalar[DType.bool]](batch)
    var has_initial_state_h = LayoutTensor[DType.bool, layout_1d, MutAnyOrigin](
        has_initial_state_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        has_initial_state_h.ptr.store(i, Scalar[DType.bool](False))

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
        var val = A_h.ptr.load(i)
        A_h.ptr.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Scale delta to be positive
    for i in range(dim * total_length):
        var val = delta_h.ptr.load(i)
        delta_h.ptr.store(i, Scalar[dtype](abs(Float32(val)) * 0.5))

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

    # Strides for row-major layout using IndexList types
    var u_strides = IndexList[2](total_length, 1)
    var delta_strides = IndexList[2](total_length, 1)
    var A_strides = IndexList[2](dstate, 1)
    var B_strides = IndexList[3](dstate * total_length, total_length, 1)
    var C_strides = IndexList[3](dstate * total_length, total_length, 1)
    var D_strides = IndexList[1](1)
    var z_strides = IndexList[2](total_length, 1)
    var delta_bias_strides = IndexList[1](1)
    var ssm_states_strides = IndexList[3](dim * dstate, dstate, 1)
    var out_strides = IndexList[2](total_length, 1)

    # Call kernel
    varlen_selective_scan_fwd_cpu[
        dtype,
        DSTATE,
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
        u_strides,
        delta_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        z_strides,
        delta_bias_strides,
        ssm_states_strides,
        out_strides,
    )

    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    var output_to_check = z_buf if has_z else output_buf
    var output_size = dim * total_length
    for i in range(output_size):
        if abs(Float32(output_to_check.ptr.load(i))) > 1e-6:
            has_nonzero = True
            break

    if not has_nonzero:
        raise Error(
            "Output is all zeros - kernel may not be executing correctly"
        )

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
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_dt_bias: Bool = True,
    dt_softplus: Bool = False,
](
    batch: Int,
    nheads: Int,
    dim: Int,
    ngroups: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen selective state update kernel (single-step, multi-head SSM).
    """
    comptime dstate = DSTATE
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
        state_heap,
        RuntimeLayout[layout_4d].row_major(Index(batch, nheads, dim, dstate)),
    ).fill(0)

    # output: (batch, nheads, dim)
    var output_heap = alloc[Scalar[dtype]](batch * nheads * dim)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, nheads, dim)),
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
        B_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, ngroups, dstate)),
    )

    # C: (batch, ngroups, dstate)
    var C_heap = alloc[Scalar[dtype]](batch * ngroups * dstate)
    var C_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, ngroups, dstate)),
    )

    # D: (nheads, dim) or empty
    var D_size = nheads * dim if has_D else 0
    var D_heap = alloc[Scalar[dtype]](max(D_size, 1))
    var D_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        D_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(nheads if has_D else 0, dim if has_D else 0)
        ),
    )

    # z: (batch, nheads, dim) or empty
    var z_size = batch * nheads * dim if has_z else 0
    var z_heap = alloc[Scalar[dtype]](max(z_size, 1))
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(
                batch if has_z else 0,
                nheads if has_z else 0,
                dim if has_z else 0,
            )
        ),
    )

    # dt_bias: (nheads, dim) or empty
    var dt_bias_size = nheads * dim if has_dt_bias else 0
    var dt_bias_heap = alloc[Scalar[dtype]](max(dt_bias_size, 1))
    var dt_bias_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        dt_bias_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(nheads if has_dt_bias else 0, dim if has_dt_bias else 0)
        ),
    )

    # state_batch_indices: (batch,) - can be empty or identity
    var state_batch_indices_heap = alloc[Scalar[DType.int32]](batch)
    var state_batch_indices_h = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](
        state_batch_indices_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch)),
    )
    for i in range(batch):
        state_batch_indices_h.ptr.store(i, Scalar[DType.int32](i))

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
        var val = A_h.ptr.load(i)
        A_h.ptr.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Scale dt to be positive
    for i in range(batch * nheads * dim):
        var val = dt_h.ptr.load(i)
        dt_h.ptr.store(i, Scalar[dtype](abs(Float32(val)) * 0.5))

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

    # Strides for row-major layout using IndexList types
    var state_strides = IndexList[4](
        nheads * dim * dstate, dim * dstate, dstate, 1
    )
    var x_strides = IndexList[3](nheads * dim, dim, 1)
    var dt_strides = IndexList[3](nheads * dim, dim, 1)
    var dt_bias_strides = IndexList[2](dim, 1)
    var A_strides = IndexList[3](dim * dstate, dstate, 1)
    var B_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var C_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var D_strides = IndexList[2](dim, 1)
    var z_strides = IndexList[3](nheads * dim, dim, 1)
    var out_strides = IndexList[3](nheads * dim, dim, 1)

    # Call kernel
    varlen_selective_state_update_cpu[
        dtype,
        DSTATE,
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
        state_strides,
        x_strides,
        dt_strides,
        dt_bias_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        z_strides,
        out_strides,
    )

    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    for i in range(batch * nheads * dim):
        if abs(Float32(output_h.ptr.load(i))) > 1e-6:
            has_nonzero = True
            break

    if not has_nonzero:
        raise Error(
            "Output is all zeros - kernel may not be executing correctly"
        )

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


# =============================================================================
# Test functions for varlen selective scan forward
# =============================================================================


fn test_varlen_selective_scan_fwd_equal_lengths() raises:
    """Test varlen selective scan forward with equal-length sequences."""
    run_varlen_selective_scan_fwd[
        DType.float32,
        4,
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8))


fn test_varlen_selective_scan_fwd_variable_lengths() raises:
    """Test varlen selective scan forward with variable-length sequences."""
    run_varlen_selective_scan_fwd[
        DType.float32,
        4,
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=3, dim=4, ngroups=1, seq_lengths=Index(10, 6, 1))


fn test_varlen_selective_scan_fwd_without_D() raises:
    """Test varlen selective scan forward without D tensor."""
    run_varlen_selective_scan_fwd[
        DType.float32,
        4,
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8))


fn test_varlen_selective_scan_fwd_without_z() raises:
    """Test varlen selective scan forward without z tensor."""
    run_varlen_selective_scan_fwd[
        DType.float32,
        4,
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8))


fn test_varlen_selective_scan_fwd_with_delta_softplus() raises:
    """Test varlen selective scan forward with delta softplus activation."""
    run_varlen_selective_scan_fwd[
        DType.float32,
        4,
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=2, dim=4, ngroups=1, seq_lengths=Index(8, 8))


# =============================================================================
# Test functions for varlen selective state update
# =============================================================================


fn test_varlen_selective_state_update_basic() raises:
    """Test basic varlen selective state update."""
    run_varlen_selective_state_update[
        DType.float32,
        4,
        has_D=True,
        has_z=True,
        has_dt_bias=True,
        dt_softplus=False,
    ](batch=2, nheads=2, dim=4, ngroups=1)


fn test_varlen_selective_state_update_without_D() raises:
    """Test varlen selective state update without D tensor."""
    run_varlen_selective_state_update[
        DType.float32,
        4,
        has_D=False,
        has_z=True,
        has_dt_bias=True,
        dt_softplus=False,
    ](batch=2, nheads=2, dim=4, ngroups=1)


fn test_varlen_selective_state_update_without_z() raises:
    """Test varlen selective state update without z tensor."""
    run_varlen_selective_state_update[
        DType.float32,
        4,
        has_D=True,
        has_z=False,
        has_dt_bias=True,
        dt_softplus=False,
    ](batch=2, nheads=2, dim=4, ngroups=1)


fn test_varlen_selective_state_update_with_dt_softplus() raises:
    """Test varlen selective state update with dt softplus activation."""
    run_varlen_selective_state_update[
        DType.float32,
        4,
        has_D=True,
        has_z=True,
        has_dt_bias=True,
        dt_softplus=True,
    ](batch=2, nheads=2, dim=4, ngroups=1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
