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

from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from layout._fillers import random
from math import exp, exp2, log
from memory import alloc
from state_space.selective_scan import (
    ssd_combined_cpu,
)
from testing import TestSuite, assert_almost_equal

from utils.index import Index, IndexList

comptime MAX_DSTATE = 16
comptime LOG2E = 1.4426950408889634


@always_inline
fn softplus_ref(val: Float32) -> Float32:
    """Reference softplus: log(1 + exp(x)) with numerical stability."""
    if val > 20.0:
        return val
    return log(Float32(1.0) + exp(val))


@always_inline
fn silu_ref(val: Float32) -> Float32:
    """Reference SiLU: x * sigmoid(x) = x / (1 + exp(-x))."""
    if val < -20.0:
        return 0.0
    return val / (Float32(1.0) + exp(-val))


fn run_ssd_combined[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    n_groups: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test SSD combined kernel against reference implementation."""
    comptime assert DSTATE <= MAX_DSTATE, "DSTATE exceeds kernel limit"
    comptime dstate = DSTATE

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
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    # x: (batch, dim, num_chunks, 2*dstate) - checkpoint tensor
    var x_heap = alloc[Scalar[dtype]](batch * dim * n_chunks * 2 * dstate)
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, dim, n_chunks, 2 * dstate)
        ),
    ).fill(0)

    # out_z: (batch, dim, seqlen) - gated output
    var out_z_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    # residual: (batch, dim, seqlen)
    var residual_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var residual_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        residual_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # u: (batch, dim, seqlen)
    var u_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var u_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        u_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )

    # delta: (batch, dim, seqlen)
    var delta_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var delta_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        delta_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # A: (dim, dstate)
    var A_heap = alloc[Scalar[dtype]](dim * dstate)
    var A_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )

    # B: (batch, n_groups, dstate, seqlen)
    var B_heap = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )

    # C: (batch, n_groups, dstate, seqlen)
    var C_heap = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
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
        z_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )

    # delta_bias: (dim,) - optional
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var delta_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(delta_bias_size)),
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

    # Call kernel
    ssd_combined_cpu[
        dtype,
        DSTATE,
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
    )

    # Reference implementation for numerical verification
    var ref_size = batch * dim * seqlen
    var output_ref_heap = alloc[Scalar[dtype]](ref_size)
    var out_z_ref_heap = alloc[Scalar[dtype]](ref_size)
    for i in range(ref_size):
        output_ref_heap[i] = Scalar[dtype](0)
        out_z_ref_heap[i] = Scalar[dtype](0)

    for b_idx in range(batch):
        for d_idx in range(dim):
            var group_id = d_idx // group_size

            # Pre-load A values with LOG2E scaling (matches kernel)
            var A_ref = SIMD[DType.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                A_ref[n] = Float32(A_h.ptr[d_idx * dstate + n]) * LOG2E

            # Load per-dim scalars
            var gamma_val = Float32(gamma_h.ptr[d_idx])
            var D_val = Float32(0)
            if has_D:
                D_val = Float32(D_h.ptr[d_idx])
            var delta_bias_val = Float32(0)
            if has_delta_bias:
                delta_bias_val = Float32(delta_bias_h.ptr[d_idx])
            var weight_offset_val = Float32(weight_offset)

            # Initialize state to zero
            var state_ref = SIMD[DType.float32, MAX_DSTATE](0.0)

            for t in range(seqlen):
                var off_3d = b_idx * dim * seqlen + d_idx * seqlen + t
                var u_val = Float32(u_h.ptr[off_3d])
                var delta_val = Float32(delta_h.ptr[off_3d])
                var residual_val = Float32(residual_h.ptr[off_3d])

                if has_delta_bias:
                    delta_val += delta_bias_val
                if delta_softplus:
                    delta_val = softplus_ref(delta_val)

                var delta_u = delta_val * u_val

                # Load B, C values
                var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                for n in range(dstate):
                    var bc_offset = (
                        b_idx * n_groups * dstate * seqlen
                        + group_id * dstate * seqlen
                        + n * seqlen
                        + t
                    )
                    B_vals[n] = Float32(B_h.ptr[bc_offset])
                    C_vals[n] = Float32(C_h.ptr[bc_offset])

                # State update: state = state * exp2(A * delta) + B * delta * u
                var a_t = exp2(A_ref * delta_val)
                var b_t = B_vals * delta_u
                state_ref = state_ref * a_t + b_t

                # Compute selective scan output
                var ss_output = (state_ref * C_vals).reduce_add()

                if has_D:
                    ss_output += D_val * u_val

                # Combine with residual and apply gamma scaling
                var combined = residual_val + ss_output
                var normalized = combined * (gamma_val + weight_offset_val)

                if has_z:
                    var z_val = Float32(z_h.ptr[off_3d])
                    var out_z_val = normalized * silu_ref(z_val)
                    out_z_ref_heap[off_3d] = Scalar[dtype](out_z_val)
                    normalized = out_z_val

                output_ref_heap[off_3d] = Scalar[dtype](normalized)

    # Compare kernel output vs reference
    for i in range(ref_size):
        assert_almost_equal(
            output_h.ptr[i],
            output_ref_heap[i],
            rtol=rtol,
        )

    # Verify out_z when z gating is enabled
    if has_z:
        for i in range(ref_size):
            assert_almost_equal(
                out_z_h.ptr[i],
                out_z_ref_heap[i],
                rtol=rtol,
            )

    output_ref_heap.free()
    out_z_ref_heap.free()

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


fn test_ssd_combined_basic() raises:
    """Test basic ssd_combined."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


fn test_ssd_combined_without_D() raises:
    """Test ssd_combined without D."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


fn test_ssd_combined_without_z() raises:
    """Test ssd_combined without z."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


fn test_ssd_combined_without_delta_bias() raises:
    """Test ssd_combined without delta_bias."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=False,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


fn test_ssd_combined_with_delta_softplus() raises:
    """Test ssd_combined with delta_softplus."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


fn test_ssd_combined_larger_shapes() raises:
    """Test ssd_combined with larger shapes."""
    run_ssd_combined[
        DType.float32,
        8,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=4, dim=8, seqlen=16, n_groups=1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
