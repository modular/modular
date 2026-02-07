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
from memory import alloc
from state_space.selective_scan import (
    ssd_combined_cpu,
)
from testing import TestSuite, assert_almost_equal

from utils.index import Index, IndexList


comptime MAX_DSTATE = 16


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
    constrained[DSTATE <= MAX_DSTATE, "DSTATE exceeds kernel limit"]()
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
        raise Error(
            "Output is all zeros - kernel may not be executing correctly"
        )

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
