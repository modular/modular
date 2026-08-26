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
"""Op-registration smoke test for the `ssd_combined` MAX graph op.

Numerical parity is covered by `test_ssd_combined.mojo`; this file exists to
ensure the @extensibility.register wrapper in `selective_scan_ops.mojo` is built
into the state_space library (a malformed @extensibility.register block fails the
library build), and that the underlying kernel is reachable via the same
import path the wrapper uses.
"""

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from state_space.selective_scan import ssd_combined_cpu

# Importing the ops module ensures the @extensibility.register-decorated structs
# are part of the build graph for the consumer of the smoke test.
import state_space.selective_scan_ops

from std.testing import TestSuite, assert_true
from std.utils.index import Index

comptime MAX_DSTATE_LOCAL = 16


def test_ssd_combined_op_registered() raises:
    """Smoke test: ssd_combined op + underlying CPU kernel are callable.

    Constructs tiny LayoutTensor inputs, invokes the CPU implementation that
    the registered op dispatches to, and verifies the output buffer is finite
    (no NaN/Inf) and shape-correct. Full numerical parity lives elsewhere.
    """
    comptime dtype = DType.float32
    comptime DSTATE = 8
    var batch = 1
    var dim = 4
    var seqlen = 4
    var n_groups = 1
    var group_size = dim // n_groups
    var chunk_size = 2048
    var n_chunks = (seqlen + chunk_size - 1) // chunk_size

    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var output_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var output_tt = TileTensor(
        output_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var x_heap = List(
        length=batch * dim * n_chunks * 2 * DSTATE, fill=Scalar[dtype](0)
    )
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, dim, n_chunks, 2 * DSTATE)
        ),
    )
    var x_tt = TileTensor(
        x_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, n_chunks, 2 * DSTATE),
    )

    var out_z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var out_z_tt = TileTensor(
        out_z_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var residual_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var residual_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        residual_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var residual_tt = TileTensor(
        residual_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var u_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var u_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        u_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var u_tt = TileTensor(
        u_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var delta_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var delta_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        delta_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var delta_tt = TileTensor(
        delta_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var A_heap = List(length=dim * DSTATE, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, DSTATE))
    )
    var A_tt = TileTensor(
        A_heap.unsafe_ptr().as_unsafe_any_origin(), row_major(dim, DSTATE)
    )

    var B_heap = List(
        length=batch * n_groups * DSTATE * seqlen, fill=Scalar[dtype](0)
    )
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, DSTATE, seqlen)
        ),
    )
    var B_tt = TileTensor(
        B_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, n_groups, DSTATE, seqlen),
    )

    var C_heap = List(
        length=batch * n_groups * DSTATE * seqlen, fill=Scalar[dtype](0)
    )
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, DSTATE, seqlen)
        ),
    )
    var C_tt = TileTensor(
        C_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, n_groups, DSTATE, seqlen),
    )

    var D_heap = List(length=dim, fill=Scalar[dtype](0))
    var D_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        D_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    var D_tt = TileTensor(
        D_heap.unsafe_ptr().as_unsafe_any_origin(), row_major(dim)
    )

    var z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var z_tt = TileTensor(
        z_heap.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, dim, seqlen),
    )

    var delta_bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var delta_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        delta_bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    var delta_bias_tt = TileTensor(
        delta_bias_heap.unsafe_ptr().as_unsafe_any_origin(), row_major(dim)
    )

    var gamma_heap = List(length=dim, fill=Scalar[dtype](0))
    var gamma_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    var gamma_tt = TileTensor(
        gamma_heap.unsafe_ptr().as_unsafe_any_origin(), row_major(dim)
    )

    random(u_h)
    random(delta_h)
    random(residual_h)
    random(A_h)
    random(B_h)
    random(C_h)
    random(D_h)
    random(z_h)
    random(delta_bias_h)
    random(gamma_h)

    # Keep gamma positive for RMSNorm stability.
    for i in range(dim):
        gamma_h.ptr[i] = abs(gamma_h.ptr[i]) + Scalar[dtype](0.1)

    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ssd_combined_cpu[
        dtype,
        DSTATE,
        output_tt.LayoutType,
        x_tt.LayoutType,
        out_z_tt.LayoutType,
        residual_tt.LayoutType,
        u_tt.LayoutType,
        delta_tt.LayoutType,
        A_tt.LayoutType,
        B_tt.LayoutType,
        C_tt.LayoutType,
        D_tt.LayoutType,
        z_tt.LayoutType,
        delta_bias_tt.LayoutType,
        gamma_tt.LayoutType,
    ](
        batch,
        dim,
        seqlen,
        group_size,
        Int8(0),
        output_tt,
        x_tt,
        out_z_tt,
        residual_tt,
        u_tt,
        delta_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        z_tt,
        delta_bias_tt,
        gamma_tt,
        epsilon,
        weight_offset,
    )

    # Shape check + finite check.
    assert_true(output_h.dim(0) == batch)
    assert_true(output_h.dim(1) == dim)
    assert_true(output_h.dim(2) == seqlen)

    var out_size = batch * dim * seqlen
    for i in range(out_size):
        var v = Float32(output_h.ptr[i])
        assert_true(v == v, "output contains NaN")
        # Inf check: an infinite float fails (v - v) == 0.
        assert_true((v - v) == 0.0, "output contains Inf")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
