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

from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import random
from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_cpu
from std.math import exp
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def run_ssd_chunk_scan[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test the SSD inter-chunk scan kernel against an independent reference."""
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cs_count = batch * n_chunks * n_heads * head_dim * state_dim
    var cd_count = batch * n_chunks * n_heads
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var final_count = batch * n_heads * head_dim * state_dim

    # chunk_states: [batch, n_chunks, n_heads, head_dim, state_dim]
    var cs_heap = List(length=cs_count, fill=Scalar[dtype](0))
    var cs_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        cs_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    # chunk_decays: [batch, n_chunks, n_heads]
    var cd_heap = List(length=cd_count, fill=Scalar[dtype](0))
    var cd_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        cd_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, n_chunks, n_heads)),
    )

    # entering: [batch, n_chunks, n_heads, head_dim, state_dim]
    var ent_heap = List(length=ent_count, fill=Scalar[dtype](0))
    var ent_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    # final: [batch, n_heads, head_dim, state_dim]
    var final_heap = List(length=final_count, fill=Scalar[dtype](0))
    var final_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        final_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_heads, head_dim, state_dim)
        ),
    )

    # Initialize inputs.
    random(cs_h)
    random(cd_h)

    # Make chunk_decays negative (a real SSM decays). `random` fills in roughly
    # [-1, 1]; map it into [-1.2, -0.3] like a real per-chunk total decay.
    for i in range(cd_count):
        var v = cd_h.ptr[i].cast[DType.float32]()
        # Map [-1, 1] -> [-1.2, -0.3].
        var scaled = Float32(-0.75) + Float32(0.45) * v
        cd_h.ptr[i] = Scalar[dtype](scaled)

    # Call the kernel.
    ssd_chunk_scan_fwd_cpu[
        dtype,
        cs_h.layout,
        cd_h.layout,
        ent_h.layout,
        final_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        head_dim,
        state_dim,
        cs_h,
        cd_h,
        ent_h,
        final_h,
    )

    # Independent naive reference. Recompute strides from scratch and run the
    # sequential recurrence per (batch, head):
    #   state init 0
    #   entering[c] = state
    #   state = chunk_states[c] + exp(chunk_decays[c]) * state
    #   final = state after the last chunk
    var ref_ent = List(length=ent_count, fill=Scalar[dtype](0))
    var ref_final = List(length=final_count, fill=Scalar[dtype](0))

    var cs_h_stride = head_dim * state_dim
    var cs_c_stride = n_heads * cs_h_stride
    var cs_b_stride = n_chunks * cs_c_stride

    var cd_c_stride = n_heads
    var cd_b_stride = n_chunks * cd_c_stride

    var ent_h_stride = head_dim * state_dim
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    var final_h_stride = head_dim * state_dim
    var final_b_stride = n_heads * final_h_stride

    for b in range(batch):
        for h in range(n_heads):
            var state = List[Float32](length=head_dim * state_dim, fill=0.0)
            for c in range(n_chunks):
                var ent_base = (
                    b * ent_b_stride + c * ent_c_stride + h * ent_h_stride
                )
                for pn in range(head_dim * state_dim):
                    ref_ent[ent_base + pn] = Scalar[dtype](state[pn])

                var cd_off = b * cd_b_stride + c * cd_c_stride + h
                var decay = exp(cd_h.ptr[cd_off].cast[DType.float32]())

                var cs_base = (
                    b * cs_b_stride + c * cs_c_stride + h * cs_h_stride
                )
                for pn in range(head_dim * state_dim):
                    var cs_val = cs_h.ptr[cs_base + pn].cast[DType.float32]()
                    state[pn] = cs_val + decay * state[pn]

            var final_base = b * final_b_stride + h * final_h_stride
            for pn in range(head_dim * state_dim):
                ref_final[final_base + pn] = Scalar[dtype](state[pn])

    # Compare kernel output against the reference.
    for i in range(ent_count):
        assert_almost_equal(ent_h.ptr[i], ref_ent[i], rtol=rtol)
    for i in range(final_count):
        assert_almost_equal(final_h.ptr[i], ref_final[i], rtol=rtol)


def test_ssd_chunk_scan_golden() raises:
    """Golden-parity test against the canonical PyTorch SSD reference.

    Pins ``ssd_chunk_scan_fwd_cpu`` to a FIXED tiny input whose expected output
    was computed independently by the inter-chunk recurrence (stage 3 of
    ``ssd_minimal_discrete``).

    Shapes: batch=1, n_chunks=3, n_heads=1, head_dim P=2, state_dim N=3;
    all tensors row-major.
    """
    comptime dtype = DType.float32
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    comptime batch = 1
    comptime n_chunks = 3
    comptime n_heads = 1
    comptime head_dim = 2
    comptime state_dim = 3

    # Fixed literals (row-major). chunk_states is C x P x N; chunk_decays is C.
    # entering_expected / final_expected are the reference's outputs.
    var cs_vals = [
        Float32(-0.90),
        0.57,
        0.91,
        0.43,
        1.29,
        -0.17,
        -0.86,
        -0.66,
        -0.70,
        -0.10,
        -0.62,
        -0.63,
        -0.58,
        0.59,
        0.12,
        0.09,
        1.04,
        -0.18,
    ]
    var cd_vals = [Float32(-0.61), -1.03, -0.79]
    var entering_expected = [
        Float32(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.90,
        0.57,
        0.91,
        0.43,
        1.29,
        -0.17,
        -1.181306,
        -0.456506,
        -0.375124,
        0.053513,
        -0.159461,
        -0.690691,
    ]
    var final_expected = [
        Float32(-1.116130),
        0.382817,
        -0.050248,
        0.114287,
        0.967629,
        -0.493467,
    ]

    var cs_count = batch * n_chunks * n_heads * head_dim * state_dim
    var cd_count = batch * n_chunks * n_heads
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var final_count = batch * n_heads * head_dim * state_dim

    # Build LayoutTensors from the fixed literals (not random).
    var cs_heap = List(length=cs_count, fill=Scalar[dtype](0))
    var cs_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        cs_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    for i in range(cs_count):
        cs_h.ptr[i] = Scalar[dtype](cs_vals[i])

    var cd_heap = List(length=cd_count, fill=Scalar[dtype](0))
    var cd_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        cd_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, n_chunks, n_heads)),
    )
    for i in range(cd_count):
        cd_h.ptr[i] = Scalar[dtype](cd_vals[i])

    var ent_heap = List(length=ent_count, fill=Scalar[dtype](0))
    var ent_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    var final_heap = List(length=final_count, fill=Scalar[dtype](0))
    var final_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        final_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_heads, head_dim, state_dim)
        ),
    )

    ssd_chunk_scan_fwd_cpu[
        dtype,
        cs_h.layout,
        cd_h.layout,
        ent_h.layout,
        final_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        head_dim,
        state_dim,
        cs_h,
        cd_h,
        ent_h,
        final_h,
    )

    # Pin to the canonical PyTorch reference numerics.
    for i in range(ent_count):
        assert_almost_equal(
            ent_h.ptr[i],
            Scalar[dtype](entering_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )
    for i in range(final_count):
        assert_almost_equal(
            final_h.ptr[i],
            Scalar[dtype](final_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_chunk_scan_basic() raises:
    """Basic single-batch, single-head case against the naive reference."""
    run_ssd_chunk_scan[DType.float32](
        batch=1,
        n_chunks=4,
        n_heads=2,
        head_dim=8,
        state_dim=16,
    )


def test_ssd_chunk_scan_multi_batch() raises:
    """Multiple batch elements and heads against the naive reference."""
    run_ssd_chunk_scan[DType.float32](
        batch=2,
        n_chunks=5,
        n_heads=3,
        head_dim=4,
        state_dim=8,
    )


def test_ssd_chunk_scan_single_chunk() raises:
    """Degenerate single chunk (entering is all zeros, final == chunk_states).
    """
    run_ssd_chunk_scan[DType.float32](
        batch=2,
        n_chunks=1,
        n_heads=2,
        head_dim=4,
        state_dim=4,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
