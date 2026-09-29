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
from state_space.ssd_chunk_combine import ssd_output_recombination_fwd_cpu
from std.math import exp
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def run_ssd_output_recombination[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test the SSD output-recombination kernel against an independent ref."""
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    # C: [batch, n_chunks, n_heads, chunk_len, state_dim]
    var c_heap = List(length=c_count, fill=Scalar[dtype](0))
    var c_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        c_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )

    # entering_state: [batch, n_chunks, n_heads, head_dim, state_dim]
    var ent_heap = List(length=ent_count, fill=Scalar[dtype](0))
    var ent_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    # A: [batch, n_chunks, n_heads, chunk_len]
    var a_heap = List(length=a_count, fill=Scalar[dtype](0))
    var a_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        a_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )

    # Y_diag: [batch, n_chunks, n_heads, chunk_len, head_dim]
    var yd_heap = List(length=y_count, fill=Scalar[dtype](0))
    var yd_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        yd_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # Y: [batch, n_chunks, n_heads, chunk_len, head_dim]
    var y_heap = List(length=y_count, fill=Scalar[dtype](0))
    var y_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        y_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # Initialize inputs.
    random(c_h)
    random(ent_h)
    random(a_h)
    random(yd_h)

    # Make A negative (a real SSM decays). `random` fills in roughly [-1, 1];
    # map it into [-0.6, -0.1] like a real per-token in-chunk log-decay.
    for i in range(a_count):
        var v = a_h.ptr[i].cast[DType.float32]()
        # Map [-1, 1] -> [-0.6, -0.1].
        var scaled = Float32(-0.35) + Float32(0.25) * v
        a_h.ptr[i] = Scalar[dtype](scaled)

    # Call the kernel.
    ssd_output_recombination_fwd_cpu[
        dtype,
        c_h.layout,
        ent_h.layout,
        a_h.layout,
        yd_h.layout,
        y_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        head_dim,
        state_dim,
        c_h,
        ent_h,
        a_h,
        yd_h,
        y_h,
    )

    # Independent naive reference. Recompute strides from scratch:
    #   A_cumsum[l] = sum_{k<=l} A[k]
    #   decay[l]    = exp(A_cumsum[l])
    #   Y_off[l,p]  = decay[l] * sum_n C[l,n] * entering[p,n]
    #   Y[l,p]      = Y_diag[l,p] + Y_off[l,p]
    var ref_y = List(length=y_count, fill=Scalar[dtype](0))

    var c_l_stride = state_dim
    var c_h_stride = chunk_len * c_l_stride
    var c_c_stride = n_heads * c_h_stride
    var c_b_stride = n_chunks * c_c_stride

    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var y_l_stride = head_dim
    var y_h_stride = chunk_len * y_l_stride
    var y_c_stride = n_heads * y_h_stride
    var y_b_stride = n_chunks * y_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var c_base = b * c_b_stride + c * c_c_stride + h * c_h_stride
                var ent_base = (
                    b * ent_b_stride + c * ent_c_stride + h * ent_h_stride
                )
                var y_base = b * y_b_stride + c * y_c_stride + h * y_h_stride

                var a_cumsum = Float32(0.0)
                for l in range(chunk_len):
                    a_cumsum += a_h.ptr[a_base + l].cast[DType.float32]()
                    var decay = exp(a_cumsum)
                    for p in range(head_dim):
                        var acc = Float32(0.0)
                        for n in range(state_dim):
                            var c_val = c_h.ptr[
                                c_base + l * c_l_stride + n
                            ].cast[DType.float32]()
                            var ent_val = ent_h.ptr[
                                ent_base + p * ent_p_stride + n
                            ].cast[DType.float32]()
                            acc += c_val * ent_val
                        var y_off = decay * acc
                        var yd_val = yd_h.ptr[y_base + l * y_l_stride + p].cast[
                            DType.float32
                        ]()
                        ref_y[y_base + l * y_l_stride + p] = Scalar[dtype](
                            yd_val + y_off
                        )

    # Compare kernel output against the reference.
    for i in range(y_count):
        assert_almost_equal(y_h.ptr[i], ref_y[i], rtol=rtol)


def test_ssd_output_recombination_golden() raises:
    """Golden-parity test against the canonical PyTorch SSD reference.

    Pins ``ssd_output_recombination_fwd_cpu`` to a FIXED tiny input whose
    expected output was computed independently by the stage-4 output
    recombination of ``ssd_minimal_discrete``.

    Shapes: batch=1, n_chunks=1, n_heads=1, chunk_len L=4, state_dim N=3,
    head_dim P=2; all tensors row-major.
    """
    comptime dtype = DType.float32
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

    # Fixed literals (row-major).
    # C is L x N; entering is P x N; A is L; Y_diag is L x P.
    var c_vals = [
        Float32(0.35),
        -0.57,
        -0.39,
        -0.29,
        -0.45,
        0.92,
        -0.76,
        1.06,
        -0.12,
        0.58,
        0.47,
        0.30,
    ]
    var ent_vals = [
        Float32(0.56),
        2.29,
        2.11,
        0.30,
        0.03,
        -0.57,
    ]
    var a_vals = [Float32(-0.51), -0.14, -0.32, -0.54]
    var yd_vals = [
        Float32(0.24),
        0.69,
        1.40,
        -0.14,
        -0.35,
        -2.05,
        -0.44,
        -1.53,
    ]
    var y_expected = [
        Float32(-0.920277),
        0.876274,
        1.790647,
        -0.466226,
        0.312865,
        -2.098447,
        0.009353,
        -1.526222,
    ]

    var c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    var c_heap = List(length=c_count, fill=Scalar[dtype](0))
    var c_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        c_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    for i in range(c_count):
        c_h.ptr[i] = Scalar[dtype](c_vals[i])

    var ent_heap = List(length=ent_count, fill=Scalar[dtype](0))
    var ent_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    for i in range(ent_count):
        ent_h.ptr[i] = Scalar[dtype](ent_vals[i])

    var a_heap = List(length=a_count, fill=Scalar[dtype](0))
    var a_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        a_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    for i in range(a_count):
        a_h.ptr[i] = Scalar[dtype](a_vals[i])

    var yd_heap = List(length=y_count, fill=Scalar[dtype](0))
    var yd_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        yd_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    for i in range(y_count):
        yd_h.ptr[i] = Scalar[dtype](yd_vals[i])

    var y_heap = List(length=y_count, fill=Scalar[dtype](0))
    var y_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        y_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    ssd_output_recombination_fwd_cpu[
        dtype,
        c_h.layout,
        ent_h.layout,
        a_h.layout,
        yd_h.layout,
        y_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        head_dim,
        state_dim,
        c_h,
        ent_h,
        a_h,
        yd_h,
        y_h,
    )

    # Pin to the canonical PyTorch reference numerics.
    for i in range(y_count):
        assert_almost_equal(
            y_h.ptr[i],
            Scalar[dtype](y_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_output_recombination_basic() raises:
    """Basic single-batch, single-head case against the naive reference."""
    run_ssd_output_recombination[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        head_dim=8,
        state_dim=16,
    )


def test_ssd_output_recombination_multi_batch() raises:
    """Multiple batch elements and heads against the naive reference."""
    run_ssd_output_recombination[DType.float32](
        batch=2,
        n_chunks=3,
        n_heads=3,
        chunk_len=5,
        head_dim=4,
        state_dim=8,
    )


def test_ssd_output_recombination_single_chunk() raises:
    """Single chunk, single token edge case against the naive reference."""
    run_ssd_output_recombination[DType.float32](
        batch=2,
        n_chunks=1,
        n_heads=2,
        chunk_len=1,
        head_dim=4,
        state_dim=4,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
