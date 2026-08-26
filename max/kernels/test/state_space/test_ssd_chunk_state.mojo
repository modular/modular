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
from state_space.ssd_chunk_state import ssd_chunk_state_fwd_cpu
from std.math import exp
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def run_ssd_chunk_state[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test the SSD chunk-state kernel against an independent reference."""
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var state_count = batch * n_chunks * n_heads * head_dim * state_dim

    # B: [batch, n_chunks, n_heads, chunk_len, state_dim]
    var B_heap = List(length=b_count, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )

    # X: [batch, n_chunks, n_heads, chunk_len, head_dim]
    var X_heap = List(length=x_count, fill=Scalar[dtype](0))
    var X_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # A: [batch, n_chunks, n_heads, chunk_len]
    var A_heap = List(length=a_count, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )

    # state: [batch, n_chunks, n_heads, head_dim, state_dim]
    var state_heap = List(length=state_count, fill=Scalar[dtype](0))
    var state_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        state_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    # Initialize inputs.
    random(B_h)
    random(X_h)
    random(A_h)

    # Make A negative-ish (a real SSM decays), so exp(cumsum diffs) stays
    # bounded. `random` fills in roughly [-1, 1]; map it into a small negative
    # range like real Mamba2 dt*A values.
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        # Map [-1, 1] -> [-0.6, -0.1].
        var scaled = Float32(-0.35) + Float32(0.25) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    # Call the kernel.
    ssd_chunk_state_fwd_cpu[
        dtype,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        state_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        B_h,
        X_h,
        A_h,
        state_h,
    )

    # Independent naive reference. Recompute strides from scratch and use a
    # straightforward loop of the formula:
    #   A_cumsum[l] = sum_{k<=l} A[k]
    #   decay[l]    = exp(A_cumsum[L-1] - A_cumsum[l])
    #   state[p,n]  = sum_l B[l,n] * decay[l] * X[l,p]
    var ref_heap = List(length=state_count, fill=Scalar[dtype](0))

    var b_h_stride = chunk_len * state_dim
    var b_c_stride = n_heads * b_h_stride
    var b_b_stride = n_chunks * b_c_stride

    var x_h_stride = chunk_len * head_dim
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var state_h_stride = head_dim * state_dim
    var state_c_stride = n_heads * state_h_stride
    var state_b_stride = n_chunks * state_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var b_base = b * b_b_stride + c * b_c_stride + h * b_h_stride
                var x_base = b * x_b_stride + c * x_c_stride + h * x_h_stride
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var state_base = (
                    b * state_b_stride + c * state_c_stride + h * state_h_stride
                )

                # Prefix sum of A over the chunk.
                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                var cum_end = cumsum[chunk_len - 1]

                for p in range(head_dim):
                    for n in range(state_dim):
                        var s_acc = Float32(0.0)
                        for l in range(chunk_len):
                            var bv = B_h.ptr[b_base + l * state_dim + n].cast[
                                DType.float32
                            ]()
                            var xv = X_h.ptr[x_base + l * head_dim + p].cast[
                                DType.float32
                            ]()
                            var decay = exp(cum_end - cumsum[l])
                            s_acc += bv * decay * xv
                        ref_heap[state_base + p * state_dim + n] = Scalar[
                            dtype
                        ](s_acc)

    # Compare kernel output against the reference.
    for i in range(state_count):
        assert_almost_equal(
            state_h.ptr[i],
            ref_heap[i],
            rtol=rtol,
        )


def test_ssd_chunk_state_golden() raises:
    """Golden-parity test against the canonical PyTorch SSD reference.

    Pins ``ssd_chunk_state_fwd_cpu`` to a FIXED tiny input whose expected
    output was computed independently by the chunk end-state reduction (stage 2
    of ``ssd_minimal_discrete``).

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

    # Fixed literals (row-major). These mirror the values in the parity
    # reference; state_expected is the reference's output for these inputs.
    var B_vals = [
        Float32(0.74),
        1.95,
        -0.70,
        -1.30,
        -0.51,
        -0.27,
        0.25,
        0.48,
        0.45,
        -0.96,
        1.50,
        -0.31,
    ]
    var X_vals = [
        Float32(-0.23),
        -1.07,
        0.16,
        0.12,
        0.38,
        -0.12,
        0.89,
        -0.49,
    ]
    var A_vals = [Float32(-0.45), -0.18, -0.36, -0.50]
    var state_expected = [
        Float32(-0.944955),
        1.252577,
        -0.133558,
        0.106325,
        -1.533317,
        0.370175,
    ]

    var b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var state_count = batch * n_chunks * n_heads * head_dim * state_dim

    # Build LayoutTensors from the fixed literals (not random).
    var B_heap = List(length=b_count, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    for i in range(b_count):
        B_h.ptr[i] = Scalar[dtype](B_vals[i])

    var X_heap = List(length=x_count, fill=Scalar[dtype](0))
    var X_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    for i in range(x_count):
        X_h.ptr[i] = Scalar[dtype](X_vals[i])

    var A_heap = List(length=a_count, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    for i in range(a_count):
        A_h.ptr[i] = Scalar[dtype](A_vals[i])

    var state_heap = List(length=state_count, fill=Scalar[dtype](0))
    var state_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        state_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    ssd_chunk_state_fwd_cpu[
        dtype,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        state_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        B_h,
        X_h,
        A_h,
        state_h,
    )

    # Pin to the canonical PyTorch reference numerics.
    for i in range(state_count):
        assert_almost_equal(
            state_h.ptr[i],
            Scalar[dtype](state_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_chunk_state_basic() raises:
    """Basic single-batch, single-chunk case."""
    run_ssd_chunk_state[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=16,
        head_dim=8,
    )


def test_ssd_chunk_state_multi_chunk() raises:
    """Multiple chunks and heads."""
    run_ssd_chunk_state[DType.float32](
        batch=1,
        n_chunks=3,
        n_heads=2,
        chunk_len=16,
        state_dim=16,
        head_dim=8,
    )


def test_ssd_chunk_state_multi_batch() raises:
    """Multiple batch elements."""
    run_ssd_chunk_state[DType.float32](
        batch=2,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=4,
    )


def test_ssd_chunk_state_chunk_len_one() raises:
    """Degenerate chunk length of one (decay is exactly 1)."""
    run_ssd_chunk_state[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=1,
        chunk_len=1,
        state_dim=4,
        head_dim=4,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
