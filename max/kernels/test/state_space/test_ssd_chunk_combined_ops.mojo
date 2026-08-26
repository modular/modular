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
"""End-to-end parity test for the fused SSD ``ssd_chunk_scan_combined`` op.

Mirrors the golden literals from RFC 0002's pipeline checkpoint
(b=1, seqlen=4, n_heads=1, head_dim=2, state_dim=3, chunk_size=2),
where the expected output was independently computed via the vendored
``ssd_minimal_discrete`` reference. The test calls the CPU implementation
that backs the ``@extensibility.register("ssd_chunk_scan_combined")`` op and
asserts the per-element output matches the reference golden within the
parity tolerance (rtol=1e-3, atol=1e-3).

The `@extensibility.register` registration itself is validated at build time:
if the struct's ``execute`` signature does not match the extensibility ABI,
the ``//max/kernels/src/state_space:state_space`` library fails to compile.
"""

from state_space.ssd_chunk_combined_ops import _ssd_chunk_scan_combined_cpu
from std.testing import TestSuite, assert_almost_equal


def test_ssd_chunk_scan_combined_cpu_golden_parity() raises:
    """Parity vs golden: b=1, seqlen=4, n_heads=1, head_dim=2, state_dim=3.

    Golden Y_expected (row-major, shape [batch, seqlen, n_heads, head_dim]):
        [ 0.126999, -0.028039,
          0.191223, -0.003887,
         -0.370202, -0.111640,
         -0.005157, -0.252940 ]
    """
    comptime dtype = DType.float32

    var batch = 1
    var seqlen = 4
    var n_heads = 1
    var head_dim = 2
    var state_dim = 3
    var chunk_size = 2

    var x_size = batch * seqlen * n_heads * head_dim
    var dt_size = batch * seqlen * n_heads
    var A_size = n_heads
    var bc_size = batch * seqlen * n_heads * state_dim
    var y_size = x_size
    var final_state_size = batch * n_heads * head_dim * state_dim

    # x: [batch, seqlen, n_heads, head_dim], row-major
    var x = List[Scalar[dtype]](length=x_size, fill=Scalar[dtype](0))
    var x_vals: List[Float32] = [
        0.77,
        -0.17,
        -0.66,
        -0.39,
        0.31,
        -0.62,
        -1.19,
        -1.28,
    ]
    for i in range(x_size):
        x[i] = x_vals[i].cast[dtype]()

    # dt: [batch, seqlen, n_heads]
    var dt = List[Scalar[dtype]](length=dt_size, fill=Scalar[dtype](0))
    var dt_vals: List[Float32] = [0.33, 0.26, 0.12, 0.24]
    for i in range(dt_size):
        dt[i] = dt_vals[i].cast[dtype]()

    # A: [n_heads]
    var A = List[Scalar[dtype]](length=A_size, fill=Scalar[dtype](0))
    A[0] = Scalar[dtype](-1.2)

    # B: [batch, seqlen, n_heads, state_dim]
    var B = List[Scalar[dtype]](length=bc_size, fill=Scalar[dtype](0))
    var B_vals: List[Float32] = [
        -0.91,
        0.56,
        -1.13,
        -0.41,
        0.59,
        0.28,
        1.46,
        0.67,
        0.19,
        -0.05,
        0.00,
        -0.72,
    ]
    for i in range(bc_size):
        B[i] = B_vals[i].cast[dtype]()

    # C: [batch, seqlen, n_heads, state_dim]
    var C = List[Scalar[dtype]](length=bc_size, fill=Scalar[dtype](0))
    var C_vals: List[Float32] = [
        0.64,
        -0.61,
        -1.26,
        -1.05,
        -1.03,
        -0.35,
        1.92,
        0.30,
        1.47,
        -0.55,
        1.06,
        -0.74,
    ]
    for i in range(bc_size):
        C[i] = C_vals[i].cast[dtype]()

    var Y = List[Scalar[dtype]](length=y_size, fill=Scalar[dtype](0))
    var final_state = List[Scalar[dtype]](
        length=final_state_size, fill=Scalar[dtype](0)
    )

    _ssd_chunk_scan_combined_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        chunk_size,
        x.unsafe_ptr(),
        dt.unsafe_ptr(),
        A.unsafe_ptr(),
        B.unsafe_ptr(),
        C.unsafe_ptr(),
        Y.unsafe_ptr(),
        final_state.unsafe_ptr(),
    )

    # Expected golden, row-major [batch, seqlen, n_heads, head_dim].
    var expected: List[Float32] = [
        0.126999,
        -0.028039,
        0.191223,
        -0.003887,
        -0.370202,
        -0.111640,
        -0.005157,
        -0.252940,
    ]

    for i in range(y_size):
        assert_almost_equal(
            Y[i],
            expected[i].cast[dtype](),
            atol=1e-3,
            rtol=1e-3,
        )

    # Expected golden for final_state, row-major
    # [batch, n_heads, head_dim, state_dim]. Computed by the vendored
    # ``ssd_minimal_discrete`` reference for the same inputs (see
    # .planning/parity/ssd_minimal_ref.py).
    var expected_final_state: List[Float32] = [
        -0.009206,
        0.020579,
        0.043290,
        -0.014832,
        -0.091143,
        0.222278,
    ]

    for i in range(final_state_size):
        assert_almost_equal(
            final_state[i],
            expected_final_state[i].cast[dtype](),
            atol=1e-3,
            rtol=1e-3,
        )


def test_ssd_chunk_scan_combined_cpu_chunk_divides_seqlen() raises:
    """The op must raise when seqlen is not divisible by chunk_size."""
    comptime dtype = DType.float32
    var x = List[Scalar[dtype]](length=4, fill=Scalar[dtype](0))
    var dt = List[Scalar[dtype]](length=4, fill=Scalar[dtype](0))
    var A = List[Scalar[dtype]](length=1, fill=Scalar[dtype](0))
    var B = List[Scalar[dtype]](length=4, fill=Scalar[dtype](0))
    var C = List[Scalar[dtype]](length=4, fill=Scalar[dtype](0))
    var Y = List[Scalar[dtype]](length=4, fill=Scalar[dtype](0))
    var final_state = List[Scalar[dtype]](length=1, fill=Scalar[dtype](0))
    var raised = False
    try:
        _ssd_chunk_scan_combined_cpu[dtype](
            1,  # batch
            4,  # seqlen
            1,  # n_heads
            1,  # head_dim
            1,  # state_dim
            3,  # chunk_size (4 % 3 != 0)
            x.unsafe_ptr(),
            dt.unsafe_ptr(),
            A.unsafe_ptr(),
            B.unsafe_ptr(),
            C.unsafe_ptr(),
            Y.unsafe_ptr(),
            final_state.unsafe_ptr(),
        )
    except _:
        raised = True
    if not raised:
        raise Error("Expected divisibility check to raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
