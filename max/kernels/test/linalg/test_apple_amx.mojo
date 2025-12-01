# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
#
# This file only tests the Apple AMX matmul functionality which is defined as a
# A^T.B where A and B are 16x16 Float32 matrices.
#
# ===----------------------------------------------------------------------=== #

from layout import Layout, LayoutTensor
from linalg.arch.cpu.apple_amx_intrinsics import *
from testing import *


fn fill_a(buf: LayoutTensor[mut=True, *_, **_]):
    # Fills the A matrix with the following values row + 2*col
    for i in range(buf.dim[0]()):
        for j in range(buf.dim[1]()):
            buf[i, j] = Scalar[buf.dtype](i // (j + 1) + j)


fn fill_b(buf: LayoutTensor[mut=True, *_, **_]):
    # Fills the A matrix with the following values row/(col + 1) + col
    for i in range(buf.dim[0]()):
        for j in range(buf.dim[1]()):
            buf[i, j] = Scalar[buf.dtype](i // (j + 1) + j)


fn clear_c(buf: LayoutTensor[mut=True, *_, **_]):
    _ = buf.fill(0)


def test_dot_at_b[dtype: DType, M: Int, N: Int]():
    # Create LayoutTensors with static shapes
    var a_matrix = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ].stack_allocation()
    var b_matrix = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ].stack_allocation()
    var c_matrix = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ].stack_allocation()

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    dot_at_b(c_matrix, a_matrix, b_matrix)

    for m in range(c_matrix.dim[0]()):
        for n in range(c_matrix.dim[1]()):
            var golden = Scalar[dtype](0)
            for k in range(a_matrix.dim[1]()):
                golden += a_matrix[k, m][0] * b_matrix[k, n][0]
            assert_almost_equal(
                c_matrix[m, n],
                golden,
                msg=String("invalid value at m=", m, ",n=", n),
            )


def main():
    test_dot_at_b[DType.float32, 16, 16]()
    test_dot_at_b[DType.float16, 32, 32]()
