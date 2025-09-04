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

from math import ceildiv
from sys import argv

from gpu import block_dim
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from layout import Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from linalg import vendor_blas

from utils.index import IndexList

from testing import assert_almost_equal


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


fn kernel_1[
    M: Int,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    BLOCKSIZE: Int = 32,
](
    c: LayoutTensor[mut=True, DType.bfloat16, Layout.row_major(M, N)],
    a: LayoutTensor[mut=False, DType.bfloat16, Layout.row_major(M, K)],
    b: LayoutTensor[mut=False, DType.bfloat16, Layout.row_major(K, N)],
):
    var row = block_dim.y * block_idx.y + (thread_idx.y)
    var col = block_dim.x * block_idx.x + (thread_idx.x)

    if row < UInt(M) and col < UInt(N):
        # Still accumulate in float32 for precision
        var acc: Float32 = 0

        for k in range(K):
            var a_val = rebind[Float32](a[row, k].cast[DType.float32]())
            var b_val = rebind[Float32](b[k, col].cast[DType.float32]())
            acc += a_val * b_val

        c[row, col] = acc.cast[DType.bfloat16]()


def test_kernel_1[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
    benchmark: Bool = False,
    prob_shape: IndexList[3] = IndexList[3](1, 1, 1),
](ctx: DeviceContext):
    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    print(M, "x", N, "x", K)

    var a = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    random(a.tensor[update=False]())
    alias b_layout = Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    random(b.tensor[update=False]())
    var c = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var c_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    alias b_vendor_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b_vendor = ManagedLayoutTensor[b_type, b_vendor_layout](ctx)

    @parameter
    if transpose_b:
        var b_tensor = b.tensor[update=False]()
        var b_vendor_tensor = b_vendor.tensor[update=True]()
        for k in range(K):
            for n in range(N):
                b_vendor_tensor[n, k] = b_tensor[k, n]
    else:
        var b_tensor = b.tensor[update=False]()
        var b_vendor_tensor = b_vendor.tensor[update=True]()
        for k in range(K):
            for n in range(N):
                b_vendor_tensor[k, n] = b_tensor[k, n]

    alias kernel = kernel_1[
        M, N, K, transpose_b=transpose_b, BLOCKSIZE=BLOCKSIZE
    ]
    # Use 1D thread block for memory coalescing
    alias BLOCKSIZE = 32

    ctx.enqueue_function[kernel](
        c.device_tensor(),
        a.device_tensor(),
        b.device_tensor(),
        grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
        block_dim=(BLOCKSIZE, BLOCKSIZE),
    )

    ctx.synchronize()

    if benchmark:
        alias num_runs = 50
        alias num_warmup = 20

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            ctx.enqueue_function[kernel](
                c.device_tensor(),
                a.device_tensor(),
                b.device_tensor(),
                grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
                block_dim=(BLOCKSIZE, BLOCKSIZE),
            )

        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_ref.device_buffer(),  # returns an NDBuffer[dtype, 2, MutableAnyOrigin]
            a.device_buffer(),
            b_vendor.device_buffer(),
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        c_host = c.tensor()
        c_host_ref = c_ref.tensor()

        for m in range(M):
            for n in range(N):
                assert_almost_equal(
                    c_host[m, n],
                    c_host_ref[m, n],
                    atol=1e-2,
                    rtol=5e-2,
                    msg=String(m) + ", " + String(n),
                )
        print("TEST PASSED")

    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^
    _ = b_vendor^


def main():
    with DeviceContext() as ctx:
        if is_benchmark():
            test_kernel_1[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                transpose_b=True,
                prob_shape = IndexList[3](4096, 4096, 4096),
                benchmark=True,
            ](ctx)
            return

        # Test with transpose_b=True
        print("Testing with transpose_b=True")
        test_kernel_1[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            transpose_b=True,
            prob_shape = IndexList[3](4096, 4096, 4096),
        ](ctx)
