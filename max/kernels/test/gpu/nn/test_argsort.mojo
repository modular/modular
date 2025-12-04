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


from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from memory import LegacyUnsafePointer as UnsafePointer
from nn.argsort import argsort
from testing import assert_equal

from utils.index import IndexList


fn linear_filler(i: Int, n: Int) -> Float32:
    return i


fn reverse_filler(i: Int, n: Int) -> Float32:
    return n - i


fn test_argsort[
    dtype: DType = DType.float32,
    *,
    filler: fn (Int, Int) -> Float32,
    ascending: Bool = True,
](ctx: DeviceContext, N: Int) raises:
    # Allocate host memory
    comptime layout = Layout.row_major[1]()
    var input_host_ptr = UnsafePointer[Scalar[dtype]].alloc(N)
    var input_host = LayoutTensor[dtype, layout](
        input_host_ptr,
        RuntimeLayout[layout].row_major(IndexList[1](N)),
    )

    for i in range(N):
        input_host_ptr[i] = filler(i, N).cast[dtype]()

    # Allocate device buffers
    var device_indices = ctx.enqueue_create_buffer[DType.int64](N)
    var device_input = ctx.enqueue_create_buffer[dtype](N)
    ctx.enqueue_copy(device_input, input_host_ptr)

    # Create device LayoutTensors
    var device_indices_tensor = LayoutTensor[DType.int64, layout, MutAnyOrigin](
        device_indices.unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[1](N)),
    )
    var device_input_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](
        device_input.unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[1](N)),
    )

    argsort[ascending=ascending, target="gpu"](
        device_indices_tensor, device_input_tensor, ctx
    )

    # Copy results back
    var indices_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(N)
    ctx.enqueue_copy(indices_host_ptr, device_indices)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var expected_indices_ptr = UnsafePointer[Scalar[DType.int64]].alloc(N)
    var expected_indices = LayoutTensor[DType.int64, layout](
        expected_indices_ptr,
        RuntimeLayout[layout].row_major(IndexList[1](N)),
    )
    argsort[ascending=ascending](expected_indices, input_host)

    for i in range(N):
        assert_equal(
            indices_host_ptr[i],
            expected_indices_ptr[i],
            msg=String(
                "indices[",
                i,
                "] = ",
                indices_host_ptr[i],
                " expected_indices[",
                i,
                "] = ",
                expected_indices_ptr[i],
                " N = ",
                N,
                " ascending = ",
                ascending,
                " at position ",
                i,
            ),
        )

    # Cleanup host memory
    input_host_ptr.free()
    indices_host_ptr.free()
    expected_indices_ptr.free()

    # Cleanup device buffers
    _ = device_indices^
    _ = device_input^


fn test_argsort_helper[
    *,
    dtype: DType,
    filler: fn (Int, Int) -> Float32,
    ascending: Bool,
](ctx: DeviceContext) raises:
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=3731)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=4096)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=102_400)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=16_384)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=1024)


def main():
    with DeviceContext() as ctx:  # argmax tests
        test_argsort_helper[
            dtype = DType.float32, filler=linear_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype = DType.float32, filler=linear_filler, ascending=False
        ](ctx)
        test_argsort_helper[
            dtype = DType.float32, filler=reverse_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype = DType.float32, filler=reverse_filler, ascending=False
        ](ctx)
