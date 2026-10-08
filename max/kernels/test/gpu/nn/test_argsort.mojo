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


from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major

from nn.argsort import argsort
from std.testing import assert_equal, assert_false, assert_true


def linear_filler(i: Int, n: Int) -> Float32:
    return Float32(i)


def reverse_filler(i: Int, n: Int) -> Float32:
    return Float32(n - i)


def scrambled_filler(i: Int, n: Int) -> Float32:
    return Float32((i * 73 + 19) % n - n // 2)


def repeated_filler(i: Int, n: Int) -> Float32:
    return Float32((i * 73) % 17 - 8)


def test_argsort[
    dtype: DType = .float32,
    *,
    filler: def(Int, Int) thin -> Float32,
    ascending: Bool = True,
](ctx: DeviceContext, N: Int) raises:
    # Allocate host memory
    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](N)
    var input_host = TileTensor(
        input_host_ptr,
        row_major(N),
    )

    for i in range(N):
        input_host_ptr[i] = filler(i, N).cast[dtype]()

    # Allocate device buffers
    var device_indices = ctx.enqueue_create_buffer[.int64](N)
    var device_input = ctx.enqueue_create_buffer[dtype](N)
    ctx.enqueue_copy(device_input, input_host_ptr)

    # Create device TileTensors
    var device_indices_tensor = TileTensor(
        device_indices,
        row_major(N),
    )
    var device_input_tensor = TileTensor(
        device_input,
        row_major(N),
    )

    argsort[ascending=ascending, target="gpu"](
        device_indices_tensor, device_input_tensor, ctx
    )

    # Copy results back
    var indices_host_ptr = ctx.enqueue_create_host_buffer[.int64](N)
    ctx.enqueue_copy(indices_host_ptr, device_indices)
    var input_after = ctx.enqueue_create_host_buffer[dtype](N)
    ctx.enqueue_copy(input_after, device_input)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var expected_indices_ptr = ctx.enqueue_create_host_buffer[.int64](N)
    var expected_indices = TileTensor(
        expected_indices_ptr,
        row_major(N),
    )
    argsort[ascending=ascending](expected_indices, input_host)

    # Equal keys may have different index orders on CPU and GPU. Check the
    # gathered values and the complete permutation instead of tie order.
    var seen = List[Bool](length=N, fill=False)
    for i in range(N):
        var index = Int(indices_host_ptr[i])
        assert_true(
            index >= 0 and index < N,
            msg=String(t"index {index} is out of range for N={N}"),
        )
        assert_false(
            seen[index],
            msg=String(t"duplicate index {index} for N={N}"),
        )
        seen[index] = True
        assert_equal(
            input_host_ptr[index],
            input_host_ptr[Int(expected_indices_ptr[i])],
            msg=String(
                t"indices[{i}] = {indices_host_ptr[i]} expected_indices[{i}] ="
                t" {expected_indices_ptr[i]} N = {N} ascending = {ascending} at"
                t" position {i}"
            ),
        )
        assert_equal(input_after[i], input_host_ptr[i])

    # Cleanup device buffers
    _ = device_indices^
    _ = device_input^


def test_argsort_helper[
    *,
    dtype: DType,
    filler: def(Int, Int) thin -> Float32,
    ascending: Bool,
](ctx: DeviceContext) raises:
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=3731)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=4096)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=102_400)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=16_384)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=1024)


def test_argsort_multiblock[
    *, dtype: DType, ascending: Bool
](ctx: DeviceContext) raises:
    for n in [255, 256, 257, 300, 448, 511, 512, 513, 1024, 4096]:
        test_argsort[dtype, filler=scrambled_filler, ascending=ascending](
            ctx, n
        )
        test_argsort[dtype, filler=repeated_filler, ascending=ascending](ctx, n)


def main() raises:
    with DeviceContext() as ctx:
        test_argsort_multiblock[dtype=DType.float32, ascending=True](ctx)
        test_argsort_multiblock[dtype=DType.float32, ascending=False](ctx)
        test_argsort_multiblock[dtype=DType.int64, ascending=True](ctx)
        test_argsort_multiblock[dtype=DType.int64, ascending=False](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=linear_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=linear_filler, ascending=False
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=reverse_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=reverse_filler, ascending=False
        ](ctx)
