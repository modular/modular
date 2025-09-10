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


from gpu.host import DeviceContext, HostBuffer
from testing import assert_equal
from gpu import block_idx, thread_idx, block_dim
from random import random_float64


fn simd_add_kernel[
    width: Int
](
    a_span: UnsafePointer[Float32],
    b_span: UnsafePointer[Float32],
    c_span: UnsafePointer[Float32],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.load[width=width](idx)
    var vector_b = b_span.load[width=width](idx)
    var vector_c = vector_a + vector_b
    c_span.store[width=width](idx, vector_c)


fn simd_mult_kernel[
    width: Int
](
    a_span: UnsafePointer[Float32],
    b_span: UnsafePointer[Float32],
    c_span: UnsafePointer[Float32],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.load[width=width](idx)
    var vector_b = b_span.load[width=width](idx)
    var vector_c = vector_a * vector_b
    c_span.store[width=width](idx, vector_c)


fn simd_fma_kernel[
    width: Int
](
    a_span: UnsafePointer[Float32],
    b_span: UnsafePointer[Float32],
    c_span: UnsafePointer[Float32],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.load[width=width](idx)
    var vector_b = b_span.load[width=width](idx)
    var vector_c = c_span.load[width=width](idx)
    vector_c = vector_a.fma(vector_b, vector_c)

    c_span.store[width=width](idx, vector_c)


fn host_elementwise_add(
    a: HostBuffer[DType.float32],
    b: HostBuffer[DType.float32],
    mut c: HostBuffer[DType.float32],
    size: Int,
):
    for i in range(size):
        c[i] = a[i] + b[i]


fn host_elementwise_mult(
    a: HostBuffer[DType.float32],
    b: HostBuffer[DType.float32],
    mut c: HostBuffer[DType.float32],
    size: Int,
):
    for i in range(size):
        c[i] = a[i] * b[i]


fn host_elementwise_fma(
    a: HostBuffer[DType.float32],
    b: HostBuffer[DType.float32],
    mut c: HostBuffer[DType.float32],
    size: Int,
):
    for i in range(size):
        var c_temp = a[i] * b[i] + c[i]
        c[i] = c_temp


def test_arithmetic[
    width: Int, mode: String
](ctx: DeviceContext,):
    alias thread_count = 32
    alias block_count = 1
    alias buff_size = thread_count * block_count * width

    var a_host = ctx.enqueue_create_host_buffer[DType.float32](buff_size)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](buff_size)
    var c_host = ctx.enqueue_create_host_buffer[DType.float32](buff_size)

    ctx.synchronize()

    for i in range(buff_size):
        a_host[i] = random_float64(-1.0, 1.0).cast[DType.float32]()
        b_host[i] = random_float64(-1.0, 1.0).cast[DType.float32]()
        c_host[i] = 0.0

    # Create device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[DType.float32](buff_size)
    var b_device_buffer = ctx.enqueue_create_buffer[DType.float32](buff_size)
    var c_device_buffer = ctx.enqueue_create_buffer[DType.float32](buff_size)

    # Copy data from host to device
    ctx.enqueue_copy(a_device_buffer, a_host.unsafe_ptr())
    ctx.enqueue_copy(b_device_buffer, b_host.unsafe_ptr())
    ctx.enqueue_copy(c_device_buffer, c_host.unsafe_ptr())

    # Compute expected result on host
    var c_expected = ctx.enqueue_create_host_buffer[DType.float32](
        buff_size
    ).enqueue_fill(0)
    ctx.synchronize()

    @parameter
    if mode == "add":
        alias kernel = simd_add_kernel[width]

        ctx.enqueue_function_checked[kernel, kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_add(a_host, b_host, c_expected, buff_size)

    elif mode == "mult":
        alias kernel = simd_mult_kernel[width]

        ctx.enqueue_function_checked[kernel, kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_mult(a_host, b_host, c_expected, buff_size)

    else:
        alias kernel = simd_fma_kernel[width]

        # Execute kernel on GPU
        ctx.enqueue_function_checked[kernel, kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_fma(a_host, b_host, c_expected, buff_size)

    # Copy result back from device to host
    var c_result = ctx.enqueue_create_host_buffer[DType.float32](buff_size)
    ctx.enqueue_copy(c_result.unsafe_ptr(), c_device_buffer)
    ctx.synchronize()

    # Compare results
    for i in range(buff_size):
        assert_equal(c_result[i], c_expected[i])


def main():
    with DeviceContext() as ctx:
        test_arithmetic[2, "add"](ctx)
        test_arithmetic[4, "add"](ctx)
        test_arithmetic[8, "add"](ctx)
        test_arithmetic[2, "mult"](ctx)
        test_arithmetic[4, "mult"](ctx)
        test_arithmetic[8, "mult"](ctx)
        test_arithmetic[2, "fma"](ctx)
        test_arithmetic[4, "fma"](ctx)
        test_arithmetic[8, "fma"](ctx)
