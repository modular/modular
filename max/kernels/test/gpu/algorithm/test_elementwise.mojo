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
# RUN: %mojo-no-debug %s

from math import exp
from sys import is_nvidia_gpu, simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host._compile import get_gpu_target
from testing import assert_equal

from utils import IndexList
from utils.index import Index


fn run_elementwise[type: DType](ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = get_gpu_target()]()

    var in_host = NDBuffer[
        type, 2, MutableAnyOrigin, DimList(2, 8)
    ].stack_allocation()
    var out_host = NDBuffer[
        type, 2, MutableAnyOrigin, DimList(2, 8)
    ].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(2):
        for j in range(8):
            in_host[Index(i, j)] = i + j

    var in_device = ctx.enqueue_create_buffer[type](flattened_length)
    var out_device = ctx.enqueue_create_buffer[type](flattened_length)

    in_device.enqueue_copy_from(in_host.data)

    var in_buffer = NDBuffer[type, 2](in_device._unsafe_ptr(), Index(2, 8))
    var out_buffer = NDBuffer[type, 2](out_device._unsafe_ptr(), Index(2, 8))

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[2]](idx0)
        out_buffer.store(
            idx,
            in_buffer.load[width=simd_width](idx) + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        IndexList[2](2, 8),
        ctx,
    )

    out_device.enqueue_copy_to(out_host.data)

    ctx.synchronize()

    var expected_vals = List[Scalar[type]](
        42.0,
        43.0,
        44.0,
        45.0,
        46.0,
        47.0,
        48.0,
        49.0,
        43.0,
        44.0,
        45.0,
        46.0,
        47.0,
        48.0,
        49.0,
        50.0,
    )
    for i in range(2):
        for j in range(8):
            assert_equal(
                out_host[Index(i, j)],
                expected_vals[i * 8 + j],
            )

    _ = in_device
    _ = out_device


fn run_elementwise_uneven_simd[type: DType](ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = get_gpu_target()]()
    var in_host = NDBuffer[
        type, 2, MutableAnyOrigin, DimList(3, 3)
    ].stack_allocation()
    var out_host = NDBuffer[
        type, 2, MutableAnyOrigin, DimList(3, 3)
    ].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(3):
        for j in range(3):
            in_host[Index(i, j)] = i + j

    var in_device = ctx.enqueue_create_buffer[type](flattened_length)
    var out_device = ctx.enqueue_create_buffer[type](flattened_length)

    in_device.enqueue_copy_from(in_host.data)

    var in_buffer = NDBuffer[type, 2](in_device._unsafe_ptr(), Index(3, 3))
    var out_buffer = NDBuffer[type, 2](out_device._unsafe_ptr(), Index(3, 3))

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[2]](idx0)

        out_buffer.store(
            idx,
            in_buffer.load[width=simd_width](idx) + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        IndexList[2](3, 3),
        ctx,
    )
    out_device.enqueue_copy_to(out_host.data)
    ctx.synchronize()

    var expected_vals = List[Scalar[type]](
        42.0, 43.0, 44.0, 43.0, 44.0, 45.0, 44.0, 45.0, 46.0
    )
    for i in range(3):
        for j in range(3):
            assert_equal(
                out_host[Index(i, j)],
                expected_vals[i * 3 + j],
            )

    _ = in_device
    _ = out_device


fn run_elementwise_transpose_copy[type: DType](ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = get_gpu_target()]()
    var in_host = NDBuffer[
        type, 3, MutableAnyOrigin, DimList(2, 4, 5)
    ].stack_allocation()
    var out_host = NDBuffer[
        type, 3, MutableAnyOrigin, DimList(4, 2, 5)
    ].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(2):
        for j in range(4):
            for k in range(5):
                in_host[Index(i, j, k)] = i * 4 * 5 + j * 5 + k

    var in_device = ctx.enqueue_create_buffer[type](flattened_length)
    var out_device = ctx.enqueue_create_buffer[type](flattened_length)

    in_device.enqueue_copy_from(in_host.data)

    var in_buffer_transposed = NDBuffer[type, 3](
        in_device._unsafe_ptr(), Index(4, 2, 5), Index(5, 20, 1)
    )
    var out_buffer = NDBuffer[type, 3](out_device._unsafe_ptr(), Index(4, 2, 5))

    @always_inline
    @__copy_capture(in_buffer_transposed, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)

        # We need to perform unaligned loads because the non-uniform strides
        # being used for in_buffer.
        out_buffer.store(
            idx, in_buffer_transposed.load[width=simd_width, alignment=1](idx)
        )

    elementwise[func, 1, target="gpu"](
        IndexList[3](4, 2, 5),
        ctx,
    )

    out_device.enqueue_copy_to(out_host.data)
    ctx.synchronize()

    var expected_vals = List[Scalar[type]](
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        35.0,
        36.0,
        37.0,
        38.0,
        39.0,
    )
    for i in range(4):
        for j in range(2):
            for k in range(5):
                assert_equal(
                    out_host[Index(i, j, k)],
                    expected_vals[i * 2 * 5 + j * 5 + k],
                )

    _ = in_device
    _ = out_device


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float32](ctx)
        run_elementwise_uneven_simd[DType.float32](ctx)
        run_elementwise_transpose_copy[DType.float32](ctx)
        run_elementwise[DType.bfloat16](ctx)
        run_elementwise_uneven_simd[DType.bfloat16](ctx)
        run_elementwise_transpose_copy[DType.bfloat16](ctx)
        run_elementwise[DType.float16](ctx)
        run_elementwise_uneven_simd[DType.float16](ctx)
        run_elementwise_transpose_copy[DType.float16](ctx)
