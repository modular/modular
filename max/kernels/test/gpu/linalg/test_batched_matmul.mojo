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


from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from linalg.bmm import _batched_matmul_gpu

from utils import Index, IndexList


# CHECK-LABEL: test_batched_matmul
fn test_batched_matmul(ctx: DeviceContext) raises:
    print("== test_batched_matmul")

    alias b = 2
    alias m = 2
    alias n = 2
    alias k = 4

    var lhs_host = NDBuffer[
        DType.float32, 3, MutableAnyOrigin, DimList(b, m, k)
    ].stack_allocation()
    var rhs_host = NDBuffer[
        DType.float32, 3, MutableAnyOrigin, DimList(b, k, n)
    ].stack_allocation()
    var dst_host = NDBuffer[
        DType.float32, 3, MutableAnyOrigin, DimList(b, m, n)
    ].stack_allocation()

    var csum: Float32 = 0.0
    for bi in range(b):
        for mi in range(m):
            for ki in range(k):
                lhs_host[Index(bi, mi, ki)] = csum
                csum += 1.0

    csum = 0.0
    for bi in range(b):
        for ki in range(k):
            for ni in range(n):
                rhs_host[Index(bi, ki, ni)] = csum
                csum += 1.0

    csum = 0.0
    for bi in range(b):
        for mi in range(m):
            for ni in range(n):
                dst_host[Index(bi, mi, ni)] = 0.0

    var lhs_device = ctx.enqueue_create_buffer[DType.float32](
        lhs_host.num_elements()
    )
    var rhs_device = ctx.enqueue_create_buffer[DType.float32](
        rhs_host.num_elements()
    )
    var dst_device = ctx.enqueue_create_buffer[DType.float32](
        dst_host.num_elements()
    )

    var lhs_buffer = NDBuffer[DType.float32, 3](
        rhs_device._unsafe_ptr(), Index(b, m, k)
    )
    var rhs_buffer = NDBuffer[DType.float32, 3](
        lhs_device._unsafe_ptr(), Index(b, k, n)
    )
    var dst_buffer = NDBuffer[DType.float32, 3](
        dst_device._unsafe_ptr(), Index(b, m, n)
    )

    ctx.enqueue_copy(lhs_device, lhs_host.data)
    ctx.enqueue_copy(rhs_device, rhs_host.data)
    ctx.enqueue_copy(dst_device, dst_host.data)

    @always_inline
    @__copy_capture(dst_buffer)
    @parameter
    fn elementwise_epilogue_empty_fn[
        c_type: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[c_type, width],) -> None:
        dst_buffer.store(
            Index(idx[0], idx[1], idx[2]), val.cast[dst_buffer.type]() + 2
        )

    _batched_matmul_gpu[
        transpose_b=False,
        elementwise_epilogue_fn=elementwise_epilogue_empty_fn,
    ](dst_buffer, lhs_buffer, rhs_buffer, ctx)

    ctx.enqueue_copy(dst_host.data, dst_device)

    ctx.synchronize()

    # CHECK: [30.0, 36.0],
    # CHECK: [78.0, 100.0]],
    # CHECK: [430.0, 468.0],
    # CHECK: [606.0, 660.0]
    print(dst_host)


def main():
    with DeviceContext() as ctx:
        test_batched_matmul(ctx)
