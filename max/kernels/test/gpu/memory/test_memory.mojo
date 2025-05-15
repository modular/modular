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

from math import iota

from gpu.host import DeviceContext
from memory import UnsafePointer


# CHECK-LABEL: test_memset_async
fn test_memset_async(ctx: DeviceContext) raises:
    print("== test_memset_async")

    @parameter
    @always_inline
    fn test_memset[type: DType](val: Scalar[type]) raises:
        alias length = 4
        var data = UnsafePointer[Scalar[type]].alloc(length)
        var data_device = ctx.enqueue_create_buffer[type](length)
        ctx.enqueue_copy(data_device, data)
        # iota(data, length, 0)
        ctx.enqueue_memset(data_device, val)
        ctx.enqueue_copy(data, data_device)
        ctx.synchronize()
        for i in range(length):
            print(data[i])

        data.free()

    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 3
    # CHECK: 3
    # CHECK: 3
    # CHECK: 3
    test_memset[DType.float32](1.0)
    test_memset[DType.float16](1.0)
    test_memset[DType.int8](3)


def main():
    with DeviceContext() as ctx:
        test_memset_async(ctx)
