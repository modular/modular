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
from testing import *


fn kernel(res: UnsafePointer[UInt32]):
    res[] = 0


# Here the argument is a host pointer and not a device pointer, so we expect
# an error about an illegal memory address.
def test_function_error(ctx: DeviceContext):
    # CHECK: test_function_error
    print("== test_function_error")
    try:
        var res_host = UnsafePointer[UInt32].alloc(1)
        ctx.enqueue_function[kernel](res_host, block_dim=(1), grid_dim=(1))
        ctx.synchronize()
        res_host.free()
    except e:
        # This error should occur at the synchronize call as the kernel launches
        # async by default.
        # CHECK: open-source/max/max/kernels/test/gpu/device_context/test_function_error.mojo:30:24
        print(e)


def main():
    with DeviceContext() as ctx:
        # CHECK: To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true
        test_function_error(ctx)
