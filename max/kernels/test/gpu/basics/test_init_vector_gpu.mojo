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


from gpu import *
from gpu.host import DeviceContext
from buffer import DimList
from internal_utils import (
    Timer,
    initialize,
    InitializationType,
    HostNDBuffer,
    init_vector_launch,
)
from testing import assert_equal

from random import seed


@no_inline
fn test_vec_init[
    dtype: DType, block_dim: Int = 256
](length: Int, init_type: InitializationType, context: DeviceContext) raises:
    var timer = Timer()
    var out_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var out_device = context.enqueue_create_buffer[dtype](length)
    timer.measure("create-buffer")

    init_vector_launch[dtype](out_device, length, init_type, context)

    timer.measure("vector_init_launch")
    context.synchronize()
    context.enqueue_copy(out_host, out_device)
    timer.measure("copy+sync")

    # verification for uniform_distribution is not supported!
    if init_type in [
        InitializationType.zero,
        InitializationType.one,
        InitializationType.arange,
    ]:
        var verification_data = HostNDBuffer[dtype, 1](DimList(length))
        seed(0)
        initialize(verification_data.tensor, init_type)
        for i in range(length):
            assert_equal(verification_data.tensor.data[i], out_host[i])

    out_host.free()
    timer.print()


def main():
    alias block_dim = 256
    alias dtype = DType.float32
    var length = 32 * 1024
    with DeviceContext() as ctx:
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.zero, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.one, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.arange, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length,
            init_type=InitializationType.uniform_distribution,
            context=ctx,
        )
