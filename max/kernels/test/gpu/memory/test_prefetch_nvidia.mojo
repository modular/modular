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

# `test_prefetch.mojo` checks which instruction each address space lowers to.
# This runs them on the GPU, so `ptxas` and the hardware accept every form.

from std.memory import stack_allocation
from std.sys import prefetch
from std.testing import assert_equal

from max.gpu.host import DeviceContext

comptime _N = 4


def _kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var shared = stack_allocation[_N, Float32, address_space=.SHARED]()
    var local = stack_allocation[_N, Float32, address_space=.LOCAL]()
    for i in range(_N):
        shared[i] = data[i]
        local[i] = data[i] * 2

    prefetch(data)
    prefetch(data.unsafe_address_space_cast[.GLOBAL]())
    prefetch(shared)
    prefetch(local)

    for i in range(_N):
        output[i] = shared[i] + local[i]


def main() raises:
    with DeviceContext() as ctx:
        var data = ctx.enqueue_create_buffer[DType.float32](_N)
        with data.map_to_host() as host:
            for i in range(_N):
                host[i] = Float32(i + 1)
        var output = ctx.enqueue_create_buffer[DType.float32](_N)
        ctx.enqueue_function[_kernel](data, output, grid_dim=1, block_dim=1)
        with output.map_to_host() as host:
            for i in range(_N):
                assert_equal(host[i], Float32(3 * (i + 1)))
