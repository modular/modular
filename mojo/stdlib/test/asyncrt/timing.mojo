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

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceContext


@parameter
fn _timed_iter_func(context: DeviceContext, iter: Int) raises:
    alias length = 64

    var in_host = context.enqueue_create_host_buffer[DType.float32](length)
    var out_host = context.enqueue_create_host_buffer[DType.float32](length)
    var in_dev = context.enqueue_create_buffer[DType.float32](length)
    var out_dev = context.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i + iter
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(in_dev)
    in_dev.enqueue_copy_to(out_dev)
    out_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    context.synchronize()

    for i in range(length):
        expect_eq(
            out_host[i], i + iter, "at index ", i, " the value is ", out_host[i]
        )


@parameter
fn _timed_func(context: DeviceContext) raises:
    _timed_iter_func(context, 2)


fn main() raises:
    var ctx = create_test_device_context()
    print("Running test_timing(" + ctx.name() + "):")

    # Measure the time to run the function 100 times.
    var elapsed_time = ctx.execution_time[_timed_func](100)
    print("Elapsed time for _timed_func: ", elapsed_time / 1e9, "s")

    elapsed_time = ctx.execution_time_iter[_timed_iter_func](100)
    print("Elapsed time for _timed_iter_func: ", elapsed_time / 1e9, "s")
    print("Done.")
