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

from math import ceildiv
from os import Atomic

from gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, thread_idx
from gpu.host.info import is_cpu
from gpu.memory import AddressSpace
from memory import stack_allocation
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, ManagedTensorSlice, OutputTensor

from utils import StaticTuple

alias bin_width = Int(UInt8.MAX)


fn _histogram_cpu(output: ManagedTensorSlice, input: ManagedTensorSlice):
    for i in range(input.dim_size(0)):
        output[Int(input[i])] += 1


fn _histogram_gpu(
    output: ManagedTensorSlice,
    input: ManagedTensorSlice,
    ctx_ptr: DeviceContextPtr,
) raises:
    alias bin_width = Int(UInt8.MAX) + 1
    alias block_dim = bin_width

    # Set the maximum number of threads per block to the block dimension.
    # This is equivalent to the `__launch_bounds__` attribute in CUDA.
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](block_dim)
    )
    fn kernel(
        output: UnsafePointer[Int64], input: UnsafePointer[UInt8], n: Int
    ):
        var tid = global_idx.x

        if tid >= UInt(n):
            return

        # Allocate shared memory for the histogram
        var shared_mem = stack_allocation[
            bin_width, Int64, address_space = AddressSpace.SHARED
        ]()

        # Initialize the shared memory to 0
        shared_mem[thread_idx.x] = 0

        # Synchronize all threads to ensure that the shared memory is initialized
        barrier()

        # Increment the shared memory for the current thread
        _ = Atomic.fetch_add(shared_mem + Int(input[tid]), 1)

        # Synchronize all threads to ensure that the shared memory is updated
        barrier()

        # Increment the output for the current thread
        _ = Atomic.fetch_add(output + thread_idx.x, shared_mem[thread_idx.x])

    var n = input.dim_size(0)

    var grid_dim = ceildiv(n, block_dim)

    var ctx = ctx_ptr.get_device_context()

    ctx.enqueue_function[kernel](
        output.unsafe_ptr(),
        input.unsafe_ptr(),
        n,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@compiler.register("histogram")
struct Histogram:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor[dtype = DType.int64, rank=1],
        input: InputTensor[dtype = DType.uint8, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            _histogram_cpu(output, input)
        else:
            _histogram_gpu(output, input, ctx)
