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

from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

from utils.index import IndexList


fn _vector_addition_cpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    rhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
):
    # Warning: This is an extremely inefficient implementation! It's merely an
    # instructional example of how a dedicated CPU-only path can be specified
    # for basic vector addition.
    var vector_length = out.dim_size(0)
    for i in range(vector_length):
        var idx = IndexList[out.rank](i)
        var result = lhs.load[1](idx) + rhs.load[1](idx)
        out.store[1](idx, result)


fn _vector_addition_gpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    rhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
) raises:
    # Note: The following has not been tuned for any GPU hardware, and is an
    # instructional example for how a simple GPU function can be constructed
    # and dispatched.
    alias BLOCK_SIZE = 16
    var gpu_ctx = ctx.get_device_context()
    var vector_length = out.dim_size(0)

    # The function that will be launched and distributed across GPU threads.
    @parameter
    fn vector_addition_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < length:
            var idx = IndexList[out.rank](tid)
            var result = lhs.load[1](idx) + rhs.load[1](idx)
            out.store[1](idx, result)

    # The vector is divided up into blocks, making sure there's an extra
    # full block for any remainder.
    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)

    # The GPU function is compiled and enqueued to run on the GPU across the
    # 1-D vector, split into blocks of `BLOCK_SIZE` width.
    gpu_ctx.enqueue_function[vector_addition_gpu_kernel](
        vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@compiler.register("vector_addition")
struct VectorAddition:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        out: OutputTensor[rank=1],
        lhs: InputTensor[type = out.type, rank = out.rank],
        rhs: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # For a simple elementwise operation like this, the `foreach` function
        # does much more rigorous hardware-specific tuning. We recommend using
        # that abstraction, with this example serving purely as an illustration
        # of how lower-level functions can be used to program GPUs via Mojo.

        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "cpu":
            _vector_addition_cpu(out, lhs, rhs, ctx)
        elif target == "gpu":
            _vector_addition_gpu(out, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)
