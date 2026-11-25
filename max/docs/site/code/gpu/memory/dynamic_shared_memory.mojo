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
# DOC: gpu/memory.mdx

"""Sizes a block's shared memory at launch and checks that the kernel can use
all of it.

The kernel stages an array in dynamically sized shared memory and writes it
back reversed. The host asks for 128 KiB, or the device's maximum if that is
smaller, so the launch succeeds on every GPU.
"""

from std.sys import has_accelerator
from std.sys.info import align_of, size_of

from max.gpu import block_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.memory import external_memory
from max.gpu.sync import barrier

comptime dtype = DType.float32
comptime BLOCK_SIZE = 256
comptime TARGET_BYTES = 128 * 1024


def reverse_in_shared_memory(
    input: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    count: Int32,
):
    # The launch's `shared_mem_bytes` argument sets the size of this memory.
    var smem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=align_of[Scalar[dtype]](),
    ]()
    var n = Int(count)
    for i in range(thread_idx.x, n, block_dim.x):
        smem[unsafe_offset=i] = input[unsafe_offset=i]
    barrier()
    for i in range(thread_idx.x, n, block_dim.x):
        output[unsafe_offset=i] = smem[unsafe_offset=n - 1 - i]


def main() raises:
    comptime assert has_accelerator(), "This example requires a GPU"

    var ctx = DeviceContext()
    # Only NVIDIA GPUs let a kernel opt in to more than the default. On AMD and
    # Apple GPUs, the default is also the maximum.
    var max_bytes = ctx.device().max_shared_memory_per_block()
    if ctx.api() == "cuda":
        max_bytes = ctx.device().get_attribute(
            DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
    var smem_bytes = min(TARGET_BYTES, max_bytes)
    var count = smem_bytes // size_of[Scalar[dtype]]()

    var host = ctx.enqueue_create_host_buffer[dtype](count)
    var dev_in = ctx.enqueue_create_buffer[dtype](count)
    var dev_out = ctx.enqueue_create_buffer[dtype](count)
    ctx.synchronize()
    for i in range(count):
        host[i] = Float32(i)
    ctx.enqueue_copy(dev_in, host)

    ctx.enqueue_function[reverse_in_shared_memory](
        dev_in,
        dev_out,
        Int32(count),
        grid_dim=1,
        block_dim=BLOCK_SIZE,
        shared_mem_bytes=smem_bytes,
    )
    ctx.enqueue_copy(host, dev_out)
    ctx.synchronize()

    for i in range(count):
        if host[i] != Float32(count - 1 - i):
            raise Error(t"output[{i}] is {host[i]}, expected {count - 1 - i}")
    print(
        "Reversed",
        count,
        "values through",
        smem_bytes,
        "bytes of shared memory",
    )
    _ = dev_in^
    _ = dev_out^
