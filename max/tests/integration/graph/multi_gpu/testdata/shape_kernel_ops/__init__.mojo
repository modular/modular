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
"""Shape-specialized kernels for the multi-GPU execute stress test.

Each batch-size bucket launches its own kernel image, as a shape-dispatched
GEMM does, and each requests more dynamic shared memory than the default
48 KB, so its first load opts in through `cuFuncSetAttribute`.
"""

from std.math import ceildiv

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu import global_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier

# Variants per op instance, and the token width of each bucket: 256 buckets of
# 32 tokens cover a step of up to 8192 tokens.
comptime NUM_VARIANTS = 256
comptime BUCKET_TOKENS = 32
comptime BLOCK_THREADS = 256
# Above the 48 KB a kernel gets without opting in.
comptime SMEM_BYTES = 64 * 1024


def _shape_kernel[
    variant: Int, salt: Int
](
    output: Pointer[Float32, MutAnyOrigin],
    input: Pointer[Float32, MutAnyOrigin],
    n_dev: Int32,
):
    var smem = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]().unsafe_bitcast[Float32]()
    var n = Int(n_dev)
    var tid = global_idx.x
    var lane = thread_idx.x
    var v: Float32 = 0
    if tid < n:
        v = input[unsafe_offset=tid]
    smem[unsafe_offset=lane] = v * Float32(variant + 1) + Float32(salt)
    barrier()
    if tid < n:
        output[unsafe_offset=tid] = smem[
            unsafe_offset=(lane + 1) % BLOCK_THREADS
        ]


def _launch[
    variant: Int, salt: Int
](
    output: OutputTensor[dtype=.float32, rank=2, ...],
    input: InputTensor[dtype=.float32, rank=2, ...],
    ctx: DeviceContext,
) raises:
    var n = input.dim_size(0) * input.dim_size(1)
    ctx.enqueue_function[_shape_kernel[variant, salt]](
        output.to_device_buffer(ctx),
        input.to_device_buffer(ctx),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK_THREADS),
        block_dim=BLOCK_THREADS,
        shared_mem_bytes=SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_BYTES)
        ),
    )


@extensibility.register("shape_kernel")
struct ShapeKernel:
    """Launches the variant for the input's token bucket.

    `salt` gives each graph position its own set of images.
    """

    @staticmethod
    def execute[
        target: StaticString, salt: Int
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        input: InputTensor[dtype=.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target != "gpu":
            raise Error("shape_kernel runs on GPUs only")
        var bucket = (input.dim_size(0) - 1) // BUCKET_TOKENS
        if bucket >= NUM_VARIANTS:
            bucket = NUM_VARIANTS - 1
        comptime for variant in range(NUM_VARIANTS):
            if bucket == variant:
                _launch[variant, salt](output, input, ctx)
