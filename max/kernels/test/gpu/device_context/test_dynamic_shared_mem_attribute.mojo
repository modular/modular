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

from max.gpu import thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation
from std.sys import size_of
from std.testing import TestSuite, assert_equal

comptime BLOCK = 32
comptime KB = 1024


comptime STATIC_SMEM_BYTES = 20 * KB
comptime STATIC_WORDS = STATIC_SMEM_BYTES // size_of[Float32]()


def test_explicit_attribute_on_any_backend() raises:
    # HAL's Metal and HIP plugins reject this attribute.
    def dynamic_only_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        var dynamic_smem = external_memory[
            Float32, address_space=.SHARED, alignment=4
        ]()
        var t = thread_idx.x
        dynamic_smem[t] = Float32(t)
        barrier()
        data[t] = dynamic_smem[BLOCK - 1 - t]

    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[.float32](BLOCK)
        ctx.enqueue_function[dynamic_only_kernel](
            out,
            grid_dim=1,
            block_dim=BLOCK,
            shared_mem_bytes=8 * KB,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(8 * KB),
        )
        with out.map_to_host() as host:
            for t in range(BLOCK):
                assert_equal(host[t], Float32(BLOCK - 1 - t))


def test_static_counts_toward_default_cap() raises:
    # 40KB is under 48KB, but CUDA's default cap shrinks by the static 20KB.
    def static_and_dynamic_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        var static_smem = unsafe_stack_allocation[
            STATIC_WORDS, Float32, address_space=.SHARED
        ]()
        var dynamic_smem = external_memory[
            Float32, address_space=.SHARED, alignment=4
        ]()
        var t = thread_idx.x
        # Use every static word so the compiler keeps the allocation.
        for i in range(t, STATIC_WORDS, BLOCK):
            static_smem[i] = Float32(i)
        dynamic_smem[t] = Float32(t)
        barrier()
        data[t] = static_smem[STATIC_WORDS - 1 - t] + dynamic_smem[t]

    with DeviceContext() as ctx:
        if ctx.api() != "cuda":
            return
        var out = ctx.enqueue_create_buffer[.float32](BLOCK)
        ctx.enqueue_function[static_and_dynamic_kernel](
            out,
            grid_dim=1,
            block_dim=BLOCK,
            shared_mem_bytes=40 * KB,
        )
        with out.map_to_host() as host:
            for t in range(BLOCK):
                assert_equal(
                    host[t], Float32(STATIC_WORDS - 1 - t) + Float32(t)
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
