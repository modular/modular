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
"""Real GPU launch test for `Span`'s `address_space` parameter.

A kernel writes GPU shared memory through `Span` indexing and reads it back
through an address-space-generic helper, checking that slicing and `as_imm`
preserve the address space.
"""

from std.gpu import barrier, thread_idx
from std.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

comptime TILE_SIZE = 32


# Address-space-generic: the same body sums a shared-memory tile and a
# generic-address-space span.
def _tile_sum(tile: Span[Float32, _, address_space=_]) -> Float32:
    var acc = Float32(0)
    for i in range(len(tile)):
        acc += tile[i]
    return acc


def _kernel(out_ptr: UnsafePointer[Float32, MutAnyOrigin]):
    var smem = unsafe_stack_allocation[
        TILE_SIZE, Float32, address_space=AddressSpace.SHARED
    ]()
    var tile = Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ](unsafe_ptr=smem, length=TILE_SIZE)

    # Write into shared memory through `Span` indexing.
    tile[thread_idx.x] = Float32(thread_idx.x + 1)
    barrier()

    if thread_idx.x == 0:
        out_ptr[unsafe_offset=0] = _tile_sum(tile)
        # Slicing preserves the address space.
        out_ptr[unsafe_offset=1] = _tile_sum(tile[0 : TILE_SIZE // 2])
        # `as_imm` preserves the address space.
        out_ptr[unsafe_offset=2] = tile.as_imm()[5]


def main() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[DType.float32](3)
        var compiled = ctx.compile_function[_kernel]()
        ctx.enqueue_function(
            compiled, out_device, grid_dim=(1,), block_dim=(TILE_SIZE,)
        )
        ctx.synchronize()

        with out_device.map_to_host() as out_host:
            # 1 + 2 + ... + 32
            assert_equal(out_host[0], 528.0)
            # 1 + 2 + ... + 16
            assert_equal(out_host[1], 136.0)
            assert_equal(out_host[2], 6.0)
