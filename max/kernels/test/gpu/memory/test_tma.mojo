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

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TMADescriptor, create_tma_descriptor
from gpu.id import block_idx
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
)
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
)
from memory import UnsafePointer, stack_allocation

from utils.index import Index


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
fn kernel_copy_async_tma(descriptor: TMADescriptor):
    var shmem = stack_allocation[
        16, DType.float32, alignment=16, address_space = _GPUAddressSpace.SHARED
    ]()
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()
    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    mbarrier_arrive_expect_tx_shared(mbar, 64)
    cp_async_bulk_tensor_shared_cluster_global(
        shmem, descriptor_ptr, mbar, Index(block_idx.x * 4, block_idx.y * 4)
    )
    mbarrier_try_wait_parity_shared(mbar, 0, 10000000)

    _printf[
        "(%lu, %lu) : %g %g %g %g; %g %g %g %g; %g %g %g %g; %g %g %g %g\n"
    ](
        block_idx.x,
        block_idx.y,
        shmem[0].cast[DType.float64](),
        shmem[1].cast[DType.float64](),
        shmem[2].cast[DType.float64](),
        shmem[3].cast[DType.float64](),
        shmem[4].cast[DType.float64](),
        shmem[5].cast[DType.float64](),
        shmem[6].cast[DType.float64](),
        shmem[7].cast[DType.float64](),
        shmem[8].cast[DType.float64](),
        shmem[9].cast[DType.float64](),
        shmem[10].cast[DType.float64](),
        shmem[11].cast[DType.float64](),
        shmem[12].cast[DType.float64](),
        shmem[13].cast[DType.float64](),
        shmem[14].cast[DType.float64](),
        shmem[15].cast[DType.float64](),
    )


# CHECK-LABEL: test_tma_tile_copy
# CHECK-DAG: (0, 0) : 0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27
# CHECK-DAG: (1, 0) : 4 5 6 7; 12 13 14 15; 20 21 22 23; 28 29 30 31
# CHECK-DAG: (0, 1) : 32 33 34 35; 40 41 42 43; 48 49 50 51; 56 57 58 59
# CHECK-DAG: (1, 1) : 36 37 38 39; 44 45 46 47; 52 53 54 55; 60 61 62 63
def test_tma_tile_copy(ctx: DeviceContext):
    print("== test_tma_tile_copy")
    var gmem_host = UnsafePointer[Float32].alloc(8 * 8)
    for i in range(64):
        gmem_host[i] = i

    var gmem_dev = ctx.enqueue_create_buffer[DType.float32](8 * 8)

    ctx.enqueue_copy(gmem_dev, gmem_host)

    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_dev, (8, 8), (8, 1), (4, 4)
    )

    ctx.enqueue_function[kernel_copy_async_tma](
        descriptor, grid_dim=(2, 2), block_dim=(1)
    )
    ctx.synchronize()
    gmem_host.free()


def main():
    with DeviceContext() as ctx:
        test_tma_tile_copy(ctx)
