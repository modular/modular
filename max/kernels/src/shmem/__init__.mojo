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
"""This module implements a subset of OpenSHMEM functionality for Mojo. It will
abstract over both NVSHMEM and ROCSHMEM, exposing a similar API to DeviceContext
with a symmetric heap that is accessible by inter-node and intra-node GPUs.

```mojo
from testing import assert_equal
from shmem import shmem_my_pe, shmem_n_pes, shmem_int_p, DeviceContextSHMEM


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    shmem_int_p(destination, mype, peer)


def main():
    with DeviceContextSHMEM() as shmem:
        var destination = shmem.enqueue_create_buffer[DType.int32](1)
        shmem.enqueue_function[simple_shift_kernel](
            destination.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        shmem.barrier_all()

        var msg = Int32(0)
        destination.enqueue_copy_to(UnsafePointer(to=msg))

        print("PE:", shmem.my_pe(), "received message:", msg)
        shmem.synchronize()

        assert_equal(msg, (shmem.my_pe() + 1) % shmem.n_pes())
```
"""
from .buffer import SHMEMBuffer
from .host import (
    DeviceContextSHMEM,
    shmem_init,
    shmem_malloc,
    shmem_calloc,
    shmem_module_init,
    shmem_barrier_all_on_stream,
    shmem_module_init,
    shmem_free,
    shmem_finalize,
    shmem_team_my_pe,
    SHMEM_TEAM_MODE,
)
from sys.ffi import c_int
from .device_and_host import shmem_my_pe, shmem_n_pes
from .device import shmem_int_p, shmem_int_g, shmem_barrier_all
