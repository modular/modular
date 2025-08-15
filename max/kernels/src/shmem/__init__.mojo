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
from shmem import shmem_my_pe, shmem_n_pes, shmem_p, SHMEMContext


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    shmem_p(destination, mype, peer)


def main():
    with SHMEMContext() as ctx:
        var destination = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.enqueue_function[simple_shift_kernel](
            destination.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        ctx.barrier_all()

        var msg = Int32(0)
        destination.enqueue_copy_to(UnsafePointer(to=msg))

        ctx.synchronize()

        print("PE:", ctx.my_pe(), "received message:", msg)

        assert_equal(msg, (ctx.my_pe() + 1) % ctx.n_pes())
```
"""
from .shmem_buffer import SHMEMBuffer
from .shmem_context import SHMEMContext
from .shmem_api import (
    shmem_barrier_all_on_stream,
    shmem_barrier_all,
    shmem_calloc,
    shmem_fence,
    shmem_finalize,
    shmem_free,
    shmem_g,
    shmem_init,
    shmem_malloc,
    shmem_module_init,
    shmem_module_init,
    shmem_my_pe,
    shmem_n_pes,
    shmem_p,
    shmem_put,
    shmem_put_signal_nbi,
    shmem_signal_op,
    shmem_signal_wait_until,
    shmem_team_my_pe,
    SHMEM_CMP_EQ,
    SHMEM_CMP_GE,
    SHMEM_CMP_GT,
    SHMEM_CMP_LE,
    SHMEM_CMP_LT,
    SHMEM_CMP_NE,
    SHMEM_CMP_SENTINEL,
    SHMEM_SIGNAL_ADD,
    SHMEM_SIGNAL_SET,
    SHMEM_TEAM_INVALID,
    SHMEM_TEAM_NODE,
    SHMEM_TEAM_SHARED,
    SHMEM_TEAM_WORLD,
    SHMEMScope,
)
