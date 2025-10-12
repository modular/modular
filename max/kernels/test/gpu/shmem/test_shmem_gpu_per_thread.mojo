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
"""Shows how to initialize SHMEM from Mojo. This uses `shmem_launch` to spawn
one thread for each GPU, and takes care of mpi initialization and
deinitialization. If running on a single node you can run this compiled binary
directly without mpirun.

See `test_shmem_gpu_per_process.mojo` for how you can launch one GPU per process
using mpirun.
"""

# RUN: %mojo-build %s -o %t
# RUN: %t

from testing import assert_equal
from shmem import *
from shmem._nvshmem import *
from pathlib import cwd, Path
from os.path import dirname
from pathlib import Path, cwd
from subprocess import run
from sys.param_env import env_get_string

from python import Python
from shmem import SHMEMContext, shmem_my_pe, shmem_n_pes, shmem_p
from testing import assert_equal


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes
    print("GPU mype:", mype, "peer:", peer)

    # Send this PE ID to a peer
    shmem_p(destination, mype, peer)


# Must have this signature to use `shmem_launch`
def simple_shift(ctx: SHMEMContext):
    # Set up buffers to test devices are communicating with the correct IDs
    var target_device = ctx.enqueue_create_buffer[DType.int32](1)
    var target_host = ctx.enqueue_create_host_buffer[DType.int32](1)

    ctx.enqueue_function[simple_shift_kernel](
        target_device, grid_dim=1, block_dim=1
    )
    ctx.barrier_all()

    target_device.enqueue_copy_to(target_host)
    ctx.synchronize()

    # Get the mype and npes across all nodes to test communication
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + npes - 1) % npes
    assert_equal(target_host[0], peer)


def main():
    shmem_launch[simple_shift]()
