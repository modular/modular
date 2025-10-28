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
"""Shows how to initialize SHMEM from Mojo, this compiles a program that is
managed via an mpi runner such as mpirun. mpirun takes care of spawning one
process per GPU and running the binary in parallel for each GPU.

See `test_shmem_gpu_per_thread.mojo` for how you can launch one GPU per thread,
if running on a single node you can run the compiled binary directly without
mpirun.
"""

# RUN: NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

from gpu.host import DeviceBuffer, DeviceContext
from os.path import dirname
from pathlib import Path
from shmem import *
from sys.param_env import env_get_string
from testing import assert_equal


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes
    print("GPU mype:", mype, "peer:", peer)

    # Send this PE ID to a peer
    shmem_p(destination, mype, peer)


def main():
    # Initializes SHMEM/MPI and finalizes at the end of the scope
    with SHMEMContext() as ctx:
        # Set up buffers to test devices are communicating with the correct IDs
        var target_device = ctx.enqueue_create_buffer[DType.int32](1)
        var target_host = ctx.enqueue_create_host_buffer[DType.int32](1)

        # SHMEMContext takes care of initializing device state into
        # `simple_shift_kernel` constant memory
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
