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
# REQUIRES: NVIDIA-GPU

# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n 1 %t

from testing import assert_equal
from shmem import (
    shmem_my_pe,
    shmem_n_pes,
    shmem_int_p,
    DeviceContextSHMEM,
)
from pathlib import cwd, Path
from os.path import dirname
from os import listdir, getenv, setenv
from subprocess import run
from sys.param_env import env_get_string
from python import Python


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
