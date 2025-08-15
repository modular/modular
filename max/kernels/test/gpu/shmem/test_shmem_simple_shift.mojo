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

# RUN: NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

from testing import assert_equal
from shmem import (
    shmem_my_pe,
    shmem_n_pes,
    shmem_p,
    SHMEMContext,
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

    shmem_p(destination, mype, peer)


def main():
    var ctx = SHMEMContext()
    var destination = ctx.enqueue_create_buffer[DType.int32](1)

    ctx.enqueue_function[simple_shift_kernel](
        destination.unsafe_ptr(), grid_dim=1, block_dim=1
    )

    ctx.barrier_all()

    var msg = Int32(0)
    destination.enqueue_copy_to(UnsafePointer(to=msg))

    ctx.synchronize()

    var mype = shmem_my_pe()
    print("PE:", mype, "received message:", msg)
    assert_equal(msg, (mype + 1) % shmem_n_pes())
