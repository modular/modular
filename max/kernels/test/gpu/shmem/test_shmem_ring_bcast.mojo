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

# RUN: %mojo-no-debug %s

# RUN: NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

from testing import assert_equal
from shmem import *
from pathlib import cwd, Path
from os.path import dirname
from os import listdir, getenv, setenv
from subprocess import run
from sys.param_env import env_get_string
from python import Python
from gpu.host.dim import Dim
from gpu.host.device_attribute import DeviceAttribute
from sys.ffi import c_int


fn ring_bcast(
    data: UnsafePointer[c_int],
    nelem: Int,
    root: c_int,
    psync: UnsafePointer[UInt64],
):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    if mype == root:
        psync[0] = 1

    shmem_signal_wait_until(psync, SHMEM_CMP_NE, 0)

    if mype == npes - 1:
        return

    shmem_put(data, data, UInt(nelem), peer)
    shmem_fence()
    shmem_signal_op(psync, 1, SHMEM_SIGNAL_SET, peer)

    psync[0] = 0


def main():
    alias data_len = 32

    with SHMEMContext() as ctx:
        var destination = ctx.enqueue_create_buffer[DType.int32](1)

        var data = ctx.enqueue_create_buffer[DType.int32](data_len)
        var data_h = UnsafePointer[Int32].alloc(data_len)
        var psync = shmem_calloc[DType.uint64](1)

        for i in range(data_len):
            data_h[i] = shmem_my_pe() + i

        data.enqueue_copy_from(data_h)

        var root = 0
        ctx.barrier_all()
        ctx.enqueue_function_collective[ring_bcast](
            data.unsafe_ptr(),
            data_len,
            root,
            psync,
            grid_dim=1,
            block_dim=1,
        )
        ctx.barrier_all()

        data.enqueue_copy_to(data_h)
        ctx.synchronize()

        var mype = shmem_my_pe()
        for i in range(data_len):
            assert_equal(
                data_h[i],
                Int32(i),
                String(
                    "PE",
                    mype,
                    "error, data[",
                    i,
                    "] = ",
                    data_h[i],
                    " expected ",
                    i,
                ),
            )
