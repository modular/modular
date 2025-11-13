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
# RUN: NUM_GPUS=$(rocm-smi --showid --json | jq 'keys | length')
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

from shmem import *
from testing import assert_equal


# TODO: this is just testing host-side comms work with rocm enabled mpirun.
# once device-side calls and shmem tests are working, remove this test.
def main():
    shmem_init()

    var mem = shmem_malloc[DType.int32](1)

    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    # Send this PE ID to a peer
    shmem_p(mem, mype, peer)
    shmem_barrier_all()

    var expected = (mype + npes - 1) % npes
    assert_equal(mem[0], expected)

    shmem_finalize()
    MPI_Finalize()
