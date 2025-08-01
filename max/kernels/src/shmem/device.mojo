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

from sys.ffi import external_call, c_int
from sys import is_nvidia_gpu, CompilationTarget

from ._nvshmem_device import nvshmem_int_p, nvshmem_int_g, nvshmem_barrier_all


fn shmem_int_p(destination: UnsafePointer[c_int], mype: c_int, peer: c_int):
    """Puts a single integer value to a destination PE.

    This is a point-to-point communication primitive that writes a value
    to the symmetric memory of a remote PE.

    Args:
        destination: Pointer to the destination location in symmetric memory.
        mype: The value to write to the destination.
        peer: The PE number of the destination PE.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_int_p(destination, mype, peer)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_int_p"]()


fn shmem_int_g(destination: UnsafePointer[c_int], mype: c_int, peer: c_int):
    """Gets a single integer value from a source PE.

    This is a point-to-point communication primitive that reads a value
    from the symmetric memory of a remote PE.

    Args:
        destination: Pointer where the retrieved value will be stored.
        mype: The memory location to read from on the source PE.
        peer: The PE number of the source PE.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_int_g(destination, mype, peer)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_int_g"]()


fn shmem_barrier_all():
    """Performs a barrier synchronization across all PEs from device code.

    All PEs must call this function before any PE can proceed past the barrier.
    This is the device-side barrier function for use within GPU kernels.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_barrier_all()
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_barrier_all"
        ]()
