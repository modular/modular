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

from sys import is_nvidia_gpu, has_nvidia_gpu_accelerator, CompilationTarget
from sys.ffi import external_call, c_int

from ._nvshmem_host import nvshmem_my_pe_host, nvshmem_n_pes_host
from ._nvshmem_device import (
    nvshmem_int_p,
    nvshmem_int_g,
    nvshmem_barrier_all,
    nvshmem_my_pe,
    nvshmem_n_pes,
)


fn shmem_my_pe() -> c_int:
    """Returns the PE (Processing Element) number of the calling PE.

    This function works on both device (GPU) and host contexts, automatically
    selecting the appropriate implementation based on the compilation target.

    Returns:
        The PE number (process rank) of this processing element.
    """

    @parameter
    if is_nvidia_gpu():
        return nvshmem_my_pe()
    elif has_nvidia_gpu_accelerator():
        return nvshmem_my_pe_host()
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_my_pe",]()
        return {}


fn shmem_n_pes() -> c_int:
    """Returns the total number of PEs (Processing Elements) in the job.

    This function works on both device (GPU) and host contexts, automatically
    selecting the appropriate implementation based on the compilation target.

    Returns:
        The total number of processing elements in the SHMEM job.
    """

    @parameter
    if is_nvidia_gpu():
        return nvshmem_n_pes()
    elif has_nvidia_gpu_accelerator():
        return nvshmem_n_pes_host()
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_n_pes",]()
        return {}
