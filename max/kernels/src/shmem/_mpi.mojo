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

from memory import (
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer as UnsafePointer,
)
from pathlib import Path
from os import getenv, abort
from sys.ffi import (
    _find_dylib,
    _get_dylib_function,
    _Global,
    OwnedDLHandle,
    c_int,
    RTLD,
)
from sys.info import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias MPI_LIBRARY = _Global["MPI_LIBRARY", _init_mpi_dylib]


fn mpi_lib_name() -> String:
    @parameter
    if has_nvidia_gpu_accelerator():
        return "nvshmem_bootstrap_mpi.so.3"
    else:
        return "libmpi.so"


fn _init_mpi_dylib() -> OwnedDLHandle:
    var lib = mpi_lib_name()
    # If provided, allow an override directory for nvshmem bootstrap libs.
    # Example:
    #   export MODULAR_SHMEM_LIB_DIR="/path/to/venv/lib"
    # will dlopen the library from:
    #   /path/to/venv/lib/libmpi.so
    if dir_name := getenv("MODULAR_SHMEM_LIB_DIR"):
        lib = String(Path(dir_name) / lib)

    var flags = RTLD.NOW | RTLD.GLOBAL

    # AMD interaction with libmpi needs the handle to stay alive after MPI_Finalize
    @parameter
    if has_amd_gpu_accelerator():
        flags = flags | RTLD.NODELETE

    try:
        return OwnedDLHandle(path=lib, flags=flags)
    except e:
        abort(String("failed to load MPI library: ", e))
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
fn _get_mpi_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() raises -> result_type:
    return _get_dylib_function[
        MPI_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types and constants
# ===-----------------------------------------------------------------------===#

alias MPIComm = UnsafePointer[OpaquePointer]

alias MPI_THREAD_SINGLE = 0
alias MPI_THREAD_FUNNELED = 1
alias MPI_THREAD_SERIALIZED = 2
alias MPI_THREAD_MULTIPLE = 3

# ===-----------------------------------------------------------------------===#
# Function bindings
# ===-----------------------------------------------------------------------===#


fn MPI_Init(mut argc: Int, mut argv: VariadicList[StaticString]) raises:
    """Initialize MPI."""
    var result = _get_mpi_function[
        "MPI_Init",
        fn (
            UnsafePointer[Int], UnsafePointer[VariadicList[StaticString]]
        ) -> c_int,
    ]()(UnsafePointer(to=argc), UnsafePointer(to=argv))
    if result != 0:
        raise Error("failed to MPI_Init with error code:", result)


fn MPI_Init_thread(
    mut argc: Int,
    mut argv: VariadicList[StaticString],
    required: c_int,
    provided: UnsafePointer[c_int],
) raises:
    """Initialize MPI."""
    var result = _get_mpi_function[
        "MPI_Init_thread",
        fn (
            UnsafePointer[Int],
            UnsafePointer[VariadicList[StaticString]],
            c_int,
            UnsafePointer[c_int],
        ) -> c_int,
    ]()(UnsafePointer(to=argc), UnsafePointer(to=argv), required, provided)
    if result != 0:
        raise Error("failed to MPI_Init_thread with error code:", result)


fn MPI_Finalize() raises:
    """Finalize MPI."""
    var result = _get_mpi_function[
        "MPI_Finalize",
        fn () -> c_int,
    ]()()
    if result != 0:
        raise Error("failed to finalize MPI with error code:", result)


fn MPI_Comm_split(
    comm: MPIComm, color: c_int, key: c_int, newcomm: UnsafePointer[MPIComm]
) raises:
    """Split a communicator into multiple subcommunicators."""
    var result = _get_mpi_function[
        "MPI_Comm_split",
        fn (MPIComm, c_int, c_int, UnsafePointer[MPIComm]) -> c_int,
    ]()(comm, color, key, newcomm)
    if result != 0:
        raise Error("failed to MPI_Comm_split with error code:", result)


fn MPI_Comm_rank(comm: MPIComm, rank: UnsafePointer[c_int]) raises:
    """Get the rank of the current process in the communicator."""
    var result = _get_mpi_function[
        "MPI_Comm_rank",
        fn (MPIComm, UnsafePointer[c_int]) -> c_int,
    ]()(comm, rank)
    if result != 0:
        raise Error("failed to get MPI_Comm_rank with error code:", result)


fn MPI_Comm_size(comm: MPIComm, size: UnsafePointer[c_int]) raises:
    """Get the size of the communicator."""
    var result = _get_mpi_function[
        "MPI_Comm_size",
        fn (MPIComm, UnsafePointer[c_int]) -> c_int,
    ]()(comm, size)
    if result != 0:
        raise Error("failed to get MPI_Comm_size with error code:", result)


fn get_mpi_comm_world() raises -> MPIComm:
    """Get the MPI_COMM_WORLD communicator."""
    var handle = MPI_LIBRARY.get_or_create_ptr()[].handle()
    var comm_world_ptr = handle.get_symbol[OpaquePointer](
        cstr_name="ompi_mpi_comm_world".unsafe_cstr_ptr()
    )
    return comm_world_ptr
