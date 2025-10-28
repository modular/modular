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

from pathlib import Path
from os import getenv
from sys.ffi import (
    _find_dylib,
    _get_dylib_function,
    _Global,
    _OwnedDLHandle,
    c_int,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias MPI_LIBRARY_PATHS = List[Path](
    "nvshmem_bootstrap_mpi.so.3.0.0",
    "nvshmem_bootstrap_mpi.so.3",
    "nvshmem_bootstrap_mpi.so",
)

alias MPI_LIBRARY = _Global["MPI_LIBRARY", _init_mpi_dylib]


fn _init_mpi_dylib() -> _OwnedDLHandle:
    var candidates = materialize[MPI_LIBRARY_PATHS]()

    # If provided, allow an override directory for nvshmem bootstrap libs.
    # Example:
    #   export MODULAR_NVSHMEM_LIB_DIR="/path/to/venv/lib"
    # This will try:
    #   /path/to/venv/lib/nvshmem_bootstrap_mpi.so.3[.0.0]
    var dir = Path(getenv("MODULAR_NVSHMEM_LIB_DIR"))
    if dir:
        var prefixed = List[Path](
            dir / "/nvshmem_bootstrap_mpi.so.3.0.0",
            dir / "/nvshmem_bootstrap_mpi.so.3",
            dir / "/nvshmem_bootstrap_mpi.so",
        )
        for p in candidates:
            prefixed.append(p)
        candidates = prefixed^

    return _find_dylib["MPI"](candidates)


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
        raise Error("failed to initialize MPI with error code:", result)


fn MPI_Finalize() raises:
    """Finalize MPI."""
    var result = _get_mpi_function[
        "MPI_Finalize",
        fn () -> c_int,
    ]()()
    if result != 0:
        raise Error("failed to finalize MPI with error code:", result)


fn MPI_Comm_rank(comm: MPIComm, rank: UnsafePointer[c_int]) raises -> c_int:
    """Get the rank of the current process in the communicator."""
    return _get_mpi_function[
        "MPI_Comm_rank",
        fn (MPIComm, UnsafePointer[c_int]) -> c_int,
    ]()(comm, rank)


fn MPI_Comm_size(comm: MPIComm, size: UnsafePointer[c_int]) raises -> c_int:
    """Get the size of the communicator."""
    return _get_mpi_function[
        "MPI_Comm_size",
        fn (MPIComm, UnsafePointer[c_int]) -> c_int,
    ]()(comm, size)


fn get_mpi_comm_world() raises -> MPIComm:
    """Get the MPI_COMM_WORLD communicator."""
    var handle = MPI_LIBRARY.get_or_create_ptr()[].handle()
    var comm_world_ptr = handle.get_symbol[OpaquePointer](
        cstr_name="ompi_mpi_comm_world".unsafe_cstr_ptr()
    )
    return comm_world_ptr
