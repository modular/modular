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
from gpu.host._nvidia_cuda import CUDA, CUstream, CUmodule
from gpu.memory import _GPUAddressSpace as AddressSpace
from ._mpi import MPIComm
from sys import sizeof
from os import abort
from sys.ffi import (
    _find_dylib,
    _Global,
    _OwnedDLHandle,
    external_call,
    _get_dylib_function,
    c_int,
    c_size_t,
)
from collections.string.string_slice import _get_kgen_string
from sys.param_env import env_get_string

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias NVSHMEM_LIBRARY_PATHS = List[Path](
    "libnvshmem_host.so.3.3.9",
    "libnvshmem_host.so.3",
    "libnvshmem_host.so",
    "/usr/lib/x86_64-linux-gnu/nvshmem/12/libnvshmem_host.so.3.3.9",
    "/usr/lib/x86_64-linux-gnu/nvshmem/12/libnvshmem_host.so.3",
    "/usr/lib/x86_64-linux-gnu/nvshmem/12/libnvshmem_host.so",
)

alias NVSHMEM_LIBRARY = _Global[
    "NVSHMEM_LIBRARY", _OwnedDLHandle, _init_nvshmem_dylib
]


fn _init_nvshmem_dylib() -> _OwnedDLHandle:
    return _find_dylib["NVSHMEM"](NVSHMEM_LIBRARY_PATHS)


@always_inline
fn _get_nvshmem_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _get_dylib_function[
        NVSHMEM_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types and constants
# ===-----------------------------------------------------------------------===#

alias NVSHMEM_SUCCESS = 0
alias NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1
alias NVSHMEMX_TEAM_NODE = 2

# NVSHMEM status constants (used for internal status checking)
alias NVSHMEM_STATUS_NOT_INITIALIZED = 0
alias NVSHMEM_STATUS_IS_BOOTSTRAPPED = 1
alias NVSHMEM_STATUS_IS_INITIALIZED = 2

# NVSHMEM error constants
alias NVSHMEMX_ERROR_INTERNAL = 1

# NVSHMEM thread support constants
alias NVSHMEM_THREAD_SINGLE: c_int = 0
alias NVSHMEM_THREAD_FUNNELED: c_int = 1
alias NVSHMEM_THREAD_SERIALIZED: c_int = 2
alias NVSHMEM_THREAD_MULTIPLE: c_int = 3
alias NVSHMEM_THREAD_TYPE_SENTINEL: c_int = c_int.MAX


# Structs
struct NVSHMEMXInitAttr:
    var version: c_int
    var mpi_comm: UnsafePointer[MPIComm]
    var args: NVSHMEMXInitArgs

    fn __init__(out self, mpi_comm: UnsafePointer[MPIComm]):
        constrained[
            sizeof[Self]() == 144, "NVSHMEMXInitAttr must be 144 bytes"
        ]()
        self.version = (1 << 16) + sizeof[NVSHMEMXInitAttr]()
        self.mpi_comm = mpi_comm
        self.args = NVSHMEMXInitArgs()


struct NVSHMEMXInitArgs:
    var version: c_int
    var uid_args: NVSHMEMXUniqueIDArgs
    var content: InlineArray[Byte, 96]

    fn __init__(out self):
        constrained[
            sizeof[Self]() == 128, "NVSHMEMXInitArgs must be 128 bytes"
        ]()
        self.version = (1 << 16) + sizeof[NVSHMEMXInitArgs]()
        self.uid_args = NVSHMEMXUniqueIDArgs()
        self.content = InlineArray[Byte, 96](fill=0)


struct NVSHMEMXUniqueIDArgs:
    var version: c_int
    var id: UnsafePointer[NVSHMEMXUniqueID]
    var myrank: c_int
    var nranks: c_int

    fn __init__(out self):
        constrained[
            sizeof[Self]() == 24, "NVSHMEMXUniqueIDArgs must be 24 bytes"
        ]()
        self.version = (1 << 16) + sizeof[NVSHMEMXUniqueIDArgs]()
        self.id = UnsafePointer[NVSHMEMXUniqueID]()
        self.myrank = 0
        self.nranks = 0


struct NVSHMEMXUniqueID:
    var version: c_int
    var internal: InlineArray[Byte, 124]

    fn __init__(out self):
        constrained[
            sizeof[Self]() == 128, "nvshmemx_uniqueid_t must be 128 bytes"
        ]()
        self.version = (1 << 16) + sizeof[NVSHMEMXUniqueID]()
        self.internal = InlineArray[Byte, 124](fill=0)


@register_passable
struct NVSHMEMIVersion:
    var major: c_int
    var minor: c_int
    var patch: c_int

    fn __init__(out self):
        self.major = 3
        self.minor = 3
        self.patch = 9


# Device state structure for simplified state management
struct NvshmemiDeviceOnlyState:
    var is_initialized: Bool
    var cuda_device_id: c_int

    fn __init__(out self, device_id: c_int = -1):
        self.is_initialized = False
        self.cuda_device_id = device_id


# ===-----------------------------------------------------------------------===#
# Function bindings
# ===-----------------------------------------------------------------------===#


fn nvshmemx_hostlib_init_attr(
    flags: UInt32,
    attr: UnsafePointer[NVSHMEMXInitAttr],
) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_hostlib_init_attr",
        fn (UInt32, UnsafePointer[NVSHMEMXInitAttr]) -> c_int,
    ]()(flags, attr)


fn nvshmemid_hostlib_init_attr(
    requested: c_int,
    provided: UnsafePointer[c_int],
    bootstrap_flags: UInt32,
    attr: UnsafePointer[NVSHMEMXInitAttr],
    nvshmem_device_lib_version: NVSHMEMIVersion,
    cb: OpaquePointer,
) -> c_int:
    return _get_nvshmem_function[
        "nvshmemid_hostlib_init_attr",
        fn (
            c_int,
            UnsafePointer[c_int],
            UInt32,
            UnsafePointer[NVSHMEMXInitAttr],
            NVSHMEMIVersion,
            OpaquePointer,
        ) -> c_int,
    ]()(
        requested,
        provided,
        bootstrap_flags,
        attr,
        nvshmem_device_lib_version,
        cb,
    )


fn nvshmemx_cumodule_init(module: CUmodule) -> c_int:
    """Initialize CUDA module/device state for NVSHMEM.

    Args:
        module: The CUmodule handle to register with NVSHMEM.

    Returns:
        Status code (0 = success).
    """
    return _get_nvshmem_function[
        "nvshmemx_cumodule_init",
        fn (CUmodule) -> c_int,
    ]()(module)


fn nvshmemx_team_init() -> c_int:
    """Initialize NVSHMEM teams."""
    return _get_nvshmem_function[
        "nvshmemx_team_init",
        fn () -> c_int,
    ]()()


fn nvshmem_sync_all():
    """Synchronize all PEs - final synchronization step."""
    _get_nvshmem_function[
        "nvshmem_sync_all",
        fn () -> NoneType,
    ]()()


fn nvshmemx_init_status() -> c_int:
    """Get the current NVSHMEM public initialization status."""
    return _get_nvshmem_function[
        "nvshmemx_init_status",
        fn () -> c_int,
    ]()()


fn nvshmemid_init_status() -> c_int:
    """Get the current NVSHMEM initialization status.

    Returns:
        Current status: NVSHMEM_STATUS_NOT_INITIALIZED,
        NVSHMEM_STATUS_IS_BOOTSTRAPPED, or NVSHMEM_STATUS_IS_INITIALIZED.
    """
    return _get_nvshmem_function[
        "nvshmemid_init_status",
        fn () -> c_int,
    ]()()


# nvshmem inline header function: include/host/nvshmemx_api.h:55
fn nvshmemx_init_attr(
    flags: UInt32, attributes: UnsafePointer[NVSHMEMXInitAttr]
) -> c_int:
    """Initialize NVSHMEM with attributes."""
    var status = nvshmemx_hostlib_init_attr(flags, attributes)
    if status != 0:
        print("ERROR: Host library initialization failed with status", status)
    return status


fn nvshmemx_init_init_attr_ver_only(attr: UnsafePointer[NVSHMEMXInitAttr]):
    attr[].version = (1 << 16) + sizeof[NVSHMEMXInitAttr]()
    attr[].args.version = (1 << 16) + sizeof[NVSHMEMXInitArgs]()
    attr[].args.uid_args.version = (1 << 16) + sizeof[NVSHMEMXUniqueIDArgs]()


fn nvshmemx_hostlib_finalize():
    """Finalize NVSHMEM hostlib."""
    _get_nvshmem_function[
        "nvshmemx_hostlib_finalize",
        fn () -> NoneType,
    ]()()


fn nvshmem_team_my_pe(team: c_int) -> c_int:
    """Get the PE ID within a team."""
    return _get_nvshmem_function[
        "nvshmem_team_my_pe",
        fn (c_int) -> c_int,
    ]()(team)


fn nvshmem_my_pe_host() -> c_int:
    """Get the PE ID of the calling PE."""
    return _get_nvshmem_function[
        "nvshmem_my_pe",
        fn () -> c_int,
    ]()()


fn nvshmem_n_pes_host() -> c_int:
    """Get the total number of PEs."""
    return _get_nvshmem_function[
        "nvshmem_n_pes",
        fn () -> c_int,
    ]()()


fn nvshmem_malloc[dtype: DType](size: c_size_t) -> UnsafePointer[Scalar[dtype]]:
    """Allocate symmetric memory."""
    return _get_nvshmem_function[
        "nvshmem_malloc",
        fn (c_size_t) -> UnsafePointer[Scalar[dtype]],
    ]()(size)


fn nvshmem_calloc[
    dtype: DType
](count: c_size_t, size: c_size_t) -> UnsafePointer[Scalar[dtype]]:
    """Allocate symmetric memory while zero initializing it."""
    return _get_nvshmem_function[
        "nvshmem_calloc",
        fn (c_size_t, c_size_t) -> UnsafePointer[Scalar[dtype]],
    ]()(count, size)


fn nvshmem_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    """Free symmetric memory."""
    _get_nvshmem_function[
        "nvshmem_free",
        fn (UnsafePointer[Scalar[dtype]]) -> NoneType,
    ]()(ptr)


fn nvshmem_int_p(dest: OpaquePointer, value: c_int, pe: c_int):
    """Put a single integer to remote PE."""
    _get_nvshmem_function[
        "nvshmem_int_p",
        fn (OpaquePointer, c_int, c_int) -> NoneType,
    ]()(dest, value, pe)


fn nvshmem_barrier_all():
    """Synchronize all PEs with barrier."""
    _get_nvshmem_function[
        "nvshmem_barrier_all",
        fn () -> NoneType,
    ]()()


fn nvshmemx_barrier_all_on_stream(stream: CUstream):
    """Synchronize all PEs."""
    _get_nvshmem_function[
        "nvshmemx_barrier_all_on_stream",
        fn (CUstream) -> NoneType,
    ]()(stream)
