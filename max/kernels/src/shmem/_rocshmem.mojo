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
from collections.string.string_slice import get_static_string
from memory import LegacyUnsafePointer as UnsafePointer
from os import abort, getenv
from pathlib import Path
from sys import argv, size_of
from sys.ffi import (
    _find_dylib,
    _get_dylib_function,
    _Global,
    OwnedDLHandle,
    c_int,
    c_uint,
    c_size_t,
    external_call,
    RTLD,
)
from sys.info import CompilationTarget, is_nvidia_gpu, is_amd_gpu

from gpu.host import DeviceContext
from gpu.host._amdgpu_hip import hipStream_t
from gpu.host._nvidia_cuda import CUmodule, CUstream

from ._mpi import (
    MPI_Comm_rank,
    MPI_Init,
    MPIComm,
    get_mpi_comm_world,
    MPI_THREAD_MULTIPLE,
    MPI_Init_thread,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


@register_passable
struct ROCSHEMIVersion:
    var major: c_int
    var minor: c_int
    var patch: c_int

    fn __init__(out self):
        self.major = 3
        self.minor = 4
        self.patch = 5


comptime ROCSHMEM_LIBRARY = _Global["ROCSHMEM_LIBRARY", _init_rocshmem_dylib]


fn _init_rocshmem_dylib() -> OwnedDLHandle:
    var lib = "librocshmem.so"
    # If provided, allow an override directory for nvshmem bootstrap libs.
    # Example:
    #   export MODULAR_SHMEM_LIB_DIR="/path/to/venv/lib"
    # will dlopen the library from:
    #   /path/to/venv/lib/librocshmem.so
    if dir_name := getenv("MODULAR_SHMEM_LIB_DIR"):
        lib = String(Path(dir_name) / lib)
    try:
        return OwnedDLHandle(
            path=lib,
            flags=RTLD.NOW | RTLD.GLOBAL | RTLD.NODELETE,
        )
    except e:
        abort(String("failed to load ROCSHMEM library: ", e))
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
fn _get_rocshmem_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    try:
        return _get_dylib_function[
            ROCSHMEM_LIBRARY(),
            func_name,
            result_type,
        ]()
    except e:
        return abort[result_type](String(e))


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#

# TODO: verify constants and structs in https://github.com/modular/rocSHMEM
comptime rocshmem_team_id_t = Int32

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime ROCSHMEM_SUCCESS = 0

comptime ROCSHMEM_INIT_WITH_MPI_COMM = 1 << 1

comptime CHANNEL_BUF_SIZE: c_int = 1 << 22
comptime CHANNEL_BUF_SIZE_LOG: c_int = 22
comptime CHANNEL_ENTRY_BYTES: c_int = 8

comptime ROCSHMEM_ERROR_INTERNAL = 1
comptime ROCSHMEM_MAX_NAME_LEN: c_int = 256

comptime ROCSHMEM_THREAD_SINGLE: c_int = 0
comptime ROCSHMEM_THREAD_FUNNELED: c_int = 1
comptime ROCSHMEM_THREAD_SERIALIZED: c_int = 2
comptime ROCSHMEM_THREAD_MULTIPLE: c_int = 3
comptime ROCSHMEM_THREAD_TYPE_SENTINEL: c_int = c_int.MAX

comptime ROCSHMEM_CMP_EQ: c_int = 0
comptime ROCSHMEM_CMP_NE: c_int = 1
comptime ROCSHMEM_CMP_GT: c_int = 2
comptime ROCSHMEM_CMP_LE: c_int = 3
comptime ROCSHMEM_CMP_LT: c_int = 4
comptime ROCSHMEM_CMP_GE: c_int = 5
comptime ROCSHMEM_CMP_SENTINEL: c_int = c_int.MAX

comptime PROXY_GLOBAL_EXIT_INIT: c_int = 1
comptime PROXY_GLOBAL_EXIT_REQUESTED: c_int = 2
comptime PROXY_GLOBAL_EXIT_FINISHED: c_int = 3
comptime PROXY_GLOBAL_EXIT_MAX_STATE: c_int = c_int.MAX

comptime PROXY_DMA_REQ_BYTES: c_int = 32
comptime PROXY_AMO_REQ_BYTES: c_int = 40
comptime PROXY_INLINE_REQ_BYTES: c_int = 24

comptime ROCSHMEM_STATUS_NOT_INITIALIZED: c_int = 0
comptime ROCSHMEM_STATUS_IS_BOOTSTRAPPED: c_int = 1
comptime ROCSHMEM_STATUS_IS_INITIALIZED: c_int = 2
comptime ROCSHMEM_STATUS_LIMITED_MPG: c_int = 4
comptime ROCSHMEM_STATUS_FULL_MPG: c_int = 5
comptime ROCSHMEM_STATUS_INVALID: c_int = c_int.MAX

comptime ROCSHMEM_SIGNAL_SET: c_int = 9
comptime ROCSHMEM_SIGNAL_ADD: c_int = 10

comptime ROCSHMEM_TEAM_INVALID: rocshmem_team_id_t = -1
comptime ROCSHMEM_TEAM_WORLD: rocshmem_team_id_t = 0
comptime ROCSHMEM_TEAM_WORLD_INDEX: rocshmem_team_id_t = 0
comptime ROCSHMEM_TEAM_SHARED: rocshmem_team_id_t = 1
comptime ROCSHMEM_TEAM_SHARED_INDEX: rocshmem_team_id_t = 1
comptime ROCSHMEM_TEAM_NODE: rocshmem_team_id_t = 2
comptime ROCSHMEM_TEAM_NODE_INDEX: rocshmem_team_id_t = 2
comptime ROCSHMEM_TEAM_SAME_MYPE_NODE: rocshmem_team_id_t = 3
comptime ROCSHMEM_TEAM_SAME_MYPE_NODE_INDEX: rocshmem_team_id_t = 3
comptime ROCSHMEMI_TEAM_SAME_GPU: rocshmem_team_id_t = 4
comptime ROCSHMEM_TEAM_SAME_GPU_INDEX: rocshmem_team_id_t = 4
comptime ROCSHMEMI_TEAM_GPU_LEADERS: rocshmem_team_id_t = 5
comptime ROCSHMEM_TEAM_GPU_LEADERS_INDEX: rocshmem_team_id_t = 5
comptime ROCSHMEM_TEAMS_MIN: rocshmem_team_id_t = 6
comptime ROCSHMEM_TEAM_INDEX_MAX: rocshmem_team_id_t = rocshmem_team_id_t.MAX


# Structs
struct ROCSHMEMInitAttr:
    var version: c_int
    var mpi_comm: UnsafePointer[MPIComm]
    var args: ROCSHMEMInitArgs

    fn __init__(out self, mpi_comm: UnsafePointer[MPIComm]):
        constrained[
            size_of[Self]() == 144, "ROCSHMEMInitAttr must be 144 bytes"
        ]()
        self.version = (1 << 16) + size_of[ROCSHMEMInitAttr]()
        self.mpi_comm = mpi_comm
        self.args = ROCSHMEMInitArgs()


struct ROCSHMEMInitArgs:
    var version: c_int
    var uid_args: ROCSHMEMUniqueIDArgs
    var content: InlineArray[Byte, 96]

    fn __init__(out self):
        constrained[
            size_of[Self]() == 128, "ROCSHMEMInitArgs must be 128 bytes"
        ]()
        self.version = (1 << 16) + size_of[ROCSHMEMInitArgs]()
        self.uid_args = ROCSHMEMUniqueIDArgs()
        self.content = InlineArray[Byte, 96](fill=0)


struct ROCSHMEMUniqueIDArgs:
    var version: c_int
    var id: UnsafePointer[ROCSHMEMUniqueID]
    var myrank: c_int
    var nranks: c_int

    fn __init__(out self):
        constrained[
            size_of[Self]() == 24, "ROCSHMEMUniqueIDArgs must be 24 bytes"
        ]()
        self.version = (1 << 16) + size_of[ROCSHMEMUniqueIDArgs]()
        self.id = UnsafePointer[ROCSHMEMUniqueID]()
        self.myrank = 0
        self.nranks = 0


struct ROCSHMEMUniqueID:
    var version: c_int
    var internal: InlineArray[Byte, 124]

    fn __init__(out self):
        constrained[
            size_of[Self]() == 128, "rocshmem_uniqueid_t must be 128 bytes"
        ]()
        self.version = (1 << 16) + size_of[ROCSHMEMUniqueID]()
        self.internal = InlineArray[Byte, 124](fill=0)


fn _dtype_to_rocshmem_type[
    prefix: StaticString,
    dtype: DType,
    suffix: StaticString,
]() -> StaticString:
    """
    Returns the ROCSHMEM name for the given dtype surrounded by the given prefix
    and suffix, for calling the correct symbol on the device-side bitcode.


    c_name               rocshmem_name  bitwidth
    -------------------------------------------
    float                float         32
    double               double        64
    half                 half          16
    char                 char          8
    signed char          schar         8
    short                short         16
    int                  int           32
    long                 long          64
    long long            longlong      64
    unsigned char        uchar         8
    unsigned short       ushort        16
    unsigned int         uint          32
    unsigned long        ulong         64
    unsigned long long   ulonglong     64

    Unsuported:
    int8_t               int8          8
    int16_t              int16         16
    int32_t              int32         32
    int64_t              int64         64
    uint8_t              uint8         8
    uint16_t             uint16        16
    uint32_t             uint32        32
    uint64_t             uint64        64
    size_t               size          64
    ptrdiff_t            ptrdiff       64
    """

    @parameter
    if dtype is DType.float16:
        return get_static_string[prefix, "half", suffix]()
    elif dtype is DType.float32:
        return get_static_string[prefix, "float", suffix]()
    elif dtype is DType.float64:
        return get_static_string[prefix, "double", suffix]()
    elif dtype is DType.int8:
        return get_static_string[prefix, "schar", suffix]()
    elif dtype is DType.uint8:
        return get_static_string[prefix, "char", suffix]()
    elif dtype is DType.int16:
        return get_static_string[prefix, "short", suffix]()
    elif dtype is DType.uint16:
        return get_static_string[prefix, "ushort", suffix]()
    elif dtype is DType.int32:
        return get_static_string[prefix, "int", suffix]()
    elif dtype is DType.uint32:
        return get_static_string[prefix, "uint", suffix]()
    elif dtype is DType.int64:
        return get_static_string[prefix, "long", suffix]()
    elif dtype is DType.uint64:
        return get_static_string[prefix, "ulong", suffix]()
    elif dtype is DType.int:
        return get_static_string[prefix, "longlong", suffix]()
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString, operation="_dtype_to_rocshmem_type"
        ]()


# ===-----------------------------------------------------------------------===#
# 1: Library Setup, Exit, and Query
# ===-----------------------------------------------------------------------===#


fn _rocshmem_init() raises:
    _get_rocshmem_function[
        "rocshmem_init",
        fn () -> NoneType,
    ]()()


fn rocshmem_init() raises:
    var _argv = argv()
    var argc = len(_argv)

    var world_rank, world_nranks = c_int(0), c_int(0)
    var provided = c_int(0)

    MPI_Init_thread(
        argc, _argv, MPI_THREAD_MULTIPLE, UnsafePointer(to=provided)
    )
    if provided != MPI_THREAD_MULTIPLE:
        raise Error("MPI_THREAD_MULTIPLE support disabled.")

    _rocshmem_init()


fn rocshmem_init_thread(
    ctx: DeviceContext, number_of_devices_node: Int = -1
) raises:
    raise Error("shmem_init_thread is not implemented for ROCSHMEM")


fn rocshmem_init_attr(
    flags: UInt32,
    attr: UnsafePointer[ROCSHMEMInitAttr],
) -> c_int:
    return _get_rocshmem_function[
        "rocshmem_init_attr",
        fn (UInt32, UnsafePointer[ROCSHMEMInitAttr]) -> c_int,
    ]()(flags, attr)


fn rocshmem_get_uniqueid(uid: UnsafePointer[ROCSHMEMUniqueID]) -> c_int:
    return _get_rocshmem_function[
        "rocshmem_get_uniqueid",
        fn (UnsafePointer[ROCSHMEMUniqueID]) -> c_int,
    ]()(uid)


fn rocshmem_finalize():
    _get_rocshmem_function[
        "rocshmem_finalize",
        fn () -> NoneType,
    ]()()


fn rocshmem_my_pe() -> c_int:
    @parameter
    if is_amd_gpu():
        return external_call["rocshmem_my_pe", c_int]()
    else:
        return _get_rocshmem_function[
            "rocshmem_my_pe",
            fn () -> c_int,
        ]()()


fn rocshmem_n_pes() -> c_int:
    @parameter
    if is_nvidia_gpu():
        return external_call["nvshmem_n_pes", c_int]()
    elif is_amd_gpu():
        return external_call["rocshmem_n_pes", c_int]()
    else:
        return _get_rocshmem_function[
            "rocshmem_n_pes",
            fn () -> c_int,
        ]()()


# ===----------------------------------------------------------------------=== #
# 3: Memory Management
# ===----------------------------------------------------------------------=== #


fn rocshmem_malloc[
    dtype: DType
](size: c_size_t) -> UnsafePointer[Scalar[dtype]]:
    return _get_rocshmem_function[
        "rocshmem_malloc",
        fn (c_size_t) -> UnsafePointer[Scalar[dtype]],
    ]()(size)


fn rocshmem_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    _get_rocshmem_function[
        "rocshmem_free",
        fn (UnsafePointer[Scalar[dtype]]) -> NoneType,
    ]()(ptr)


# ===----------------------------------------------------------------------=== #
# 4: Team Management
# ===----------------------------------------------------------------------=== #


fn rocshmem_team_my_pe(team: c_int) -> c_int:
    return _get_rocshmem_function[
        "rocshmem_team_my_pe",
        fn (c_int) -> c_int,
    ]()(team)


# ===----------------------------------------------------------------------=== #
# 6: Remote Memory Access (RMA)
# ===----------------------------------------------------------------------=== #


fn rocshmem_put[
    dtype: DType, //,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_put"]()

    @parameter
    if is_amd_gpu():
        external_call[symbol, NoneType](dest, source, nelems, pe)
    else:
        _get_rocshmem_function[
            symbol,
            fn (
                UnsafePointer[Scalar[dtype]],
                UnsafePointer[Scalar[dtype]],
                c_size_t,
                c_int,
            ) -> NoneType,
        ]()(dest, source, nelems, pe)


fn rocshmem_put_nbi[
    dtype: DType, //,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_put_nbi"]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn rocshmem_p[
    dtype: DType
](dest: UnsafePointer[Scalar[dtype]], value: Scalar[dtype], pe: c_int):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_p"]()

    @parameter
    if is_amd_gpu():
        external_call[symbol, NoneType](dest, value, pe)
    else:
        _get_rocshmem_function[
            symbol,
            fn (
                UnsafePointer[Scalar[dtype]],
                Scalar[dtype],
                c_int,
            ) -> NoneType,
        ]()(dest, value, pe)


fn rocshmem_get[
    dtype: DType, //,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_get"]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn rocshmem_get_nbi[
    dtype: DType, //,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_get_nbi"]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn rocshmem_g[
    dtype: DType
](source: UnsafePointer[Scalar[dtype]], pe: c_int) -> Scalar[dtype]:
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_g"]()
    return external_call[symbol, Scalar[dtype]](source, pe)


# ===----------------------------------------------------------------------=== #
# 8: Signaling Operations
# ===----------------------------------------------------------------------=== #


fn rocshmem_put_signal_nbi[
    dtype: DType
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: Int,
    sig_addr: UnsafePointer[UInt64],
    signal: UInt64,
    sig_op: c_int,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type[
        "rocshmem_", dtype, "_put_signal_nbi"
    ]()
    external_call[symbol, NoneType](
        dest, source, nelems, sig_addr, signal, sig_op, pe
    )


# ===----------------------------------------------------------------------=== #
# 10: Collective Communication
# ===----------------------------------------------------------------------=== #


fn rocshmem_sync_all():
    _get_rocshmem_function[
        "rocshmem_sync_all",
        fn () -> NoneType,
    ]()()


fn rocshmem_barrier_all():
    @parameter
    if is_amd_gpu():
        external_call["rocshmem_barrier_all", NoneType]()
    else:
        _get_rocshmem_function[
            "rocshmem_barrier_all",
            fn () -> NoneType,
        ]()()


fn rocshmem_barrier_all_wave(stream: hipStream_t):
    _get_rocshmem_function[
        "rocshmem_barrier_all_wave",
        fn (hipStream_t) -> NoneType,
    ]()(stream)


# ===----------------------------------------------------------------------=== #
# 11: Point-To-Point Synchronization
# ===----------------------------------------------------------------------=== #


fn rocshmem_signal_wait_until[
    dtype: DType
](sig_addr: UnsafePointer[UInt64], cmp: c_int, cmp_value: UInt64):
    comptime symbol = _dtype_to_rocshmem_type[
        "rocshmem_", dtype, "_wait_until"
    ]()
    external_call[symbol, NoneType](sig_addr, cmp, cmp_value)


# ===----------------------------------------------------------------------=== #
# 12: Memory Ordering
# ===----------------------------------------------------------------------=== #


@extern("rocshmem_fence")
fn rocshmem_fence():
    ...
