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
from os import abort
from pathlib import Path
from sys import argv, size_of
from sys.ffi import (
    _find_dylib,
    _get_dylib_function,
    _Global,
    _OwnedDLHandle,
    c_int,
    c_uint,
    c_size_t,
    external_call,
)
from sys.info import CompilationTarget, is_nvidia_gpu

from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import CUmodule, CUstream

from ._mpi import MPI_Comm_rank, MPI_Init, MPIComm, get_mpi_comm_world

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias NVSHMEM_LIBRARY_PATHS = List[Path](
    "libnvshmem_host.so.3.4.5",
    "libnvshmem_host.so.3",
    "libnvshmem_host.so",
)


@register_passable
struct NVSHMEMIVersion:
    var major: c_int
    var minor: c_int
    var patch: c_int

    fn __init__(out self):
        self.major = 3
        self.minor = 4
        self.patch = 5


fn _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the NVShmem libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[NVSHMEM_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


alias NVSHMEM_LIBRARY = _Global[
    "NVSHMEM_LIBRARY", _init_nvshmem_dylib, on_error_msg=_on_error_msg
]


fn _init_nvshmem_dylib() -> _OwnedDLHandle:
    return _find_dylib["NVSHMEM"](materialize[NVSHMEM_LIBRARY_PATHS]())


@always_inline
fn _get_nvshmem_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    try:
        return _get_dylib_function[
            NVSHMEM_LIBRARY(),
            func_name,
            result_type,
        ]()
    except e:
        return abort[result_type](String(e))


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#

alias nvshmem_team_id_t = Int32

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

alias NVSHMEM_SUCCESS = 0

alias NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1

alias CHANNEL_BUF_SIZE: c_int = 1 << 22
alias CHANNEL_BUF_SIZE_LOG: c_int = 22
alias CHANNEL_ENTRY_BYTES: c_int = 8

alias NVSHMEMX_ERROR_INTERNAL = 1
alias NVSHMEM_MAX_NAME_LEN: c_int = 256

alias NVSHMEM_THREAD_SINGLE: c_int = 0
alias NVSHMEM_THREAD_FUNNELED: c_int = 1
alias NVSHMEM_THREAD_SERIALIZED: c_int = 2
alias NVSHMEM_THREAD_MULTIPLE: c_int = 3
alias NVSHMEM_THREAD_TYPE_SENTINEL: c_int = c_int.MAX

alias NVSHMEM_CMP_EQ: c_int = 0
alias NVSHMEM_CMP_NE: c_int = 1
alias NVSHMEM_CMP_GT: c_int = 2
alias NVSHMEM_CMP_LE: c_int = 3
alias NVSHMEM_CMP_LT: c_int = 4
alias NVSHMEM_CMP_GE: c_int = 5
alias NVSHMEM_CMP_SENTINEL: c_int = c_int.MAX

alias PROXY_GLOBAL_EXIT_INIT: c_int = 1
alias PROXY_GLOBAL_EXIT_REQUESTED: c_int = 2
alias PROXY_GLOBAL_EXIT_FINISHED: c_int = 3
alias PROXY_GLOBAL_EXIT_MAX_STATE: c_int = c_int.MAX

alias PROXY_DMA_REQ_BYTES: c_int = 32
alias PROXY_AMO_REQ_BYTES: c_int = 40
alias PROXY_INLINE_REQ_BYTES: c_int = 24

alias NVSHMEM_STATUS_NOT_INITIALIZED: c_int = 0
alias NVSHMEM_STATUS_IS_BOOTSTRAPPED: c_int = 1
alias NVSHMEM_STATUS_IS_INITIALIZED: c_int = 2
alias NVSHMEM_STATUS_LIMITED_MPG: c_int = 4
alias NVSHMEM_STATUS_FULL_MPG: c_int = 5
alias NVSHMEM_STATUS_INVALID: c_int = c_int.MAX

alias NVSHMEM_SIGNAL_SET: c_int = 9
alias NVSHMEM_SIGNAL_ADD: c_int = 10

alias NVSHMEM_TEAM_INVALID: nvshmem_team_id_t = -1
alias NVSHMEM_TEAM_WORLD: nvshmem_team_id_t = 0
alias NVSHMEM_TEAM_WORLD_INDEX: nvshmem_team_id_t = 0
alias NVSHMEM_TEAM_SHARED: nvshmem_team_id_t = 1
alias NVSHMEM_TEAM_SHARED_INDEX: nvshmem_team_id_t = 1
alias NVSHMEMX_TEAM_NODE: nvshmem_team_id_t = 2
alias NVSHMEM_TEAM_NODE_INDEX: nvshmem_team_id_t = 2
alias NVSHMEMX_TEAM_SAME_MYPE_NODE: nvshmem_team_id_t = 3
alias NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX: nvshmem_team_id_t = 3
alias NVSHMEMI_TEAM_SAME_GPU: nvshmem_team_id_t = 4
alias NVSHMEM_TEAM_SAME_GPU_INDEX: nvshmem_team_id_t = 4
alias NVSHMEMI_TEAM_GPU_LEADERS: nvshmem_team_id_t = 5
alias NVSHMEM_TEAM_GPU_LEADERS_INDEX: nvshmem_team_id_t = 5
alias NVSHMEM_TEAMS_MIN: nvshmem_team_id_t = 6
alias NVSHMEM_TEAM_INDEX_MAX: nvshmem_team_id_t = nvshmem_team_id_t.MAX


# Structs
struct NVSHMEMXInitAttr:
    var version: c_int
    var mpi_comm: UnsafePointer[MPIComm]
    var args: NVSHMEMXInitArgs

    fn __init__(out self, mpi_comm: UnsafePointer[MPIComm]):
        constrained[
            size_of[Self]() == 144, "NVSHMEMXInitAttr must be 144 bytes"
        ]()
        self.version = (1 << 16) + size_of[NVSHMEMXInitAttr]()
        self.mpi_comm = mpi_comm
        self.args = NVSHMEMXInitArgs()


struct NVSHMEMXInitArgs:
    var version: c_int
    var uid_args: NVSHMEMXUniqueIDArgs
    var content: InlineArray[Byte, 96]

    fn __init__(out self):
        constrained[
            size_of[Self]() == 128, "NVSHMEMXInitArgs must be 128 bytes"
        ]()
        self.version = (1 << 16) + size_of[NVSHMEMXInitArgs]()
        self.uid_args = NVSHMEMXUniqueIDArgs()
        self.content = InlineArray[Byte, 96](fill=0)


struct NVSHMEMXUniqueIDArgs:
    var version: c_int
    var id: UnsafePointer[NVSHMEMXUniqueID]
    var myrank: c_int
    var nranks: c_int

    fn __init__(out self):
        constrained[
            size_of[Self]() == 24, "NVSHMEMXUniqueIDArgs must be 24 bytes"
        ]()
        self.version = (1 << 16) + size_of[NVSHMEMXUniqueIDArgs]()
        self.id = UnsafePointer[NVSHMEMXUniqueID]()
        self.myrank = 0
        self.nranks = 0


struct NVSHMEMXUniqueID:
    var version: c_int
    var internal: InlineArray[Byte, 124]

    fn __init__(out self):
        constrained[
            size_of[Self]() == 128, "nvshmemx_uniqueid_t must be 128 bytes"
        ]()
        self.version = (1 << 16) + size_of[NVSHMEMXUniqueID]()
        self.internal = InlineArray[Byte, 124](fill=0)


fn _get_prefix[scope: SHMEMScope]() -> StaticString:
    @parameter
    if scope == SHMEMScope.default:
        return "nvshmem_"
    else:
        return "nvshmemx_"


fn _dtype_to_nvshmem_type[
    prefix: StaticString,
    dtype: DType,
    suffix: StaticString,
    scope: StaticString = "",
]() -> StaticString:
    """
    Returns the NVSHMEM name for the given dtype surrounded by the given prefix
    and suffix, for calling the correct symbol on the device-side bitcode.

    The below mappings come from
    https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#remote-memory-access as
    of nvshmem 3.3, and the bitwidths were generated from CUDA 12.6 on x86-64
    Ubuntu 22.04.

    c_name               nvshmem_name  bitwidth
    -------------------------------------------
    float                float         32
    double               double        64
    __nv_bfloat16        bfloat16      16
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
        return get_static_string[prefix, "half", suffix, scope]()
    elif dtype is DType.bfloat16:
        return get_static_string[prefix, "bfloat16", suffix, scope]()
    elif dtype is DType.float32:
        return get_static_string[prefix, "float", suffix, scope]()
    elif dtype is DType.float64:
        return get_static_string[prefix, "double", suffix, scope]()
    elif dtype is DType.int8:
        return get_static_string[prefix, "int8", suffix, scope]()
    elif dtype is DType.uint8:
        return get_static_string[prefix, "uint8", suffix, scope]()
    elif dtype is DType.int16:
        return get_static_string[prefix, "int16", suffix, scope]()
    elif dtype is DType.uint16:
        return get_static_string[prefix, "uint16", suffix, scope]()
    elif dtype is DType.int32:
        return get_static_string[prefix, "int32", suffix, scope]()
    elif dtype is DType.uint32:
        return get_static_string[prefix, "uint32", suffix, scope]()
    elif dtype is DType.int64:
        return get_static_string[prefix, "int64", suffix, scope]()
    elif dtype is DType.uint64:
        return get_static_string[prefix, "uint64", suffix, scope]()
    elif dtype is DType.int:
        return get_static_string[prefix, "size", suffix, scope]()
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString, operation="_dtype_to_nvshmem_type"
        ]()


# ===-----------------------------------------------------------------------===#
# 1: Library Setup, Exit, and Query
# https://docs.nvidia.com/nvshmem/api/gen/api/setup.html
# ===-----------------------------------------------------------------------===#


# Run one GPU per process
fn nvshmemx_init() raises:
    var _argv = argv()
    var argc = len(_argv)
    MPI_Init(argc, _argv)

    # Get MPI rank and size
    var rank = c_int(0)
    var mpi_comm = get_mpi_comm_world()

    _ = MPI_Comm_rank(mpi_comm, UnsafePointer(to=rank))
    # Set CUDA device early - needed for CUDA-related NVSHMEM initialization
    var ctx = DeviceContext(device_id=Int(rank))
    ctx.set_as_current()

    # Initialize NVSHMEM with MPI
    var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
    # For single process per GPU, fallback to one device per process per node.
    attr.args.uid_args.myrank = 0
    attr.args.uid_args.nranks = 1

    _ = nvshmemx_hostlib_init_attr(
        NVSHMEMX_INIT_WITH_MPI_COMM, UnsafePointer(to=attr)
    )

    # Check initialization status
    if nvshmemx_init_status() != 2:
        raise Error("failed to initialize NVSHMEM")


# Modular specific init, run one GPU per thread and return associated DeviceContext
fn nvshmemx_init(mype_node: Int, npes_node: Int) raises -> DeviceContext:
    var ctx = DeviceContext(mype_node)
    ctx.set_as_current()

    # Initialize NVSHMEM with MPI
    var mpi_comm = get_mpi_comm_world()
    var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
    attr.args.uid_args.myrank = mype_node
    attr.args.uid_args.nranks = npes_node

    var status = nvshmemx_hostlib_init_attr(
        NVSHMEMX_INIT_WITH_MPI_COMM, UnsafePointer(to=attr)
    )
    if status:
        raise Error("failed to initialize NVSHMEM with status:", status)
    # Check initialization status
    status = nvshmemx_init_status()
    if status != 2:
        raise Error("failed to initialize NVSHMEM with status:", status)

    return ctx


fn nvshmemx_hostlib_init_attr(
    flags: UInt32,
    attr: UnsafePointer[NVSHMEMXInitAttr],
) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_hostlib_init_attr",
        fn (UInt32, UnsafePointer[NVSHMEMXInitAttr]) -> c_int,
    ]()(flags, attr)


fn nvshmemx_hostlib_finalize():
    _get_nvshmem_function[
        "nvshmemx_hostlib_finalize",
        fn () -> NoneType,
    ]()()


fn nvshmemx_cumodule_init(module: CUmodule) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_cumodule_init",
        fn (CUmodule) -> c_int,
    ]()(module)


fn nvshmemx_cumodule_finalize(module: CUmodule) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_cumodule_finalize",
        fn (CUmodule) -> c_int,
    ]()(module)


fn nvshmemx_init_status() -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_init_status",
        fn () -> c_int,
    ]()()


fn nvshmem_my_pe() -> c_int:
    @parameter
    if is_nvidia_gpu():
        return external_call["nvshmem_my_pe", c_int]()
    else:
        return _get_nvshmem_function[
            "nvshmem_my_pe",
            fn () -> c_int,
        ]()()


fn nvshmem_n_pes() -> c_int:
    @parameter
    if is_nvidia_gpu():
        return external_call["nvshmem_n_pes", c_int]()
    else:
        return _get_nvshmem_function[
            "nvshmem_n_pes",
            fn () -> c_int,
        ]()()


# ===----------------------------------------------------------------------=== #
# 3: Memory Management
# https://docs.nvidia.com/nvshmem/api/gen/api/memory.html
# ===----------------------------------------------------------------------=== #


fn nvshmem_malloc[dtype: DType](size: c_size_t) -> UnsafePointer[Scalar[dtype]]:
    return _get_nvshmem_function[
        "nvshmem_malloc",
        fn (c_size_t) -> UnsafePointer[Scalar[dtype]],
    ]()(size)


fn nvshmem_calloc[
    dtype: DType
](count: c_size_t, size: c_size_t) -> UnsafePointer[Scalar[dtype]]:
    return _get_nvshmem_function[
        "nvshmem_calloc",
        fn (c_size_t, c_size_t) -> UnsafePointer[Scalar[dtype]],
    ]()(count, size)


fn nvshmem_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    _get_nvshmem_function[
        "nvshmem_free",
        fn (UnsafePointer[Scalar[dtype]]) -> NoneType,
    ]()(ptr)


# ===----------------------------------------------------------------------=== #
# 4: Team Management
# https://docs.nvidia.com/nvshmem/api/gen/api/symmetric.html
# ===----------------------------------------------------------------------=== #


fn nvshmem_team_my_pe(team: c_int) -> c_int:
    return _get_nvshmem_function[
        "nvshmem_team_my_pe",
        fn (c_int) -> c_int,
    ]()(team)


fn nvshmemx_team_init() -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_team_init",
        fn () -> c_int,
    ]()()


# ===----------------------------------------------------------------------=== #
# 6: Remote Memory Access (RMA)
# https://docs.nvidia.com/nvshmem/api/gen/api/rma.html
# ===----------------------------------------------------------------------=== #


fn nvshmem_put[
    dtype: DType, //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    alias symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_put", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn nvshmem_put_nbi[
    dtype: DType, //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    alias symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_put_nbi", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn nvshmem_p[
    dtype: DType
](dest: UnsafePointer[Scalar[dtype]], value: Scalar[dtype], pe: c_int):
    alias symbol = _dtype_to_nvshmem_type["nvshmem_", dtype, "_p"]()
    external_call[symbol, NoneType](dest, value, pe)


fn nvshmem_get[
    dtype: DType, //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    alias symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_get", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn nvshmem_get_nbi[
    dtype: DType, //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    alias symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_get_nbi", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn nvshmem_g[
    dtype: DType
](source: UnsafePointer[Scalar[dtype]], pe: c_int) -> Scalar[dtype]:
    alias symbol = _dtype_to_nvshmem_type["nvshmem_", dtype, "_g"]()
    return external_call[symbol, Scalar[dtype]](source, pe)


# ===----------------------------------------------------------------------=== #
# 8: Signaling Operations
# https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmemx_signal_op")
fn nvshmemx_signal_op(
    sig_addr: UnsafePointer[UInt64], signal: UInt64, sig_op: c_int, pe: c_int
):
    ...


fn nvshmem_put_signal_nbi[
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
    alias symbol = _dtype_to_nvshmem_type[
        "nvshmem_", dtype, "_put_signal_nbi"
    ]()
    external_call[symbol, NoneType](
        dest, source, nelems, sig_addr, signal, sig_op, pe
    )


# ===----------------------------------------------------------------------=== #
# 10: Collective Communication
# https://docs.nvidia.com/nvshmem/api/gen/api/collective.html
# ===----------------------------------------------------------------------=== #


fn nvshmem_sync_all():
    _get_nvshmem_function[
        "nvshmem_sync_all",
        fn () -> NoneType,
    ]()()


fn nvshmem_barrier_all():
    @parameter
    if is_nvidia_gpu():
        external_call["nvshmem_barrier_all", NoneType]()
    else:
        _get_nvshmem_function[
            "nvshmem_barrier_all",
            fn () -> NoneType,
        ]()()


fn nvshmemx_barrier_all_on_stream(stream: CUstream):
    _get_nvshmem_function[
        "nvshmemx_barrier_all_on_stream",
        fn (CUstream) -> NoneType,
    ]()(stream)


# ===----------------------------------------------------------------------=== #
# 11: Point-To-Point Synchronization
# https://docs.nvidia.com/nvshmem/api/gen/api/sync.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmem_signal_wait_until")
fn nvshmem_signal_wait_until(
    sig_addr: UnsafePointer[UInt64], cmp: c_int, cmp_value: UInt64
):
    ...


# ===----------------------------------------------------------------------=== #
# 12: Memory Ordering
# https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmem_fence")
fn nvshmem_fence():
    ...
