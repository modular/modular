# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from collections.string import atol
from collections.string.string_slice import get_static_string
from gpu.host import DeviceContext
from gpu.host._amdgpu_hip import hipStream_t, HIP
from gpu.host._nvidia_cuda import CUmodule, CUstream
from os import abort, getenv
from pathlib import Path
from sys import argv, size_of, has_amd_gpu_accelerator
from sys.info import CompilationTarget, is_nvidia_gpu, is_amd_gpu
from ffi import (
    _find_dylib,
    _get_dylib_function,
    _Global,
    c_int,
    c_size_t,
    c_uint,
    CStringSlice,
    external_call,
    OwnedDLHandle,
    RTLD,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


struct ROCSHEMIVersion(RegisterPassable):
    var major: c_int
    var minor: c_int
    var patch: c_int

    fn __init__(out self):
        self.major = 3
        self.minor = 4
        self.patch = 5


comptime ROCSHMEM_LIBRARY = _Global["ROCSHMEM_LIBRARY", _init_rocshmem_dylib]


fn _init_rocshmem_dylib() -> OwnedDLHandle:
    var lib = "librocshmem_host.so"
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


@always_inline
fn _get_rocshmem_function[
    func_name: StaticString, result_type: __TypeOfAllTypes
]() -> result_type:
    try:
        return _get_dylib_function[
            ROCSHMEM_LIBRARY(),
            func_name,
            result_type,
        ]()
    except e:
        abort(String(e))


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#

comptime rocshmem_team_id_t = Int32

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime ROCSHMEM_SUCCESS = 0

comptime ROCSHMEM_INIT_WITH_UNIQUEID: UInt32 = 1

comptime CHANNEL_BUF_SIZE: c_int = 1 << 22
comptime CHANNEL_BUF_SIZE_LOG: c_int = 22
comptime CHANNEL_ENTRY_BYTES: c_int = 8

comptime ROCSHMEM_ERROR_INTERNAL = 1
comptime ROCSHMEM_MAX_NAME_LEN: c_int = 256

comptime ROCSHMEM_THREAD_SINGLE: c_int = 0
comptime ROCSHMEM_THREAD_FUNNELED: c_int = 1
comptime ROCSHMEM_THREAD_WG_FUNNELED: c_int = 2
comptime ROCSHMEM_THREAD_SERIALIZED: c_int = 3
comptime ROCSHMEM_THREAD_MULTIPLE: c_int = 4

comptime ROCSHMEM_CMP_EQ: c_int = 0
comptime ROCSHMEM_CMP_NE: c_int = 1
comptime ROCSHMEM_CMP_GT: c_int = 2
comptime ROCSHMEM_CMP_GE: c_int = 3
comptime ROCSHMEM_CMP_LT: c_int = 4
comptime ROCSHMEM_CMP_LE: c_int = 5

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

comptime ROCSHMEM_SIGNAL_SET: c_int = 0
comptime ROCSHMEM_SIGNAL_ADD: c_int = 1

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
struct ROCSHMEMUniqueID(ImplicitlyCopyable):
    """Unique ID for a process (ROCSHMEM_UNIQUE_ID_BYTES = 128 bytes)."""

    var data: InlineArray[Byte, 128]

    fn __init__(out self):
        self.data = InlineArray[Byte, 128](fill=0)

    fn __copyinit__(out self, copy: Self):
        self.data = copy.data.copy()


struct ROCSHMEMInitAttr(ImplicitlyCopyable):
    """Data structure used for attribute based initialization.

    Maps to rocshmem_init_attr_t:
        int32_t rank;
        int32_t nranks;
        rocshmem_uniqueid_t uid;  // 128 bytes
        void* mpi_comm;

    mpi_comm is always a null pointer in our implementation, as we bootstrap with TCP.
    """

    var rank: Int32
    var nranks: Int32
    var uid: ROCSHMEMUniqueID
    var mpi_comm: UnsafePointer[NoneType, ImmutAnyOrigin]

    fn __init__(out self):
        comptime assert (
            size_of[Self]() == 144
        ), "ROCSHMEMInitAttr must be 144 bytes"
        self.rank = 0
        self.nranks = 0
        self.uid = ROCSHMEMUniqueID()
        self.mpi_comm = UnsafePointer[NoneType, ImmutAnyOrigin]()

    fn __init__(out self, rank: Int32, nranks: Int32, uid: ROCSHMEMUniqueID):
        self.rank = rank
        self.nranks = nranks
        self.uid = uid
        # Null pointer, we're not using MPI
        self.mpi_comm = UnsafePointer[NoneType, ImmutAnyOrigin]()


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
    if dtype == DType.float16:
        return get_static_string[prefix, "half", suffix]()
    elif dtype == DType.float32:
        return get_static_string[prefix, "float", suffix]()
    elif dtype == DType.float64:
        return get_static_string[prefix, "double", suffix]()
    elif dtype == DType.int8:
        return get_static_string[prefix, "schar", suffix]()
    elif dtype == DType.uint8:
        return get_static_string[prefix, "char", suffix]()
    elif dtype == DType.int16:
        return get_static_string[prefix, "short", suffix]()
    elif dtype == DType.uint16:
        return get_static_string[prefix, "ushort", suffix]()
    elif dtype == DType.int32:
        return get_static_string[prefix, "int", suffix]()
    elif dtype == DType.uint32:
        return get_static_string[prefix, "uint", suffix]()
    elif dtype == DType.int64:
        return get_static_string[prefix, "long", suffix]()
    elif dtype == DType.uint64:
        return get_static_string[prefix, "ulong", suffix]()
    elif dtype == DType.int:
        return get_static_string[prefix, "longlong", suffix]()
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString, operation = __get_current_function_name()
        ]()


# ===-----------------------------------------------------------------------===#
# 1: Library Setup, Exit, and Query
# ===-----------------------------------------------------------------------===#


fn _rocshmem_init() raises:
    _get_rocshmem_function[
        "rocshmem_init",
        fn() -> NoneType,
    ]()()


fn rocshmem_init_thread(
    ctx: DeviceContext,
    uid: UnsafePointer[ROCSHMEMUniqueID, MutAnyOrigin],
    var node_id: Int = -1,
    var total_nodes: Int = -1,
    var gpus_per_node: Int = -1,
) raises:
    """Initialize rocSHMEM for the given device using a unique ID for TCP bootstrap.

    Computes a global rank from the node ID, GPUs per node, and device ID, then
    initializes rocSHMEM with attribute-based initialization using the provided
    unique ID. Parameters default to -1, in which case values are read from
    environment variables (`SHMEM_NODE_ID`, `SHMEM_TOTAL_NODES`,
    `SHMEM_GPUS_PER_NODE`). If `SHMEM_GPUS_PER_NODE` is also unset, the number
    of attached GPUs is detected automatically.

    Args:
        ctx: The device context to initialize rocSHMEM on. This device is set
            as the current device before initialization.
        uid: Pointer to a `ROCSHMEMUniqueID` shared across all participating
            PEs, typically created by `rocshmem_create_uniqueid` on one PE
            and broadcast to the others.
        node_id: The ID of this node in the cluster. Defaults to -1, which
            reads from `SHMEM_NODE_ID` (default: 0).
        total_nodes: The total number of nodes in the cluster. Defaults to -1,
            which reads from `SHMEM_TOTAL_NODES` (default: 1).
        gpus_per_node: The number of GPUs per node. Defaults to -1, which
            reads from `SHMEM_GPUS_PER_NODE`, falling back to the number of
            detected devices.
    """
    ctx.set_as_current()
    # Check env vars if the defaults are not overridden
    if total_nodes == -1:
        total_nodes = atol(getenv("SHMEM_TOTAL_NODES", "1"))
    if node_id == -1:
        node_id = atol(getenv("SHMEM_NODE_ID", "0"))
    if gpus_per_node == -1:
        gpus_per_node = atol(getenv("SHMEM_GPUS_PER_NODE", "-1"))
        # If not defined by argument or env var, use the number of attached GPUs
        if gpus_per_node == -1:
            gpus_per_node = DeviceContext.number_of_devices()

    var global_rank = node_id * gpus_per_node + Int(ctx.id())
    var total_gpus = gpus_per_node * total_nodes

    var attr = ROCSHMEMInitAttr()
    rocshmem_set_attr_uniqueid_args(
        c_int(global_rank), c_int(total_gpus), uid, UnsafePointer(to=attr)
    )
    rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, UnsafePointer(to=attr))


fn rocshmem_create_uniqueid() raises -> ROCSHMEMUniqueID:
    """Create a unique ID for rocSHMEM TCP bootstrap.

    Generates a `ROCSHMEMUniqueID` that must be identical across all participating
    PEs to establish communication. One PE should call this function and
    broadcast the result to all others before calling `rocshmem_init_thread`.

    - `SHMEM_SERVER_IP`: Bootstrap server IP address (default: "0.0.0.0" for running on single node).
    - `SHMEM_SERVER_PORT`: Bootstrap server port (default: 44434).

    Returns:
        A `ROCSHMEMUniqueID` to be passed to `rocshmem_init_thread`.
    """
    var ip = getenv("SHMEM_SERVER_IP", "0.0.0.0")
    var port = c_int(atol(getenv("SHMEM_SERVER_PORT", "44434")))
    var uid = ROCSHMEMUniqueID()
    _get_rocshmem_function[
        "rocshmem_create_uniqueid",
        fn(
            CStringSlice[origin_of(ip)],
            c_int,
            UnsafePointer[ROCSHMEMUniqueID, origin_of(uid)],
        ) -> None,
    ]()(ip.as_c_string_slice(), port, UnsafePointer(to=uid))
    return uid


fn rocshmem_set_attr_uniqueid_args(
    rank: c_int,
    nranks: c_int,
    uid: UnsafePointer[ROCSHMEMUniqueID, MutAnyOrigin],
    attr: UnsafePointer[ROCSHMEMInitAttr, MutAnyOrigin],
) raises:
    """Populate a `ROCSHMEMInitAttr` with rank, size, and unique ID for
    attribute-based initialization.

    Args:
        rank: The global rank of this PE.
        nranks: The total number of PEs across all nodes.
        uid: Pointer to the shared `ROCSHMEMUniqueID` obtained from
            `rocshmem_create_uniqueid`.
        attr: Pointer to a `ROCSHMEMInitAttr` to be populated. The resulting
            attr is passed to `rocshmem_init_attr`.

    Raises:
        Error: If the underlying rocSHMEM call returns a non-zero error code.
    """
    var result = _get_rocshmem_function[
        "rocshmem_set_attr_uniqueid_args",
        fn(
            c_int,
            c_int,
            UnsafePointer[ROCSHMEMUniqueID, MutAnyOrigin],
            UnsafePointer[ROCSHMEMInitAttr, MutAnyOrigin],
        ) -> c_int,
    ]()(rank, nranks, uid, attr)
    if result:
        raise Error(
            "rocshmem_set_attr_uniqueid_args failed with error code:", result
        )


fn rocshmem_init_attr(
    flags: UInt32,
    attr: UnsafePointer[ROCSHMEMInitAttr, MutAnyOrigin],
) raises:
    var result = _get_rocshmem_function[
        "rocshmem_init_attr",
        fn(UInt32, UnsafePointer[ROCSHMEMInitAttr, MutAnyOrigin]) -> c_int,
    ]()(flags, attr)
    if result:
        raise Error("rocshmem_init_attr failed with error code:", result)


fn rocshmem_get_uniqueid(
    uid: UnsafePointer[ROCSHMEMUniqueID, MutAnyOrigin]
) raises:
    var result = _get_rocshmem_function[
        "rocshmem_get_uniqueid",
        fn(UnsafePointer[ROCSHMEMUniqueID, MutAnyOrigin]) -> c_int,
    ]()(uid)
    if result:
        raise Error("rocshmem_get_uniqueid failed with error code:", result)


fn rocshmem_finalize():
    _get_rocshmem_function[
        "rocshmem_finalize",
        fn() -> NoneType,
    ]()()


fn rocshmemx_hipmodule_init[T: AnyType](module: T) raises:
    """Initialize rocSHMEM device state in a dynamically loaded HIP module.

    This is the AMD equivalent of NVSHMEM's nvshmemx_cumodule_init().
    It copies the initialized ROCSHMEM_CTX_DEFAULT from the host library
    to the specified HIP module's device memory.

    Parameters:
        T: The HIP module type (usually HIP.hipModule_t).

    Args:
        module: The HIP module handle to initialize.
    """
    var result = _get_rocshmem_function[
        "rocshmemx_hipmodule_init",
        fn(T) -> c_int,
    ]()(module)
    if result:
        raise Error("rocshmemx_hipmodule_init failed with error code:", result)


fn rocshmem_my_pe() -> c_int:
    @parameter
    if is_amd_gpu():
        return external_call["rocshmem_my_pe", c_int]()
    else:
        return _get_rocshmem_function[
            "rocshmem_my_pe",
            fn() -> c_int,
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
            fn() -> c_int,
        ]()()


# ===----------------------------------------------------------------------=== #
# 3: Memory Management
# ===----------------------------------------------------------------------=== #


fn rocshmem_malloc[
    dtype: DType
](size: c_size_t) raises -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
    var ptr = _get_rocshmem_function[
        "rocshmem_malloc",
        fn(c_size_t) -> UnsafePointer[Scalar[dtype], MutExternalOrigin],
    ]()(size)

    return _check_rocshmem_allocation(ptr, "rochsmem_malloc", size)


fn rocshmem_calloc[
    dtype: DType
](count: c_size_t, size: c_size_t) raises -> UnsafePointer[
    Scalar[dtype], MutExternalOrigin
]:
    var ptr = _get_rocshmem_function[
        "rocshmem_calloc",
        fn(
            c_size_t, c_size_t
        ) -> UnsafePointer[Scalar[dtype], MutExternalOrigin],
    ]()(count, size)

    return _check_rocshmem_allocation(ptr, "rochsmem_calloc", count * size)


fn _check_rocshmem_allocation[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    func_name: StaticString,
    requested_bytes: c_size_t,
) raises -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
    if not ptr:
        raise Error(
            func_name,
            " failed to allocate ",
            requested_bytes,
            " bytes of ",
            dtype,
            (
                ". Increase allocatable bytes by setting the ROCSHMEM_HEAP_SIZE"
                " env var"
            ),
        )
    return ptr


fn rocshmem_free[
    dtype: DType, //
](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin]):
    _get_rocshmem_function[
        "rocshmem_free",
        fn(type_of(ptr)) -> NoneType,
    ]()(ptr)


# ===----------------------------------------------------------------------=== #
# 4: Team Management
# ===----------------------------------------------------------------------=== #


fn rocshmem_team_my_pe(team: c_int) -> c_int:
    return _get_rocshmem_function[
        "rocshmem_team_my_pe",
        fn(c_int) -> c_int,
    ]()(team)


# ===----------------------------------------------------------------------=== #
# 6: Remote Memory Access (RMA)
# ===----------------------------------------------------------------------=== #


fn rocshmem_put[
    dtype: DType,
    //,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_put"]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn rocshmem_put_nbi[
    dtype: DType,
    //,
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
](
    dest: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_p"]()

    @parameter
    if is_amd_gpu():
        external_call[symbol, NoneType](dest, value, pe)
    else:
        _get_rocshmem_function[
            symbol,
            fn(
                UnsafePointer[Scalar[dtype], MutAnyOrigin],
                Scalar[dtype],
                c_int,
            ) -> NoneType,
        ]()(dest, value, pe)


fn rocshmem_get[
    dtype: DType,
    //,
](
    dest: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_rocshmem_type["rocshmem_", dtype, "_get"]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


fn rocshmem_get_nbi[
    dtype: DType,
    //,
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


fn rocshmemx_signal_op(
    sig_addr: UnsafePointer[UInt64],
    signal: UInt64,
    sig_op: c_int,
    pe: c_int,
):
    """Atomically update a remote signal value.

    The rocshmemx_signal_op operation atomically updates sig_addr with signal
    using operation sig_op on the specified PE. This operation can be used
    together with wait and test routines for efficient point-to-point
    synchronization.

    Args:
        sig_addr: Symmetric address of the signal word to be updated.
        signal: The value used to update sig_addr.
        sig_op: Operation used to update sig_addr (ROCSHMEM_SIGNAL_SET or
            ROCSHMEM_SIGNAL_ADD).
        pe: PE number of the remote PE.
    """
    external_call["rocshmemx_signal_op", NoneType](sig_addr, signal, sig_op, pe)


# ===----------------------------------------------------------------------=== #
# 10: Collective Communication
# ===----------------------------------------------------------------------=== #


fn rocshmem_sync_all():
    _get_rocshmem_function[
        "rocshmem_sync_all",
        fn() -> NoneType,
    ]()()


fn rocshmem_barrier_all():
    @parameter
    if is_amd_gpu():
        external_call["rocshmem_barrier_all", NoneType]()
    else:
        _get_rocshmem_function[
            "rocshmem_barrier_all",
            fn() -> NoneType,
        ]()()


fn rocshmem_barrier_all_on_stream(stream: hipStream_t):
    _get_rocshmem_function[
        "rocshmem_barrier_all_on_stream",
        fn(hipStream_t) -> NoneType,
    ]()(stream)


# ===----------------------------------------------------------------------=== #
# 11: Point-To-Point Synchronization
# ===----------------------------------------------------------------------=== #


fn rocshmem_signal_wait_until[
    dtype: DType
](sig_addr: UnsafePointer[Scalar[dtype]], cmp: c_int, cmp_value: Scalar[dtype]):
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
