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
"""This is a work-in-progress implementation of the OpenSHMEM spec, in the future
both ROCSHMEM and NVSHMEM will be supported.

You can find the current specification at
http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.6.pdf

The headings below corrosspond to section 9: OpenSHMEM Library API.
"""

from sys.ffi import external_call, c_int, c_size_t
from sys import is_nvidia_gpu, CompilationTarget
from collections.optional import OptionalReg
from gpu.host.launch_attribute import LaunchAttributeID, LaunchAttributeValue
from gpu.host import (
    ConstantMemoryMapping,
    DeviceAttribute,
    DeviceContext,
    DeviceFunction,
    DeviceStream,
    Dim,
    FuncAttribute,
    LaunchAttribute,
)
from gpu.host._nvidia_cuda import (
    CUDA,
    CUDA_MODULE,
)
from gpu.host.device_context import (
    _DumpPath,
    _checked,
    _CharPtr,
    _DeviceContextPtr,
)
from os import getenv, setenv
from sys import (
    CompilationTarget,
    is_amd_gpu,
    has_nvidia_gpu_accelerator,
    size_of,
    argv,
)
from sys.ffi import c_int, external_call
from ._mpi import (
    get_mpi_comm_world,
    MPI_Init,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Finalize,
)
from ._nvshmem import (
    nvshmem_barrier_all,
    nvshmem_calloc,
    nvshmem_fence,
    nvshmem_free,
    nvshmem_g,
    nvshmem_get,
    nvshmem_get_nbi,
    nvshmem_malloc,
    nvshmem_my_pe,
    nvshmem_n_pes,
    nvshmem_p,
    nvshmem_put,
    nvshmem_put_nbi,
    nvshmem_put_signal_nbi,
    nvshmem_signal_wait_until,
    nvshmem_team_my_pe,
    nvshmemx_barrier_all_on_stream,
    nvshmemx_cumodule_init,
    nvshmemx_hostlib_finalize,
    nvshmemx_init_status,
    nvshmemx_init,
    nvshmemx_signal_op,
    NVSHMEMXInitAttr,
    NVSHMEM_TEAM_INVALID,
    NVSHMEM_TEAM_SHARED,
    NVSHMEM_TEAM_WORLD,
    NVSHMEMX_INIT_WITH_MPI_COMM,
    NVSHMEMX_TEAM_NODE,
    NVSHMEM_CMP_EQ,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE,
    NVSHMEM_CMP_SENTINEL,
    NVSHMEM_SIGNAL_SET,
    NVSHMEM_SIGNAL_ADD,
)

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

alias shmem_team_t = c_int


struct SHMEMScope(EqualityComparable, ImplicitlyCopyable, Movable):
    """Enables following the OpenSHMEM spec by default for put/get/iput/iget
    etc. While allowing NVIDIA extensions for block and warp scopes by passing a
    parameter."""

    var value: StaticString

    alias default = Self("")
    """Execute RMA operation at global scope"""
    alias block = Self("_block")
    """Execute RMA operation at thread block scope (NVIDIA extension)"""
    alias warp = Self("_warp")
    """Execute RMA operation at warp scope (NVIDIA extension)"""

    fn __init__(out self, value: StaticString):
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

alias SHMEM_TEAM_INVALID: shmem_team_t = NVSHMEM_TEAM_INVALID
alias SHMEM_TEAM_SHARED: shmem_team_t = NVSHMEM_TEAM_SHARED
alias SHMEM_TEAM_NODE: shmem_team_t = NVSHMEMX_TEAM_NODE
alias SHMEM_TEAM_WORLD: shmem_team_t = NVSHMEM_TEAM_WORLD

alias SHMEM_CMP_EQ: c_int = NVSHMEM_CMP_EQ
alias SHMEM_CMP_NE: c_int = NVSHMEM_CMP_NE
alias SHMEM_CMP_GT: c_int = NVSHMEM_CMP_GT
alias SHMEM_CMP_LE: c_int = NVSHMEM_CMP_LE
alias SHMEM_CMP_LT: c_int = NVSHMEM_CMP_LT
alias SHMEM_CMP_GE: c_int = NVSHMEM_CMP_GE
alias SHMEM_CMP_SENTINEL: c_int = NVSHMEM_CMP_SENTINEL

alias SHMEM_SIGNAL_SET: c_int = NVSHMEM_SIGNAL_SET
alias SHMEM_SIGNAL_ADD: c_int = NVSHMEM_SIGNAL_ADD


# ===----------------------------------------------------------------------=== #
# 1: Library Setup, Exit, and Query Routines
# ===----------------------------------------------------------------------=== #


fn shmem_init() raises:
    """A collective operation that allocates and initializes the resources used
    by the SHMEM library.

    `shmem_init` allocates and initializes resources used by the SHMEM library.
    It is a collective operation that all PEs must call before any other SHMEM
    routine may be called, except `shmem_query_initialized` which checks the
    current initialized state of the library. In the SHMEM program which it
    initialized, each call to `shmem_init` must be matched with a corresponding
    call to `shmem_finalize`.

    The `shmem_init` and `shmem_init_thread` initialization routines may be
    called multiple times within an SHMEM program. A corresponding call to
    `shmem_finalize` must be made for each call to an SHMEM initialization
    routine. The SHMEM library must not be finalized until after the last call
    to `shmem_finalize` and may be re-initialized with a subsequent call to an
    initialization routine.

    Raises:
        If SHMEM initialization fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return nvshmemx_init()
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_init"
        ]()


fn shmem_finalize():
    """A collective operation that releases all resources used by SHMEM.

    `shmem_finalize` ends the SHMEM portion of a program previously initialized
    by `shmem_init` or `shmem_init_thread`. This is a collective operation that
    requires all PEs to participate in the call.

    A SHMEM program may perform a series of matching initialization and
    finalization calls. The last call to `shmem_finalize` in this series
    releases all resources used by the SHMEM library. This call destroys all
    teams created by the SHMEM program. As a result, all shareable contexts are
    destroyed.

    The last call to `shmem_finalize` performs an implicit global barrier to
    ensure that pending communications are completed and that no resources are
    released until all PEs have entered `shmem_finalize`. All other calls to
    `shmem_finalize` perform an operation semantically equivalent to
    `shmem_barrier_all` and return without freeing any SHMEM resources.

    The last call to `shmem_finalize` causes the SHMEM library to enter an
    uninitialized state. No further SHMEM calls may be made until an SHMEM
    initialization routine is called.

    All processes that represent the PEs will still exist after the call to
    `shmem_finalize` returns, but they will no longer have access to SHMEM
    library resources that have been released.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmemx_hostlib_finalize()
        try:
            _ = MPI_Finalize()
        except e:
            pass
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_finalize",
        ]()


fn shmem_my_pe() -> c_int:
    """Returns the number of the calling PE.

    Returns:
        The PE number of the calling PE. The result is an integer between 0 and
        npes - 1, where npes is the total number of PEs executing the current
        program.
    """

    @parameter
    if is_nvidia_gpu() or has_nvidia_gpu_accelerator():
        return nvshmem_my_pe()
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_my_pe",]()
        return {}


fn shmem_n_pes() -> c_int:
    """Returns the number of PEs running in a program.

    Returns:
        Number of PEs running in the OpenSHMEM program.
    """

    @parameter
    if is_nvidia_gpu() or has_nvidia_gpu_accelerator():
        return nvshmem_n_pes()
    else:
        return CompilationTarget.unsupported_target_error[
            c_int,
            operation="shmem_n_pes",
        ]()


# ===----------------------------------------------------------------------=== #
# 3: Memory Management
# ===----------------------------------------------------------------------=== #


fn shmem_malloc[dtype: DType](size: UInt) -> UnsafePointer[Scalar[dtype]]:
    """Collectively allocate symmetric memory.

    Parameters:
        dtype: The data type of elements to allocate memory for.

    Args:
        size: The number of elements to be allocated from the symmetric heap.

    Returns:
        The symmetric address of the allocated space; otherwise, it returns a
        null pointer.

    The `shmem_malloc` routine is a collective operation on the world team and
    returns the symmetric address of a block of at least size bytes, which shall
    be suitably aligned so that it may be assigned to a pointer to any type of
    object. This space is allocated from the symmetric heap (in contrast to
    malloc, which allocates from the private heap). When size is zero, the
    `shmem_malloc` routine performs no action and returns a null pointer;
    otherwise, `shmem_malloc` calls a procedure that is semantically equivalent to
    `shmem_barrier_all` on exit. This ensures that all PEs participate in the
    memory allocation, and that the memory on other PEs can be used as soon as
    the local PE returns. The value of the size argument must be identical on
    all PEs; otherwise, the behavior is undefined.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return nvshmem_malloc[dtype](UInt(size_of[dtype]() * size))
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_malloc"]()
        return UnsafePointer[Scalar[dtype]]()


fn shmem_calloc[
    dtype: DType
](count: UInt, size: UInt = UInt(size_of[dtype]())) -> UnsafePointer[
    Scalar[dtype]
]:
    """Collectively allocate a zeroed block of symmetric memory.

    Parameters:
        dtype: The data type of elements to allocate memory for.

    Args:
        count: The number of elements to allocate.
        size: The size in bytes of each element (defaults to size_of[dtype]()).

    Returns:
        A pointer to the lowest byte address of the allocated space; otherwise, it
        returns a null pointer.

    The `shmem_calloc` routine is a collective operation on the world team that
    allocates a region of remotely accessible memory for an array of `count`
    objects of `size` bytes each and returns a pointer to the lowest byte
    address of the allocated symmetric memory. The space is initialized to all
    bits zero. If the allocation succeeds, the pointer returned shall be
    suitably aligned so that it may be assigned to a pointer to any type of
    object. If the allocation does not succeed, or either count or size is 0,
    the return value is a null pointer. The values for count and size shall each
    be equal across all PEs calling `shmem_calloc`; otherwise, the behavior is
    undefined. When count or size is 0, the `shmem_calloc` routine returns
    without performing a barrier. Otherwise, this routine calls a procedure that
    is semantically equivalent to `shmem_barrier_all` on exit.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return nvshmem_calloc[dtype](count, size)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_calloc"]()
        return {}


fn shmem_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    """Collectively deallocate symmetric memory.

    Parameters:
        dtype: The data type of the memory being freed.

    Args:
        ptr: Symmetric address of an object in the symmetric heap.

    The shmem_free routine is a collective operation on the world team that
    causes the block to which ptr points to be deallocated, that is, made
    available for further allocation. If ptr is a null pointer, no action is
    performed; otherwise, shmem_free calls a barrier on entry. It is the user’s
    responsibility to ensure that no communication operations involving the
    given memory block are pending on other communication contexts prior to
    calling shmem_free. The value of the ptr argument must be identical on all
    PEs; otherwise, the behavior is undefined.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmem_free(ptr)
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_free",
        ]()


# ===----------------------------------------------------------------------=== #
# 4: Team Management Routines
# ===----------------------------------------------------------------------=== #


fn shmem_team_my_pe(team: shmem_team_t = SHMEM_TEAM_NODE) -> c_int:
    """Returns the number of the calling PE within a specified team.

    When team specifies a valid team, the shmem_team_my_pe routine returns the
    number of the calling PE within the specified team. The number is an integer
    between 0 and N − 1 for a team containing N PEs. Each member of the team
    has a unique number. If team compares equal to SHMEM_TEAM_INVALID, then the
    value -1 is returned. If team is otherwise invalid, the behavior is
    undefined.

    For the world team, this routine will return the same value as shmem_my_pe.

    Args:
        team: The team identifier, defaults to SHMEM_TEAM_NODE (node team).

    Returns:
        The number of the calling PE within the specified team, or the value -1
        if the team handle compares equal to SHMEM_TEAM_INVALID.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        return Int(nvshmem_team_my_pe(c_int(team)))
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_team_my_pe",
        ]()
        return 0


# ===----------------------------------------------------------------------=== #
# 6: Remote Memory Access (RMA) routines
# ===----------------------------------------------------------------------=== #


fn shmem_get[
    dtype: DType,
    scope: SHMEMScope = SHMEMScope.default,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    """Copies data from a specified PE.

    Args:
        dest: Local address of the data object to be updated.
        source: Symmetric address of the source data object.
        nelems: Number of elements in the dest and source arrays.
        pe: PE number of the remote PE relative to the team associated with the
            device.

    The get routines provide a method for copying a contiguous symmetric data
    object from a remote PE to a contiguous data object on the local PE. The
    routines return after the data has been delivered to the dest array on the
    local PE.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_get[scope](dest, source, nelems, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_get"]()


fn shmem_get_nbi[
    dtype: DType,
    scope: SHMEMScope = SHMEMScope.default,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    """Initiate a non-blocking copy of data from a specified PE.

    Args:
        dest: Local address of the data object to be updated.
        source: Symmetric address of the source data object.
        nelems: Number of elements in the dest and source arrays.
        pe: PE number of the remote PE relative to the team associated with the
            device.

    The get routines provide a method for copying a contiguous symmetric data
    object from a remote PE to a contiguous data object on the local PE. The
    routines return after initiating the operation. The operation is considered
    complete after a subsequent call to shmem_quiet. At the completion of
    shmem_quiet, the data has been delivered to the dest array on the local PE.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_get_nbi[scope](dest, source, nelems, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_get_nbi"]()


fn shmem_g[
    dtype: DType
](source: UnsafePointer[Scalar[dtype]], pe: c_int) -> Scalar[dtype]:
    """Copies one data item from a remote PE.

    Very low latency get capability for single elements.

    Args:
        source: Symmetric address of the source data object.
        pe: PE number of the remote PE on which source resides relative to the
            team associated with the given device context.

    Returns:
        A single element of dtype.
    """

    @parameter
    if is_nvidia_gpu():
        return nvshmem_g(source, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_g"]()
        return 0


fn shmem_put[
    dtype: DType, //,
    kind: SHMEMScope = SHMEMScope.default,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    """Copy data from a contiguous local data object to a data object on a
    specified PE.

    Args:
        dest: Symmetric address of the destination data object.
        source: Local address of the data object containing the data to be copied.
        nelems: Number of elements in the dest and source arrays.
        pe: PE number of the remote PE relative to the team associated
            with the device.

    The routines return after the data has been copied out of the source array
    on the local PE. The delivery of data words into the data object on the
    destination PE may occur in any order. Furthermore, two successive put
    routines may deliver data out of order unless a call to shmem_fence is
    introduced between the two calls.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_put[kind](dest, source, nelems, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_put"]()


fn shmem_put_nbi[
    dtype: DType, //,
    kind: SHMEMScope = SHMEMScope.default,
](
    dest: UnsafePointer[Scalar[dtype]],
    source: UnsafePointer[Scalar[dtype]],
    nelems: c_size_t,
    pe: c_int,
):
    """Initiate a non-blocking copy of data from a contiguous local data object
    to a data object on a specified PE.

    Args:
        dest: Symmetric address of the destination data object.
        source: Local address of the data object containing the data to be copied.
        nelems: Number of elements in the dest and source arrays.
        pe: PE number of the remote PE relative to the team associated
            with the device.

    The routines return after initiating the operation. The operation is
    considered complete after a subsequent call to shmem_quiet. At the
    completion of shmem_quiet, the data has been copied into the dest array on
    the destination PE. The delivery of data words into the data object on the
    destination PE may occur in any order. Furthermore, two successive put
    routines may deliver data out of order unless a call to shmem_fence is
    introduced between the two calls.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_put_nbi[kind](dest, source, nelems, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_put_nbi"]()


fn shmem_p[
    dtype: DType
](dest: UnsafePointer[Scalar[dtype]], value: Scalar[dtype], pe: c_int):
    """Copies one data item to a remote PE.

    Very low latency put capability for single elements. As with shmem_put,
    these routines start the remote transfer and may return before the data is
    delivered to the remote PE. Use shmem_quiet to force completion of all
    remote Put transfers.

    Args:
        dest: Symmetric address of the destination data object.
        value: The value to be transferred to dest.
        pe: PE number of the remote PE relative to the team associated with
            the device.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_p(dest, value, pe)
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_p"]()


# ===----------------------------------------------------------------------=== #
# 8: Signaling Operations
# ===----------------------------------------------------------------------=== #


fn shmem_put_signal_nbi[
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
    """
    The nonblocking put-with-signal routines provide a method for copying data
    from a contiguous local data object to a data object on a specified PE and
    subsequently updating a remote flag to signal completion.

    dest: Symmetric address of the data object to be updated on the remote PE.
        The type of dest should match that implied in the SYNOPSIS section.
    source: Local address of data object containing the data to be copied. The type
        of source should match that implied in the SYNOPSIS section.
    nelems: Number of elements in the dest and source arrays. For
        shmem_putmem_signal_nbi and shmem_ctx_putmem_signal_nbi, elements are bytes.
    sig_addr: Symmetric address of the signal data object to be updated on the remote
        PE as a signal.
    signal: Unsigned 64-bit value that is used for updating the remote signal
        data object.
    sig_op: Signal operator that represents the type of update to be performed on
        the remote signal data object.
    pe: PE number of the remote PE relative to the team associated with the
        given ctx when provided, or the default context otherwise.

    The nonblocking put-with-signal routines provide a method for copying data
    from a contiguous local data object to a data object on a specified PE and
    subsequently updating a remote flag to signal completion.

    The routines return after initiating the operation. The operation is
    considered complete after a subsequent call to shmem_quiet. At the
    completion of shmem_quiet, the data has been copied out of the source array
    on the local PE and delivered into the dest array on the destination PE.

    The delivery of signal flag on the remote PE indicates only the delivery of
    its corresponding dest data words into the data object on the remote PE.
    Furthermore, two successive nonblocking put-with-signal routines, or a
    nonblocking put-with-signal routine with another data transfer may deliver
    data out of order unless a call to shmem_fence is introduced between the two
    calls.

    The sig_op signal operator determines the type of update to be performed on
    the remote sig_addr signal data object.

    An update to the sig_addr signal data object through a nonblocking
    put-with-signal routine completes as if performed atomically as described in
    Section 9.8.1. The various options as described in Section 9.8.2 can be used
    as the sig_op signal operator.

    The dest and sig_addr data objects must both be remotely accessible. The
    sig_addr and dest could be of different kinds, for example, one could be a
    global/static C variable and the other could be allocated on the symmetric
    heap.  sig_addr and dest may not be overlapping in memory
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_put_signal_nbi(
            dest, source, nelems, sig_addr, signal, sig_op, pe
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_put_signal_nbi"
        ]()


# ===----------------------------------------------------------------------=== #
# 10: Collective Routines
# ===----------------------------------------------------------------------=== #


fn shmem_barrier_all():
    """Registers the arrival of a PE at a barrier and blocks the PE until all
    other PEs arrive at the barrier and all local updates and remote memory
    updates on the default context are completed.

    The shmem_barrier_all routine is a mechanism for synchronizing all PEs in
    the world team at once. This routine blocks the calling PE until all PEs
    have called shmem_barrier_all. In a multithreaded OpenSHMEM program, only
    the calling thread is blocked, however, it may not be called concurrently by
    multiple threads in the same PE.  Prior to synchronizing with other PEs,
    shmem_barrier_all ensures completion of all previously issued memory stores
    and remote memory updates issued on the default context via OpenSHMEM AMOs
    and RMA routine calls such as shmem_int_add, shmem_put32, shmem_put_nbi, and
    shmem_get_nbi.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_barrier_all()
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_barrier_all"
        ]()


# ===----------------------------------------------------------------------=== #
# 11: Point-to-point Synchronization Routines
# ===----------------------------------------------------------------------=== #


fn shmem_signal_wait_until(
    sig_addr: UnsafePointer[UInt64], cmp: c_int, cmp_value: UInt64
):
    """Wait for a variable on the local PE to change from a signaling operation.

    Args:
        sig_addr: Local address of the remotely accessible source signal variable.
        cmp: The comparison operator that compares sig_addr with cmp_value.
        cmp_value: The value against which the object pointed to by sig_addr
            will be compared.

    shmem_signal_wait_until operation blocks until the value contained in the
    signal data object, sig_addr, at the calling PE satisfies the wait
    condition. In a program with single-threaded or multithreaded PEs, the
    sig_addr object at the calling PE is expected only to be updated as a
    signal. This routine can be used to implement point-to-point synchronization
    between PEs or between threads within the same PE. A call to this routine
    blocks until the value of sig_addr at the calling PE satisfies the wait
    condition specified by the comparison operator, cmp, and comparison value,
    cmp_value. Implementations must ensure that shmem_signal_wait_until do not
    return before the update of the memory indicated by sig_addr is fully
    complete.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_signal_wait_until(sig_addr, cmp, cmp_value)
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_signal_wait_until"
        ]()


# ===----------------------------------------------------------------------=== #
# 12: Memory Ordering Routines
# ===----------------------------------------------------------------------=== #


fn shmem_fence():
    """Ensures ordering of delivery of operations on symmetric data objects.

    All operations on symmetric data objects issued to a particular PE on the
    given context prior to the call to shmem_fence are guaranteed to be
    delivered before any subsequent operations on symmetric data objects to the
    same PE on the same context. shmem_fence guarantees order of delivery, not
    completion. It does not guarantee order of delivery of nonblocking Get or
    values fetched by nonblocking AMO routines.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmem_fence()
    else:
        CompilationTarget.unsupported_target_error[operation="shmem_fence"]()


# ===----------------------------------------------------------------------=== #
# Outside of the OpenSHMEM spec
# ===----------------------------------------------------------------------=== #
# These are functions outside of the OpenSHMEM spec that have specific function
# symbols in NVSHMEM and ROCSHMEM. For example, NVSHMEM has a
# nvshmemx_cumodule_init for initializing the device_state into constant memory.
# Functions outside the spec are prefixed with `nvshmemx_`. These functions will
# be generalized to support both NVSHMEM and ROCSHMEM where possible.


fn shmem_signal_op(
    sig_addr: UnsafePointer[UInt64], signal: UInt64, sig_op: c_int, pe: c_int
):
    """The nvshmemx_signal_op operation atomically updates sig_addr with signal
    using operation sig_op on the specified PE. This operation can be used
    together with wait and test routines for efficient point-to-point
    synchronization.

    Args:
        sig_addr: Symmetric address of the signal word to be updated.
        signal: The value used to update sig_addr.
        sig_op: Operation used to update sig_addr with signal.
        pe: PE number of the remote PE.
    """

    @parameter
    if is_nvidia_gpu():
        nvshmemx_signal_op(sig_addr, signal, sig_op, pe)
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_signal_op"
        ]()


fn shmem_barrier_all_on_stream(stream: DeviceStream) raises:
    """
    Mechanism for synchronizing all PEs at once. This routine blocks the calling
    PE until all PEs have called nvshmem_barrier_all. In a multithreaded NVSHMEM
    program, only the calling thread is blocked, however, it may not be called
    concurrently by multiple threads in the same PE.

    Prior to synchronizing with other PEs, this function ensures
    completion of all previously issued memory stores and remote memory updates
    issued NVSHMEMAMOs and RMA routine calls such as shmem_put, shmem_g, etc.

    Args:
        stream: The stream to perform the barrier on.

    Raises:
        If the barrier operation fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        nvshmemx_barrier_all_on_stream(CUDA(stream))
    else:
        return CompilationTarget.unsupported_target_error[
            operation="shmem_barrier_all_on_stream",
        ]()


fn shmem_module_init(device_function: DeviceFunction) raises:
    """
    Intializes the device state in the compiled function module so that it’s
    able to perform NVSHMEM operations. Must have completed device
    initialization prior to calling this function.

    Args:
        device_function: The compiled device function to initialize with NVSHMEM.

    Raises:
        String: If module initialization fails.
    """

    @parameter
    if has_nvidia_gpu_accelerator():
        var func = CUDA_MODULE(device_function)
        _ = nvshmemx_cumodule_init(func)
    else:
        CompilationTarget.unsupported_target_error[
            operation="shmem_cumodule_init",
        ]()
