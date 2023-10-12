# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA host functions."""

from sys.ffi import DLHandle
from memory import stack_allocation
from memory.unsafe import bitcast
from utils.index import StaticIntTuple, Index
from sys.info import sizeof
from pathlib import Path
from math import floor
from utils._reflection import get_linkage_name

# ===----------------------------------------------------------------------===#
# Globals
# ===----------------------------------------------------------------------===#


alias CUDA_DRIVER_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so"

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _check_error(err: Result) raises:
    if err != Result.SUCCESS:
        raise Error(err.__str__())


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return String(Float32(size) / GB) + "GB"

    if size >= MB:
        return String(Float32(size) / MB) + "MB"

    if size >= KB:
        return String(Float32(size) / KB) + "KB"

    return String(size) + "B"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib() -> Pointer[AnyType]:
    let ptr = Pointer[DLHandle].alloc(1)
    let handle = DLHandle(CUDA_DRIVER_PATH)
    _ = handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
    __get_address_as_lvalue(ptr.address) = handle
    return ptr.bitcast[AnyType]()


fn _destroy_dylib(ptr: Pointer[AnyType]):
    __get_address_as_lvalue(ptr.bitcast[DLHandle]().address)._del_old()
    ptr.free()


@always_inline
fn _get_dylib() -> DLHandle:
    let ptr = external_call["KGEN_CompilerRT_GetGlobalOr", Pointer[DLHandle]](
        StringRef("CUDA"), _init_dylib, _destroy_dylib
    )
    return __get_address_as_lvalue(ptr.address)


@always_inline
fn _get_dylib_function[result_type: AnyType](name: StringRef) -> result_type:
    return _get_dylib_function[result_type](_get_dylib(), name)


@always_inline
fn _get_dylib_function[
    result_type: AnyType
](dylib: DLHandle, name: StringRef) -> result_type:
    return dylib.get_function[result_type](name)


# ===----------------------------------------------------------------------===#
# Result
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Result:
    var code: Int32

    alias SUCCESS = Result(0)
    """The API call returned with no errors. In the case of query calls, this
    also means that the operation being queried is complete (see
    ::cuEventQuery() and ::cuStreamQuery()).
    """

    alias INVALID_VALUE = Result(1)
    """This indicates that one or more of the parameters passed to the API call
    is not within an acceptable range of values.
    """

    alias OUT_OF_MEMORY = Result(2)
    """The API call failed because it was unable to allocate enough memory to
    perform the requested operation.
    """

    alias NOT_INITIALIZED = Result(3)
    """This indicates that the CUDA driver has not been initialized with
    ::cuInit() or that initialization has failed.
    """

    alias DEINITIALIZED = Result(4)
    """This indicates that the CUDA driver is in the process of shutting down.
    """

    alias PROFILER_DISABLED = Result(5)
    """This indicates profiler is not initialized for this run. This can
    happen when the application is running with external profiling tools
    like visual profiler.
    """

    alias PROFILER_NOT_INITIALIZED = Result(6)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to attempt to enable/disable the profiling via ::cuProfilerStart or
    ::cuProfilerStop without initialization.
    """

    alias PROFILER_ALREADY_STARTED = Result(7)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to call cuProfilerStart() when profiling is already enabled.
    """

    alias PROFILER_ALREADY_STOPPED = Result(8)
    """This error return is deprecated as of CUDA 5.0. It is no longer an error
    to call cuProfilerStop() when profiling is already disabled.
    """

    alias STUB_LIBRARY = Result(34)
    """This indicates that the CUDA driver that the application has loaded is a
    stub library. Applications that run with the stub rather than a real
    driver loaded will result in CUDA API returning this error.
    """

    alias DEVICE_UNAVAILABLE = Result(46)
    """This indicates that requested CUDA device is unavailable at the current
    time. Devices are often unavailable due to use of
    ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.
    """

    alias NO_DEVICE = Result(100)
    """This indicates that no CUDA-capable devices were detected by the
    installed CUDA driver.
    """

    alias INVALID_DEVICE = Result(101)
    """This indicates that the device ordinal supplied by the user does not
    correspond to a valid CUDA device or that the action requested is
    invalid for the specified device.
    """

    alias DEVICE_NOT_LICENSED = Result(102)
    """This error indicates that the Grid license is not applied.
    """

    alias INVALID_IMAGE = Result(200)
    """This indicates that the device kernel image is invalid. This can also
    indicate an invalid CUDA module.
    """

    alias INVALID_CONTEXT = Result(201)
    """This most frequently indicates that there is no context bound to the
    current thread. This can also be returned if the context passed to an
    API call is not a valid handle (such as a context that has had
    ::cuCtxDestroy() invoked on it). This can also be returned if a user
    mixes different API versions (i.e. 3010 context with 3020 API calls).
    See ::cuCtxGetApiVersion() for more details.
    """

    alias CONTEXT_ALREADY_CURRENT = Result(202)
    """This indicated that the context being supplied as a parameter to the
    API call was already the active context.
    [[depricated]]
    This error return is deprecated as of CUDA 3.2. It is no longer an
    error to attempt to push the active context via ::cuCtxPushCurrent().
    """

    alias MAP_FAILED = Result(205)
    """This indicates that a map or register operation has failed.
    """

    alias UNMAP_FAILED = Result(206)
    """This indicates that an unmap or unregister operation has failed.
    """

    alias ARRAY_IS_MAPPED = Result(207)
    """This indicates that the specified array is currently mapped and thus
    cannot be destroyed.
    """

    alias ALREADY_MAPPED = Result(208)
    """This indicates that the resource is already mapped.
    """

    alias NO_BINARY_FOR_GPU = Result(209)
    """This indicates that there is no kernel image available that is suitable
    for the device. This can occur when a user specifies code generation
    options for a particular CUDA source file that do not include the
    corresponding device configuration.
    """

    alias ALREADY_ACQUIRED = Result(210)
    """This indicates that a resource has already been acquired.
    """

    alias NOT_MAPPED = Result(211)
    """This indicates that a resource is not mapped.
    """

    alias NOT_MAPPED_AS_ARRAY = Result(212)
    """This indicates that a mapped resource is not available for access as an
    array.
    """

    alias NOT_MAPPED_AS_POINTER = Result(213)
    """This indicates that a mapped resource is not available for access as a
    pointer.
    """

    alias ECC_UNCORRECTABLE = Result(214)
    """This indicates that an uncorrectable ECC error was detected during
    execution.
    """

    alias UNSUPPORTED_LIMIT = Result(215)
    """This indicates that the ::CUlimit passed to the API call is not
    supported by the active device.
    """

    alias CONTEXT_ALREADY_IN_USE = Result(216)
    """This indicates that the ::CUcontext passed to the API call can
    only be bound to a single CPU thread at a time but is already
    bound to a CPU thread.
    """

    alias PEER_ACCESS_UNSUPPORTED = Result(217)
    """This indicates that peer access is not supported across the given
    devices.
    """

    alias INVALID_PTX = Result(218)
    """This indicates that a PTX JIT compilation failed.
    """

    alias INVALID_GRAPHICS_CONTEXT = Result(219)
    """This indicates an error with OpenGL or DirectX context.
    """

    alias NVLINK_UNCORRECTABLE = Result(220)
    """This indicates that an uncorrectable NVLink error was detected during the
    execution.
    """

    alias JIT_COMPILER_NOT_FOUND = Result(221)
    """This indicates that the PTX JIT compiler library was not found.
    """

    alias UNSUPPORTED_PTX_VERSION = Result(222)
    """This indicates that the provided PTX was compiled with an unsupported
    toolchain.
    """

    alias JIT_COMPILATION_DISABLED = Result(223)
    """This indicates that the PTX JIT compilation was disabled.
    """

    alias UNSUPPORTED_EXEC_AFFINITY = Result(224)
    """This indicates that the ::CUexecAffinityType passed to the API call is
    not supported by the active device.
    """

    alias UNSUPPORTED_DEVSIDE_SYNC = Result(225)
    """This indicates that the code to be compiled by the PTX JIT contains
    unsupported call to cudaDeviceSynchronize.
    """

    alias INVALID_SOURCE = Result(300)
    """This indicates that the device kernel source is invalid. This includes
    compilation/linker errors encountered in device code or user error.
    """

    alias FILE_NOT_FOUND = Result(301)
    """This indicates that the file specified was not found.
    """

    alias SHARED_OBJECT_SYMBOL_NOT_FOUND = Result(302)
    """This indicates that a link to a shared object failed to resolve.
    """

    alias SHARED_OBJECT_INIT_FAILED = Result(303)
    """This indicates that initialization of a shared object failed.
    """

    alias OPERATING_SYSTEM = Result(304)
    """This indicates that an OS call failed.
    """

    alias INVALID_HANDLE = Result(400)
    """This indicates that a resource handle passed to the API call was not
    valid. Resource handles are opaque types like ::CUstream and ::CUevent.
    """

    alias ILLEGAL_STATE = Result(401)
    """This indicates that a resource required by the API call is not in a
    valid state to perform the requested operation.
    """

    alias NOT_FOUND = Result(500)
    """This indicates that a named symbol was not found. Examples of symbols
    are global/constant variable names, driver function names, texture names,
    and surface names.
    """

    alias NOT_READY = Result(600)
    """This indicates that asynchronous operations issued previously have not
    completed yet. This result is not actually an error, but must be indicated
    differently than ::SUCCESS (which indicates completion). Calls that
    may return this value include ::cuEventQuery() and ::cuStreamQuery().
    """

    alias ILLEGAL_ADDRESS = Result(700)
    """While executing a kernel, the device encountered a
    load or store instruction on an invalid memory address.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_OUT_OF_RESOURCES = Result(701)
    """This indicates that a launch did not occur because it did not have
    appropriate resources. This error usually indicates that the user has
    attempted to pass too many arguments to the device kernel, or the
    kernel launch specifies too many threads for the kernel's register
    count. Passing arguments of the wrong size (i.e. a 64-bit pointer
    when a 32-bit int is expected) is equivalent to passing too many
    arguments and can also result in this error.
    """

    alias LAUNCH_TIMEOUT = Result(702)
    """This indicates that the device kernel took too long to execute. This can
    only occur if timeouts are enabled - see the device attribute
    ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_INCOMPATIBLE_TEXTURING = Result(703)
    """This error indicates a kernel launch that uses an incompatible texturing
    mode.
    """

    alias PEER_ACCESS_ALREADY_ENABLED = Result(704)
    """This error indicates that a call to ::cuCtxEnablePeerAccess() is
    trying to re-enable peer access to a context which has already
    had peer access to it enabled.
    """

    alias PEER_ACCESS_NOT_ENABLED = Result(705)
    """This error indicates that ::cuCtxDisablePeerAccess() is
    trying to disable peer access which has not been enabled yet
    via ::cuCtxEnablePeerAccess().
    """

    alias PRIMARY_CONTEXT_ACTIVE = Result(708)
    """This error indicates that the primary context for the specified device
    has already been initialized.
    """

    alias CONTEXT_IS_DESTROYED = Result(709)
    """This error indicates that the context current to the calling thread
    has been destroyed using ::cuCtxDestroy, or is a primary context which
    has not yet been initialized.
    """

    alias ASSERT = Result(710)
    """A device-side assert triggered during kernel execution. The context
    cannot be used anymore, and must be destroyed. All existing device
    memory allocations from this context are invalid and must be
    reconstructed if the program is to continue using CUDA.
    """

    alias TOO_MANY_PEERS = Result(711)
    """This error indicates that the hardware resources required to enable
    peer access have been exhausted for one or more of the devices
    passed to ::cuCtxEnablePeerAccess().
    """

    alias HOST_MEMORY_ALREADY_REGISTERED = Result(712)
    """This error indicates that the memory range passed to ::cuMemHostRegister
    has already been registered.
    """

    alias HOST_MEMORY_NOT_REGISTERED = Result(713)
    """This error indicates that the pointer passed to ::cuMemHostUnregister()
    does not correspond to any currently registered memory region.
    """

    alias HARDWARE_STACK_ERROR = Result(714)
    """While executing a kernel, the device encountered a stack error.
    This can be due to stack corruption or exceeding the stack size limit.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias ILLEGAL_INSTRUCTION = Result(715)
    """While executing a kernel, the device encountered an illegal instruction.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias MISALIGNED_ADDRESS = Result(716)
    """While executing a kernel, the device encountered a load or store
    instruction on a memory address which is not aligned.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias INVALID_ADDRESS_SPACE = Result(717)
    """While executing a kernel, the device encountered an instruction
    which can only operate on memory locations in certain address spaces
    (global, shared, or local), but was supplied a memory address not belonging
    to an allowed address space.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias INVALID_PC = Result(718)
    """While executing a kernel, the device program counter wrapped its address
    space. This leaves the process in an inconsistent state and any further CUDA
    work will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias LAUNCH_FAILED = Result(719)
    """An exception occurred on the device while executing a kernel. Common
    causes include dereferencing an invalid device pointer and accessing
    out of bounds shared memory. Less common cases can be system specific - more
    information about these cases can be found in the system specific user guide.
    This leaves the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process must be
    terminated and relaunched.
    """

    alias COOPERATIVE_LAUNCH_TOO_LARGE = Result(720)
    """This error indicates that the number of blocks launched per grid for a
    kernel that was launched via either ::cuLaunchCooperativeKernel or
    ::cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks
    as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor or
    ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of
    multiprocessors as specified by the device attribute
    ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
    """

    alias NOT_PERMITTED = Result(800)
    """This error indicates that the attempted operation is not permitted.
    """

    alias NOT_SUPPORTED = Result(801)
    """This error indicates that the attempted operation is not supported
    on the current system or device.
    """

    alias SYSTEM_NOT_READY = Result(802)
    """This error indicates that the system is not yet ready to start any CUDA
    work.  To continue using CUDA, verify the system configuration is in a
    valid state and all required driver daemons are actively running.
    More information about this error can be found in the system specific
    user guide.
    """

    alias SYSTEM_DRIVER_MISMATCH = Result(803)
    """This error indicates that there is a mismatch between the versions of
    the display driver and the CUDA driver. Refer to the compatibility
    documentation for supported versions.
    """

    alias COMPAT_NOT_SUPPORTED_ON_DEVICE = Result(804)
    """This error indicates that the system was upgraded to run with forward
    compatibility ut the visible hardware detected by CUDA does not support this
    configuration. Refer to the compatibility documentation for the supported
    hardware matrix or ensure that only supported hardware is visible during
    initialization via the CUDA_VISIBLE_DEVICES environment variable.
    """

    alias MPS_CONNECTION_FAILED = Result(805)
    """This error indicates that the MPS client failed to connect to the MPS
    control daemon or the MPS server.
    """

    alias MPS_RPC_FAILURE = Result(806)
    """This error indicates that the remote procedural call between the MPS
    server and the MPS client failed.
    """

    alias MPS_SERVER_NOT_READY = Result(807)
    """This error indicates that the MPS server is not ready to accept new MPS
    client requests. This error can be returned when the MPS server is in the
    process of recovering from a fatal failure.
    """

    alias MPS_MAX_CLIENTS_REACHED = Result(808)
    """This error indicates that the hardware resources required to create MPS
    client have been exhausted.
    """

    alias MPS_MAX_CONNECTIONS_REACHED = Result(809)
    """This error indicates the the hardware resources required to support
    device connections have been exhausted.
    """

    alias MPS_CLIENT_TERMINATED = Result(810)
    """This error indicates that the MPS client has been terminated by the
    server. To continue using CUDA, the process must be terminated and
    relaunched.
    """

    alias CDP_NOT_SUPPORTED = Result(811)
    """This error indicates that the module is using CUDA Dynamic Parallelism,
    but the current configuration, like MPS, does not support it.
    """

    alias CDP_VERSION_MISMATCH = Result(812)
    """This error indicates that a module contains an unsupported interaction
    between different versions of CUDA Dynamic Parallelism.
    """

    alias STREAM_CAPTURE_UNSUPPORTED = Result(900)
    """This error indicates that the operation is not permitted when
    the stream is capturing.
    """

    alias STREAM_CAPTURE_INVALIDATED = Result(901)
    """This error indicates that the current capture sequence on the stream
    has been invalidated due to a previous error.
    """

    alias STREAM_CAPTURE_MERGE = Result(902)
    """This error indicates that the operation would have resulted in a merge
    of two independent capture sequences.
    """

    alias STREAM_CAPTURE_UNMATCHED = Result(903)
    """This error indicates that the capture was not initiated in this stream.
    """

    alias STREAM_CAPTURE_UNJOINED = Result(904)
    """This error indicates that the capture sequence contains a fork that was
    not joined to the primary stream.
    """

    alias STREAM_CAPTURE_ISOLATION = Result(905)
    """This error indicates that a dependency would have been created which
    crosses the capture sequence boundary. Only implicit in-stream ordering
    dependencies are allowed to cross the boundary.
    """

    alias STREAM_CAPTURE_IMPLICIT = Result(906)
    """This error indicates a disallowed implicit dependency on a current
    capture sequence from cudaStreamLegacy.
    """

    alias CAPTURED_EVENT = Result(907)
    """This error indicates that the operation is not permitted on an event
    which was last recorded in a capturing stream.
    """

    alias STREAM_CAPTURE_WRONG_THREAD = Result(908)
    """A stream capture sequence not initiated with the
    ::CU_STREAM_CAPTURE_MODE_RELAXED argument to ::cuStreamBeginCapture was
    passed to ::cuStreamEndCapture in a different thread.
    """

    alias TIMEOUT = Result(909)
    """This error indicates that the timeout specified for the wait operation
    has lapsed.
    """

    alias GRAPH_EXEC_UPDATE_FAILURE = Result(910)
    """This error indicates that the graph update was not performed because it
    included changes which violated constraints specific to instantiated graph
    update.
    """

    alias EXTERNAL_DEVICE = Result(911)
    """This indicates that an async error has occurred in a device outside of
    CUDA. If CUDA was waiting for an external device's signal before consuming
    shared data, the external device signaled an error indicating that the data
    is not valid for consumption. This leaves the process in an inconsistent
    state and any further CUDA work will return the same error. To continue
    using CUDA, the process must be terminated and relaunched.
    """

    alias INVALID_CLUSTER_SIZE = Result(912)
    """Indicates a kernel launch error due to cluster misconfiguration.
    """

    alias UNKNOWN = Result(999)
    """This indicates that an unknown internal error has occurred.
    """

    fn __init__(code: Int32) -> Self:
        return Self {code: code}

    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> StringRef:
        if self == Result.SUCCESS:
            return "SUCCESS"
        elif self == Result.INVALID_VALUE:
            return "INVALID_VALUE"
        elif self == Result.OUT_OF_MEMORY:
            return "OUT_OF_MEMORY"
        elif self == Result.NOT_INITIALIZED:
            return "NOT_INITIALIZED"
        elif self == Result.DEINITIALIZED:
            return "DEINITIALIZED"
        elif self == Result.PROFILER_DISABLED:
            return "PROFILER_DISABLED"
        elif self == Result.PROFILER_NOT_INITIALIZED:
            return "PROFILER_NOT_INITIALIZED"
        elif self == Result.PROFILER_ALREADY_STARTED:
            return "PROFILER_ALREADY_STARTED"
        elif self == Result.PROFILER_ALREADY_STOPPED:
            return "PROFILER_ALREADY_STOPPED"
        elif self == Result.STUB_LIBRARY:
            return "STUB_LIBRARY"
        elif self == Result.DEVICE_UNAVAILABLE:
            return "DEVICE_UNAVAILABLE"
        elif self == Result.NO_DEVICE:
            return "NO_DEVICE"
        elif self == Result.INVALID_DEVICE:
            return "INVALID_DEVICE"
        elif self == Result.DEVICE_NOT_LICENSED:
            return "DEVICE_NOT_LICENSED"
        elif self == Result.INVALID_IMAGE:
            return "INVALID_IMAGE"
        elif self == Result.INVALID_CONTEXT:
            return "INVALID_CONTEXT"
        elif self == Result.CONTEXT_ALREADY_CURRENT:
            return "CONTEXT_ALREADY_CURRENT"
        elif self == Result.MAP_FAILED:
            return "MAP_FAILED"
        elif self == Result.UNMAP_FAILED:
            return "UNMAP_FAILED"
        elif self == Result.ARRAY_IS_MAPPED:
            return "ARRAY_IS_MAPPED"
        elif self == Result.ALREADY_MAPPED:
            return "ALREADY_MAPPED"
        elif self == Result.NO_BINARY_FOR_GPU:
            return "NO_BINARY_FOR_GPU"
        elif self == Result.ALREADY_ACQUIRED:
            return "ALREADY_ACQUIRED"
        elif self == Result.NOT_MAPPED:
            return "NOT_MAPPED"
        elif self == Result.NOT_MAPPED_AS_ARRAY:
            return "NOT_MAPPED_AS_ARRAY"
        elif self == Result.NOT_MAPPED_AS_POINTER:
            return "NOT_MAPPED_AS_POINTER"
        elif self == Result.ECC_UNCORRECTABLE:
            return "ECC_UNCORRECTABLE"
        elif self == Result.UNSUPPORTED_LIMIT:
            return "UNSUPPORTED_LIMIT"
        elif self == Result.CONTEXT_ALREADY_IN_USE:
            return "CONTEXT_ALREADY_IN_USE"
        elif self == Result.PEER_ACCESS_UNSUPPORTED:
            return "PEER_ACCESS_UNSUPPORTED"
        elif self == Result.INVALID_PTX:
            return "INVALID_PTX"
        elif self == Result.INVALID_GRAPHICS_CONTEXT:
            return "INVALID_GRAPHICS_CONTEXT"
        elif self == Result.NVLINK_UNCORRECTABLE:
            return "NVLINK_UNCORRECTABLE"
        elif self == Result.JIT_COMPILER_NOT_FOUND:
            return "JIT_COMPILER_NOT_FOUND"
        elif self == Result.UNSUPPORTED_PTX_VERSION:
            return "UNSUPPORTED_PTX_VERSION"
        elif self == Result.JIT_COMPILATION_DISABLED:
            return "JIT_COMPILATION_DISABLED"
        elif self == Result.UNSUPPORTED_EXEC_AFFINITY:
            return "UNSUPPORTED_EXEC_AFFINITY"
        elif self == Result.UNSUPPORTED_DEVSIDE_SYNC:
            return "UNSUPPORTED_DEVSIDE_SYNC"
        elif self == Result.INVALID_SOURCE:
            return "INVALID_SOURCE"
        elif self == Result.FILE_NOT_FOUND:
            return "FILE_NOT_FOUND"
        elif self == Result.SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "SHARED_OBJECT_SYMBOL_NOT_FOUND"
        elif self == Result.SHARED_OBJECT_INIT_FAILED:
            return "SHARED_OBJECT_INIT_FAILED"
        elif self == Result.OPERATING_SYSTEM:
            return "OPERATING_SYSTEM"
        elif self == Result.INVALID_HANDLE:
            return "INVALID_HANDLE"
        elif self == Result.ILLEGAL_STATE:
            return "ILLEGAL_STATE"
        elif self == Result.NOT_FOUND:
            return "NOT_FOUND"
        elif self == Result.NOT_READY:
            return "NOT_READY"
        elif self == Result.ILLEGAL_ADDRESS:
            return "ILLEGAL_ADDRESS"
        elif self == Result.LAUNCH_OUT_OF_RESOURCES:
            return "LAUNCH_OUT_OF_RESOURCES"
        elif self == Result.LAUNCH_TIMEOUT:
            return "LAUNCH_TIMEOUT"
        elif self == Result.LAUNCH_INCOMPATIBLE_TEXTURING:
            return "LAUNCH_INCOMPATIBLE_TEXTURING"
        elif self == Result.PEER_ACCESS_ALREADY_ENABLED:
            return "PEER_ACCESS_ALREADY_ENABLED"
        elif self == Result.PEER_ACCESS_NOT_ENABLED:
            return "PEER_ACCESS_NOT_ENABLED"
        elif self == Result.PRIMARY_CONTEXT_ACTIVE:
            return "PRIMARY_CONTEXT_ACTIVE"
        elif self == Result.CONTEXT_IS_DESTROYED:
            return "CONTEXT_IS_DESTROYED"
        elif self == Result.ASSERT:
            return "ASSERT"
        elif self == Result.TOO_MANY_PEERS:
            return "TOO_MANY_PEERS"
        elif self == Result.HOST_MEMORY_ALREADY_REGISTERED:
            return "HOST_MEMORY_ALREADY_REGISTERED"
        elif self == Result.HOST_MEMORY_NOT_REGISTERED:
            return "HOST_MEMORY_NOT_REGISTERED"
        elif self == Result.HARDWARE_STACK_ERROR:
            return "HARDWARE_STACK_ERROR"
        elif self == Result.ILLEGAL_INSTRUCTION:
            return "ILLEGAL_INSTRUCTION"
        elif self == Result.MISALIGNED_ADDRESS:
            return "MISALIGNED_ADDRESS"
        elif self == Result.INVALID_ADDRESS_SPACE:
            return "INVALID_ADDRESS_SPACE"
        elif self == Result.INVALID_PC:
            return "INVALID_PC"
        elif self == Result.LAUNCH_FAILED:
            return "LAUNCH_FAILED"
        elif self == Result.COOPERATIVE_LAUNCH_TOO_LARGE:
            return "COOPERATIVE_LAUNCH_TOO_LARGE"
        elif self == Result.NOT_PERMITTED:
            return "NOT_PERMITTED"
        elif self == Result.NOT_SUPPORTED:
            return "NOT_SUPPORTED"
        elif self == Result.SYSTEM_NOT_READY:
            return "SYSTEM_NOT_READY"
        elif self == Result.SYSTEM_DRIVER_MISMATCH:
            return "SYSTEM_DRIVER_MISMATCH"
        elif self == Result.COMPAT_NOT_SUPPORTED_ON_DEVICE:
            return "COMPAT_NOT_SUPPORTED_ON_DEVICE"
        elif self == Result.MPS_CONNECTION_FAILED:
            return "MPS_CONNECTION_FAILED"
        elif self == Result.MPS_RPC_FAILURE:
            return "MPS_RPC_FAILURE"
        elif self == Result.MPS_SERVER_NOT_READY:
            return "MPS_SERVER_NOT_READY"
        elif self == Result.MPS_MAX_CLIENTS_REACHED:
            return "MPS_MAX_CLIENTS_REACHED"
        elif self == Result.MPS_MAX_CONNECTIONS_REACHED:
            return "MPS_MAX_CONNECTIONS_REACHED"
        elif self == Result.MPS_CLIENT_TERMINATED:
            return "MPS_CLIENT_TERMINATED"
        elif self == Result.CDP_NOT_SUPPORTED:
            return "CDP_NOT_SUPPORTED"
        elif self == Result.CDP_VERSION_MISMATCH:
            return "CDP_VERSION_MISMATCH"
        elif self == Result.STREAM_CAPTURE_UNSUPPORTED:
            return "STREAM_CAPTURE_UNSUPPORTED"
        elif self == Result.STREAM_CAPTURE_INVALIDATED:
            return "STREAM_CAPTURE_INVALIDATED"
        elif self == Result.STREAM_CAPTURE_MERGE:
            return "STREAM_CAPTURE_MERGE"
        elif self == Result.STREAM_CAPTURE_UNMATCHED:
            return "STREAM_CAPTURE_UNMATCHED"
        elif self == Result.STREAM_CAPTURE_UNJOINED:
            return "STREAM_CAPTURE_UNJOINED"
        elif self == Result.STREAM_CAPTURE_ISOLATION:
            return "STREAM_CAPTURE_ISOLATION"
        elif self == Result.STREAM_CAPTURE_IMPLICIT:
            return "STREAM_CAPTURE_IMPLICIT"
        elif self == Result.CAPTURED_EVENT:
            return "CAPTURED_EVENT"
        elif self == Result.STREAM_CAPTURE_WRONG_THREAD:
            return "STREAM_CAPTURE_WRONG_THREAD"
        elif self == Result.TIMEOUT:
            return "TIMEOUT"
        elif self == Result.GRAPH_EXEC_UPDATE_FAILURE:
            return "GRAPH_EXEC_UPDATE_FAILURE"
        elif self == Result.EXTERNAL_DEVICE:
            return "EXTERNAL_DEVICE"
        elif self == Result.INVALID_CLUSTER_SIZE:
            return "INVALID_CLUSTER_SIZE"
        elif self == Result.UNKNOWN:
            return "UNKNOWN"
        else:
            return "<UNKNOWN>"

    fn __repr__(self) -> String:
        return self.__str__()


# ===----------------------------------------------------------------------===#
# Device Information
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct DeviceAttribute:
    var _value: Int32

    alias MAX_THREADS_PER_BLOCK = DeviceAttribute(1)
    """Maximum number of threads per block
    """

    alias MAX_BLOCK_DIM_X = DeviceAttribute(2)
    """Maximum block dimension X
    """

    alias MAX_BLOCK_DIM_Y = DeviceAttribute(3)
    """Maximum block dimension Y
    """

    alias MAX_BLOCK_DIM_Z = DeviceAttribute(4)
    """Maximum block dimension Z
    """

    alias MAX_GRID_DIM_X = DeviceAttribute(5)
    """Maximum grid dimension X
    """

    alias MAX_GRID_DIM_Y = DeviceAttribute(6)
    """Maximum grid dimension Y
    """

    alias MAX_GRID_DIM_Z = DeviceAttribute(7)
    """Maximum grid dimension Z
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK = DeviceAttribute(8)
    """Maximum shared memory available per block in bytes
    """

    alias SHARED_MEMORY_PER_BLOCK = DeviceAttribute(8)
    """Deprecated, use alias MAX_SHARED_MEMORY_PER_BLOCK
    """

    alias TOTAL_CONSTANT_MEMORY = DeviceAttribute(9)
    """Memory available on device for __constant__ variables in a CUDA C kernel
    in bytes
    """

    alias WARP_SIZE = DeviceAttribute(10)
    """Warp size in threads
    """

    alias MAX_PITCH = DeviceAttribute(11)
    """Maximum pitch in bytes allowed by memory copies
    """

    alias MAX_REGISTERS_PER_BLOCK = DeviceAttribute(12)
    """Maximum number of 32-bit registers available per block
    """

    alias REGISTERS_PER_BLOCK = DeviceAttribute(12)
    """Deprecated, use alias MAX_REGISTERS_PER_BLOCK
    """

    alias CLOCK_RATE = DeviceAttribute(13)
    """Typical clock frequency in kilohertz
    """

    alias TEXTURE_ALIGNMENT = DeviceAttribute(14)
    """Alignment requirement for textures
    """

    alias GPU_OVERLAP = DeviceAttribute(15)
    """Device can possibly copy memory and execute a kernel concurrently.
    Deprecated. Use instead alias ASYNC_ENGINE_COUNT.)
    """

    alias MULTIPROCESSOR_COUNT = DeviceAttribute(16)
    """Number of multiprocessors on device
    """

    alias KERNEL_EXEC_TIMEOUT = DeviceAttribute(17)
    """Specifies whether there is a run time limit on kernels
    """

    alias INTEGRATED = DeviceAttribute(18)
    """Device is integrated with host memory
    """

    alias CAN_MAP_HOST_MEMORY = DeviceAttribute(19)
    """Device can map host memory into CUDA address space
    """

    alias COMPUTE_MODE = DeviceAttribute(20)
    """Compute mode (See ::CUcomputemode for details))
    """

    alias MAXIMUM_TEXTURE1D_WIDTH = DeviceAttribute(21)
    """Maximum 1D texture width
    """

    alias MAXIMUM_TEXTURE2D_WIDTH = DeviceAttribute(22)
    """Maximum 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_HEIGHT = DeviceAttribute(23)
    """Maximum 2D texture height
    """

    alias MAXIMUM_TEXTURE3D_WIDTH = DeviceAttribute(24)
    """Maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT = DeviceAttribute(25)
    """Maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH = DeviceAttribute(26)
    """Maximum 3D texture depth
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH = DeviceAttribute(27)
    """Maximum 2D layered texture width
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = DeviceAttribute(28)
    """Maximum 2D layered texture height
    """

    alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS = DeviceAttribute(29)
    """Maximum layers in a 2D layered texture
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_WIDTH = DeviceAttribute(27)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = DeviceAttribute(28)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    """

    alias MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = DeviceAttribute(29)
    """Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    """

    alias SURFACE_ALIGNMENT = DeviceAttribute(30)
    """Alignment requirement for surfaces
    """

    alias CONCURRENT_KERNELS = DeviceAttribute(31)
    """Device can possibly execute multiple kernels concurrently
    """

    alias ECC_ENABLED = DeviceAttribute(32)
    """Device has ECC support enabled
    """

    alias PCI_BUS_ID = DeviceAttribute(33)
    """PCI bus ID of the device
    """

    alias PCI_DEVICE_ID = DeviceAttribute(34)
    """PCI device ID of the device
    """

    alias TCC_DRIVER = DeviceAttribute(35)
    """Device is using TCC driver model
    """

    alias MEMORY_CLOCK_RATE = DeviceAttribute(36)
    """Peak memory clock frequency in kilohertz
    """

    alias GLOBAL_MEMORY_BUS_WIDTH = DeviceAttribute(37)
    """Global memory bus width in bits
    """

    alias L2_CACHE_SIZE = DeviceAttribute(38)
    """Size of L2 cache in bytes
    """

    alias MAX_THREADS_PER_MULTIPROCESSOR = DeviceAttribute(39)
    """Maximum resident threads per multiprocessor
    """

    alias ASYNC_ENGINE_COUNT = DeviceAttribute(40)
    """Number of asynchronous engines
    """

    alias UNIFIED_ADDRESSING = DeviceAttribute(41)
    """Device shares a unified address space with the host
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_WIDTH = DeviceAttribute(42)
    """Maximum 1D layered texture width
    """

    alias MAXIMUM_TEXTURE1D_LAYERED_LAYERS = DeviceAttribute(43)
    """Maximum layers in a 1D layered texture
    """

    alias CAN_TEX2D_GATHER = DeviceAttribute(44)
    """Deprecated, do not use.)
    """

    alias MAXIMUM_TEXTURE2D_GATHER_WIDTH = DeviceAttribute(45)
    """Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE2D_GATHER_HEIGHT = DeviceAttribute(46)
    """Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    """

    alias MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = DeviceAttribute(47)
    """Alternate maximum 3D texture width
    """

    alias MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = DeviceAttribute(48)
    """Alternate maximum 3D texture height
    """

    alias MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = DeviceAttribute(49)
    """Alternate maximum 3D texture depth
    """

    alias PCI_DOMAIN_ID = DeviceAttribute(50)
    """PCI domain ID of the device
    """

    alias TEXTURE_PITCH_ALIGNMENT = DeviceAttribute(51)
    """Pitch alignment requirement for textures
    """

    alias MAXIMUM_TEXTURECUBEMAP_WIDTH = DeviceAttribute(52)
    """Maximum cubemap texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = DeviceAttribute(53)
    """Maximum cubemap layered texture width/height
    """

    alias MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = DeviceAttribute(54)
    """Maximum layers in a cubemap layered texture
    """

    alias MAXIMUM_SURFACE1D_WIDTH = DeviceAttribute(55)
    """Maximum 1D surface width
    """

    alias MAXIMUM_SURFACE2D_WIDTH = DeviceAttribute(56)
    """Maximum 2D surface width
    """

    alias MAXIMUM_SURFACE2D_HEIGHT = DeviceAttribute(57)
    """Maximum 2D surface height
    """

    alias MAXIMUM_SURFACE3D_WIDTH = DeviceAttribute(58)
    """Maximum 3D surface width
    """

    alias MAXIMUM_SURFACE3D_HEIGHT = DeviceAttribute(59)
    """Maximum 3D surface height
    """

    alias MAXIMUM_SURFACE3D_DEPTH = DeviceAttribute(60)
    """Maximum 3D surface depth
    """

    alias MAXIMUM_SURFACE1D_LAYERED_WIDTH = DeviceAttribute(61)
    """Maximum 1D layered surface width
    """

    alias MAXIMUM_SURFACE1D_LAYERED_LAYERS = DeviceAttribute(62)
    """Maximum layers in a 1D layered surface
    """

    alias MAXIMUM_SURFACE2D_LAYERED_WIDTH = DeviceAttribute(63)
    """Maximum 2D layered surface width
    """

    alias MAXIMUM_SURFACE2D_LAYERED_HEIGHT = DeviceAttribute(64)
    """Maximum 2D layered surface height
    """

    alias MAXIMUM_SURFACE2D_LAYERED_LAYERS = DeviceAttribute(65)
    """Maximum layers in a 2D layered surface
    """

    alias MAXIMUM_SURFACECUBEMAP_WIDTH = DeviceAttribute(66)
    """Maximum cubemap surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = DeviceAttribute(67)
    """Maximum cubemap layered surface width
    """

    alias MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = DeviceAttribute(68)
    """Maximum layers in a cubemap layered surface
    """

    alias MAXIMUM_TEXTURE1D_LINEAR_WIDTH = DeviceAttribute(69)
    """Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or
    cuDeviceGetTexture1DLinearMaxWidth() instead.)
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_WIDTH = DeviceAttribute(70)
    """Maximum 2D linear texture width
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = DeviceAttribute(71)
    """Maximum 2D linear texture height
    """

    alias MAXIMUM_TEXTURE2D_LINEAR_PITCH = DeviceAttribute(72)
    """Maximum 2D linear texture pitch in bytes
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = DeviceAttribute(73)
    """Maximum mipmapped 2D texture width
    """

    alias MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = DeviceAttribute(74)
    """Maximum mipmapped 2D texture height
    """

    alias COMPUTE_CAPABILITY_MAJOR = DeviceAttribute(75)
    """Major compute capability version number
    """

    alias COMPUTE_CAPABILITY_MINOR = DeviceAttribute(76)
    """Minor compute capability version number
    """

    alias MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = DeviceAttribute(77)
    """Maximum mipmapped 1D texture width
    """

    alias STREAM_PRIORITIES_SUPPORTED = DeviceAttribute(78)
    """Device supports stream priorities
    """

    alias GLOBAL_L1_CACHE_SUPPORTED = DeviceAttribute(79)
    """Device supports caching globals in L1
    """

    alias LOCAL_L1_CACHE_SUPPORTED = DeviceAttribute(80)
    """Device supports caching locals in L1
    """

    alias MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = DeviceAttribute(81)
    """Maximum shared memory available per multiprocessor in bytes
    """

    alias MAX_REGISTERS_PER_MULTIPROCESSOR = DeviceAttribute(82)
    """Maximum number of 32-bit registers available per multiprocessor
    """

    alias MANAGED_MEMORY = DeviceAttribute(83)
    """Device can allocate managed memory on this system
    """

    alias MULTI_GPU_BOARD = DeviceAttribute(84)
    """Device is on a multi-GPU board
    """

    alias MULTI_GPU_BOARD_GROUP_ID = DeviceAttribute(85)
    """Unique id for a group of devices on the same multi-GPU board
    """

    alias HOST_NATIVE_ATOMIC_SUPPORTED = DeviceAttribute(86)
    """Link between the device and the host supports native atomic operations
    (this is a placeholder attribute, and is not supported on any current
    hardware).
    """

    alias SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = DeviceAttribute(87)
    """Ratio of single precision performance (in floating-point operations per
    second) to double precision performance.
    """

    alias PAGEABLE_MEMORY_ACCESS = DeviceAttribute(88)
    """Device supports coherently accessing pageable memory without calling
    cudaHostRegister on it.
    """

    alias CONCURRENT_MANAGED_ACCESS = DeviceAttribute(89)
    """Device can coherently access managed memory concurrently with the CPU
    """

    alias COMPUTE_PREEMPTION_SUPPORTED = DeviceAttribute(90)
    """Device supports compute preemption.
    """

    alias CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = DeviceAttribute(91)
    """Device can access host registered memory at the same virtual address as
    the CPU
    """

    alias CAN_USE_STREAM_MEM_OPS_V1 = DeviceAttribute(92)
    """Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related
    APIs are supported.
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = DeviceAttribute(93)
    """Deprecated, along with v1 MemOps API, 64-bit operations are supported in
    ::cuStreamBatchMemOp and related APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = DeviceAttribute(94)
    """Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is
    supported.
    """

    alias COOPERATIVE_LAUNCH = DeviceAttribute(95)
    """Device supports launching cooperative kernels via
    ::cuLaunchCooperativeKernel
    """

    alias COOPERATIVE_MULTI_DEVICE_LAUNCH = DeviceAttribute(96)
    """Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.)
    """

    alias MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = DeviceAttribute(97)
    """Maximum optin shared memory per block
    """

    alias CAN_FLUSH_REMOTE_WRITES = DeviceAttribute(98)
    """The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the
    ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.
    See \ref CUDA_MEMOP for additional details.
    """

    alias HOST_REGISTER_SUPPORTED = DeviceAttribute(99)
    """Device supports host memory registration via ::cudaHostRegister.
    """

    alias PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = DeviceAttribute(100)
    """Device accesses pageable memory via the host's page tables.
    """

    alias DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = DeviceAttribute(101)
    """The host can directly access managed memory on the device without
    migration.
    """

    alias VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = DeviceAttribute(102)
    """Deprecated, Use alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    """

    alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = DeviceAttribute(102)
    """Device supports virtual memory management APIs like
    ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
    """

    alias HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = DeviceAttribute(103)
    """Device supports exporting memory to a posix file descriptor with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = DeviceAttribute(104)
    """Device supports exporting memory to a Win32 NT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = DeviceAttribute(105)
    """Device supports exporting memory to a Win32 KMT handle with
    ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    """

    alias MAX_BLOCKS_PER_MULTIPROCESSOR = DeviceAttribute(106)
    """Maximum number of blocks per multiprocessor
    """

    alias GENERIC_COMPRESSION_SUPPORTED = DeviceAttribute(107)
    """Device supports compression of memory
    """

    alias MAX_PERSISTING_L2_CACHE_SIZE = DeviceAttribute(108)
    """Maximum L2 persisting lines capacity setting in bytes.
    """

    alias MAX_ACCESS_POLICY_WINDOW_SIZE = DeviceAttribute(109)
    """Maximum value of CUaccessPolicyWindow::num_bytes.
    """

    alias GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = DeviceAttribute(110)
    """Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
    """

    alias RESERVED_SHARED_MEMORY_PER_BLOCK = DeviceAttribute(111)
    """Shared memory reserved by CUDA driver per block in bytes
    """

    alias SPARSE_CUDA_ARRAY_SUPPORTED = DeviceAttribute(112)
    """Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    """

    alias READ_ONLY_HOST_REGISTER_SUPPORTED = DeviceAttribute(113)
    """Device supports using the ::cuMemHostRegister flag
    ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as
    read-only to the GPU
    """

    alias TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = DeviceAttribute(114)
    """External timeline semaphore interop is supported on the device
    """

    alias MEMORY_POOLS_SUPPORTED = DeviceAttribute(115)
    """Device supports using the ::cuMemAllocAsync and ::cuMemPool family of
    APIs
    """

    alias GPU_DIRECT_RDMA_SUPPORTED = DeviceAttribute(116)
    """Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages
    (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    """

    alias GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = DeviceAttribute(117)
    """The returned attribute shall be interpreted as a bitmask, where the
    individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions
    enum
    """

    alias GPU_DIRECT_RDMA_WRITES_ORDERING = DeviceAttribute(118)
    """GPUDirect RDMA writes to the device do not need to be flushed for
    consumers within the scope indicated by the returned attribute. See
    ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
    """

    alias MEMPOOL_SUPPORTED_HANDLE_TYPES = DeviceAttribute(119)
    """Handle types supported with mempool based IPC
    """

    alias CLUSTER_LAUNCH = DeviceAttribute(120)
    """Indicates device supports cluster launch
    """

    alias DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = DeviceAttribute(121)
    """Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
    """

    alias CAN_USE_64_BIT_STREAM_MEM_OPS = DeviceAttribute(122)
    """64-bit operations are supported in ::cuStreamBatchMemOp and related
    MemOp APIs.
    """

    alias CAN_USE_STREAM_WAIT_VALUE_NOR = DeviceAttribute(123)
    """::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
    """

    alias DMA_BUF_SUPPORTED = DeviceAttribute(124)
    """Device supports buffer sharing with dma_buf mechanism.
    """

    alias IPC_EVENT_SUPPORTED = DeviceAttribute(125)
    """Device supports IPC Events.)
    """

    alias MEM_SYNC_DOMAIN_COUNT = DeviceAttribute(126)
    """Number of memory domains the device supports.
    """

    alias TENSOR_MAP_ACCESS_SUPPORTED = DeviceAttribute(127)
    """Device supports accessing memory using Tensor Map.
    """

    alias UNIFIED_FUNCTION_POINTERS = DeviceAttribute(129)
    """Device supports unified function pointers.
    """

    alias MULTICAST_SUPPORTED = DeviceAttribute(132)
    """Device supports switch multicast and reduction operations.
    """

    fn __init__(value: Int32) -> Self:
        return Self {_value: value}


fn device_count() raises -> Int:
    var res: Int32 = 0
    _check_error(
        _get_dylib_function[fn (Pointer[Int32]) -> Result]("cuDeviceGetCount")(
            Pointer.address_of(res)
        )
    )
    return res.to_int()


@value
@register_passable("trivial")
struct Device:
    var id: Int32

    fn __init__(id: Int = 0) -> Self:
        return Self {id: id}

    fn __str__(self) raises -> String:
        let dylib = _get_dylib()
        var res = String("name: ") + self._name(dylib) + "\n"
        res += (
            String("memory: ") + _human_memory(self._total_memory(dylib)) + "\n"
        )
        res += (
            String("compute_capability: ")
            + self._query(dylib, DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
            + "."
            + self._query(dylib, DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
            + "\n"
        )
        res += (
            String("clock_rate: ")
            + self._query(dylib, DeviceAttribute.CLOCK_RATE)
            + "\n"
        )
        res += (
            String("warp_size: ")
            + self._query(dylib, DeviceAttribute.WARP_SIZE)
            + "\n"
        )
        res += (
            String("max_threads_per_block: ")
            + self._query(dylib, DeviceAttribute.MAX_THREADS_PER_BLOCK)
            + "\n"
        )
        res += (
            String("max_shared_memory: ")
            + _human_memory(
                self._query(dylib, DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
            )
            + "\n"
        )
        res += (
            String("max_block: ")
            + Dim(
                self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_X),
                self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_Y),
                self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_Z),
            ).__str__()
            + "\n"
        )
        res += (
            String("max_grid: ")
            + Dim(
                self._query(dylib, DeviceAttribute.MAX_GRID_DIM_X),
                self._query(dylib, DeviceAttribute.MAX_GRID_DIM_Y),
                self._query(dylib, DeviceAttribute.MAX_GRID_DIM_Z),
            ).__str__()
            + "\n"
        )

        return res

    fn _name(self, dylib: DLHandle) -> String:
        alias buffer_size = 256
        let buffer = stack_allocation[buffer_size, DType.int8]()

        let ok = _get_dylib_function[
            fn (DTypePointer[DType.int8], Int32, Device) -> Result
        ](dylib, "cuDeviceGetName")(buffer, Int32(buffer_size), self)

        return StringRef(buffer.address)

    fn _total_memory(self, dylib: DLHandle) raises -> Int:
        var res: Int = 0
        _check_error(
            _get_dylib_function[fn (Pointer[Int], Device) -> Result](
                dylib, "cuDeviceTotalMem_v2"
            )(Pointer.address_of(res), self)
        )
        return res

    fn _query(self, dylib: DLHandle, attr: DeviceAttribute) raises -> Int:
        var res: Int32 = 0
        _check_error(
            _get_dylib_function[
                fn (Pointer[Int32], DeviceAttribute, Device) -> Result
            ](dylib, "cuDeviceGetAttribute")(
                Pointer.address_of(res), attr, self
            )
        )
        return res.to_int()


# ===----------------------------------------------------------------------===#
# Stream
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _StreamImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Stream[is_borrowed: Bool = False]:
    var stream: _StreamImpl

    fn __init__(inout self, stream: _StreamImpl):
        self.stream = stream

    fn __init__(inout self, flags: Int = 0) raises:
        var stream = _StreamImpl()

        _check_error(
            _get_dylib_function[fn (Pointer[_StreamImpl], Int32) -> Result](
                "cuStreamCreate"
            )(Pointer.address_of(stream), Int32(0))
        )

        self.stream = stream

    fn __del__(owned self) raises:
        @parameter
        if is_borrowed:
            return
        if self.stream:
            _check_error(
                _get_dylib_function[fn (_StreamImpl) -> Result](
                    "cuStreamDestroy"
                )(self.stream)
            )

    fn __moveinit__(inout self, owned existing: Self):
        self.stream = existing.stream
        existing.stream = _StreamImpl()

    fn __takeinit__(inout self, inout existing: Self):
        self.stream = existing.stream
        existing.stream = _StreamImpl()


# ===----------------------------------------------------------------------===#
# Context
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _ContextImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Context:
    var ctx: _ContextImpl

    fn __init__(inout self) raises:
        self.__init__(Device())

    fn __init__(inout self, device: Device, flags: Int = 0) raises:
        var ctx = _ContextImpl()

        _check_error(
            _get_dylib_function[
                fn (Pointer[_ContextImpl], Int32, Device) -> Result
            ]("cuCtxCreate_v2")(Pointer.address_of(ctx), flags, device)
        )
        self.ctx = ctx

    fn __del__(owned self) raises:
        if self.ctx:
            _check_error(
                _get_dylib_function[fn (_ContextImpl) -> Result](
                    "cuCtxDestroy_v2"
                )(self.ctx)
            )

    fn __enter__(owned self) -> Self:
        return self ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        existing.ctx = _ContextImpl()

    fn __takeinit__(inout self, inout existing: Self):
        self.ctx = existing.ctx
        existing.ctx = _ContextImpl()


# ===----------------------------------------------------------------------===#
# JitOptions
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct JitOptions:
    var _value: Int32

    alias MAX_REGISTERS: JitOptions = 0
    """Max number of registers that a thread may use.
      Option type: unsigned int
      Applies to: compiler only
    """

    alias THREADS_PER_BLOCK: JitOptions = 1
    """IN: Specifies minimum number of threads per block to target compilation
    for
    OUT: Returns the number of threads the compiler actually targeted.
    This restricts the resource utilization of the compiler (e.g. max
    registers) such that a block with the given number of threads should be
    able to launch based on register limitations. Note, this option does not
    currently take into account any other resource limitations, such as
    shared memory utilization.
    Cannot be combined with ::CU_JIT_TARGET.
    Option type: unsigned int
    Applies to: compiler only
    """
    alias WALL_TIME: JitOptions = 2
    """Overwrites the option value with the total wall clock time, in
      milliseconds, spent in the compiler and linker
      Option type: float
      Applies to: compiler and linker
    """
    alias INFO_LOG_BUFFER: JitOptions = 3
    """Pointer to a buffer in which to print any log messages
      that are informational in nature (the buffer size is specified via
      option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
      Option type: char *
      Applies to: compiler and linker
    """
    alias INFO_LOG_BUFFER_SIZE_BYTES: JitOptions = 4
    """IN: Log buffer size in bytes.  Log messages will be capped at this size
      (including null terminator)
      OUT: Amount of log buffer filled with messages
      Option type: unsigned int
      Applies to: compiler and linker
    """
    alias ERROR_LOG_BUFFER: JitOptions = 5
    """Pointer to a buffer in which to print any log messages that
      reflect errors (the buffer size is specified via option
      ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
      Option type: char *
      Applies to: compiler and linker
    """
    alias ERROR_LOG_BUFFER_SIZE_BYTES: JitOptions = 6
    """IN: Log buffer size in bytes.  Log messages will be capped at this size
      (including null terminator)
      OUT: Amount of log buffer filled with messages
      Option type: unsigned int
      Applies to: compiler and linker
    """
    alias OPTIMIZATION_LEVEL: JitOptions = 7
    """Level of optimizations to apply to generated code (0 - 4), with 4
      being the default and highest level of optimizations.
      Option type: unsigned int
      Applies to: compiler only
    """
    alias TARGET_FROM_CUCONTEXT: JitOptions = 8
    """No option value required. Determines the target based on the current
      attached context (default)
      Option type: No option value needed
      Applies to: compiler and linker
    """
    alias TARGET: JitOptions = 9
    """Target is chosen based on supplied ::CUjit_target.  Cannot be
      combined with ::CU_JIT_THREADS_PER_BLOCK.
      Option type: unsigned int for enumerated type ::CUjit_target
      Applies to: compiler and linker
    """
    alias FALLBACK_STRATEGY: JitOptions = 10
    """Specifies choice of fallback strategy if matching cubin is not found.
      Choice is based on supplied ::CUjit_fallback.  This option cannot be
      used with cuLink* APIs as the linker requires exact matches.
      Option type: unsigned int for enumerated type ::CUjit_fallback
      Applies to: compiler only
    """
    alias GENERATE_DEBUG_INFO: JitOptions = 11
    """Specifies whether to create debug information in output (-g)
      (0: false, default)
      Option type: int
      Applies to: compiler and linker
    """
    alias LOG_VERBOSE: JitOptions = 12
    """Generate verbose log messages (0: false, default)
      Option type: int
      Applies to: compiler and linker
    """
    alias GENERATE_LINE_INFO: JitOptions = 13
    """Generate line number information (-lineinfo) (0: false, default)
      Option type: int
      Applies to: compiler only
    """
    alias CACHE_MODE: JitOptions = 14
    """Specifies whether to enable caching explicitly (-dlcm)
      Choice is based on supplied ::CUjit_cacheMode_enum.
      Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum
      Applies to: compiler only
    """
    alias NEW_SM3X_OPT: JitOptions = 15
    """[[depricated]]
      This jit option is deprecated and should not be used.
    """
    alias FAST_COMPILE: JitOptions = 16
    """This jit option is used for internal purpose only.
    """
    alias GLOBAL_SYMBOL_NAMES: JitOptions = 17
    """Array of device symbol names that will be relocated to the corresponding
      host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.
      Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.
      When loading a device module, driver will relocate all encountered
      unresolved symbols to the host addresses.
      It is only allowed to register symbols that correspond to unresolved
      global variables.
      It is illegal to register the same device symbol at multiple addresses.
      Option type: const char **
      Applies to: dynamic linker only
    """
    alias GLOBAL_SYMBOL_ADDRESSES: JitOptions = 18
    """Array of host addresses that will be used to relocate corresponding
      device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.
      Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.
      Option type: void **
      Applies to: dynamic linker only
    """
    alias GLOBAL_SYMBOL_COUNT: JitOptions = 19
    """Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
      ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.
      Option type: unsigned int
      Applies to: dynamic linker only
    """
    alias LTO: JitOptions = 20
    """[[depricated]]
      Enable link-time optimization (-dlto) for device code (Disabled by default).
      This option is not supported on 32-bit platforms.
      Option type: int
      Applies to: compiler and linker
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias FTZ: JitOptions = 21
    """[[depricated]]
      Control single-precision denormals (-ftz) support (0: false, default).
      1 : flushes denormal values to zero
      0 : preserves denormal values
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias PREC_DIV: JitOptions = 22
    """[[depricated]]
      Control single-precision floating-point division and reciprocals
      (-prec-div) support (1: true, default).
      1 : Enables the IEEE round-to-nearest mode
      0 : Enables the fast approximation mode
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias PREC_SQRT: JitOptions = 23
    """[[depricated]]
      Control single-precision floating-point square root
      (-prec-sqrt) support (1: true, default).
      1 : Enables the IEEE round-to-nearest mode
      0 : Enables the fast approximation mode
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias FMA: JitOptions = 24
    """[[depricated]]
      Enable/Disable the contraction of floating-point multiplies
      and adds/subtracts into floating-point multiply-add (-fma)
      operations (1: Enable, default; 0: Disable).
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_KERNEL_NAMES: JitOptions = 25
    """[[depricated]]
      Array of kernel names that should be preserved at link time while others
      can be removed.
      Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.
      Note that kernel names can be mangled by the compiler in which case the
      mangled name needs to be specified.
      Wildcard "*" can be used to represent zero or more characters instead of
      specifying the full or mangled name.
      It is important to note that the wildcard "*" is also added implicitly.
      For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
      thus preserve all kernels with those names. This can be avoided by providing
      a more specific name like "barfoobaz".
      Option type: const char **
      Applies to: dynamic linker only
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_KERNEL_COUNT: JitOptions = 26
    """[[depricated]]
      Number of entries in ::CU_JIT_REFERENCED_KERNEL_NAMES array.
      Option type: unsigned int
      Applies to: dynamic linker only
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_VARIABLE_NAMES: JitOptions = 27
    """[[depricated]]
      Array of variable names (__device__ and/or __constant__) that should be
      preserved at link time while others can be removed.
      Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.
      Note that variable names can be mangled by the compiler in which case the
      mangled name needs to be specified.
      Wildcard "*" can be used to represent zero or more characters instead of
      specifying the full or mangled name.
      It is important to note that the wildcard "*" is also added implicitly.
      For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
      thus preserve all variables with those names. This can be avoided by providing
      a more specific name like "barfoobaz".
      Option type: const char **
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_VARIABLE_COUNT: JitOptions = 28
    """[[depricated]]
      Number of entries in ::CU_JIT_REFERENCED_VARIABLE_NAMES array.
      Option type: unsigned int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias OPTIMIZE_UNUSED_DEVICE_VARIABLES: JitOptions = 29
    """[[depricated]]
      This option serves as a hint to enable the JIT compiler/linker
      to remove constant (__constant__) and device (__device__) variables
      unreferenced in device code (Disabled by default).
      Note that host references to constant and device variables using APIs like
      ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
      the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias POSITION_INDEPENDENT_CODE: JitOptions = 30
    """Generate position independent code (0: false)
      Option type: int
      Applies to: compiler only
    """

    fn __init__() -> Self:
        return Self {_value: 0}

    fn __init__(value: Int) -> Self:
        return Self {_value: value}

    fn __init__(value: Int32) -> Self:
        return Self {_value: value}


# ===----------------------------------------------------------------------===#
# Module
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _ModuleImpl:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct ModuleHandle:
    var module: _ModuleImpl

    fn __init__(inout self):
        self.module = _ModuleImpl()

    fn __init__(inout self, path: Path) raises:
        var module = _ModuleImpl()
        let path_cstr = _cleanup_string(path.__str__())

        _check_error(
            _get_dylib_function[
                fn (Pointer[_ModuleImpl], DTypePointer[DType.int8]) -> Result
            ]("cuModuleLoad")(Pointer.address_of(module), path_cstr._as_ptr())
        )

        _ = path_cstr
        self.module = module

    fn __init__(
        inout self,
        content: String,
        debug: Bool = False,
        verbose: Bool = False,
    ) raises:
        var module = _ModuleImpl()
        if debug or verbose:
            alias buffer_size = 4096
            alias max_num_options = 6
            var num_options = 0

            let info_buffer = stack_allocation[buffer_size, Int8]()
            let error_buffer = stack_allocation[buffer_size, Int8]()

            let opts = stack_allocation[max_num_options, JitOptions]()
            let option_vals = stack_allocation[
                max_num_options, Pointer[AnyType]
            ]()

            opts.store(num_options, JitOptions.INFO_LOG_BUFFER)
            option_vals.store(num_options, info_buffer.bitcast[AnyType]())
            num_options += 1

            opts.store(num_options, JitOptions.INFO_LOG_BUFFER_SIZE_BYTES)
            option_vals.store(num_options, bitcast[AnyType](buffer_size))
            num_options += 1

            opts.store(num_options, JitOptions.ERROR_LOG_BUFFER)
            option_vals.store(num_options, info_buffer.bitcast[AnyType]())
            num_options += 1

            opts.store(num_options, JitOptions.ERROR_LOG_BUFFER_SIZE_BYTES)
            option_vals.store(num_options, bitcast[AnyType](buffer_size))
            num_options += 1

            if debug:
                opts.store(num_options, JitOptions.GENERATE_DEBUG_INFO)
                option_vals.store(num_options, bitcast[AnyType](1))
                num_options += 1

            # Note that content has already gone through _cleanup_asm and
            # is null terminated.
            let result = _get_dylib_function[
                # fmt: off
                fn (
                    Pointer[_ModuleImpl],
                    DTypePointer[DType.int8],
                    UInt32,
                    Pointer[JitOptions],
                    Pointer[Pointer[AnyType]]
                ) -> Result
                # fmt: on
            ]("cuModuleLoadDataEx")(
                Pointer.address_of(module),
                content._as_ptr(),
                UInt32(num_options),
                opts,
                option_vals,
            )

            if verbose:
                let info_buffer_str = StringRef(info_buffer)
                if info_buffer_str:
                    print(info_buffer_str)

                let error_buffer_str = StringRef(error_buffer)
                if error_buffer_str:
                    print(error_buffer_str)

            _check_error(result)
        else:
            # Note that content has already gone through _cleanup_asm and
            # is null terminated.
            _check_error(
                _get_dylib_function[
                    fn (
                        Pointer[_ModuleImpl], DTypePointer[DType.int8]
                    ) -> Result
                ]("cuModuleLoadData")(
                    Pointer.address_of(module), content._as_ptr()
                )
            )

        self.module = module
        _ = content

    fn __init__(inout self, content: String) raises:
        var module = _ModuleImpl()
        # Note that content has already gone through _cleanup_asm and
        # is null terminated.
        _check_error(
            _get_dylib_function[
                fn (Pointer[_ModuleImpl], DTypePointer[DType.int8]) -> Result
            ]("cuModuleLoadData")(Pointer.address_of(module), content._as_ptr())
        )
        self.module = module

    fn __del__(owned self) raises:
        if self.module:
            _check_error(
                _get_dylib_function[fn (_ModuleImpl) -> Result](
                    "cuModuleUnload"
                )(self.module)
            )

    fn load(self, name: String) raises -> FunctionHandle:
        var func = FunctionHandle()
        let name_cstr = _cleanup_string(name)

        _check_error(
            _get_dylib_function[
                # fmt: off
                fn (
                    Pointer[FunctionHandle],
                    _ModuleImpl,
                    DTypePointer[DType.int8]
                ) -> Result
                # fmt: on
            ]("cuModuleGetFunction")(
                Pointer.address_of(func), self.module, name_cstr._as_ptr()
            )
        )

        _ = name_cstr

        return func


# ===----------------------------------------------------------------------===#
# FunctionHandle
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct FunctionHandle:
    var handle: DTypePointer[DType.invalid]

    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __call__(self, grid_dim: Dim, block_dim: Dim, /, stream: Stream) raises:
        self._call_impl(
            grid_dim, block_dim, Pointer[Pointer[AnyType]](), stream=stream
        )

    fn __call__[
        T0: AnyType
    ](self, grid_dim: Dim, block_dim: Dim, arg0: T0, /, stream: Stream) raises:
        var _arg0 = arg0

        let args = stack_allocation[1, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1

        let args = stack_allocation[2, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2

        let args = stack_allocation[3, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3

        let args = stack_allocation[4, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType, T4: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4

        let args = stack_allocation[5, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5

        let args = stack_allocation[6, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())
        args.store(5, Pointer.address_of(_arg5).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6

        let args = stack_allocation[7, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())
        args.store(5, Pointer.address_of(_arg5).bitcast[AnyType]())
        args.store(6, Pointer.address_of(_arg6).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7

        let args = stack_allocation[8, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())
        args.store(5, Pointer.address_of(_arg5).bitcast[AnyType]())
        args.store(6, Pointer.address_of(_arg6).bitcast[AnyType]())
        args.store(7, Pointer.address_of(_arg7).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7
        var _arg8 = arg8

        let args = stack_allocation[9, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())
        args.store(5, Pointer.address_of(_arg5).bitcast[AnyType]())
        args.store(6, Pointer.address_of(_arg6).bitcast[AnyType]())
        args.store(7, Pointer.address_of(_arg7).bitcast[AnyType]())
        args.store(8, Pointer.address_of(_arg8).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
        T9: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        arg9: T9,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7
        var _arg8 = arg8
        var _arg9 = arg9

        let args = stack_allocation[10, Pointer[AnyType]]()
        args.store(0, Pointer.address_of(_arg0).bitcast[AnyType]())
        args.store(1, Pointer.address_of(_arg1).bitcast[AnyType]())
        args.store(2, Pointer.address_of(_arg2).bitcast[AnyType]())
        args.store(3, Pointer.address_of(_arg3).bitcast[AnyType]())
        args.store(4, Pointer.address_of(_arg4).bitcast[AnyType]())
        args.store(5, Pointer.address_of(_arg5).bitcast[AnyType]())
        args.store(6, Pointer.address_of(_arg6).bitcast[AnyType]())
        args.store(7, Pointer.address_of(_arg7).bitcast[AnyType]())
        args.store(8, Pointer.address_of(_arg8).bitcast[AnyType]())
        args.store(9, Pointer.address_of(_arg9).bitcast[AnyType]())

        self._call_impl(grid_dim, block_dim, args, stream=stream)

    fn _call_impl(
        self,
        grid_dim: Dim,
        block_dim: Dim,
        args: Pointer[Pointer[AnyType]],
        /,
        stream: Stream,
    ) raises:
        _check_error(
            _get_dylib_function[
                # fmt: off
                fn (
                  Self,
                  UInt32, # GridDimZ
                  UInt32, # GridDimY
                  UInt32, # GridDimX
                  UInt32, # BlockDimZ
                  UInt32, # BlockDimY
                  UInt32, # BlockDimX
                  UInt32, # SharedMemSize
                  _StreamImpl, # Stream
                  Pointer[Pointer[AnyType]], # Args
                  DTypePointer[DType.invalid] # Extra
                ) -> Result
                # fmt: on
            ]("cuLaunchKernel")(
                self.handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(0),
                stream.stream,
                args,
                DTypePointer[DType.invalid](),
            )
        )


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


struct Function[func_type: AnyType, func: func_type]:
    var mod_handle: ModuleHandle
    var func_handle: FunctionHandle

    fn __init__(inout self, debug: Bool = False, verbose: Bool = False) raises:
        alias name = get_linkage_name[func_type, func]()
        let ptx = _compile_nvptx_asm[func_type, func]()
        self.mod_handle = ModuleHandle(
            ptx, debug=debug, verbose=verbose or debug
        )
        self.func_handle = self.mod_handle.load(name)

    fn __del__(owned self) raises:
        pass

    fn __bool__(self) -> Bool:
        return self.func_handle.__bool__()

    fn __call__[
        T0: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(grid_dim, block_dim, stream=stream)

    fn __call__[
        T0: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(grid_dim, block_dim, arg0, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(grid_dim, block_dim, arg0, arg1, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(grid_dim, block_dim, arg0, arg1, arg2, stream=stream)

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim, block_dim, arg0, arg1, arg2, arg3, stream=stream
        )

    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType, T4: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim, block_dim, arg0, arg1, arg2, arg3, arg4, stream=stream
        )

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            stream=stream,
        )

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            stream=stream,
        )

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            stream=stream,
        )

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            stream=stream,
        )

    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
        T9: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        arg9: T9,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle(
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            stream=stream,
        )


# ===----------------------------------------------------------------------===#
# Dim
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Dim:
    var _value: StaticIntTuple[3]

    fn __init__(dims: Tuple[Int]) -> Self:
        return Self(dims.get[0, Int]())

    fn __init__(dims: Tuple[Int, Int]) -> Self:
        return Self(dims.get[1, Int](), dims.get[0, Int]())

    fn __init__(dims: Tuple[Int, Int, Int]) -> Self:
        return Self(dims.get[2, Int](), dims.get[1, Int](), dims.get[0, Int]())

    fn __init__(z: Int, y: Int, x: Int) -> Self:
        return Self {_value: Index(x, y, z)}

    fn __init__(y: Int, x: Int) -> Self:
        return Self(1, y, x)

    fn __init__(x: Int) -> Self:
        return Self(1, 1, x)

    fn __getitem__(self, idx: Int) -> Int:
        return self._value[idx]

    fn __str__(self) -> String:
        var res = String("(") + self.x() + ", "
        if self.y() != 1 or self.z() != 1:
            res += self.y()
            if self.z() != 1:
                res += ", " + String(self.z())
        res += ")"
        return res

    fn __repr__(self) -> String:
        return self.__str__()

    fn z(self) -> Int:
        return self[2]

    fn y(self) -> Int:
        return self[1]

    fn x(self) -> Int:
        return self[0]


# ===----------------------------------------------------------------------===#
# Memory
# ===----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> Pointer[type]:
    var ptr = Pointer[UInt32]()
    _check_error(
        _get_dylib_function[fn (Pointer[Pointer[UInt32]], Int) -> Result](
            "cuMemAlloc_v2"
        )(Pointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> DTypePointer[type]:
    let res = _malloc[SIMD[type, 1]](count)
    return DTypePointer[type](res.address)


fn _free[type: AnyType](ptr: Pointer[type]) raises:
    _check_error(
        _get_dylib_function[fn (Pointer[UInt32]) -> Result]("cuMemFree_v2")(
            ptr.bitcast[UInt32]()
        )
    )


fn _free[type: DType](ptr: DTypePointer[type]) raises:
    _free(ptr._as_scalar_pointer())


fn _copy_host_to_device[
    type: AnyType
](device_dest: Pointer[type], host_src: Pointer[type], count: Int) raises:
    _check_error(
        _get_dylib_function[
            fn (Pointer[UInt32], Pointer[AnyType], Int) -> Result
        ]("cuMemcpyHtoD_v2")(
            device_dest.bitcast[UInt32](),
            host_src.bitcast[AnyType](),
            count * sizeof[type](),
        )
    )


fn _copy_host_to_device[
    type: DType
](
    device_dest: DTypePointer[type], host_src: DTypePointer[type], count: Int
) raises:
    _copy_host_to_device[SIMD[type, 1]](
        device_dest._as_scalar_pointer(),
        host_src._as_scalar_pointer(),
        count,
    )


fn _copy_device_to_host[
    type: AnyType
](host_dest: Pointer[type], device_src: Pointer[type], count: Int) raises:
    _check_error(
        _get_dylib_function[
            fn (Pointer[AnyType], Pointer[UInt32], Int) -> Result
        ]("cuMemcpyDtoH_v2")(
            host_dest.bitcast[AnyType](),
            device_src.bitcast[UInt32](),
            count * sizeof[type](),
        )
    )


fn _copy_device_to_host[
    type: DType
](
    host_dest: DTypePointer[type], device_src: DTypePointer[type], count: Int
) raises:
    _copy_device_to_host[SIMD[type, 1]](
        host_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
    )


# ===----------------------------------------------------------------------===#
# Synchronize
# ===----------------------------------------------------------------------===#


fn synchronize() raises:
    _check_error(_get_dylib_function[fn () -> Result]("cuCtxSynchronize")())


# ===----------------------------------------------------------------------===#
# Compilation
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nvtx_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_75", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


@always_inline
fn __compile_nvptx_asm_impl[
    func_type: AnyType, func: func_type->asm: StringLiteral
]():
    param_return[
        __mlir_attr[
            `#kgen.param.expr<compile_assembly,`,
            _get_nvtx_target(),
            `, `,
            func,
            `> : !kgen.string`,
        ]
    ]


@always_inline
fn _compile_nvptx_asm[func_type: AnyType, func: func_type]() -> String:
    alias asm: StringLiteral
    __compile_nvptx_asm_impl[func_type, func -> asm]()
    return _cleanup_asm(asm)


@always_inline
fn _cleanup_asm(asm: String) -> String:
    return _cleanup_string(asm.replace(".version 6.3\n", ".version 8.1\n"))


@always_inline
fn _cleanup_string(name: String) -> String:
    return name + "\0"
