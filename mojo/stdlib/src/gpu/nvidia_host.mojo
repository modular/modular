# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA host functions."""

from sys.ffi import RTLD, DLHandle
from memory import stack_allocation
from utils.index import StaticIntTuple, Index
from math import floor

# ===----------------------------------------------------------------------===#
# Globals
# ===----------------------------------------------------------------------===#


alias CUDA_DRIVER_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so"

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


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


fn _init_dylib() -> Pointer[DLHandle]:
    let ptr = Pointer[DLHandle].alloc(1)
    let handle = DLHandle(CUDA_DRIVER_PATH, RTLD.NOW | RTLD.GLOBAL)
    _ = handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
    __get_address_as_lvalue(ptr.address) = handle
    return ptr


fn _destroy_dylib(ptr: Pointer[DLHandle]):
    __get_address_as_lvalue(ptr.address)._del_old()
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
    \\deprecated
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
            String("max_block: [")
            + self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_Z)
            + ", "
            + self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_Y)
            + ", "
            + self._query(dylib, DeviceAttribute.MAX_BLOCK_DIM_X)
            + "]\n"
        )
        res += (
            String("max_grid: [")
            + self._query(dylib, DeviceAttribute.MAX_GRID_DIM_Z)
            + ", "
            + self._query(dylib, DeviceAttribute.MAX_GRID_DIM_Y)
            + ", "
            + self._query(dylib, DeviceAttribute.MAX_GRID_DIM_X)
            + "]\n"
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


struct Module:
    var module: _ModuleImpl

    fn __init__(inout self):
        self.module = _ModuleImpl()

    fn __init__(inout self, path: String) raises:
        var module = _ModuleImpl()

        _check_error(
            _get_dylib_function[
                fn (Pointer[_ModuleImpl], DTypePointer[DType.int8]) -> Result
            ]("cuModuleLoad")(Pointer.address_of(module), path._as_ptr())
        )
        self.module = module

    fn __del__(owned self) raises:
        if self.module:
            _check_error(
                _get_dylib_function[fn (_ModuleImpl) -> Result](
                    "cuModuleUnload"
                )(self.module)
            )


# ===----------------------------------------------------------------------===#
# Dim
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Dim:
    var _value: StaticIntTuple[3]

    fn __init__(x: Int, y: Int = 1, z: Int = 1) -> Self:
        return Self {_value: Index(x, y, z)}

    fn __getitem__(self, idx: Int) -> Int:
        return self._value[idx]

    fn __str__(self) -> String:
        var res = String("(") + String(self[0]) + String(", ")
        if self[1] != 1 or self[2] != 1:
            res += String(self[1])
        if self[2] != 1:
            res += String(", ") + String(self[2])
        res += String(")")
        return res

    fn __repr__(self) -> String:
        return self.__str__()
