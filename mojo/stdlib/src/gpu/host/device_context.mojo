# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List, Optional
from os import abort
from sys import external_call, sizeof
from sys.ffi import c_size_t
from sys.param_env import env_get_int, is_defined

from gpu.host._compile import _get_nvptx_target
from gpu.host.context import Context
from gpu.host.cuda_instance import CudaInstance, LaunchAttribute
from gpu.host.device import Device
from gpu.host.event import Event
from gpu.host.function import Function
from gpu.host.memory import _memset_async
from gpu.host.stream import Stream

from ._compile import _get_nvptx_fn_name
from ._utils import _check_error, _StreamHandle

# In device_context.mojo we define Device{Context,Buffer,Function}V1, the old
# Mojo versions. The C++ versions Device{Context,Buffer,Function}V2 are in
# device_context_v2.mojo. Finally, device_context_variant.mojo defines
# Device{Context,Buffer,Function}, which dynamically selects V1 or V2 using a
# command-line flag. Import them here so users can continue to import from
# gpu.host.device_context. Eventually, device_context_v2.mojo will be renamed to
# replace this file.
from .device_context_variant import DeviceBuffer, DeviceContext, DeviceFunction


# TODO: Figure a way to resolve circular dependency between the gpu and runtime
# packages in the corresponding CMakes and sub below from runtime.tracing
fn _build_info_asyncrt_max_profiling_level() -> OptionalReg[Int]:
    @parameter
    if not is_defined["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]():
        return None
    return env_get_int["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]()


@value
struct KernelProfilingInfoElement(CollectionElement):
    """Struct to handle kernel profiling info for a single kernel.
    Objects of this type form a list that belongs to the KernelProfilingInfo
    object along with a pointer to it.

    Supported operations include: initialization, printing all elements of a
    list, clearing all elements of a list.

    This struct can be extended with further information down the road (e.g.,
    record host-to-device, device-to-host copies timing info).
    """

    var name: String
    var time: Int

    fn __init__(inout self, name: String, time: Int):
        self.name = name
        self.time = time

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


@value
struct KernelProfilingInfo:
    """Struct that incorporates a list of KernelProfilingInfoElement's and a
    pointer to it.

    This is passed to DeviceContext and through the pointer and list append
    operation the list of KernelProfilingInfoElements is constructed and
    operated upon.
    """

    # TODO: Make this parameterizable. KERN-636.
    alias out_file = "kernel_profile_info.log"

    var kernelProfilingList: List[KernelProfilingInfoElement]

    fn __init__(inout self):
        self.kernelProfilingList = List[KernelProfilingInfoElement]()

    fn append(inout self, element: KernelProfilingInfoElement):
        self.kernelProfilingList.append(element)

    fn print(self):
        for m in self.kernelProfilingList:
            print(
                "Function: ",
                m[].name,
                " Time (nsec): ",
                m[].time,
            )

    fn write(self, path: Path = Self.out_file) raises:
        var report = String("Function name, Time (nsec)\n")
        for m in self.kernelProfilingList:
            report += m[].name + ", " + str(m[].time) + "\n"

        path.write_text(report)

    fn clear(inout self):
        self.kernelProfilingList.clear()

    fn get_kernel_from_list(
        self, name: String
    ) -> OptionalReg[UnsafePointer[KernelProfilingInfoElement]]:
        for i in reversed(range(len(self.kernelProfilingList))):
            if self.kernelProfilingList[i].name == name:
                return UnsafePointer.address_of(self.kernelProfilingList[i])
        return None


@value
struct DeviceBufferV1[type: DType](Sized):
    var ptr: UnsafePointer[Scalar[type]]
    var ctx_ptr: UnsafePointer[DeviceContextV1]
    var size: Int
    var owning: Bool

    @parameter
    fn v1(self) -> ref [__lifetime_of(self)] Self:
        return self

    fn __init__(inout self, ctx: DeviceContextV1, size: Int) raises:
        """This init takes in a constructed DeviceContext and schedules an owned buffer allocation
        using the stream in the device context.
        """
        self.ctx_ptr = UnsafePointer[DeviceContextV1].address_of(ctx)
        self.ctx_ptr[].cuda_context.set_current()
        self.ptr = self.ctx_ptr[].cuda_context.malloc_async[Scalar[type]](
            size, self.ctx_ptr[].cuda_stream
        )
        self.size = size
        self.owning = True

    fn __init__(
        inout self,
        ctx: DeviceContextV1,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        self.ctx_ptr = UnsafePointer[DeviceContextV1].address_of(ctx)
        self.ptr = ptr
        self.size = size
        self.owning = owning

    fn __init__(inout self):
        self.ctx_ptr = UnsafePointer[DeviceContextV1]()
        self.ptr = UnsafePointer[Scalar[type]]()
        self.size = 0
        self.owning = False

    fn __copyinit__(inout self, existing: Self):
        self.ctx_ptr = existing.ctx_ptr
        self.ptr = existing.ptr
        self.size = existing.size
        self.owning = False

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx_ptr = existing.ctx_ptr
        self.ptr = existing.ptr
        self.size = existing.size
        self.owning = existing.owning
        existing.ctx_ptr = UnsafePointer[DeviceContextV1]()
        existing.ptr = UnsafePointer[Scalar[type]]()
        existing.size = 0
        existing.owning = False

    @always_inline
    fn __del__(owned self):
        """This function schedules an owned buffer free using the stream in the device context.
        """
        try:
            if self.owning and self.ptr:
                self.ctx_ptr[].cuda_context.set_current()
                self.ctx_ptr[].cuda_context.free_async(
                    self.ptr, self.ctx_ptr[].cuda_stream
                )
        except e:
            abort(e)

    fn __len__(self) -> Int:
        return self.size

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBufferV1[view_type]:
        if sizeof[view_type]() * (offset + size) > sizeof[type]() * self.size:
            raise Error("offset and size exceed original buffer size")
        var sub_buffer = DeviceBufferV1[view_type]()
        sub_buffer.ctx_ptr = self.ctx_ptr
        sub_buffer.ptr = rebind[UnsafePointer[Scalar[view_type]]](
            self.ptr
        ).offset(offset)
        sub_buffer.size = size
        sub_buffer.owning = False
        return sub_buffer

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        var tmp = self.ptr
        self.ptr = UnsafePointer[Scalar[type]]()
        return tmp


@value
struct DeviceFunctionV1[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    dump_ptx: Variant[Bool, Path, fn () capturing -> Path] = False,
    dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
    dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
    _is_failable: Bool = False,
    _ptxas_info_verbose: Bool = False,
]:
    var ctx_ptr: UnsafePointer[DeviceContextV1]
    var cuda_function: Function[
        func,
        dump_ptx=dump_ptx,
        dump_llvm=dump_llvm,
        dump_sass=dump_sass,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ]
    alias fn_name = _get_nvptx_fn_name[func]()

    @parameter
    fn v1(self) -> ref [__lifetime_of(self)] Self:
        return self

    fn __init__(
        inout self,
        ctx: DeviceContextV1,
        *,
        max_registers: OptionalReg[Int] = None,
        threads_per_block: OptionalReg[Int] = None,
        cache_mode: OptionalReg[CacheMode] = None,
        cache_config: OptionalReg[CacheConfig] = None,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        self.ctx_ptr = UnsafePointer[DeviceContextV1].address_of(ctx)
        self.cuda_function = Function[
            func,
            dump_ptx=dump_ptx,
            dump_llvm=dump_llvm,
            dump_sass=dump_sass,
            target=target,
            _is_failable=_is_failable,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](
            self.ctx_ptr,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
        )


@value
struct DeviceContextV1:
    # NOTE: The fields' declaration order matches the destruction order
    var cuda_stream: Stream
    var cuda_context: Context
    var cuda_instance: CudaInstance
    var profiling_info_ptr: UnsafePointer[KernelProfilingInfo]
    # Profiling is enabled only when the optional returned by the below call is
    # not None and has a non-zero value (this will be True for a
    # cmake-modular-profiling build).
    alias profiling_enabled = _build_info_asyncrt_max_profiling_level().or_else(
        -1
    ) > 0

    # We only support CUDA architectures sm_80 and above.
    alias MIN_COMPUTE_CAPABILITY = 80

    # We only support CUDA versions 12.0 and above.
    alias MIN_DRIVER_VERSION = 12.0

    @parameter
    fn v1(self) -> ref [__lifetime_of(self)] Self:
        return self

    # Default initializer for all existing cases outside MGP; this currently
    # includes tests, benchmarks, Driver API. The tests and benchmarks (all of
    # which, except the test_deviceContext_profiling.mojo) would have the
    # profiling_enabled = False, which is OK. But for the Driver API, we would
    # want profiling to occur for appropriate builds when it substitutes the
    # current MGP implementation.
    fn __init__(
        inout self, kind: StringLiteral = "cuda", gpu_id: Int = 0
    ) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(Device(self.cuda_instance, gpu_id))
        self.cuda_stream = Stream(self.cuda_context)

        @parameter
        if self.profiling_enabled:
            self.profiling_info_ptr = UnsafePointer[KernelProfilingInfo].alloc(
                1
            )
            self.profiling_info_ptr.init_pointee_move(KernelProfilingInfo())
        else:
            self.profiling_info_ptr = UnsafePointer[KernelProfilingInfo]()

    fn __enter__(owned self) -> Self:
        return self^

    fn malloc_host[
        type: AnyType
    ](self, size: Int) raises -> UnsafePointer[type]:
        """Allocates a pinned memory area registered with the device."""
        return self.cuda_context.malloc_host[type](size)

    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        """Frees memory allocated with malloc_host()."""
        self.cuda_context.free_host(ptr)

    fn enqueue_create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV1[type]:
        """Enqueues a buffer creation using the DeviceBuffer constructor."""
        return DeviceBufferV1[type](self, size)

    fn create_buffer_sync[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV1[type]:
        """Creates a buffer synchronously using the DeviceBuffer constructor."""
        var result = DeviceBufferV1[type](self, size)
        self.synchronize()
        return result

    fn create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV1[type]:
        """Enqueues a buffer creation using the DeviceBuffer constructor."""
        return DeviceBufferV1[type](self, size)

    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_ptx: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
        target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
        _is_failable: Bool = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        max_registers: OptionalReg[Int] = None,
        threads_per_block: OptionalReg[Int] = None,
        cache_mode: OptionalReg[CacheMode] = None,
        cache_config: OptionalReg[CacheConfig] = None,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises -> DeviceFunctionV1[
        func,
        dump_ptx=dump_ptx,
        dump_llvm=dump_llvm,
        dump_sass=dump_sass,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ] as result:
        return __type_of(result)(
            self,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
        )

    @parameter
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunctionV1,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: Int = 0,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self._enqueue_function(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    fn _enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunctionV1,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: Int = 0,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self.cuda_context.set_current()

        var kernel_time: Int
        var stream = self.cuda_stream

        @parameter
        if self.profiling_enabled:
            var start = Event(self.cuda_context)
            var end = Event(self.cuda_context)
            kernel_time = 0
            start.record(stream)
            f.cuda_function._call_pack(
                args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                stream=stream,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )
            end.record(stream)
            end.sync()
            kernel_time = int(start.elapsed(end) * 1e6)
            var list_item = self.profiling_info_ptr[].get_kernel_from_list(
                f.fn_name
            )
            # If kernel (as found by name) exists in the list, add the timing
            # info in the existing entry. Otherwise, create a new entry.
            if list_item:
                (list_item.value())[].time += kernel_time
            else:
                self.profiling_info_ptr[].append(
                    KernelProfilingInfoElement(f.fn_name, kernel_time)
                )
        else:
            f.cuda_function._call_pack(
                args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                shared_mem_bytes=shared_mem_bytes,
                stream=stream,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )

    fn execution_time[
        func: fn (Self) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        self.cuda_context.set_current()
        var stream = self.cuda_stream
        var start = Event(self.cuda_context)
        var end = Event(self.cuda_context)
        start.record(stream)
        for _ in range(num_iters):
            func(self)
        end.record(stream)
        end.sync()
        return int(start.elapsed(end) * 1e6)

    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        self.cuda_context.set_current()
        var stream = self.cuda_stream
        var start = Event(self.cuda_context)
        var end = Event(self.cuda_context)
        start.record(stream)
        for i in range(num_iters):
            func(self, i)
        end.record(stream)
        end.sync()
        return int(start.elapsed(end) * 1e6)

    fn enqueue_copy_to_device[
        type: DType
    ](self, buf: DeviceBufferV1[type], ptr: UnsafePointer[Scalar[type]]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_host_to_device_async(
            buf.ptr, ptr, len(buf), self.cuda_stream
        )

    fn enqueue_copy_from_device[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV1[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_device_to_host_async(
            ptr, buf.ptr, len(buf), self.cuda_stream
        )

    fn enqueue_copy_device_to_device[
        type: DType
    ](self, dst: DeviceBufferV1[type], src: DeviceBufferV1[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_device_to_device_async(
            dst.ptr, src.ptr, len(dst), self.cuda_stream
        )

    fn enqueue_copy_device_to_device[
        type: DType
    ](
        self,
        dst: UnsafePointer[Scalar[type]],
        src: UnsafePointer[Scalar[type]],
        size: Int,
    ) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_device_to_device_async(
            dst, src, size, self.cuda_stream
        )

    fn copy_to_device_sync[
        type: DType
    ](self, buf: DeviceBufferV1[type], ptr: UnsafePointer[Scalar[type]]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_host_to_device(buf.ptr, ptr, len(buf))

    fn copy_from_device_sync[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV1[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_device_to_host(ptr, buf.ptr, len(buf))

    fn copy_device_to_device_sync[
        type: DType
    ](self, dst: DeviceBufferV1[type], src: DeviceBufferV1[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.copy_device_to_device_async(
            dst.ptr, src.ptr, len(dst), self.cuda_stream
        )
        self.synchronize()

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV1[type], val: Scalar[type]) raises:
        self.cuda_context.set_current()
        _memset_async[type](dst.ptr, val, dst.size, self.cuda_stream)

    fn synchronize(self) raises:
        self.cuda_context.set_current()
        self.cuda_stream.synchronize()

    fn print_kernel_timing_info(self):
        """Print profiling info associated with this DeviceContext."""

        @parameter
        if self.profiling_enabled:
            self.profiling_info_ptr[].print()
        else:
            print(
                "<print_kernel_timing_info>: Kernel profiling has not been"
                " enabled."
            )

    fn dump_kernel_timing_info(self) raises:
        """Prints out profiling info associated with this DeviceContext."""

        @parameter
        if self.profiling_enabled:
            self.profiling_info_ptr[].write()
        else:
            print(
                "<dump_kernel_timing_info>: Kernel profiling has not been"
                " enabled."
            )

    fn clear_kernel_timing_info(self):
        """Clear profiling info associated with this DeviceContext."""

        @parameter
        if self.profiling_enabled:
            self.profiling_info_ptr[].clear()
        else:
            print(
                "<clear_kernel_timing_info>: Kernel profiling has not been"
                " enabled."
            )

    fn is_compatible(self) raises:
        """Returns whether the current CUDA device is compatible with MAX."""
        if self.compute_capability() < Self.MIN_COMPUTE_CAPABILITY:
            raise Error("MAX only supports CUDA Ampere architectures or higher")

        if self.cuda_context.get_version() < Self.MIN_DRIVER_VERSION:
            raise Error("MAX not supported using current GPU driver")

    fn compute_capability(self) raises -> Int:
        return self.cuda_context.device._query(
            DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
        ) * 10 + self.cuda_context.device._query(
            DeviceAttribute.COMPUTE_CAPABILITY_MINOR
        )

    fn get_memory_info(self) raises -> (c_size_t, c_size_t):
        self.cuda_context.set_current()
        return self.cuda_context.get_memory_info()
