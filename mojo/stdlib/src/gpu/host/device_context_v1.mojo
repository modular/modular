# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List, Optional
from os import abort
from pathlib import Path
from sys import external_call, sizeof
from sys.ffi import c_size_t
from utils import StringRef, Variant

from gpu.host._compile import _get_gpu_target, _get_nvptx_fn_name
from gpu.host.launch_attribute import LaunchAttribute
from gpu.host._utils_v1 import _check_error, _StreamHandle
from gpu.host.context_v1 import Context
from gpu.host.cuda_instance_v1 import CudaInstance
from gpu.host.device_attribute import DeviceAttribute
from gpu.host.device_v1 import DeviceV1
from gpu.host.event_v1 import Event
from gpu.host.function_v1 import Function
from gpu.host.stream_v1 import Stream

# In device_context.mojo we define Device{Context,Buffer,Function}V1, the old
# Mojo versions. The C++ versions Device{Context,Buffer,Function}V2 are in
# device_context_v2.mojo. Finally, device_context_variant.mojo defines
# Device{Context,Buffer,Function}, which dynamically selects V1 or V2 using a
# command-line flag. Import them here so users can continue to import from
# gpu.host.device_context. Eventually, device_context_v2.mojo will be renamed to
# replace this file.
from .device_context_variant import DeviceBuffer, DeviceContext, DeviceFunction


@value
struct DeviceBufferV1[type: DType](Sized):
    var ptr: UnsafePointer[Scalar[type]]
    var ctx_ptr: UnsafePointer[DeviceContextV1]
    var size: Int
    var owning: Bool

    @parameter
    fn v1(self) -> ref [__origin_of(self)] Self:
        return self

    fn __init__(out self, ctx: DeviceContextV1, size: Int) raises:
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

    fn __init__(
        inout self,
        ctx_ptr: UnsafePointer[DeviceContextV1],
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        self.ctx_ptr = ctx_ptr
        self.ptr = ptr
        self.size = size
        self.owning = owning

    fn __copyinit__(out self, existing: Self):
        self.ctx_ptr = existing.ctx_ptr
        self.ptr = existing.ptr
        self.size = existing.size
        self.owning = False

    fn __moveinit__(out self, owned existing: Self):
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
        return DeviceBufferV1[view_type](
            self.ctx_ptr,
            rebind[UnsafePointer[Scalar[view_type]]](self.ptr).offset(offset),
            size,
            owning=False,
        )

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        var tmp = self.ptr
        self.ptr = UnsafePointer[Scalar[type]]()
        return tmp


@value
struct DeviceFunctionV1[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    target: __mlir_type.`!kgen.target` = _get_gpu_target(),
    _is_failable: Bool = False,
    _ptxas_info_verbose: Bool = False,
]:
    var ctx_ptr: UnsafePointer[DeviceContextV1]
    var cuda_function: Function[
        func,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ]
    alias fn_name = _get_nvptx_fn_name[func]()

    @parameter
    fn v1(self) -> ref [__origin_of(self)] Self:
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

    fn dump_rep[
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        _dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
    ](self) raises:
        self.cuda_function.dump_rep[
            dump_asm=dump_asm, dump_llvm=dump_llvm, _dump_sass=_dump_sass
        ]()


@value
struct DeviceContextV1:
    # NOTE: The fields' declaration order matches the destruction order
    var cuda_stream: Stream
    var cuda_context: Context
    var cuda_instance: CudaInstance

    # We only support CUDA architectures sm_80 and above.
    alias MIN_COMPUTE_CAPABILITY = 80

    # We only support CUDA versions 12.0 and above.
    alias MIN_DRIVER_VERSION = 12.0

    @parameter
    fn v1(self) -> ref [__origin_of(self)] Self:
        return self

    # Default initializer for all existing cases outside MGP; this currently
    # includes tests, benchmarks, Driver API. The tests and benchmarks (all of
    # which, except the test_deviceContext_profiling.mojo) would have the
    # profiling_enabled = False, which is OK. But for the Driver API, we would
    # want profiling to occur for appropriate builds when it substitutes the
    # current MGP implementation.
    fn __init__(out self, kind: StringRef = "cuda", gpu_id: Int = 0) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(DeviceV1(self.cuda_instance, gpu_id))
        self.cuda_stream = Stream(self.cuda_context)

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

    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        _dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
        target: __mlir_type.`!kgen.target` = _get_gpu_target(),
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
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ] as result:
        result = __type_of(result)(
            self,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
        )
        result.dump_rep[
            dump_asm=dump_asm, dump_llvm=dump_llvm, _dump_sass=_dump_sass
        ]()

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
        shared_mem_bytes: OptionalReg[Int] = None,
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
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self.cuda_context.set_current()
        var stream = self.cuda_stream
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

    fn enqueue_memset[
        type: DType
    ](self, dst: DeviceBufferV1[type], val: Scalar[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.memset_async[type](
            dst.ptr, val, dst.size, self.cuda_stream
        )

    fn memset_sync[
        type: DType
    ](self, dst: DeviceBufferV1[type], val: Scalar[type]) raises:
        self.cuda_context.set_current()
        self.cuda_context.memset_async[type](
            dst.ptr, val, dst.size, self.cuda_stream
        )
        self.synchronize()

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV1[type], val: Scalar[type]) raises:
        self.enqueue_memset[type](dst, val)

    fn synchronize(self) raises:
        self.cuda_context.set_current()
        self.cuda_stream.synchronize()

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
