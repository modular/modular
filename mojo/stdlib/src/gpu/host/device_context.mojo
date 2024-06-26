# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._compile import _get_nvptx_fn_name
from collections import List
from gpu.host.context import Context
from gpu.host.cuda_instance import CudaInstance
from gpu.host.device import Device
from gpu.host.event import Event
from gpu.host.function import Function
from gpu.host.stream import Stream
from ._utils import _check_error, _StreamHandle


@value
struct KernelProfilingInfo:
    """Struct that incorporates a list of KernelProfilingInfoElement's and a
    pointer to it.

    This is passed to DeviceContext and through the pointer and list append
    operation the list of KernelProfilingInfoElements is constructed and
    operated upon.
    """

    var kernelProfilingList: List[KernelProfilingInfoElement]
    var KernelProfilingListPtr: UnsafePointer[List[KernelProfilingInfoElement]]

    fn __init__(inout self):
        self.kernelProfilingList = List[KernelProfilingInfoElement]()
        self.KernelProfilingListPtr = UnsafePointer[
            List[KernelProfilingInfoElement]
        ].address_of(self.kernelProfilingList)


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

    @staticmethod
    fn print(kernelInfoList: List[KernelProfilingInfoElement]):
        for m in kernelInfoList:
            print(
                "Function: ",
                m[].name,
                " Time (nsec): ",
                m[].time,
            )

    @staticmethod
    fn get_kernel_from_list(
        name: String,
        kernelInfoList: UnsafePointer[List[KernelProfilingInfoElement]],
    ) -> Optional[UnsafePointer[KernelProfilingInfoElement]]:
        for i in range(len(kernelInfoList[])):
            if (kernelInfoList[])[i].name == name:
                return UnsafePointer.address_of((kernelInfoList[])[i])
        return None

    @staticmethod
    fn clear(kernelInfoList: UnsafePointer[List[KernelProfilingInfoElement]]):
        kernelInfoList[].clear()


@value
struct DeviceBuffer[type: DType](Sized):
    var ptr: DTypePointer[type]
    var ctx_ptr: UnsafePointer[DeviceContext]
    var size: Int
    var owning: Bool

    fn __init__(inout self, ctx: DeviceContext, size: Int) raises:
        self.ctx_ptr = UnsafePointer[DeviceContext].address_of(ctx)
        self.ptr = self.ctx_ptr[].cuda_context.malloc[type](size)
        self.size = size
        self.owning = True

    fn __init__(
        inout self,
        ctx: DeviceContext,
        ptr: DTypePointer[type],
        size: Int,
        *,
        owning: Bool,
    ):
        self.ctx_ptr = UnsafePointer[DeviceContext].address_of(ctx)
        self.ptr = ptr
        self.size = size
        self.owning = owning

    fn __init__(inout self):
        self.ctx_ptr = UnsafePointer[DeviceContext]()
        self.ptr = DTypePointer[type]()
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
        existing.ctx_ptr = UnsafePointer[DeviceContext]()
        existing.ptr = DTypePointer[type]()
        existing.size = 0
        existing.owning = False

    fn __del__(owned self):
        try:
            if self.owning and self.ptr:
                self.ctx_ptr[].cuda_context.free(self.ptr)
                self.ptr = DTypePointer[type]()
                self.size = 0
        except e:
            print("something went wrong", e)

    fn __len__(self) -> Int:
        return self.size

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        if sizeof[view_type]() * (offset + size) > sizeof[type]() * self.size:
            raise Error("offset and size exceed original buffer size")
        var sub_buffer = DeviceBuffer[view_type]()
        sub_buffer.ctx_ptr = self.ctx_ptr
        sub_buffer.ptr = rebind[DTypePointer[view_type]](self.ptr).offset(
            offset
        )
        sub_buffer.size = size
        sub_buffer.owning = False
        return sub_buffer

    fn take_ptr(owned self) -> DTypePointer[type]:
        var tmp = self.ptr
        self.ptr = DTypePointer[type]()
        return tmp


@value
struct DeviceFunction[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    _is_failable: Bool = False,
]:
    var ctx_ptr: UnsafePointer[DeviceContext]
    var cuda_function: Function[func, _is_failable=_is_failable]
    alias fn_name = _get_nvptx_fn_name[func]()

    fn __init__(
        inout self,
        ctx: DeviceContext,
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
    ) raises:
        self.ctx_ptr = UnsafePointer[DeviceContext].address_of(ctx)
        self.cuda_function = Function[func, _is_failable=_is_failable](
            self.ctx_ptr[].cuda_context,
            debug,
            verbose,
            dump_ptx,
            dump_llvm,
            max_registers,
            threads_per_block,
            cache_config,
            func_attribute,
        )


@value
struct DeviceContext:
    # NOTE: The fields' declaration order matches the destruction order
    var cuda_stream: Stream
    var cuda_context: Context
    var cuda_instance: CudaInstance
    var profiling_enabled: Bool
    var profiling_info: KernelProfilingInfo

    fn __init__(
        inout self,
        *,
        profiling_enabled: Bool = False,
        profiling_info: KernelProfilingInfo = KernelProfilingInfo(),
    ) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(Device(self.cuda_instance))
        self.cuda_stream = Stream(self.cuda_context)
        self.profiling_enabled = profiling_enabled
        self.profiling_info = profiling_info

    fn __init__(
        inout self,
        cuda_instance: CudaInstance,
        cuda_context: Context,
        cuda_stream: Stream,
        *,
        profiling_enabled: Bool = False,
        profiling_info: KernelProfilingInfo = KernelProfilingInfo(),
    ):
        self.cuda_instance = cuda_instance
        self.cuda_context = cuda_context
        self.cuda_stream = cuda_stream
        self.profiling_enabled = profiling_enabled
        self.profiling_info = profiling_info

    fn __init__(
        inout self,
        cuda_stream: Stream,
        *,
        profiling_enabled: Bool = False,
        profiling_info: KernelProfilingInfo = KernelProfilingInfo(),
    ) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(Device(self.cuda_instance))
        self.cuda_stream = cuda_stream
        self.profiling_enabled = profiling_enabled
        self.profiling_info = profiling_info

    fn __enter__(owned self) -> Self:
        return self^

    fn create_buffer[type: DType](self, size: Int) raises -> DeviceBuffer[type]:
        return DeviceBuffer[type](self, size)

    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        _is_failable: Bool = False,
    ](
        self,
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
    ) raises -> DeviceFunction[func, _is_failable=_is_failable]:
        return DeviceFunction[func, _is_failable=_is_failable](
            self,
            debug=debug,
            verbose=verbose,
            dump_ptx=dump_ptx,
            dump_llvm=dump_llvm,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_config=cache_config,
            func_attribute=func_attribute,
        )

    @parameter
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int = 0,
    ) raises:
        var kernel_time: Int
        var stream = self.cuda_stream
        var start = Event(self.cuda_context)
        var end = Event(self.cuda_context)
        if self.profiling_enabled:
            kernel_time = 0
            start.record(stream)
        f.cuda_function._call_pack(
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=shared_mem_bytes,
            stream=self.cuda_stream,
        )
        if self.profiling_enabled:
            end.record(stream)
            end.sync()
            kernel_time = int(start.elapsed(end) * 1e6)
            var listItem = KernelProfilingInfoElement.get_kernel_from_list(
                f.fn_name, self.profiling_info.KernelProfilingListPtr
            )
            # If kernel (as found by name) exists in the list, add the timing
            # info in the existing entry. Otherwise, create a new entry.
            if listItem:
                (listItem.value())[].time += kernel_time
            else:
                self.profiling_info.KernelProfilingListPtr[].append(
                    KernelProfilingInfoElement(f.fn_name, kernel_time)
                )

    fn execution_time[
        func: fn (DeviceContext) raises capturing -> None
    ](self, num_iters: Int) raises -> Int:
        var kernel_time: Int = 0
        try:
            var stream = self.cuda_stream
            var start = Event(self.cuda_context)
            var end = Event(self.cuda_context)
            start.record(stream)
            for _ in range(num_iters):
                func(self)
            end.record(stream)
            end.sync()
            kernel_time = int(start.elapsed(end) * 1e6)
        except e:
            abort(e)
        return kernel_time

    fn enqueue_copy_to_device[
        type: DType
    ](self, buf: DeviceBuffer[type], ptr: DTypePointer[type]) raises:
        self.cuda_context.copy_host_to_device_async(
            buf.ptr, ptr, len(buf), self.cuda_stream
        )

    fn enqueue_copy_from_device[
        type: DType
    ](self, ptr: DTypePointer[type], buf: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_host_async(
            ptr, buf.ptr, len(buf), self.cuda_stream
        )

    fn enqueue_copy_device_to_device[
        type: DType
    ](self, dst: DeviceBuffer[type], src: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_device_async(
            dst.ptr, src.ptr, len(dst), self.cuda_stream
        )

    fn copy_to_device_sync[
        type: DType
    ](self, buf: DeviceBuffer[type], ptr: DTypePointer[type]) raises:
        self.cuda_context.copy_host_to_device(buf.ptr, ptr, len(buf))

    fn copy_from_device_sync[
        type: DType
    ](self, ptr: DTypePointer[type], buf: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_host(ptr, buf.ptr, len(buf))

    fn copy_device_to_device_sync[
        type: DType
    ](self, dst: DeviceBuffer[type], src: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_device_async(
            dst.ptr, src.ptr, len(dst), self.cuda_stream
        )
        self.synchronize()

    fn synchronize(self) raises:
        self.cuda_stream.synchronize()

    fn print_kernel_timing_info(self):
        """Print profiling info associated with this DeviceContext."""

        KernelProfilingInfoElement.print(
            self.profiling_info.KernelProfilingListPtr[]
        )

    fn clear_kernel_timing_info(self):
        """Clear profiling info associated with this DeviceContext."""

        KernelProfilingInfoElement.clear(
            self.profiling_info.KernelProfilingListPtr
        )
