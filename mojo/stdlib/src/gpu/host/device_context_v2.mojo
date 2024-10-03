# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Implementation of the C++ backed DeviceContext in Mojo
# WIP, just stubs for now

from ._compile import (
    _compile_code,
    _get_nvptx_target,
)
from memory import stack_allocation


alias _DeviceContextPtr = UnsafePointer[NoneType]
alias _DeviceBufferPtr = UnsafePointer[NoneType]
alias _DeviceFunctionPtr = UnsafePointer[NoneType]
alias _DeviceTimerPtr = UnsafePointer[NoneType]
alias _CharPtr = UnsafePointer[UInt8]
alias _VoidPtr = UnsafePointer[NoneType]
alias _SizeT = UInt64

# Define helper methods to call AsyncRT bindings.


fn not_implemented_yet[msg: StringLiteral]():
    # Uncomment to convert runtime errors into compile-time errors:
    # constrained[False, msg]()
    abort(msg)


fn _checked(err_msg: _CharPtr) raises:
    if err_msg:
        err_str = String(StringRef(err_msg))
        external_call["free", NoneType, _CharPtr](err_msg)
        raise Error(err_str)


struct _DeviceTimer:
    var _handle: _DeviceTimerPtr

    fn __init__(inout self, ptr: _DeviceTimerPtr):
        self._handle = ptr

    fn __del__(owned self):
        # void AsyncRT_DeviceTimer_release(const DviceTimer *timer)
        external_call["AsyncRT_DeviceTimer_release", NoneType, _DeviceTimerPtr](
            self._handle
        )


struct DeviceBufferV2[type: DType](Sized):
    # _device_ptr must be the first word in the struct to enable passing of
    # DeviceBufferV2 to kernels. The first word is passed to the kernel and
    # it needs to contain the value registered with the driver.
    var _device_ptr: UnsafePointer[Scalar[type]]
    var _handle: _DeviceBufferPtr

    fn __init__(inout self, ctx: DeviceContextV2, size: Int) raises:
        """This init takes in a constructed DeviceContext and schedules an owned buffer allocation
        using the stream in the device context.
        """
        # const char *AsyncRT_DeviceContext_createBuffer(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
        alias elem_size = sizeof[type]()
        var result = _DeviceBufferPtr()
        var device_ptr = UnsafePointer[Scalar[type]]()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[type]]],
                _DeviceContextPtr,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer.address_of(result),
                UnsafePointer.address_of(device_ptr),
                ctx._handle,
                size,
                elem_size,
            )
        )
        self._device_ptr = device_ptr
        self._handle = result

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        not_implemented_yet["##### DeviceBufferV2.__init__ - 2"]()
        self._device_ptr = UnsafePointer[Scalar[type]]()
        self._handle = _DeviceBufferPtr()

    fn __init__(inout self):
        not_implemented_yet["##### DeviceBufferV2.__init__ - 3"]()
        self._device_ptr = UnsafePointer[Scalar[type]]()
        self._handle = _DeviceBufferPtr()

    fn __copyinit__(inout self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceContextPtr,
        ](existing._handle)
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle

    fn __moveinit__(inout self, owned existing: Self):
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle
        existing._device_ptr = UnsafePointer[Scalar[type]]()
        existing._handle = _DeviceBufferPtr()

    @always_inline
    fn __del__(owned self):
        """This function schedules an owned buffer free using the stream in the device context.
        """
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceContextPtr
        ](
            self._handle,
        )

    fn __len__(self) -> Int:
        # int64_t AsyncRT_DeviceBuffer_len(const DeviceBuffer *buffer)
        return external_call["AsyncRT_DeviceBuffer_len", Int, _DeviceBufferPtr](
            self._handle
        )

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBufferV2[view_type]:
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceBufferV2.create_sub_buffer"
        ]()
        return DeviceBufferV2[view_type]()  # FIXME

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        not_implemented_yet["##### UNIMPLEMENTED: DeviceBufferV2.take_ptr"]()
        return UnsafePointer[Scalar[type]]()  # FIXME

    fn ptr(self) -> UnsafePointer[Scalar[type]]:
        return self._device_ptr


@value
struct DeviceFunctionV2[
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
    var _handle: _DeviceFunctionPtr
    alias _func_impl = _compile_code[
        func, is_failable=_is_failable, emission_kind="asm", target=target
    ]()

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        *,
        max_registers: OptionalReg[Int] = None,
        threads_per_block: OptionalReg[Int] = None,
        cache_mode: OptionalReg[CacheMode] = None,
        cache_config: OptionalReg[CacheConfig] = None,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        # const char *AsyncRT_DeviceContext_compileFunction(const DeviceFunction **result, const DeviceContext *ctx, const char *function_name, const void *data)
        var result = _DeviceFunctionPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_compileFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
            ](
                UnsafePointer.address_of(result),
                ctx._handle,
                self._func_impl.function_name.unsafe_ptr(),
                self._func_impl.asm.unsafe_ptr(),
            )
        )
        self._handle = result

    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContextV2,
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
        alias num_args = len(VariadicList(Ts))
        alias num_captures = self._func_impl.num_captures
        alias populate = self._func_impl.populate

        var dense_args_addrs = stack_allocation[
            num_captures + num_args, UnsafePointer[NoneType]
        ]()

        @parameter
        if num_captures > 0:
            # Call the populate function to initialize the first values in the arguments array.
            populate(
                rebind[UnsafePointer[NoneType]](
                    dense_args_addrs.bitcast[NoneType]()
                )
            )

        @parameter
        for i in range(num_args):
            alias arg_offset = num_captures + i
            var first_word_addr = UnsafePointer.address_of(args[i])
            dense_args_addrs[arg_offset] = first_word_addr.bitcast[NoneType]()

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ, void **args)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueFunctionDirect",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceFunctionPtr,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UnsafePointer[UnsafePointer[NoneType]],
            ](
                ctx._handle,
                self._handle,
                grid_dim.x(),
                grid_dim.y(),
                grid_dim.z(),
                block_dim.x(),
                block_dim.y(),
                block_dim.z(),
                dense_args_addrs,
            )
        )


struct DeviceContextV2:
    """DeviceContext backed by a C++ implementation."""

    var _handle: _DeviceContextPtr

    fn __init__(
        inout self, device_kind: StringLiteral = "cuda", device_id: Int = 0
    ) raises:
        # const char * AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind, int id)
        var result = _DeviceContextPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                _CharPtr,
                UnsafePointer[_DeviceContextPtr],
                _CharPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                device_kind.unsafe_ptr(),
                device_id,
            )
        )
        self._handle = result

    fn __copyinit__(inout self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceContext_retain(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_retain",
            NoneType,
            _DeviceContextPtr,
        ](existing._handle)
        self._handle = existing._handle

    fn __moveinit__(inout self, owned existing: Self):
        self._handle = existing._handle
        existing._handle = _DeviceContextPtr()

    fn __del__(owned self):
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr,
        ](self._handle)

    fn __enter__(owned self) -> Self:
        return self^

    fn malloc_host[
        type: AnyType
    ](self, size: Int) raises -> UnsafePointer[type]:
        # const char * AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx, size_t size)
        alias elem_size = sizeof[type]()
        var result = UnsafePointer[type]()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_mallocHost",
                _CharPtr,
                UnsafePointer[UnsafePointer[type]],
                _DeviceContextPtr,
                _SizeT,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                size * elem_size,
            )
        )
        return result

    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        # const char * AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_freeHost",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[type],
            ](
                self._handle,
                ptr,
            )
        )

    fn create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV2[type]:
        """Creates a buffer using the DeviceBuffer constructor."""
        return DeviceBufferV2[type](self, size)

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
    ) raises -> DeviceFunctionV2[
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
        f: DeviceFunctionV2,
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
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.enqueue_function"
        ]()

    @parameter
    fn _enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunctionV2,
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
        f._call_with_pack(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    fn execution_time[
        func: fn (Self) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func(self)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer.address_of(elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for i in range(num_iters):
            func(self, i)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer.address_of(elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    fn enqueue_copy_to_device[
        type: DType
    ](self, buf: DeviceBufferV2[type], ptr: UnsafePointer[Scalar[type]]) raises:
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                buf._handle,
                ptr,
            )
        )

    fn enqueue_copy_from_device[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV2[type]) raises:
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                ptr,
                buf._handle,
            )
        )

    fn enqueue_copy_device_to_device[
        type: DType
    ](self, dst: DeviceBufferV2[type], src: DeviceBufferV2[type]) raises:
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst._handle,
                src._handle,
            )
        )

    fn copy_to_device_sync[
        type: DType
    ](self, buf: DeviceBufferV2[type], ptr: UnsafePointer[Scalar[type]]) raises:
        # const char * AsyncRT_DeviceContext_HtoD_sync(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                buf._handle,
                ptr,
            )
        )

    fn copy_from_device_sync[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV2[type]) raises:
        # const char * AsyncRT_DeviceContext_DtoH_sync(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_sync",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                ptr,
                buf._handle,
            )
        )

    fn copy_device_to_device_sync[
        type: DType
    ](self, dst: DeviceBufferV2[type], src: DeviceBufferV2[type]) raises:
        # const char * AsyncRT_DeviceContext_DtoD_sync(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst._handle,
                src._handle,
            )
        )

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        not_implemented_yet["##### UNIMPLEMENTED: DeviceContextV2.memset"]()

    fn synchronize(self) raises:
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._handle,
            )
        )

    fn get_memory_info(self) raises -> (c_size_t, c_size_t):
        var free = c_size_t(0)
        var total = c_size_t(0)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getMemoryInfo",
                _CharPtr,
                UnsafePointer[c_size_t],
                UnsafePointer[c_size_t],
            ](
                UnsafePointer.address_of(free),
                UnsafePointer.address_of(total),
            )
        )

        return (free, total)

    fn print_kernel_timing_info(self):
        """Print profiling info associated with this DeviceContext."""
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.print_kernel_timing_info"
        ]()

    fn dump_kernel_timing_info(self) raises:
        """Prints out profiling info associated with this DeviceContext."""
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.dump_kernel_timing_info"
        ]()

    fn clear_kernel_timing_info(self):
        """Clear profiling info associated with this DeviceContext."""
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.clear_kernel_timing_info"
        ]()

    fn is_compatible(self) raises:
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.is_compatible"
        ]()

    fn compute_capability(self) raises -> Int:
        not_implemented_yet[
            "##### UNIMPLEMENTED: DeviceContextV2.compute_capability"
        ]()
        return 0  # FIXME
