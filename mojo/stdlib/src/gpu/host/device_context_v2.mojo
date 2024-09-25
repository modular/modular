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
alias _CharPtr = UnsafePointer[UInt8]
alias _VoidPtr = UnsafePointer[NoneType]
alias _SizeT = UInt64

# Define helper methods to call AsyncRT bindings.


fn _checked(err_msg: _CharPtr) raises:
    if err_msg:
        err_str = String(StringRef(err_msg))
        external_call["free", NoneType, _CharPtr](err_msg)
        raise Error(err_str)


@value
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
        # const char * AsyncRT_DeviceContext_createBuffer
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     size_t len, size_t elem_size)
        alias elem_size = sizeof[type]()
        var result = _DeviceBufferPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                _DeviceContextPtr,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer.address_of(result),
                ctx._ptr,
                size,
                elem_size,
            )
        )
        # void *AsyncRT_DeviceBuffer_ptr(const DeviceBuffer *buffer)
        self._device_ptr = external_call[
            "AsyncRT_DeviceBuffer_ptr",
            UnsafePointer[Scalar[type]],
            _DeviceBufferPtr,
        ](
            result,
        )
        self._handle = result

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        constrained[False, "##### DeviceBufferV2.__init__ - 2"]()
        self._device_ptr = UnsafePointer[Scalar[type]]()
        self._handle = _DeviceBufferPtr()

    fn __init__(inout self):
        constrained[False, "##### DeviceBufferV2.__init__ - 3"]()
        self._device_ptr = UnsafePointer[Scalar[type]]()
        self._handle = _DeviceBufferPtr()

    fn __copyinit__(inout self, existing: Self):
        constrained[False, "##### DeviceBufferV2.__copyinit__"]()
        self._device_ptr = UnsafePointer[Scalar[type]]()
        self._handle = _DeviceBufferPtr()

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
        constrained[False, "##### UNIMPLEMENTED: DeviceBufferV2.__len__"]()
        return 0  # FIXME

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBufferV2[view_type]:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceBufferV2.create_sub_buffer"
        ]()
        return DeviceBufferV2[view_type]()  # FIXME

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        constrained[False, "##### UNIMPLEMENTED: DeviceBufferV2.take_ptr"]()
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
    var _ptr: _DeviceFunctionPtr
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
                ctx._ptr,
                self._func_impl.function_name.unsafe_ptr(),
                self._func_impl.asm.unsafe_ptr(),
            )
        )
        self._ptr = result

    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContextV2,
        args: VariadicPack[_, AnyType, Ts],
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

        # TODO(iposva): call populate

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
                ctx._ptr,
                self._ptr,
                grid_dim.x(),
                grid_dim.y(),
                grid_dim.z(),
                block_dim.x(),
                block_dim.y(),
                block_dim.z(),
                dense_args_addrs,
            )
        )


@value
struct DeviceContextV2:
    """DeviceContext backed by a C++ implementation."""

    var _ptr: _DeviceContextPtr

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
        self._ptr = result

    fn __del__(owned self):
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr,
        ](self._ptr)

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
                self._ptr,
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
                self._ptr,
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
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.enqueue_function"
        ]()

    @parameter
    fn _enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunctionV2,
        args: VariadicPack[_, AnyType, Ts],
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
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.execution_time"
        ]()
        return 0  # FIXME

    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.execution_time_iter"
        ]()
        return 0  # FIXME

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
                self._ptr,
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
                self._ptr,
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
                self._ptr,
                dst._handle,
                src._handle,
            )
        )

    fn copy_to_device_sync[
        type: DType
    ](self, buf: DeviceBufferV2[type], ptr: UnsafePointer[Scalar[type]]) raises:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.copy_to_device_sync"
        ]()
        pass

    fn copy_from_device_sync[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV2[type]) raises:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.copy_from_device_sync"
        ]()
        pass

    fn copy_device_to_device_sync[
        type: DType
    ](self, dst: DeviceBufferV2[type], src: DeviceBufferV2[type]) raises:
        constrained[
            False,
            "##### UNIMPLEMENTED: DeviceContextV2.copy_device_to_device_sync",
        ]()
        pass

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        constrained[False, "##### UNIMPLEMENTED: DeviceContextV2.memset"]()
        pass

    fn synchronize(self) raises:
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._ptr,
            )
        )

    fn print_kernel_timing_info(self):
        """Print profiling info associated with this DeviceContext."""
        constrained[
            False,
            "##### UNIMPLEMENTED: DeviceContextV2.print_kernel_timing_info",
        ]()
        pass

    fn dump_kernel_timing_info(self) raises:
        """Prints out profiling info associated with this DeviceContext."""
        constrained[
            False,
            "##### UNIMPLEMENTED: DeviceContextV2.dump_kernel_timing_info",
        ]()
        pass

    fn clear_kernel_timing_info(self):
        """Clear profiling info associated with this DeviceContext."""
        constrained[
            False,
            "##### UNIMPLEMENTED: DeviceContextV2.clear_kernel_timing_info",
        ]()
        pass

    fn is_compatible(self) raises:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.is_compatible"
        ]()
        pass

    fn compute_capability(self) raises -> Int:
        constrained[
            False, "##### UNIMPLEMENTED: DeviceContextV2.compute_capability"
        ]()
        return 0  # FIXME
