# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Implementation of the C++ backed DeviceContext in Mojo
# WIP, just stubs for now

from memory import stack_allocation

from ._compile import (
    _compile_code,
    _compile_code_asm,
    _get_nvptx_fn_name,
    _get_nvptx_target,
    _ptxas_compile,
    _to_sass,
)

alias _DeviceContextPtr = UnsafePointer[NoneType]
alias _DeviceBufferPtr = UnsafePointer[NoneType]
alias _DeviceFunctionPtr = UnsafePointer[NoneType]
alias _DeviceTimerPtr = UnsafePointer[NoneType]
alias _CharPtr = UnsafePointer[UInt8]
alias _IntPtr = UnsafePointer[Int32]
alias _VoidPtr = UnsafePointer[NoneType]
alias _SizeT = UInt

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


@value
struct DeviceSyncMode:
    var _is_sync: Bool


struct DeviceBufferV2[type: DType](Sized):
    # _device_ptr must be the first word in the struct to enable passing of
    # DeviceBufferV2 to kernels. The first word is passed to the kernel and
    # it needs to contain the value registered with the driver.
    var _device_ptr: UnsafePointer[Scalar[type]]
    var _handle: _DeviceBufferPtr

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        size: Int,
        sync_mode: DeviceSyncMode,
    ) raises:
        """This init takes in a constructed DeviceContext and schedules an owned buffer allocation
        using the stream in the device context.
        """
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        var device_ptr = UnsafePointer[Scalar[type]]()

        if sync_mode._is_sync:
            # const char *AsyncRT_DeviceContext_createBuffer_sync(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_sync",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[UnsafePointer[Scalar[type]]],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer.address_of(cpp_handle),
                    UnsafePointer.address_of(device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )
        else:
            # const char *AsyncRT_DeviceContext_createBuffer_async(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_async",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[UnsafePointer[Scalar[type]]],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer.address_of(cpp_handle),
                    UnsafePointer.address_of(device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )

        self._device_ptr = device_ptr
        self._handle = cpp_handle

    fn __init__(
        inout self,
        handle: _DeviceBufferPtr,
        device_ptr: UnsafePointer[Scalar[type]],
    ):
        self._device_ptr = device_ptr
        self._handle = handle

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr],
            _DeviceContextPtr,
            UnsafePointer[Scalar[type]],
            _SizeT,
            _SizeT,
            Bool,
        ](
            UnsafePointer.address_of(cpp_handle),
            ctx._handle,
            ptr,
            size,
            elem_size,
            owning,
        )

        self._device_ptr = ptr
        self._handle = cpp_handle

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

    @always_inline
    fn __del__(owned self):
        """This function schedules an owned buffer free using the stream in the device context.
        """
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr
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
        alias elem_size = sizeof[view_type]()
        var new_handle = _DeviceBufferPtr()
        var new_device_ptr = UnsafePointer[Scalar[view_type]]()
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[view_type]]],
                _DeviceBufferPtr,
                _SizeT,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer.address_of(new_handle),
                UnsafePointer.address_of(new_device_ptr),
                self._handle,
                offset,
                size,
                elem_size,
            )
        )
        return DeviceBufferV2[view_type](new_handle, new_device_ptr)

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr", NoneType, _DeviceBufferPtr
        ](self._handle)
        var result = self._device_ptr
        self._device_ptr = UnsafePointer[Scalar[type]]()
        return result

    fn get_ptr(self) -> UnsafePointer[Scalar[type]]:
        return self._device_ptr

    fn __getattr__[name: StringLiteral](self) -> UnsafePointer[Scalar[type]]:
        @parameter
        if name == "ptr":
            return self.get_ptr()

        abort("Unsupported attr for DeviceBufferV2: " + name)
        return UnsafePointer[Scalar[type]]()


struct DeviceFunctionV2[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
    _is_failable: Bool = False,
    _ptxas_info_verbose: Bool = False,
]:
    var _handle: _DeviceFunctionPtr
    alias _func_impl = _compile_code[
        func, is_failable=_is_failable, emission_kind="asm", target=target
    ]()

    fn __copyinit__(inout self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr,
        ](existing._handle)
        self._handle = existing._handle

    fn __moveinit__(inout self, owned existing: Self):
        self._handle = existing._handle

    fn __del__(owned self):
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr,
        ](self._handle)

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
        alias debug_level = env_get_string["DEBUG_LEVEL", "none"]()
        alias optimization_level = env_get_int["OPTIMIZATION_LEVEL", 4]()

        if max_registers:
            print(
                "DeviceFunctionV2.__init__: max_registers = "
                + str(max_registers.value())
            )
            not_implemented_yet["DeviceFunctionV2.__init__: max_registers"]()

        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                print(
                    "DeviceFunctionV2.__init__: func_attribute = ["
                    + str(func_attribute.value().attribute.code)
                    + ", "
                    + str(func_attribute.value().value)
                    + "]"
                )
                not_implemented_yet[
                    "DeviceFunctionV2.__init__: func_attribute"
                ]()

        # const char *AsyncRT_DeviceContext_compileFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *function_name, const void *data,
        #     int32_t max_registers, int32_t threads_per_block,
        #     int32_t cache_mode, int32_t cache_config, int32_t max_dynamic_shared_bytes,
        #     const char* debug_level, int32_t optimization_level)
        var result = _DeviceFunctionPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_compileFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
                Int32,
                Int32,
                Int32,
                Int32,
                Int32,
                _CharPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                ctx._handle,
                self._func_impl.function_name.unsafe_ptr(),
                self._func_impl.asm.unsafe_ptr(),
                max_registers.or_else(-1),
                threads_per_block.or_else(-1),
                int(cache_mode.or_else(-1)),
                cache_config.or_else(CacheConfig(-1)).code,
                max_dynamic_shared_size_bytes,
                debug_level.unsafe_cstr_ptr().bitcast[UInt8](),
                optimization_level,
            )
        )
        self._handle = result

    fn _copy_to_constant_memory(
        self, borrowed mapping: ConstantMemoryMapping
    ) raises:
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(const DeviceFunction *func, const char *name,
        #                                                         const void *data, size_t byte_size)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CharPtr,
                _DeviceFunctionPtr,
                _CharPtr,
                _VoidPtr,
                _SizeT,
            ](
                self._handle,
                mapping.name.unsafe_cstr_ptr().bitcast[UInt8](),
                mapping.ptr,
                mapping.byte_count,
            )
        )

    @staticmethod
    fn _dump_q[val: Variant[Bool, Path, fn () capturing -> Path]]() -> Bool:
        @parameter
        if val.isa[Bool]():
            return val.unsafe_get[Bool]()
        elif val.isa[Path]():
            return val.unsafe_get[Path]() != Path("")
        return val.isa[fn () capturing -> Path]()

    @staticmethod
    fn _cleanup_asm(s: StringLiteral) -> StringLiteral:
        return s.replace("\t// begin inline asm\n", "").replace(
            "\t// end inline asm\n", ""
        )

    @no_inline
    @staticmethod
    fn dump_rep[
        dump_ptx: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
    ]() raises:
        @parameter
        if _ptxas_info_verbose:
            alias ptx = Self._func_impl.asm
            print(_ptxas_compile[target](ptx, options="-v"))

        @parameter
        if Self._dump_q[dump_ptx]():
            alias ptx = Self._cleanup_asm(Self._func_impl.asm)

            @parameter
            if dump_ptx.isa[fn () capturing -> Path]():
                alias dump_ptx_fn = dump_ptx.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_ptx_fn().write_text(ptx)
            elif dump_ptx.isa[Path]():
                dump_ptx.unsafe_get[Path]().write_text(ptx)
            else:
                print(ptx)

        @parameter
        if Self._dump_q[dump_sass]():
            alias ptx = Self._cleanup_asm(Self._func_impl.asm)
            var sass = _to_sass[target](ptx)

            @parameter
            if dump_sass.isa[fn () capturing -> Path]():
                alias dump_sass_fn = dump_sass.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_sass_fn().write_text(sass)
            elif dump_sass.isa[Path]():
                dump_sass.unsafe_get[Path]().write_text(sass)
            else:
                print(sass)

        @parameter
        if Self._dump_q[dump_llvm]():
            alias llvm = _compile_code_asm[
                Self.func, emission_kind="llvm-opt"
            ]()

            @parameter
            if dump_llvm.isa[fn () capturing -> Path]():
                alias dump_llvm_fn = dump_llvm.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_llvm_fn().write_text(llvm)
            elif dump_llvm.isa[Path]():
                dump_llvm.unsafe_get[Path]().write_text(llvm)
            else:
                print(llvm)

    @always_inline
    @parameter
    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContextV2,
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

        if cluster_dim:
            not_implemented_yet[
                "DeviceFunctionV2._call_with_pack: cluster_dim"
            ]()

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)
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
                UInt32,
                UnsafePointer[LaunchAttribute],
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
                shared_mem_bytes.or_else(0),
                attributes.unsafe_ptr(),
                len(attributes),
                dense_args_addrs,
            )
        )

    fn test_only_num_captures(self) -> Int:
        return self._func_impl.num_captures

    fn test_only_get_attribute(self, attr: Attribute) raises -> Int:
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_TEST_ONLY_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_TEST_ONLY_getAttribute",
                _CharPtr,
                UnsafePointer[Int32],
                _DeviceFunctionPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                attr.code,
            )
        )
        return int(result)


struct DeviceContextV2:
    """DeviceContext backed by a C++ implementation."""

    alias SYNC = DeviceSyncMode(True)
    alias ASYNC = DeviceSyncMode(False)

    var _handle: _DeviceContextPtr

    fn __init__(
        inout self, device_kind: StringLiteral = "cuda", device_id: Int = 0
    ) raises:
        # const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind, int id)
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

    fn enqueue_create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV2[type]:
        """Enqueues a buffer creation using the DeviceBuffer constructor."""
        return DeviceBufferV2[type](self, size, Self.ASYNC)

    fn create_buffer_sync[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV2[type]:
        """Creates a buffer synchronously using the DeviceBuffer constructor."""
        var result = DeviceBufferV2[type](self, size, Self.SYNC)
        self.synchronize()
        return result

    fn create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV2[type]:
        """Enqueues a buffer creation using the DeviceBuffer constructor."""
        return DeviceBufferV2[type](self, size, Self.ASYNC)

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
            dump_ptx=dump_ptx, dump_llvm=dump_llvm, dump_sass=dump_sass
        ]()

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
        f: DeviceFunctionV2,
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

    fn enqueue_memset[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
            "bitwidth of memset type must be one of [8,16,32]",
        ]()
        var value: UInt32

        @parameter
        if bitwidth == 8:
            value = UInt32(int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt32(int(bitcast[DType.uint16, 1](val)))
        else:
            value = bitcast[DType.uint32, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx, const DeviceBuffer *dst, uint32_t val, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt32,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    fn memset_sync[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
            "bitwidth of memset type must be one of [8,16,32]",
        ]()
        var value: UInt32

        @parameter
        if bitwidth == 8:
            value = UInt32(int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt32(int(bitcast[DType.uint16, 1](val)))
        else:
            value = bitcast[DType.uint32, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_sync(const DeviceContext *ctx, const DeviceBuffer *dst, uint32_t val, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt32,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        self.enqueue_memset[type](dst, val)

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

    fn is_compatible(self) raises:
        _checked(
            external_call[
                "AsyncRT_DeviceContext_isCompatibleWithMAX",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._handle,
            )
        )

    fn compute_capability(self) raises -> Int:
        var compute_capability: Int32 = 0
        _checked(
            external_call[
                "AsyncRT_DeviceContext_computeCapability",
                _CharPtr,
                _DeviceContextPtr,
                _IntPtr,
            ](self._handle, UnsafePointer.address_of(compute_capability))
        )
        return int(compute_capability)

    fn get_memory_info(self) raises -> (_SizeT, _SizeT):
        var free = _SizeT(0)
        var total = _SizeT(0)
        print("Calling AsyncRT_DeviceContext_getMemoryInfo")
        # const char *AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free, size_t *total)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getMemoryInfo",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[_SizeT],
                UnsafePointer[_SizeT],
            ](
                self._handle,
                UnsafePointer.address_of(free),
                UnsafePointer.address_of(total),
            )
        )

        return (free, total)
