# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from sys.ffi import c_size_t
from sys.param_env import is_defined

from gpu.host._compile import _get_nvptx_target

# The structs in this file are thin wrappers that delegate to either
# DeviceContextV1 (the old, Mojo implementation) or DeviceContextV2 (the new,
# C++ backed implementation).
# We don't have polymorphism in Mojo (yet!) so we need to wrap each method.

# RUNP-457: Revert back to V1 directly for now
alias DeviceFunction = DeviceFunctionV1
alias DeviceBuffer = DeviceBufferV1
alias DeviceContext = DeviceContextV1


# Runtime switch to select Device context V1 (mojo) or V2 (C++)
@parameter
fn _device_ctx_v2() -> Bool:
    return is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()


struct DeviceFunctionVariant[
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
    alias V1 = DeviceFunctionV1[
        func,
        dump_ptx=dump_ptx,
        dump_llvm=dump_llvm,
        dump_sass=dump_sass,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ]
    alias V2 = DeviceFunctionV2[
        func,
        dump_ptx=dump_ptx,
        dump_llvm=dump_llvm,
        dump_sass=dump_sass,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ]
    var _impl: Variant[Self.V1, Self.V2]

    @parameter
    fn v1(self) -> ref [__lifetime_of(self._impl)] Self.V1:
        return self._impl[Self.V1]

    @parameter
    fn v2(self) -> ref [__lifetime_of(self._impl)] Self.V2:
        return self._impl[Self.V2]

    fn __init__(inout self, owned impl: Self.V1):
        self._impl = impl^

    fn __init__(inout self, owned impl: Self.V2):
        self._impl = impl^

    fn __copyinit__(inout self, existing: Self):
        self._impl = existing._impl

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl^


struct DeviceBufferVariant[type: DType](Sized):
    alias V1 = DeviceBufferV1[type]
    alias V2 = DeviceBufferV2[type]

    var _impl: Variant[Self.V1, Self.V2]

    @parameter
    fn v1(self) -> ref [__lifetime_of(self._impl)] Self.V1:
        return self._impl[Self.V1]

    @parameter
    fn v2(self) -> ref [__lifetime_of(self._impl)] Self.V2:
        return self._impl[Self.V2]

    def __init__(inout self, owned impl: Self.V1):
        self._impl = impl^

    def __init__(inout self, owned impl: Self.V2):
        self._impl = impl^

    fn __init__(inout self, ctx: DeviceContextVariant, size: Int) raises:
        @parameter
        if _device_ctx_v2():
            # TODO(iposva): Should we expose the sync mode to the public API?
            self._impl = Self.V2(ctx.v2(), size, DeviceContextV2.SYNC)
        else:
            self._impl = Self.V1(ctx.v1(), size)

    fn __init__(
        inout self,
        ctx: DeviceContextVariant,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        @parameter
        if _device_ctx_v2():
            self._impl = Self.V2(ctx.v2(), ptr, size, owning=owning)
        else:
            self._impl = Self.V1(ctx.v1(), ptr, size, owning=owning)

    fn __init__(inout self):
        @parameter
        if _device_ctx_v2():
            self._impl = Self.V2()
        else:
            self._impl = Self.V1()

    fn __copyinit__(inout self, existing: Self):
        self._impl = existing._impl

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl^

    fn __len__(self) -> Int:
        @parameter
        if _device_ctx_v2():
            return self.v2().__len__()
        else:
            return self.v1().__len__()

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBufferVariant[view_type]:
        @parameter
        if _device_ctx_v2():
            return self.v2().create_sub_buffer[view_type](offset, size)
        else:
            return self.v1().create_sub_buffer[view_type](offset, size)

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        @parameter
        if _device_ctx_v2():
            return self._impl.take[Self.V2]().take_ptr()
        else:
            return self._impl.take[Self.V1]().take_ptr()

    # DeviceContextV1 allows direct access to the raw device pointer
    # DeviceContextV2 will make this a method
    fn __getattr__[name: StringLiteral](self) -> UnsafePointer[Scalar[type]]:
        @parameter
        if name == "ptr":

            @parameter
            if _device_ctx_v2():
                return self.v2().get_ptr()
            else:
                return self.v1().ptr

        abort("Unsupported attr for DeviceBufferVariant: " + name)
        return UnsafePointer[Scalar[type]]()


struct DeviceContextVariant:
    """Wrapper struct to select V1 or V2 implementation at compile time."""

    alias V1 = DeviceContextV1
    alias V2 = DeviceContextV2
    var _impl: Variant[Self.V1, Self.V2]

    @parameter
    fn v1(self) -> ref [__lifetime_of(self._impl)] Self.V1:
        return self._impl[Self.V1]

    @parameter
    fn v2(self) -> ref [__lifetime_of(self._impl)] Self.V2:
        return self._impl[Self.V2]

    fn __init__(
        inout self, kind: StringLiteral = "cuda", gpu_id: Int = 0
    ) raises:
        @parameter
        if _device_ctx_v2():
            self._impl = Self.V2(kind, gpu_id)
        else:
            self._impl = Self.V1(kind, gpu_id)

    fn __copyinit__(inout self, existing: Self):
        self._impl = existing._impl

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl^

    fn __enter__(owned self) -> Self:
        return self^

    fn name(self) -> String:
        @parameter
        if _device_ctx_v2():
            return "DeviceContextV2"
        else:
            return "DeviceContextV1"

    fn malloc_host[
        type: AnyType
    ](self, size: Int) raises -> UnsafePointer[type]:
        @parameter
        if _device_ctx_v2():
            return self.v2().malloc_host[type](size)
        else:
            return self.v1().malloc_host[type](size)

    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().free_host[type](ptr)
        else:
            self.v1().free_host[type](ptr)

    fn enqueue_create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferVariant[type]:
        """Enqueues a buffer creation using the DeviceBuffer constructor."""

        @parameter
        if _device_ctx_v2():
            return self.v2().enqueue_create_buffer[type](size)
        else:
            return self.v1().enqueue_create_buffer[type](size)

    fn create_buffer_sync[
        type: DType
    ](self, size: Int) raises -> DeviceBufferVariant[type]:
        """Creates a buffer synchronously using the DeviceBuffer constructor."""

        @parameter
        if _device_ctx_v2():
            return self.v2().create_buffer_sync[type](size)
        else:
            return self.v1().create_buffer_sync[type](size)

    fn create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferVariant[type]:
        @parameter
        if _device_ctx_v2():
            return self.v2().enqueue_create_buffer[type](size)
        else:
            return self.v1().enqueue_create_buffer[type](size)

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
    ) raises -> DeviceFunctionVariant[
        func,
        dump_ptx=dump_ptx,
        dump_llvm=dump_llvm,
        dump_sass=dump_sass,
        target=target,
        _is_failable=_is_failable,
        _ptxas_info_verbose=_ptxas_info_verbose,
    ]:
        @parameter
        if _device_ctx_v2():
            return self.v2().compile_function[
                func,
                dump_ptx=dump_ptx,
                dump_llvm=dump_llvm,
                dump_sass=dump_sass,
                target=target,
                _is_failable=_is_failable,
                _ptxas_info_verbose=_ptxas_info_verbose,
            ](
                max_registers=max_registers,
                threads_per_block=threads_per_block,
                cache_mode=cache_mode,
                cache_config=cache_config,
                func_attribute=func_attribute,
            )
        else:
            return self.v1().compile_function[
                func,
                dump_ptx=dump_ptx,
                dump_llvm=dump_llvm,
                dump_sass=dump_sass,
                target=target,
                _is_failable=_is_failable,
                _ptxas_info_verbose=_ptxas_info_verbose,
            ](
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
        f: DeviceFunctionVariant,
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
        f: DeviceFunctionVariant,
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
        @parameter
        if _device_ctx_v2():
            return self.v2()._enqueue_function[*Ts](
                f.v2(),
                args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )
        else:
            return self.v1()._enqueue_function[*Ts](
                f.v1(),
                args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )

    fn execution_time[
        func: fn (Self) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        @parameter
        if _device_ctx_v2():

            @always_inline
            @parameter
            fn wrap_func(this: Self.V2) raises -> None:
                return func(self)

            return self.v2().execution_time[wrap_func](num_iters)
        else:

            @always_inline
            @parameter
            fn wrap_func(this: Self.V1) raises -> None:
                return func(self)

            return self.v1().execution_time[wrap_func](num_iters)

    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        @parameter
        if _device_ctx_v2():

            @always_inline
            @parameter
            fn wrap_func(
                this: Self.V2, num_iters: Int
            ) capturing raises -> None:
                return func(self, num_iters)

            return self.v2().execution_time_iter[wrap_func](num_iters)
        else:

            @always_inline
            @parameter
            fn wrap_func(
                this: Self.V1, num_iters: Int
            ) capturing raises -> None:
                return func(self, num_iters)

            return self.v1().execution_time_iter[wrap_func](num_iters)

    fn enqueue_copy_to_device[
        type: DType
    ](
        self, buf: DeviceBufferVariant[type], ptr: UnsafePointer[Scalar[type]]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().enqueue_copy_to_device[type](buf.v2(), ptr)
        else:
            self.v1().enqueue_copy_to_device[type](buf.v1(), ptr)

    fn enqueue_copy_from_device[
        type: DType
    ](
        self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferVariant[type]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().enqueue_copy_from_device[type](ptr, buf.v2())
        else:
            self.v1().enqueue_copy_from_device[type](ptr, buf.v1())

    fn enqueue_copy_device_to_device[
        type: DType
    ](
        self, dst: DeviceBufferVariant[type], src: DeviceBufferVariant[type]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().enqueue_copy_device_to_device[type](dst.v2(), src.v2())
        else:
            self.v1().enqueue_copy_device_to_device[type](dst.v1(), src.v1())

    fn enqueue_copy_device_to_device[
        type: DType
    ](
        self,
        dst: UnsafePointer[Scalar[type]],
        src: UnsafePointer[Scalar[type]],
        size: Int,
    ) raises:
        @parameter
        if _device_ctx_v2():
            # Not implemented on DeviceContextV2, wrap in buffers first
            var dst_buf = DeviceBufferV2(self.v2(), dst, size, owning=False)
            var src_buf = DeviceBufferV2(self.v2(), src, size, owning=False)
            self.v2().enqueue_copy_device_to_device[type](dst_buf, src_buf)
        else:
            self.v1().enqueue_copy_device_to_device[type](dst, src, size)

    fn copy_to_device_sync[
        type: DType
    ](
        self, buf: DeviceBufferVariant[type], ptr: UnsafePointer[Scalar[type]]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().copy_to_device_sync(buf.v2(), ptr)
        else:
            self.v1().copy_to_device_sync(buf.v1(), ptr)

    fn copy_from_device_sync[
        type: DType
    ](
        self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferVariant[type]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().copy_from_device_sync(ptr, buf.v2())
        else:
            self.v1().copy_from_device_sync(ptr, buf.v1())

    fn copy_device_to_device_sync[
        type: DType
    ](
        self, dst: DeviceBufferVariant[type], src: DeviceBufferVariant[type]
    ) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().copy_device_to_device_sync(dst.v2(), src.v2())
        else:
            self.v1().copy_device_to_device_sync(dst.v1(), src.v1())

    fn enqueue_memset[
        type: DType
    ](self, dst: DeviceBufferVariant[type], val: Scalar[type]) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().enqueue_memset[type](dst.v2(), val)
        else:
            self.v1().enqueue_memset[type](dst.v1(), val)

    fn memset_sync[
        type: DType
    ](self, dst: DeviceBufferVariant[type], val: Scalar[type]) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().memset_sync[type](dst.v2(), val)
        else:
            self.v1().memset_sync[type](dst.v1(), val)

    fn memset[
        type: DType
    ](self, dst: DeviceBufferVariant[type], val: Scalar[type]) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().enqueue_memset[type](dst.v2(), val)
        else:
            self.v1().enqueue_memset[type](dst.v1(), val)

    fn synchronize(self) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().synchronize()
        else:
            self.v1().synchronize()

    fn print_kernel_timing_info(self):
        @parameter
        if _device_ctx_v2():
            self.v2().print_kernel_timing_info()
        else:
            self.v1().print_kernel_timing_info()

    fn dump_kernel_timing_info(self) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().dump_kernel_timing_info()
        else:
            self.v1().dump_kernel_timing_info()

    fn clear_kernel_timing_info(self):
        @parameter
        if _device_ctx_v2():
            self.v2().clear_kernel_timing_info()
        else:
            self.v1().clear_kernel_timing_info()

    fn is_compatible(self) raises:
        @parameter
        if _device_ctx_v2():
            self.v2().is_compatible()
        else:
            self.v1().is_compatible()

    fn compute_capability(self) raises -> Int:
        @parameter
        if _device_ctx_v2():
            return self.v2().compute_capability()
        else:
            return self.v1().compute_capability()

    fn get_memory_info(self) raises -> (c_size_t, c_size_t):
        @parameter
        if _device_ctx_v2():
            return self.v2().get_memory_info()
        else:
            return self.v1().get_memory_info()
