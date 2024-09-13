# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Implementation of the C++ backed DeviceContext in Mojo
# WIP, just stubs for now

from gpu.host._compile import _get_nvptx_target


@value
struct DeviceBufferV2[type: DType](Sized):
    fn __init__(inout self, ctx: DeviceContextV2, size: Int) raises:
        """This init takes in a constructed DeviceContext and schedules an owned buffer allocation
        using the stream in the device context.
        """
        pass

    fn __init__(
        inout self,
        ctx: DeviceContextV2,
        ptr: UnsafePointer[Scalar[type]],
        size: Int,
        *,
        owning: Bool,
    ):
        pass

    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    @always_inline
    fn __del__(owned self):
        """This function schedules an owned buffer free using the stream in the device context.
        """
        pass

    fn __len__(self) -> Int:
        return 0  # FIXME

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBufferV2[view_type]:
        return DeviceBufferV2[view_type]()  # FIXME

    fn take_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        return UnsafePointer[Scalar[type]]()  # FIXME

    fn ptr(self) -> UnsafePointer[Scalar[type]]:
        return UnsafePointer[Scalar[type]]()  # FIXME


@value
struct DeviceFunctionV2[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    dump_ptx: Variant[Path, Bool] = False,
    dump_llvm: Variant[Path, Bool] = False,
    dump_sass: Variant[Path, Bool] = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
    _is_failable: Bool = False,
    _ptxas_info_verbose: Bool = False,
]:
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
        pass


@value
struct DeviceContextV2:
    """Device Context backed by a C++ implementation."""

    fn __init__(inout self, kind: String):
        pass

    fn __del__(owned self):
        pass

    fn __enter__(owned self) -> Self:
        return self^

    fn malloc_host[
        type: AnyType
    ](self, size: Int) raises -> UnsafePointer[type]:
        return UnsafePointer[type]()

    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        pass

    fn create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBufferV2[type]:
        """Creates a buffer using the DeviceBuffer constructor."""
        return DeviceBufferV2[type](self, size)

    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        dump_sass: Variant[Path, Bool] = False,
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
        pass

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
        pass

    fn execution_time[
        func: fn (Self) raises capturing -> None
    ](self, num_iters: Int) raises -> Int:
        return 0  # FIXME

    fn execution_time_iter[
        func: fn (Self, Int) raises capturing -> None
    ](self, num_iters: Int) raises -> Int:
        return 0  # FIXME

    fn enqueue_copy_to_device[
        type: DType
    ](self, buf: DeviceBufferV2[type], ptr: UnsafePointer[Scalar[type]]) raises:
        pass

    fn enqueue_copy_from_device[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV2[type]) raises:
        pass

    fn enqueue_copy_device_to_device[
        type: DType
    ](self, dst: DeviceBufferV2[type], src: DeviceBufferV2[type]) raises:
        pass

    fn copy_to_device_sync[
        type: DType
    ](self, buf: DeviceBufferV2[type], ptr: UnsafePointer[Scalar[type]]) raises:
        pass

    fn copy_from_device_sync[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], buf: DeviceBufferV2[type]) raises:
        pass

    fn copy_device_to_device_sync[
        type: DType
    ](self, dst: DeviceBufferV2[type], src: DeviceBufferV2[type]) raises:
        pass

    fn memset[
        type: DType
    ](self, dst: DeviceBufferV2[type], val: Scalar[type]) raises:
        pass

    fn synchronize(self) raises:
        pass

    fn print_kernel_timing_info(self):
        """Print profiling info associated with this DeviceContext."""
        pass

    fn dump_kernel_timing_info(self) raises:
        """Prints out profiling info associated with this DeviceContext."""
        pass

    fn clear_kernel_timing_info(self):
        """Clear profiling info associated with this DeviceContext."""
        pass

    fn is_compatible(self) raises:
        pass

    fn compute_capability(self) raises -> Int:
        return 0  # FIXME
