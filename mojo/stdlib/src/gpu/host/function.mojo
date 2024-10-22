# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the function type."""

from collections import Dict, List, OptionalReg
from math.math import align_up
from pathlib import Path
from sys import is_defined
from sys.intrinsics import _mlirtype_is_eq

from builtin._location import __call_location
from gpu.host.device_context import DeviceBuffer
from memory import stack_allocation

from utils.lock import BlockingScopedLock, BlockingSpinLock
from utils.variant import Variant

from ._compile import (
    _compile_code,
    _compile_code_asm,
    _get_nvptx_fn_name,
    _get_nvptx_target,
    _ptxas_compile,
    _to_sass,
)
from ._utils import (
    CudaHandle,
    _check_error,
    _FunctionHandle,
    _ModuleHandle,
    _StreamHandle,
)
from .context import Context
from .cuda_instance import LaunchConfig
from .dim import Dim
from .module import Module
from .stream import Stream

# ===----------------------------------------------------------------------===#
# Attribute
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Attribute:
    var code: Int32

    alias MAX_THREADS_PER_BLOCK = Self(0)
    """The maximum number of threads per block, beyond which a launch of the
    function would fail. This number depends on both the function and the device
    on which the function is currently loaded."""

    alias SHARED_SIZE_BYTES = Self(1)
    """The size in bytes of statically-allocated shared memory required by this
    function. This does not include dynamically-allocated shared memory
    requested by the user at runtime."""

    alias CONST_SIZE_BYTES = Self(2)
    """The size in bytes of user-allocated constant memory required by this
    function."""

    alias LOCAL_SIZE_BYTES = Self(3)
    """The size in bytes of local memory used by each thread of this function."""

    alias NUM_REGS = Self(4)
    """The number of registers used by each thread of this function."""

    alias PTX_VERSION = Self(5)
    """The PTX virtual architecture version for which the function was compiled.
    This value is the major PTX version * 10 + the minor PTX version, so a PTX
    version 1.3 function would return the value 13. Note that this may return
    the undefined value of 0 for cubins compiled prior to CUDA 3.0.."""

    alias BINARY_VERSION = Self(6)
    """The binary architecture version for which the function was compiled.
    This value is the major binary version * 10 + the minor binary version,
    so a binary version 1.3 function would return the value 13. Note that this
    will return a value of 10 for legacy cubins that do not have a properly-
    encoded binary architecture version.."""

    alias CACHE_MODE_CA = Self(7)
    """The attribute to indicate whether the function has been compiled with
    user specified option "-Xptxas --dlcm=ca" set ."""

    alias MAX_DYNAMIC_SHARED_SIZE_BYTES = Self(8)
    """The maximum size in bytes of dynamically-allocated shared memory that
    can be used by this function. If the user-specified dynamic shared memory
    size is larger than this value."""

    alias PREFERRED_SHARED_MEMORY_CARVEOUT = Self(9)
    """On devices where the L1 cache and shared memory use the same hardware
    resources, this sets the shared memory carveout preference, in percent of
    the total shared memory."""

    alias CLUSTER_SIZE_MUST_BE_SET = Self(10)
    """If this attribute is set, the kernel must launch with a valid cluster
    size specified."""

    alias REQUIRED_CLUSTER_WIDTH = Self(11)
    """The required cluster width in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias REQUIRED_CLUSTER_HEIGHT = Self(12)
    """The required cluster height in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias REQUIRED_CLUSTER_DEPTH = Self(13)
    """The required cluster depth in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias NON_PORTABLE_CLUSTER_SIZE_ALLOWED = Self(14)
    """Whether the function can be launched with non-portable cluster size. 1 is
    allowed, 0 is disallowed. A non-portable cluster size may only function on
    the specific SKUs the program is tested on. The launch might fail if the
    program is run on a different hardware platform.CUDA API provides
    cudaOccupancyMaxActiveClusters to assist with checking whether the desired
    size can be launched on the current device.Portable Cluster SizeA portable
    cluster size is guaranteed to be functional on all compute capabilities
    higher than the target compute capability. The portable cluster size for
    sm_90 is 8 blocks per cluster."""

    alias CLUSTER_SCHEDULING_POLICY_PREFERENCE = Self(15)
    """The block scheduling policy of a function. The value type is
    CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        return not (self is other)


# ===----------------------------------------------------------------------===#
# CacheConfig
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct CacheConfig(CollectionElement, EqualityComparable):
    var code: Int32

    alias PREFER_NONE = Self(0)
    """No preference for shared memory or L1 (default)."""

    alias PREFER_SHARED = Self(1)
    """Prefer larger shared memory and smaller L1 cache."""

    alias PREFER_L1 = Self(2)
    """Prefer larger L1 cache and smaller shared memory."""

    alias PREFER_EQUAL = Self(3)
    """Prefer equal sized L1 cache and shared memory."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


# ===----------------------------------------------------------------------===#
# Function Attribute
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct FuncAttribute(CollectionElement, EqualityComparable):
    """Implement Cuda's CUfunction_attribute enum.
    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc.

    Only add 'max_dynamic_shared_size_bytes`.
    """

    var attribute: Attribute
    var value: Int32

    alias NULL = FuncAttribute(Attribute(-1), -1)

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.attribute == other.attribute and self.value == other.value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    @staticmethod
    fn CACHE_MODE_CA(val: Bool) -> FuncAttribute:
        """Indicates whether the function has been compiled with user specified
        option CacheMode.L1_CACHE_DISABLED set."""
        return FuncAttribute(Attribute.CACHE_MODE_CA, int(val))

    @always_inline
    @staticmethod
    fn MAX_DYNAMIC_SHARED_SIZE_BYTES(val: UInt32) -> FuncAttribute:
        """The maximum size in bytes of dynamically-allocated shared memory that
        can be used by this function. If the user-specified dynamic shared memory
        size is larger than this value, the launch will fail."""
        return FuncAttribute(
            Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, val.cast[DType.int32]()
        )

    @always_inline
    @staticmethod
    fn PREFERRED_SHARED_MEMORY_CARVEOUT(val: Int32) -> FuncAttribute:
        """On devices where the L1 cache and shared memory use the same hardware
        resources, this sets the shared memory carveout preference, in percent
        of the total shared memory."""
        return FuncAttribute(Attribute.PREFERRED_SHARED_MEMORY_CARVEOUT, val)


# ===----------------------------------------------------------------------===#
# Cached Function Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct _CachedFunctionInfo(Boolable):
    var mod_handle: _ModuleHandle
    var func_handle: _FunctionHandle
    var error: Error

    fn __init__(inout self):
        self.mod_handle = _ModuleHandle()
        self.func_handle = _FunctionHandle()
        self.error = Error()

    fn __init__(inout self, error: Error):
        self.mod_handle = _ModuleHandle()
        self.func_handle = _FunctionHandle()
        self.error = error

    fn __init__(
        inout self, mod_handle: _ModuleHandle, func_handle: _FunctionHandle
    ):
        self.mod_handle = mod_handle
        self.func_handle = func_handle
        self.error = Error()

    fn __bool__(self) -> Bool:
        return self.func_handle.__bool__()


struct FunctionCache:
    var dict: Dict[String, _CachedFunctionInfo]
    var lock: BlockingSpinLock

    fn __init__(inout self):
        self.dict = Dict[String, _CachedFunctionInfo]()
        self.lock = BlockingSpinLock()

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.dict = existing.dict^
        self.lock = BlockingSpinLock()

    fn __contains__(self, value: StringLiteral) -> Bool:
        return value in self.dict


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


fn _dump_q[val: Variant[Bool, Path, fn () capturing -> Path]]() -> Bool:
    @parameter
    if val.isa[Bool]():
        return val.unsafe_get[Bool]()
    elif val.isa[Path]():
        return val.unsafe_get[Path]() != Path("")
    return val.isa[fn () capturing -> Path]()


fn _cleanup_asm(s: StringLiteral) -> StringLiteral:
    return s.replace("\t// begin inline asm\n", "").replace(
        "\t// end inline asm\n", ""
    )


@value
struct Function[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
    _is_failable: Bool = False,
    _ptxas_info_verbose: Bool = False,
]:
    var info: _CachedFunctionInfo
    var cuda_dll: CudaDLL
    var cuda_function_cache: UnsafePointer[FunctionCache]

    alias _impl = _compile_code[
        func, is_failable=_is_failable, emission_kind="asm", target=target
    ]()

    @always_inline
    fn __init__(
        inout self,
        ctx_ptr: UnsafePointer[DeviceContextV1],
        *,
        max_registers: OptionalReg[Int] = None,
        threads_per_block: OptionalReg[Int] = None,
        cache_mode: OptionalReg[CacheMode] = None,
        cache_config: OptionalReg[CacheConfig] = None,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        self.__init__(
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
            # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
            cuda_dll=ctx_ptr[].cuda_context.cuda_dll,
            cuda_function_cache=ctx_ptr[].cuda_context.cuda_function_cache,
            device_context_ptr=ctx_ptr,
            cuda_dll_ptr=UnsafePointer[CudaDLL].address_of(
                ctx_ptr[].cuda_context.cuda_dll
            ),
        )

    fn __init__(
        inout self,
        cuda_dll: CudaDLL,
        cuda_dll_ptr: UnsafePointer[CudaDLL],
        *,
        max_registers: OptionalReg[Int] = None,
        threads_per_block: OptionalReg[Int] = None,
        cache_mode: OptionalReg[CacheMode] = None,
        cache_config: OptionalReg[CacheConfig] = None,
        func_attribute: OptionalReg[FuncAttribute] = None,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
        device_context_ptr: UnsafePointer[DeviceContextV1] = UnsafePointer[
            DeviceContextV1
        ](),
    ) raises:
        @parameter
        if _is_failable and self._impl.is_error:
            raise self._impl.error_msg

        self.cuda_dll = cuda_dll
        self.cuda_function_cache = cuda_function_cache
        self.info = Self._get_cached_function_info[func_type, func](
            device_context_ptr=device_context_ptr,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
            cuda_function_cache=cuda_function_cache,
        )

    fn __init__(
        inout self,
        module: Module,
        name: String,
        cuda_dll: CudaDLL,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
    ) raises:
        @parameter
        if _is_failable and self._impl.is_error:
            raise self._impl.error_msg

        self.cuda_dll = cuda_dll
        self.cuda_function_cache = cuda_function_cache
        var function_handle = module.load(name)
        if not function_handle:
            raise "Unable to load the CUDA function"

        self.info = _CachedFunctionInfo(module.module, function_handle)

    fn get_attribute(self, attr: Attribute) raises -> Int:
        var res = Int32(0)
        _check_error(
            self.cuda_dll.cuFuncGetAttribute(
                UnsafePointer.address_of(res), attr.code, self.info.func_handle
            )
        )
        return int(res)

    @no_inline
    @staticmethod
    fn dump_rep[
        dump_ptx: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
    ]() raises:
        @parameter
        if _ptxas_info_verbose:
            alias ptx = Self._impl.asm
            print(_ptxas_compile[target](ptx, options="-v"))

        @parameter
        if _dump_q[dump_ptx]():
            alias ptx = _cleanup_asm(Self._impl.asm)

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
        if _dump_q[dump_sass]():
            alias ptx = _cleanup_asm(Self._impl.asm)
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
        if _dump_q[dump_llvm]():
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
    fn __bool__(self) -> Bool:
        return self.info.__bool__()

    @always_inline
    @parameter
    fn __call__[
        *Ts: AnyType
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Stream,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self._call_pack(
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            stream=stream,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @always_inline
    @parameter
    fn _call_pack[
        *Ts: AnyType
    ](
        self,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        stream: Stream,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        alias num_args = len(VariadicList(Ts))
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        var args_stack = stack_allocation[
            num_captures + num_args, UnsafePointer[NoneType]
        ]()

        populate(
            rebind[UnsafePointer[NoneType]](args_stack.bitcast[NoneType]())
        )

        @parameter
        for i in range(num_args):
            alias arg_offset = num_captures + i
            var elt_addr = UnsafePointer.address_of(args[i])
            args_stack[arg_offset] = elt_addr.bitcast[NoneType]()

        self.__call_impl(
            args_stack,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            stream=stream,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @always_inline
    fn __call_impl(
        self,
        args: UnsafePointer[UnsafePointer[NoneType]],
        *,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Stream,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        if constant_memory:
            for i in range(len(constant_memory)):
                var entry = constant_memory[i]
                var device_ptr = UnsafePointer[NoneType]()
                _check_error(
                    self.cuda_dll.cuModuleGetGlobal(
                        UnsafePointer.address_of(device_ptr),
                        UnsafePointer[Int](),
                        self.info.mod_handle,
                        entry.name.unsafe_cstr_ptr(),
                    )
                )
                _check_error(
                    self.cuda_dll.cuMemcpyHtoDAsync(
                        device_ptr,
                        entry.ptr.bitcast[Int](),
                        entry.byte_count,
                        stream.stream,
                    )
                )

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        var config = LaunchConfig(
            grid_dim_x=grid_dim.x(),
            grid_dim_y=grid_dim.y(),
            grid_dim_z=grid_dim.z(),
            block_dim_x=block_dim.x(),
            block_dim_y=block_dim.y(),
            block_dim_z=block_dim.z(),
            shared_mem_bytes=shared_mem_bytes.or_else(0),
            stream=stream.stream,
            attrs=attributes.unsafe_ptr(),
            num_attrs=len(attributes),
        )
        _check_error(
            self.cuda_dll.cuLaunchKernelEx(
                UnsafePointer.address_of(config),
                self.info.func_handle,
                args,
                UnsafePointer[NoneType](),
            ),
            msg=Self._impl.function_name,
            location=__call_location(),
        )

    @staticmethod
    fn init_fn[
        func_type: AnyTrivialRegType, func: func_type
    ](
        device_context_ptr: UnsafePointer[DeviceContextV1],
        *,
        max_registers: OptionalReg[Int],
        threads_per_block: OptionalReg[Int],
        cache_config: OptionalReg[CacheConfig],
        func_attribute: OptionalReg[FuncAttribute],
        cache_mode: OptionalReg[CacheMode],
    ) raises -> _CachedFunctionInfo:
        alias _impl = Self._impl
        alias fn_name = _impl.function_name

        # Set the current context in case this is called outside the main
        # thread.
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        device_context_ptr[].cuda_context.set_current()

        var cuda_dll = device_context_ptr[].cuda_instance.cuda_dll

        var module = Module(
            _impl.asm,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cuda_dll=cuda_dll,
        )
        var func_handle = module.load(fn_name)

        if cache_config:
            _check_error(
                cuda_dll.cuFuncSetCacheConfig(
                    func_handle, cache_config.value().code
                )
            )
        if func_attribute:
            _check_error(
                cuda_dll.cuFuncSetAttribute(
                    func_handle,
                    func_attribute.value().attribute.code,
                    func_attribute.value().value,
                )
            )
        return _CachedFunctionInfo(module._steal_handle(), func_handle)

    @staticmethod
    fn _destroy_fn(cached_value_ptr: UnsafePointer[NoneType]):
        if not cached_value_ptr:
            return
        # We do not need to destroy the module, since it will be destroyed once the
        # CUDA context is destroyed.
        cached_value_ptr.free()

    @staticmethod
    @always_inline
    fn _get_cached_function_info[
        func_type: AnyTrivialRegType, func: func_type
    ](
        device_context_ptr: UnsafePointer[DeviceContextV1],
        *,
        max_registers: OptionalReg[Int],
        threads_per_block: OptionalReg[Int],
        cache_config: OptionalReg[CacheConfig],
        func_attribute: OptionalReg[FuncAttribute],
        cache_mode: OptionalReg[CacheMode],
        cuda_function_cache: UnsafePointer[FunctionCache],
    ) raises -> _CachedFunctionInfo:
        alias fn_name = _get_nvptx_fn_name[func]()
        with BlockingScopedLock(cuda_function_cache[].lock):
            # FIXME: (MSTDL-694) The following sporadically fails in commit
            # test (unhandled exception).

            # if entry:
            #     return entry.value()

            # var info_ptr = init_fn(payload)
            # self.dict[name] = info_ptr
            # return info_ptr

            # FIXME: (MSTDL-694) This code is unnecessairly expensive,
            # but it won't fail in commit tests.
            if fn_name in cuda_function_cache[]:
                return cuda_function_cache[].dict[fn_name]

        var info = Self.init_fn[func_type, func](
            device_context_ptr=device_context_ptr,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_config=cache_config,
            func_attribute=func_attribute,
            cache_mode=cache_mode,
        )
        with BlockingScopedLock(cuda_function_cache[].lock):
            cuda_function_cache[].dict[fn_name] = info
            return info
