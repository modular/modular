# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the function type."""

from collections import Dict, List, Optional
from math.math import align_up
from pathlib import Path
from sys.intrinsics import _mlirtype_is_eq
from sys import is_defined

from builtin._location import __call_location
from gpu.host.device_context import DeviceBuffer
from memory import stack_allocation

from utils.lock import BlockingScopedLock, BlockingSpinLock
from utils.variant import Variant

from ._compile import (
    _compile_code,
    _get_nvptx_fn_name,
    _get_nvptx_target,
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

    var code: Int32
    var value: Int32

    alias NULL = FuncAttribute(-1, -1)

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code and self.value == other.value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    @staticmethod
    fn CACHE_MODE_CA(val: Bool) -> FuncAttribute:
        """Indicates whether the function has been compiled with user specified
        option CacheMode.L1_CACHE_DISABLED set."""
        return FuncAttribute(7, int(val))

    @always_inline
    @staticmethod
    fn MAX_DYNAMIC_SHARED_SIZE_BYTES(val: Int32) -> FuncAttribute:
        """The maximum size in bytes of dynamically-allocated shared memory that
        can be used by this function. If the user-specified dynamic shared memory
        size is larger than this value, the launch will fail."""
        return FuncAttribute(8, val)

    @always_inline
    @staticmethod
    fn PREFERRED_SHARED_MEMORY_CARVEOUT(val: Int32) -> FuncAttribute:
        """On devices where the L1 cache and shared memory use the same hardware
        resources, this sets the shared memory carveout preference, in percent
        of the total shared memory."""
        return FuncAttribute(9, val)


# ===----------------------------------------------------------------------===#
# Cached Function Info
# ===----------------------------------------------------------------------===#


alias _populate_fn_type = fn (UnsafePointer[NoneType]) capturing -> None


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


@value
struct _CachedFunctionPayload:
    var max_registers: Optional[Int]
    var threads_per_block: Optional[Int]
    var cache_mode: Optional[CacheMode]
    var cache_config: Optional[CacheConfig]
    var func_attribute: Optional[FuncAttribute]
    var device_context_ptr: UnsafePointer[DeviceContext]
    var cuda_dll_ptr: UnsafePointer[CudaDLL]

    fn __init__(
        inout self,
        max_registers: Optional[Int],
        threads_per_block: Optional[Int],
        cache_mode: Optional[CacheMode],
        cache_config: Optional[CacheConfig],
        func_attribute: Optional[FuncAttribute],
        device_context_ptr: UnsafePointer[DeviceContext],
        cuda_dll_ptr: UnsafePointer[CudaDLL],
    ):
        self.max_registers = max_registers
        self.threads_per_block = threads_per_block
        self.cache_mode = cache_mode
        self.cache_config = cache_config
        self.func_attribute = func_attribute
        self.device_context_ptr = device_context_ptr
        self.cuda_dll_ptr = cuda_dll_ptr


struct FunctionCache:
    var dict: Dict[StringLiteral, _CachedFunctionInfo]
    var lock: BlockingSpinLock

    fn __init__(inout self):
        self.dict = Dict[StringLiteral, _CachedFunctionInfo]()
        self.lock = BlockingSpinLock()

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.dict = existing.dict^
        self.lock = BlockingSpinLock()

    fn get_or_create_entry[
        name: StringLiteral,
        init_fn: fn (_CachedFunctionPayload) raises -> _CachedFunctionInfo,
    ](
        inout self, payload: _CachedFunctionPayload
    ) raises -> _CachedFunctionInfo:
        with BlockingScopedLock(self.lock):
            # FIXME: (MSTDL-694) The following sporadically fails in commit test (unhandled exception).

            # var entry = self.dict.find(name)

            # if entry:
            #     return entry.value()

            # var info_ptr = init_fn(payload)
            # self.dict[name] = info_ptr
            # return info_ptr

            # FIXME: (MSTDL-694) This code is unnecessairly expensive, but it won't fail in commit tests.
            if name in self.dict:
                return self.dict[name]

            var info_ptr = init_fn(payload)
            self.dict[name] = info_ptr
            return info_ptr


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


fn _dump_q(val: Variant[Path, Bool]) -> Bool:
    if val.isa[Bool]():
        return val[Bool]
    return val[Path] != Path("")


fn _cleanup_asm(s: StringLiteral) -> StringLiteral:
    return s.replace("\t// begin inline asm\n", "").replace(
        "\t// end inline asm\n", ""
    )


@value
struct Function[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    dump_ptx: Variant[Path, Bool] = False,
    dump_llvm: Variant[Path, Bool] = False,
    dump_sass: Variant[Path, Bool] = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
    _is_failable: Bool = False,
](Boolable):
    var info: _CachedFunctionInfo
    var cuda_dll: CudaDLL
    var cuda_function_cache: UnsafePointer[FunctionCache]

    alias _impl = _compile_code[
        func, is_failable=_is_failable, emission_kind="asm", target=target
    ]()

    @always_inline
    fn __init__(
        inout self,
        ctx_ptr: UnsafePointer[DeviceContext],
        *,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_mode: Optional[CacheMode] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
    ) raises:
        self.__init__(
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_mode=cache_mode,
            cache_config=cache_config,
            func_attribute=func_attribute,
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
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_mode: Optional[CacheMode] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
        device_context_ptr: UnsafePointer[DeviceContext] = UnsafePointer[
            DeviceContext
        ](),
    ) raises:
        @parameter
        if _is_failable and self._impl.is_error:
            raise self._impl.error_msg

        self.cuda_dll = cuda_dll
        self.cuda_function_cache = cuda_function_cache

        Self._dump_rep()

        self.info = Self._get_cached_function_info[func_type, func](
            device_context_ptr=device_context_ptr,
            cuda_dll_ptr=cuda_dll_ptr,
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

        Self._dump_rep()

        var function_handle = module.load(name)
        if not function_handle:
            raise "Unable to load the CUDA function"

        self.info = _CachedFunctionInfo(module.module, function_handle)

    @no_inline
    @staticmethod
    fn _dump_rep() raises:
        @parameter
        if _dump_q(dump_ptx):
            alias ptx = _cleanup_asm(Self._impl.asm)
            if dump_ptx.isa[Path]():
                dump_ptx[Path].write_text(ptx)
            else:
                print(ptx)

        @parameter
        if _dump_q(dump_sass):
            alias ptx = _cleanup_asm(Self._impl.asm)
            var sass = _to_sass[target](ptx)
            if dump_sass.isa[Path]():
                dump_sass[Path].write_text(sass)
            else:
                print(sass)

        @parameter
        if _dump_q(dump_llvm):
            alias llvm = _compile_code[Self.func, emission_kind="llvm"]().asm

            if dump_llvm.isa[Path]():
                dump_llvm[Path].write_text(StringRef(llvm))
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
        shared_mem_bytes: Int = 0,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self._call_pack(
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
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
        args: VariadicPack[_, AnyType, Ts],
        grid_dim: Dim,
        block_dim: Dim,
        stream: Stream,
        shared_mem_bytes: Int = 0,
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
        shared_mem_bytes: Int = 0,
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
                _ = entry
                _ = device_ptr

        var config = LaunchConfig(
            grid_dim_x=grid_dim.x(),
            grid_dim_y=grid_dim.y(),
            grid_dim_z=grid_dim.z(),
            block_dim_x=block_dim.x(),
            block_dim_y=block_dim.y(),
            block_dim_z=block_dim.z(),
            shared_mem_bytes=shared_mem_bytes,
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
        _ = config
        _ = attributes^
        _ = constant_memory^

    @staticmethod
    fn init_fn[
        func_type: AnyTrivialRegType, func: func_type
    ](payload: _CachedFunctionPayload) raises -> _CachedFunctionInfo:
        alias _impl = Self._impl
        alias fn_name = _impl.function_name

        var module = Module(
            _impl.asm,
            max_registers=payload.max_registers,
            threads_per_block=payload.threads_per_block,
            cache_mode=payload.cache_mode,
            cuda_dll=payload.cuda_dll_ptr[],
        )
        var func_handle = module.load(fn_name)

        var cuda_dll = payload.device_context_ptr[].cuda_instance.cuda_dll

        if payload.cache_config:
            _check_error(
                cuda_dll.cuFuncSetCacheConfig(
                    func_handle, payload.cache_config.value().code
                )
            )
        if payload.func_attribute:
            _check_error(
                cuda_dll.cuFuncSetAttribute(
                    func_handle,
                    payload.func_attribute.value().code,
                    payload.func_attribute.value().value,
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
        device_context_ptr: UnsafePointer[DeviceContext],
        cuda_dll_ptr: UnsafePointer[CudaDLL],
        *,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
        cache_mode: Optional[CacheMode] = None,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
    ) raises -> _CachedFunctionInfo:
        alias fn_name = _get_nvptx_fn_name[func]()

        var payload = _CachedFunctionPayload(
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_config=cache_config,
            func_attribute=func_attribute,
            device_context_ptr=device_context_ptr,
            cache_mode=cache_mode,
            cuda_dll_ptr=cuda_dll_ptr,
        )

        var info = cuda_function_cache[].get_or_create_entry[
            fn_name,
            Self.init_fn[func_type, func],
        ](payload)

        _ = payload

        return info
