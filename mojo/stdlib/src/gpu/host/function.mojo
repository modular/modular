# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the function type."""

from collections.optional import Optional
from math.math import align_up
from pathlib import Path
from sys.ffi import _get_global
from sys.intrinsics import _mlirtype_is_eq

from memory import stack_allocation
from memory.unsafe import DTypePointer, Pointer

from utils.variant import Variant

from ._compile import _compile_code, _get_nvptx_fn_name
from ._utils import _check_error, _get_dylib_function
from .dim import Dim
from .module import ModuleHandle, _ModuleImpl
from .stream import Stream, _StreamImpl

# ===----------------------------------------------------------------------===#
# CacheConfig
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct CacheConfig(CollectionElement, EqualityComparable):
    var code: Int32

    alias PREFER_NONE = CacheConfig(0)
    """No preference for shared memory or L1 (default)."""

    alias PREFER_SHARED = CacheConfig(1)
    """Prefer larger shared memory and smaller L1 cache."""

    alias PREFER_L1 = CacheConfig(2)
    """Prefer larger L1 cache and smaller shared memory."""

    alias PREFER_EQUAL = CacheConfig(3)
    """Prefer equal sized L1 cache and shared memory."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


# ===----------------------------------------------------------------------===#
# FunctionHandle
# ===----------------------------------------------------------------------===#

alias _populate_fn_type = fn (Pointer[NoneType]) capturing -> None


@value
@register_passable("trivial")
struct FunctionHandle(Boolable):
    var handle: DTypePointer[DType.invalid]

    @always_inline
    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    @always_inline
    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    @always_inline
    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        *Ts: AnyType,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        self._call_impl_pack[num_captures, populate](
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            stream=stream,
        )

    @always_inline
    fn _call_impl_pack[
        num_captures: Int,
        populate: _populate_fn_type,
        *Ts: AnyType,
        elt_is_mutable: __mlir_type.i1,
        lifetime: AnyLifetime[elt_is_mutable].type,
    ](
        self,
        # TODO(unpacking): this is just because we can't forward packs!
        args: VariadicPack[elt_is_mutable, lifetime, AnyType, Ts],
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        var args_stack = stack_allocation[
            num_captures + len(VariadicList(Ts)), Pointer[NoneType]
        ]()
        populate(args_stack.bitcast[NoneType]())

        @parameter
        fn unrolled[i: Int]():
            var arg_offset = num_captures + i
            args_stack[arg_offset] = (
                args.get_element[i]().get_legacy_pointer().bitcast[NoneType]()
            )

        unroll[unrolled, args.__len__()]()

        self.__call_impl(
            args_stack, grid_dim=grid_dim, block_dim=block_dim, stream=stream
        )

    @always_inline
    fn __call_impl(
        self,
        args: Pointer[Pointer[NoneType]],
        *,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        var stream_value = stream.value().stream if stream else Stream()
        _check_error(
            _get_dylib_function[
                "cuLaunchKernel",
                fn (
                    Self,
                    UInt32,  # GridDimZ
                    UInt32,  # GridDimY
                    UInt32,  # GridDimX
                    UInt32,  # BlockDimZ
                    UInt32,  # BlockDimY
                    UInt32,  # BlockDimX
                    UInt32,  # SharedMemSize
                    _StreamImpl,
                    Pointer[Pointer[NoneType]],  # Args
                    DTypePointer[DType.invalid],  # Extra
                ) -> Result,
            ]()(
                self.handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(0),
                stream_value.stream,
                args,
                DTypePointer[DType.invalid](),
            )
        )
        # if we created the stream, we should sync
        if not stream:
            stream_value.synchronize()


# ===----------------------------------------------------------------------===#
# Cached Function Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct _CachedFunctionInfo(Boolable):
    var mod_handle: _ModuleImpl
    var func_handle: FunctionHandle
    var error: Error

    fn __init__(inout self):
        self.mod_handle = _ModuleImpl()
        self.func_handle = FunctionHandle()
        self.error = Error()

    fn __init__(
        inout self, mod_handle: _ModuleImpl, func_handle: FunctionHandle
    ):
        self.mod_handle = mod_handle
        self.func_handle = func_handle
        self.error = Error()

    fn __init__(inout self, error: Error):
        self.mod_handle = _ModuleImpl()
        self.func_handle = FunctionHandle()
        self.error = error

    fn __del__(owned self):
        pass

    fn __bool__(self) -> Bool:
        return self.func_handle.__bool__()


@value
@register_passable
struct _GlobalPayload:
    var debug: Bool
    var verbose: Bool
    var max_registers: Int32
    var threads_per_block: Int32
    var cache_config: Int32

    fn __init__(
        debug: Bool,
        verbose: Bool,
        max_registers: Int32,
        threads_per_block: Int32,
        cache_config: Int32,
    ) -> Self:
        return Self {
            debug: debug,
            verbose: verbose,
            max_registers: max_registers,
            threads_per_block: threads_per_block,
            cache_config: cache_config,
        }


fn _init_fn[
    func_type: AnyRegType, func: func_type
](payload_ptr: Pointer[NoneType]) -> Pointer[NoneType]:
    try:
        var payload = payload_ptr.bitcast[_GlobalPayload]().load()

        alias _impl = _compile_code[func_type, func, emission_kind="asm"]()
        alias fn_name = _get_nvptx_fn_name[func_type, func]()

        var mod_handle = ModuleHandle(
            _impl.asm,
            debug=payload.debug,
            verbose=payload.verbose or payload.debug,
            max_registers=Optional[Int]() if payload.max_registers
            <= 0 else Optional[Int](int(payload.max_registers)),
            threads_per_block=Optional[Int]() if payload.threads_per_block
            <= 0 else Optional[Int](int(payload.threads_per_block)),
        )
        var func_handle = mod_handle.load(fn_name)
        if payload.cache_config != -1:
            _check_error(
                _get_dylib_function[
                    "cuFuncSetCacheConfig", fn (FunctionHandle, Int32) -> Result
                ]()(func_handle, payload.cache_config)
            )
        var res = Pointer[_CachedFunctionInfo].alloc(1)
        res.store(_CachedFunctionInfo(mod_handle._steal_handle(), func_handle))
        return res.bitcast[NoneType]()
    except e:
        var res = Pointer[_CachedFunctionInfo].alloc(1)
        res.store(_CachedFunctionInfo(e))
        return res.bitcast[NoneType]()


fn _destroy_fn(cached_value_ptr: Pointer[NoneType]):
    if not cached_value_ptr:
        return
    var cached_value = cached_value_ptr.bitcast[_CachedFunctionInfo]().load()
    # We do not need to destroy the module, since it will be destroyed once the
    # CUDA context is destroyed.
    cached_value_ptr.free()


@always_inline
fn _get_global_cache_info[
    func_type: AnyRegType, func: func_type
](
    debug: Bool = False,
    verbose: Bool = False,
    max_registers: Int = -1,
    threads_per_block: Int = -1,
    cache_config: Int32 = -1,
) -> _CachedFunctionInfo:
    alias fn_name = _get_nvptx_fn_name[func_type, func]()

    var payload = _GlobalPayload(
        debug, verbose, max_registers, threads_per_block, cache_config
    )

    var info_ptr = _get_global[
        fn_name,
        _init_fn[func_type, func],
        _destroy_fn,
    ](Pointer.address_of(payload).bitcast[NoneType]())

    _ = payload

    return info_ptr.bitcast[_CachedFunctionInfo]().load()


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct Function[func_type: AnyRegType, func: func_type](Boolable):
    var info: _CachedFunctionInfo

    alias _impl = _compile_code[func_type, func, emission_kind="asm"]()

    @always_inline
    fn __init__(
        inout self,
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
    ) raises:
        fn dump_q(val: Variant[Path, Bool]) -> Bool:
            if val.isa[Bool]():
                return val.get[Bool]()[]
            return val.get[Path]()[] != ""

        if dump_q(dump_ptx):
            alias ptx = Self._impl.asm
            if dump_ptx.isa[Path]():
                with open(dump_ptx.get[Path]()[], "w") as f:
                    f.write(ptx)
            else:
                print(ptx)
        if dump_q(dump_llvm):
            alias llvm = _compile_code[
                func_type, func, emission_kind="llvm"
            ]().asm

            if dump_llvm.isa[Path]():
                with open(dump_llvm.get[Path]()[], "w") as f:
                    f.write(llvm)
            else:
                print(llvm)

        var info = _get_global_cache_info[func_type, func](
            debug=debug,
            verbose=verbose,
            max_registers=max_registers.value() if max_registers else -1,
            threads_per_block=threads_per_block.value() if threads_per_block else -1,
            cache_config=cache_config.value().code if cache_config else -1,
        )

        if info.error:
            raise info.error

        self.info = info

        if not self.info.func_handle:
            raise "Unable to load the CUDA function"

    @always_inline
    fn __del__(owned self):
        pass

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
        stream: Optional[Stream] = None,
    ) raises:
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        self.info.func_handle._call_impl_pack[num_captures, populate](
            args, grid_dim=grid_dim, block_dim=block_dim, stream=stream
        )
