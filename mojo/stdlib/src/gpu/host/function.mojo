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


@value
@register_passable("trivial")
struct AnyRegTuple[*Ts: AnyRegType]:
    alias _type = __mlir_type[
        `!kgen.pack<:variadic<`, AnyRegType, `> `, Ts, `>`
    ]
    var storage: Self._type

    @staticmethod
    fn _offset[i: Int]() -> Int:
        constrained[i >= 0, "index must be positive"]()

        @parameter
        if i == 0:
            return 0
        else:
            return align_up(
                Self._offset[i - 1]()
                + align_up(sizeof[Ts[i - 1]](), alignof[Ts[i - 1]]()),
                alignof[Ts[i]](),
            )

    fn get[i: Int, T: AnyRegType](self) -> T:
        alias offset = Self._offset[i]()
        # Copy the storage so we can get its address, because we can't take the
        # address of 'self' in a non-mutating method.
        # TODO(Ownership): we should be able to get const references.
        var selfCopy = self
        var addr = Pointer.address_of(selfCopy.storage).bitcast[Int8]().offset(
            offset
        )
        return addr.bitcast[T]().load()


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

    # TODO: Move the stream argument at the end when When Mojo will support both vararg and keyword only arguments
    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        *Ts: AnyRegType,
    ](
        self,
        borrowed *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        var values = AnyRegTuple(args)

        self._call_impl[num_captures, populate](
            values,
            grid_dim=grid_dim,
            block_dim=block_dim,
            stream=stream,
        )

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        *Ts: AnyRegType,
    ](
        self,
        inout values: AnyRegTuple[Ts],
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        alias types = VariadicList(Ts)

        var args_stack = stack_allocation[
            num_captures + len(VariadicList(Ts)), Pointer[NoneType]
        ]()
        populate(args_stack.bitcast[NoneType]())

        @parameter
        @always_inline
        fn append[i: Int]():
            alias T = types[i]
            var _val = values.get[i, T]()
            alias arg_offset = num_captures + i

            @parameter
            if _mlirtype_is_eq[T, FloatLiteral]():
                var tmp = Float32(rebind[FloatLiteral](_val))
                args_stack[arg_offset] = Pointer.address_of(tmp).bitcast[
                    NoneType
                ]()
            elif _mlirtype_is_eq[T, IntLiteral]():
                var tmp = Int(rebind[IntLiteral](_val))
                args_stack[arg_offset] = Pointer.address_of(tmp).bitcast[
                    NoneType
                ]()
            else:
                args_stack[arg_offset] = Pointer.address_of(_val).bitcast[
                    NoneType
                ]()

        unroll[append, len(types)]()

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

    fn __init__(inout self):
        self.mod_handle = _ModuleImpl()
        self.func_handle = FunctionHandle()

    fn __init__(
        inout self, mod_handle: _ModuleImpl, func_handle: FunctionHandle
    ):
        self.mod_handle = mod_handle
        self.func_handle = func_handle

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
        print("Error loading the PTX code:", e)
        return Pointer[NoneType]()


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

    if not info_ptr:
        return _CachedFunctionInfo()

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

        self.info = _get_global_cache_info[func_type, func](
            debug=debug,
            verbose=verbose,
            max_registers=max_registers.value() if max_registers else -1,
            threads_per_block=threads_per_block.value() if threads_per_block else -1,
            cache_config=cache_config.value().code if cache_config else -1,
        )

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
        *Ts: AnyRegType
    ](
        self,
        borrowed *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        stream: Optional[Stream] = None,
    ) raises:
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        var values = AnyRegTuple(args)

        self.info.func_handle._call_impl[num_captures, populate](
            values, grid_dim=grid_dim, block_dim=block_dim, stream=stream
        )
