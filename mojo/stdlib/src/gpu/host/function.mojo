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

from gpu.host.device_context import DeviceBuffer
from memory import stack_allocation
from memory.unsafe import DTypePointer

from utils.lock import BlockingSpinLock, BlockingScopedLock
from utils.variant import Variant

from ._compile import _compile_code, _get_nvptx_fn_name
from builtin._location import __call_location
from ._utils import _check_error, _FunctionHandle, _ModuleHandle, _StreamHandle
from .context import Context
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

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code and self.value == other.value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

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

    fn __init__(
        inout self, mod_handle: _ModuleHandle, func_handle: _FunctionHandle
    ):
        self.mod_handle = mod_handle
        self.func_handle = func_handle
        self.error = Error()

    fn __init__(inout self, error: Error):
        self.mod_handle = _ModuleHandle()
        self.func_handle = _FunctionHandle()
        self.error = error

    fn __bool__(self) -> Bool:
        return self.func_handle.__bool__()


@value
@register_passable("trivial")
struct _CachedFunctionPayload:
    var debug: Bool
    var verbose: Bool
    var max_registers: Int32
    var threads_per_block: Int32
    var cache_config: Int32
    var func_attribute: FuncAttribute

    fn __init__(
        debug: Bool,
        verbose: Bool,
        max_registers: Int32,
        threads_per_block: Int32,
        cache_config: Int32,
        func_attribute: FuncAttribute,
    ) -> Self:
        return Self {
            debug: debug,
            verbose: verbose,
            max_registers: max_registers,
            threads_per_block: threads_per_block,
            cache_config: cache_config,
            func_attribute: func_attribute,
        }


struct FunctionCache:
    var dict: Dict[StringLiteral, UnsafePointer[_CachedFunctionInfo]]
    var lock: BlockingSpinLock

    fn __init__(inout self):
        self.dict = Dict[StringLiteral, UnsafePointer[_CachedFunctionInfo]]()
        self.lock = BlockingSpinLock()

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.dict = existing.dict^
        self.lock = BlockingSpinLock()

    fn __del__(owned self):
        for v in self.dict.values():
            v[].destroy_pointee()

    fn get_or_create_entry[
        name: StringLiteral,
        init_fn: fn (UnsafePointer[_CachedFunctionPayload]) -> UnsafePointer[
            _CachedFunctionInfo
        ],
    ](
        inout self,
        payload: UnsafePointer[_CachedFunctionPayload] = UnsafePointer[
            _CachedFunctionPayload
        ](),
    ) -> UnsafePointer[_CachedFunctionInfo]:
        with BlockingScopedLock(self.lock):
            # FIXME: (MSTDL-694) The following sporadically fails in commit test (unhandled exception).

            # var entry = self.dict.find(name)

            # if entry:
            #     return entry.value()

            # var info_ptr = init_fn(payload)
            # self.dict[name] = info_ptr
            # return info_ptr

            # FIXME: (MSTDL-694) This code is unnecessairly expensive, but it won't fail in commit tests.
            try:
                if name in self.dict:
                    return self.dict[name]

                var info_ptr = init_fn(payload)
                self.dict[name] = info_ptr
                return info_ptr
            except:
                return UnsafePointer[_CachedFunctionInfo]()


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


@value
struct Function[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    _is_failable: Bool = False,
](Boolable):
    var info: _CachedFunctionInfo
    var cuda_dll: Optional[CudaDLL]
    var cuda_function_cache: UnsafePointer[FunctionCache]

    alias _impl = _compile_code[
        func, is_failable=_is_failable, emission_kind="asm"
    ]()

    @always_inline
    fn __init__(
        inout self,
        ctx: Context,
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
    ) raises:
        self.__init__(
            debug=debug,
            verbose=verbose,
            dump_ptx=dump_ptx,
            dump_llvm=dump_llvm,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_config=cache_config,
            func_attribute=func_attribute,
            cuda_dll=ctx.cuda_dll,
            cuda_function_cache=ctx.cuda_function_cache,
        )

    fn __init__(
        inout self,
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cache_config: Optional[CacheConfig] = None,
        func_attribute: Optional[FuncAttribute] = None,
        cuda_dll: Optional[CudaDLL] = None,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
    ) raises:
        @parameter
        if _is_failable and self._impl.is_error:
            raise self._impl.error_msg

        self.cuda_dll = cuda_dll
        self.cuda_function_cache = cuda_function_cache

        Self._dump_rep(dump_ptx, dump_llvm)

        var info = Self._get_cached_function_info[func_type, func](
            debug=debug,
            verbose=verbose,
            max_registers=max_registers.value() if max_registers else -1,
            threads_per_block=threads_per_block.value() if threads_per_block else -1,
            cache_config=cache_config.value().code if cache_config else -1,
            func_attribute=func_attribute.value() if func_attribute else FuncAttribute.NULL,
            cuda_function_cache=cuda_function_cache,
        )

        if info.error:
            raise info.error

        self.info = info

        if not self.info.func_handle:
            raise "Unable to load the CUDA function"

    fn __init__(
        inout self,
        module: Module,
        name: String,
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
        cuda_dll: Optional[CudaDLL] = None,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
    ) raises:
        @parameter
        if _is_failable and self._impl.is_error:
            raise self._impl.error_msg

        self.cuda_dll = cuda_dll
        self.cuda_function_cache = cuda_function_cache

        Self._dump_rep(dump_ptx, dump_llvm)

        var function_handle = module.load(name)
        if not function_handle:
            raise "Unable to load the CUDA function"

        self.info = _CachedFunctionInfo(module.module, function_handle)

    @staticmethod
    fn _dump_rep(
        dump_ptx: Variant[Path, Bool] = False,
        dump_llvm: Variant[Path, Bool] = False,
    ) raises:
        fn dump_q(val: Variant[Path, Bool]) -> Bool:
            if val.isa[Bool]():
                return val[Bool]
            return val[Path] != Path("")

        if dump_q(dump_ptx):
            alias ptx = Self._impl.asm
            if dump_ptx.isa[Path]():
                with open(dump_ptx[Path], "w") as f:
                    f.write(StringRef(ptx))
            else:
                print(ptx)

        if dump_q(dump_llvm):
            alias llvm = _compile_code[func, emission_kind="llvm"]().asm

            if dump_llvm.isa[Path]():
                with open(dump_llvm[Path], "w") as f:
                    f.write(StringRef(llvm))
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
        shared_mem_bytes: Int = 0,
        stream: Optional[Stream] = None,
    ) raises:
        self._call_pack(
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=shared_mem_bytes,
            stream=stream,
        )

    @always_inline
    @parameter
    fn _call_pack[
        *Ts: AnyType
    ](
        self,
        args: VariadicPack[_, _, AnyType, Ts],
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int = 0,
        stream: Optional[Stream] = None,
    ) raises:
        alias num_args = len(VariadicList(Ts))
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        var args_stack = stack_allocation[
            num_captures + num_args, UnsafePointer[NoneType]
        ]()

        populate(args_stack.bitcast[NoneType]())

        @parameter
        for i in range(num_args):
            alias arg_offset = num_captures + i
            var elt_addr = UnsafePointer.address_of(args[i])
            args_stack[arg_offset] = elt_addr.bitcast[NoneType]()

        self.__call_impl(
            args_stack,
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=shared_mem_bytes,
            stream=stream,
        )

    @always_inline
    fn __call_impl(
        self,
        args: UnsafePointer[UnsafePointer[NoneType]],
        *,
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int = 0,
        stream: Optional[Stream] = None,
    ) raises:
        var stream_value = stream.value().stream if stream else Stream()
        var cuLaunchKernel = self.cuda_dll.value().cuLaunchKernel if self.cuda_dll else cuLaunchKernel.load()
        _check_error(
            cuLaunchKernel(
                self.info.func_handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(shared_mem_bytes),
                stream_value.stream,
                args,
                DTypePointer[DType.invalid](),
            ),
            msg=Self._impl.function_name,
            location=__call_location(),
        )
        # if we created the stream, we should sync
        if not stream:
            stream_value.synchronize()

    @staticmethod
    fn init_fn[
        func_type: AnyTrivialRegType, func: func_type
    ](payload_ptr: UnsafePointer[_CachedFunctionPayload]) -> UnsafePointer[
        _CachedFunctionInfo
    ]:
        var res = UnsafePointer[_CachedFunctionInfo].alloc(1)
        try:
            var payload = payload_ptr[]

            alias _impl = _compile_code[
                func, emission_kind="asm", is_failable=_is_failable
            ]()
            alias fn_name = _impl.function_name

            var module = Module(
                _impl.asm,
                debug=payload.debug,
                verbose=payload.verbose or payload.debug,
                max_registers=Optional[Int]() if payload.max_registers
                <= 0 else Optional[Int](int(payload.max_registers)),
                threads_per_block=Optional[Int]() if payload.threads_per_block
                <= 0 else Optional[Int](int(payload.threads_per_block)),
            )
            var func_handle = module.load(fn_name)
            # FIXME: Figure out a way to get the following through self.cuda_dll
            var cuFuncSetCacheConfig = cuFuncSetCacheConfig.load()
            var cuFuncSetAttribute = cuFuncSetAttribute.load()

            if payload.cache_config != -1:
                _check_error(
                    cuFuncSetCacheConfig(func_handle, payload.cache_config)
                )
            if payload.func_attribute.code != -1:
                _check_error(
                    cuFuncSetAttribute(
                        func_handle,
                        payload.func_attribute.code,
                        payload.func_attribute.value,
                    )
                )
            res.init_pointee_move(
                _CachedFunctionInfo(module._steal_handle(), func_handle)
            )
        except e:
            res.init_pointee_move(_CachedFunctionInfo(e))
        return res.address

    @staticmethod
    fn _init_fn[
        func_type: AnyTrivialRegType, func: func_type
    ](payload_ptr: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
        return Self.init_fn[func_type, func](
            payload_ptr.bitcast[_CachedFunctionPayload]()
        ).bitcast[NoneType]()

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
        debug: Bool = False,
        verbose: Bool = False,
        max_registers: Int = -1,
        threads_per_block: Int = -1,
        cache_config: Int32 = -1,
        func_attribute: FuncAttribute = FuncAttribute.NULL,
        cuda_function_cache: UnsafePointer[FunctionCache] = UnsafePointer[
            FunctionCache
        ](),
    ) -> _CachedFunctionInfo:
        alias fn_name = _get_nvptx_fn_name[func]()

        var payload = _CachedFunctionPayload(
            debug=debug,
            verbose=verbose,
            max_registers=max_registers,
            threads_per_block=threads_per_block,
            cache_config=cache_config,
            func_attribute=func_attribute,
        )

        if cuda_function_cache:
            var info_ptr = cuda_function_cache[].get_or_create_entry[
                fn_name,
                Self.init_fn[func_type, func],
            ](UnsafePointer.address_of(payload))

            _ = payload

            return info_ptr[]
        else:
            var info_ptr = _get_global[
                fn_name,
                Self._init_fn[func_type, func],
                Self._destroy_fn,
            ](UnsafePointer.address_of(payload).bitcast[NoneType]())

            _ = payload

            return UnsafePointer(info_ptr.address).bitcast[
                _CachedFunctionInfo
            ]()[]
