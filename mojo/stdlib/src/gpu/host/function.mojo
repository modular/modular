# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the function type."""

from memory import stack_allocation
from memory.unsafe import DTypePointer, Pointer

from sys.ffi import _get_global
from utils._optional import Optional

from pathlib import Path
from ._compile import _compile_nvptx, _get_nvptx_fn_name
from ._utils import _check_error, _get_dylib_function
from .dim import Dim
from .module import ModuleHandle, _ModuleImpl
from .stream import Stream, _StreamImpl

# ===----------------------------------------------------------------------===#
# FunctionHandle
# ===----------------------------------------------------------------------===#


fn _alignto(value: Int, align: Int) -> Int:
    return (value + align - 1) // align * align


struct AnyRegTuple[*Ts: AnyRegType]:
    alias _type = __mlir_type[
        `!kgen.pack<:variadic<`, AnyRegType, `> `, Ts, `>`
    ]
    var storage: Self._type

    fn __init__(inout self, value: Self._type):
        self.storage = value

    @staticmethod
    fn _offset[i: Int]() -> Int:
        constrained[i >= 0, "index must be positive"]()

        @parameter
        if i == 0:
            return 0
        else:
            return _alignto(
                Self._offset[i - 1]()
                + _alignto(sizeof[Ts[i - 1]](), alignof[Ts[i - 1]]()),
                alignof[Ts[i]](),
            )

    fn get[i: Int, T: AnyRegType](inout self) -> T:
        alias offset = Self._offset[i]()
        let addr = Pointer.address_of(self).bitcast[Int8]().offset(offset)
        return addr.bitcast[T]().load()


alias _populate_fn_type = fn (Pointer[Pointer[NoneType]]) capturing -> None


@value
@register_passable("trivial")
struct FunctionHandle:
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
    ](self, stream: Stream, grid_dim: Dim, block_dim: Dim, *args: *Ts) raises:
        let values = AnyRegTuple(args)
        alias types = VariadicList(Ts)

        var args_stack = stack_allocation[
            num_captures + len(VariadicList(Ts)), Pointer[NoneType]
        ]()
        populate(args_stack)

        @parameter
        @always_inline
        fn append[i: Int]():
            alias T = types[i]
            var _val = values.get[i, T]()
            args_stack.store(
                num_captures + i, Pointer.address_of(_val).bitcast[NoneType]()
            )

        unroll[len(types), append]()

        self.__call_impl(grid_dim, block_dim, args_stack, stream=stream)

    @always_inline
    fn __call_impl(
        self,
        grid_dim: Dim,
        block_dim: Dim,
        args: Pointer[Pointer[NoneType]],
        /,
        stream: Stream,
    ) raises:
        _check_error(
            _get_dylib_function[
                fn (
                    Self,
                    UInt32,  # GridDimZ
                    UInt32,  # GridDimY
                    UInt32,  # GridDimX
                    UInt32,  # BlockDimZ
                    UInt32,  # BlockDimY
                    UInt32,  # BlockDimX
                    UInt32,  # SharedMemSize
                    _StreamImpl,  # Stream
                    Pointer[Pointer[NoneType]],  # Args
                    DTypePointer[DType.invalid],  # Extra
                ) -> Result
            ]("cuLaunchKernel")(
                self.handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(0),
                stream.stream,
                args,
                DTypePointer[DType.invalid](),
            )
        )


# ===----------------------------------------------------------------------===#
# _PathOrBool
# ===----------------------------------------------------------------------===#


@value
struct _PathOrBool:
    """A type that contains a Path and a Bool type, of which only one can be
    set at a given time.
    """

    var print_val: Bool
    var path: Path

    @always_inline
    fn __init__(inout self):
        self.print_val = False
        self.path = Path("")

    @always_inline
    fn __init__(inout self, print_val: Bool):
        self.print_val = print_val
        self.path = Path("")

    @always_inline
    fn __init__(inout self, path: Path):
        self.print_val = False
        self.path = path

    @always_inline
    fn __init__(inout self, path: String):
        self.print_val = False
        self.path = Path(path)

    @always_inline
    fn __bool__(self) -> Bool:
        return self.print_val or self._is_path()

    @always_inline
    fn _is_path(self) -> Bool:
        return self.path != ""


# ===----------------------------------------------------------------------===#
# Cached Function Info
# ===----------------------------------------------------------------------===#


@value
@register_passable
struct _CachedFunctionInfo:
    var mod_handle: _ModuleImpl
    var func_handle: FunctionHandle

    fn __init__() -> Self:
        return Self {mod_handle: _ModuleImpl(), func_handle: FunctionHandle()}

    fn __init__(mod_handle: _ModuleImpl, func_handle: FunctionHandle) -> Self:
        return Self {mod_handle: mod_handle, func_handle: func_handle}

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

    fn __init__(
        debug: Bool,
        verbose: Bool,
        max_registers: Int32,
        threads_per_block: Int32,
    ) -> Self:
        return Self {
            debug: debug,
            verbose: verbose,
            max_registers: max_registers,
            threads_per_block: threads_per_block,
        }


@parameter
fn _init_fn[
    func_type: AnyRegType, func: func_type
](payload_ptr: Pointer[NoneType]) -> Pointer[NoneType]:
    try:
        let payload = payload_ptr.bitcast[_GlobalPayload]().load()

        alias _impl = _compile_nvptx[func_type, func, emission_kind="asm"]()
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
        let func_handle = mod_handle.load(fn_name)

        let res = Pointer[_CachedFunctionInfo].alloc(1)
        res.store(_CachedFunctionInfo(mod_handle._steal_handle(), func_handle))
        return res.bitcast[NoneType]()
    except e:
        print("Error loading the PTX code:", e)
        return Pointer[NoneType]()


@parameter
fn _destroy_fn(cached_value_ptr: Pointer[NoneType]):
    if not cached_value_ptr:
        return
    let cached_value = cached_value_ptr.bitcast[_CachedFunctionInfo]().load()
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
) -> _CachedFunctionInfo:
    alias fn_name = _get_nvptx_fn_name[func_type, func]()

    var payload = _GlobalPayload(
        debug, verbose, max_registers, threads_per_block
    )

    let res = (
        _get_global[
            fn_name,
            _init_fn[func_type, func],
            _destroy_fn,
        ](Pointer.address_of(payload).bitcast[NoneType]())
        .bitcast[_CachedFunctionInfo]()
        .load()
    )

    _ = payload

    return res


# ===----------------------------------------------------------------------===#
# Function
# ===----------------------------------------------------------------------===#


@register_passable
struct Function[func_type: AnyRegType, func: func_type]:
    var info: _CachedFunctionInfo

    alias _impl = _compile_nvptx[func_type, func, emission_kind="asm"]()

    @always_inline
    fn __init__(
        debug: Bool = False,
        verbose: Bool = False,
        dump_ptx: _PathOrBool = _PathOrBool(),
        dump_llvm: _PathOrBool = _PathOrBool(),
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
    ) raises -> Self:
        if dump_ptx:
            alias ptx = Self._impl.asm
            if dump_ptx._is_path():
                with open(dump_ptx.path, "w") as f:
                    f.write(ptx)
            else:
                print(ptx)
        if dump_llvm:
            alias llvm = _compile_nvptx[
                func_type, func, emission_kind="llvm"
            ]().asm
            if dump_llvm._is_path():
                with open(dump_llvm.path, "w") as f:
                    f.write(llvm)
            else:
                print(llvm)

        return Self {
            info: _get_global_cache_info[func_type, func](
                debug=debug,
                verbose=verbose,
                max_registers=max_registers.value() if max_registers else -1,
                threads_per_block=threads_per_block.value() if threads_per_block else -1,
            )
        }

    @always_inline
    fn __del__(owned self):
        pass

    @always_inline
    fn __bool__(self) -> Bool:
        return self.info.__bool__()

    # TODO: Move the stream argument at the end when When Mojo will support both vararg and keyword only arguments
    @closure
    @always_inline
    fn __call__[
        *Ts: AnyRegType
    ](self, stream: Stream, grid_dim: Dim, block_dim: Dim, *args: *Ts) raises:
        let values = AnyRegTuple(args)
        alias types = VariadicList(Ts)
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        var args_stack = stack_allocation[
            num_captures + len(VariadicList(Ts)), Pointer[NoneType]
        ]()
        populate(args_stack)

        @parameter
        @always_inline
        fn append[i: Int]():
            alias T = types[i]
            var _val = values.get[i, T]()
            args_stack.store(
                num_captures + i, Pointer.address_of(_val).bitcast[NoneType]()
            )

        unroll[len(types), append]()

        self.info.func_handle.__call_impl(
            grid_dim, block_dim, args_stack, stream=stream
        )

        # self.info.func_handle._call_impl[
        #     Self._impl.num_captures, Self._impl.populate
        # ](stream, grid_dim, block_dim, AnyRegTuple(args))

    # Convenience method omitting the stream parameter
    @closure
    @always_inline
    fn __call__[
        *Ts: AnyRegType
    ](self, grid_dim: Dim, block_dim: Dim, *args: *Ts) raises:
        let stream: Stream = _StreamImpl()
        let values = AnyRegTuple(args)
        alias types = VariadicList(Ts)
        alias num_captures = Self._impl.num_captures
        alias populate = Self._impl.populate

        var args_stack = stack_allocation[
            num_captures + len(VariadicList(Ts)), Pointer[NoneType]
        ]()
        populate(args_stack)

        @parameter
        @always_inline
        fn append[i: Int]():
            alias T = types[i]
            var _val = values.get[i, T]()
            args_stack.store(
                num_captures + i, Pointer.address_of(_val).bitcast[NoneType]()
            )

        unroll[len(types), append]()

        self.info.func_handle.__call_impl(
            grid_dim, block_dim, args_stack, stream=stream
        )

        # self.info.func_handle._call_impl[
        #     Self._impl.num_captures, Self._impl.populate
        # ](stream, grid_dim, block_dim, AnyRegTuple(args))
