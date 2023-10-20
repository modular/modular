# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the function type."""

from memory import stack_allocation
from memory.unsafe import DTypePointer, Pointer

from utils._reflection import get_linkage_name

from ._compile import _cleanup_asm, _compile_nvptx_asm
from ._utils import _check_error, _get_dylib, _get_dylib_function
from .dim import Dim
from .module import ModuleHandle
from .stream import Stream, _StreamImpl

# ===----------------------------------------------------------------------===#
# FunctionHandle
# ===----------------------------------------------------------------------===#

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

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    @always_inline
    fn _call_impl[
        num_captures: Int, populate: _populate_fn_type
    ](self, grid_dim: Dim, block_dim: Dim, /, stream: Stream) raises:
        let args = stack_allocation[num_captures + 0, Pointer[NoneType]]()
        populate(args)

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int, populate: _populate_fn_type, T0: AnyType
    ](self, grid_dim: Dim, block_dim: Dim, arg0: T0, /, stream: Stream) raises:
        var _arg0 = arg0

        let args = stack_allocation[num_captures + 1, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int, populate: _populate_fn_type, T0: AnyType, T1: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1

        let args = stack_allocation[num_captures + 2, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2

        let args = stack_allocation[num_captures + 3, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3

        let args = stack_allocation[num_captures + 4, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4

        let args = stack_allocation[num_captures + 5, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5

        let args = stack_allocation[num_captures + 6, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )
        args.store(
            num_captures + 5, Pointer.address_of(_arg5).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6

        let args = stack_allocation[num_captures + 7, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )
        args.store(
            num_captures + 5, Pointer.address_of(_arg5).bitcast[NoneType]()
        )
        args.store(
            num_captures + 6, Pointer.address_of(_arg6).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7

        let args = stack_allocation[num_captures + 8, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )
        args.store(
            num_captures + 5, Pointer.address_of(_arg5).bitcast[NoneType]()
        )
        args.store(
            num_captures + 6, Pointer.address_of(_arg6).bitcast[NoneType]()
        )
        args.store(
            num_captures + 7, Pointer.address_of(_arg7).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7
        var _arg8 = arg8

        let args = stack_allocation[num_captures + 9, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )
        args.store(
            num_captures + 5, Pointer.address_of(_arg5).bitcast[NoneType]()
        )
        args.store(
            num_captures + 6, Pointer.address_of(_arg6).bitcast[NoneType]()
        )
        args.store(
            num_captures + 7, Pointer.address_of(_arg7).bitcast[NoneType]()
        )
        args.store(
            num_captures + 8, Pointer.address_of(_arg8).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

    @always_inline
    fn _call_impl[
        num_captures: Int,
        populate: _populate_fn_type,
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
        T9: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        arg9: T9,
        /,
        stream: Stream,
    ) raises:
        var _arg0 = arg0
        var _arg1 = arg1
        var _arg2 = arg2
        var _arg3 = arg3
        var _arg4 = arg4
        var _arg5 = arg5
        var _arg6 = arg6
        var _arg7 = arg7
        var _arg8 = arg8
        var _arg9 = arg9

        let args = stack_allocation[num_captures + 10, Pointer[NoneType]]()
        populate(args)
        args.store(
            num_captures + 0, Pointer.address_of(_arg0).bitcast[NoneType]()
        )
        args.store(
            num_captures + 1, Pointer.address_of(_arg1).bitcast[NoneType]()
        )
        args.store(
            num_captures + 2, Pointer.address_of(_arg2).bitcast[NoneType]()
        )
        args.store(
            num_captures + 3, Pointer.address_of(_arg3).bitcast[NoneType]()
        )
        args.store(
            num_captures + 4, Pointer.address_of(_arg4).bitcast[NoneType]()
        )
        args.store(
            num_captures + 5, Pointer.address_of(_arg5).bitcast[NoneType]()
        )
        args.store(
            num_captures + 6, Pointer.address_of(_arg6).bitcast[NoneType]()
        )
        args.store(
            num_captures + 7, Pointer.address_of(_arg7).bitcast[NoneType]()
        )
        args.store(
            num_captures + 8, Pointer.address_of(_arg8).bitcast[NoneType]()
        )
        args.store(
            num_captures + 9, Pointer.address_of(_arg9).bitcast[NoneType]()
        )

        self.__call_impl(grid_dim, block_dim, args, stream=stream)

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
# Function
# ===----------------------------------------------------------------------===#


@register_passable
struct Function[func_type: AnyType, func: func_type]:
    var mod_handle: ModuleHandle
    var func_handle: FunctionHandle

    alias _impl = _compile_nvptx_asm[func_type, func]()

    @always_inline
    fn __init__(
        debug: Bool = False,
        verbose: Bool = False,
        print_ptx: Bool = False,
    ) raises -> Self:
        alias name = get_linkage_name[func_type, func]()
        let ptx = _cleanup_asm(Self._impl.asm)
        if print_ptx:
            print(ptx)

        let mod_handle = ModuleHandle(
            ptx, debug=debug, verbose=verbose or debug
        )
        let func_handle = mod_handle.load(name)

        return Self {mod_handle: mod_handle ^, func_handle: func_handle ^}

    fn __del__(owned self) raises:
        pass

    fn __bool__(self) -> Bool:
        return self.func_handle.__bool__()

    @closure
    @always_inline
    fn __call__(
        self,
        grid_dim: Dim,
        block_dim: Dim,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, arg0, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType, T1: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, arg0, arg1, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, arg0, arg1, arg2, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, arg0, arg1, arg2, arg3, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType, T1: AnyType, T2: AnyType, T3: AnyType, T4: AnyType
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](grid_dim, block_dim, arg0, arg1, arg2, arg3, arg4, stream=stream)

    @closure
    @always_inline
    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            stream=stream,
        )

    @closure
    @always_inline
    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            stream=stream,
        )

    @closure
    @always_inline
    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            stream=stream,
        )

    @closure
    @always_inline
    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            stream=stream,
        )

    @closure
    @always_inline
    fn __call__[
        T0: AnyType,
        T1: AnyType,
        T2: AnyType,
        T3: AnyType,
        T4: AnyType,
        T5: AnyType,
        T6: AnyType,
        T7: AnyType,
        T8: AnyType,
        T9: AnyType,
    ](
        self,
        grid_dim: Dim,
        block_dim: Dim,
        arg0: T0,
        arg1: T1,
        arg2: T2,
        arg3: T3,
        arg4: T4,
        arg5: T5,
        arg6: T6,
        arg7: T7,
        arg8: T8,
        arg9: T9,
        /,
        stream: Stream = _StreamImpl(),
    ) raises:
        self.func_handle._call_impl[
            Self._impl.num_captures, Self._impl.populate
        ](
            grid_dim,
            block_dim,
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            stream=stream,
        )
