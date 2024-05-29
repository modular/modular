# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from memory.unsafe_pointer import *
from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from utils import StringRef


@value
@register_passable("trivial")
struct CString:
    """Represents `const char*` in C. Useful for binding with C APIs."""

    var ptr: DTypePointer[DType.int8]

    fn __init__(inout self, ptr: UnsafePointer[UInt8]):
        """
        Construct a `CString` from a string data pointer.

        Args:
            ptr: The string data pointer to wrap.
        """

        # TODO: Remove cast once UInt8 string types transition is complete.
        self.ptr = DTypePointer(ptr.bitcast[Int8]())

    fn get_as_string_ref(self) -> StringRef:
        """
        Get the `CString` as `StringRef`. Lifetime is tied to C API.
        For owning version use `__str__()`.
        """
        return StringRef(self.ptr)

    fn __str__(self) -> String:
        """
        Get `CString` as a owning `String`.
        """
        return String(StringRef(self.ptr))


@always_inline("nodebug")
fn exchange[T: AnyTrivialRegType](inout old_var: T, owned new_value: T) -> T:
    """
    Assign `new_value` to `old_var` and returns the value previously
    contained in `old_var`.
    """
    var old = old_var
    old_var = new_value
    return old


# ======================================================================#
#                                                                       #
# Utility structs and functions to interact with dylibs.                #
#                                                                       #
# ======================================================================#


fn handle_from_config(name: String, param: String) -> DLHandle:
    var lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", DTypePointer[DType.uint8]
    ](param._strref_dangerous())
    param._strref_keepalive()

    if not lib_path_str_ptr:
        abort("cannot get " + name + " library location from modular.cfg")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getMAXConfigValue` so that it can be destroyed by Mojo.
    var lib_path = String._from_bytes(lib_path_str_ptr)

    if not Path(lib_path).exists():
        abort("cannot load " + name + " library from " + lib_path)

    return DLHandle(lib_path)


@value
@register_passable("trivial")
struct SingleArgCallable[ResultTy: AnyTrivialRegType, ArgTy: AnyTrivialRegType]:
    var func: fn (ArgTy) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg: ArgTy) -> ResultTy:
        return self.func(arg)


@value
@register_passable("trivial")
struct TwoArgCallable[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg1: Arg1Ty, arg2: Arg2Ty) -> ResultTy:
        return self.func(arg1, arg2)


@value
@register_passable("trivial")
struct ThreeArgCallable[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty, Arg3Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg1: Arg1Ty, arg2: Arg2Ty, arg3: Arg3Ty) -> ResultTy:
        return self.func(arg1, arg2, arg3)


@value
@register_passable("trivial")
struct FourArgCallable[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
    Arg4Ty: AnyTrivialRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(
        self,
        arg1: Arg1Ty,
        arg2: Arg2Ty,
        arg3: Arg3Ty,
        arg4: Arg4Ty,
    ) -> ResultTy:
        return self.func(arg1, arg2, arg3, arg4)


@value
@register_passable("trivial")
struct FiveArgCallable[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
    Arg4Ty: AnyTrivialRegType,
    Arg5Ty: AnyTrivialRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty, Arg5Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(
        self,
        arg1: Arg1Ty,
        arg2: Arg2Ty,
        arg3: Arg3Ty,
        arg4: Arg4Ty,
        arg5: Arg5Ty,
    ) -> ResultTy:
        return self.func(arg1, arg2, arg3, arg4, arg5)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType
](lib: DLHandle, name: StringRef) -> ResultTy:
    """Call function `name` in dylib with one result and no arguments."""
    return lib.get_function[SingleArgCallable[ResultTy, NoneType]](name)(None)


@always_inline("nodebug")
fn call_dylib_func[
    ArgTy: AnyTrivialRegType
](lib: DLHandle, name: StringRef, arg: ArgTy) -> None:
    """Call function `name` in dylib with no result and one argument."""
    lib.get_function[SingleArgCallable[NoneType, ArgTy]](name)(arg)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType, ArgTy: AnyTrivialRegType
](lib: DLHandle, name: StringRef, arg: ArgTy) -> ResultTy:
    """Call function `name` in dylib with one result and one argument."""
    return lib.get_function[SingleArgCallable[ResultTy, ArgTy]](name)(arg)


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyTrivialRegType, Arg2Ty: AnyTrivialRegType
](lib: DLHandle, name: StringRef, arg1: Arg1Ty, arg2: Arg2Ty):
    """Call function `name` in dylib with no result and two arguments."""
    lib.get_function[TwoArgCallable[NoneType, Arg1Ty, Arg2Ty]](name)(arg1, arg2)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
](lib: DLHandle, name: StringRef, arg1: Arg1Ty, arg2: Arg2Ty) -> ResultTy:
    """Call function `name` in dylib with one result and two arguments."""
    return lib.get_function[TwoArgCallable[ResultTy, Arg1Ty, Arg2Ty]](name)(
        arg1, arg2
    )


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
](lib: DLHandle, name: StringRef, arg1: Arg1Ty, arg2: Arg2Ty, arg3: Arg3Ty,):
    """Call function `name` in dylib with no result and three arguments."""
    lib.get_function[ThreeArgCallable[NoneType, Arg1Ty, Arg2Ty, Arg3Ty]](name)(
        arg1, arg2, arg3
    )


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
](
    lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
) -> ResultTy:
    """Call function `name` in dylib with one result and three arguments."""
    return lib.get_function[ThreeArgCallable[ResultTy, Arg1Ty, Arg2Ty, Arg3Ty]](
        name
    )(arg1, arg2, arg3)


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
    Arg4Ty: AnyTrivialRegType,
](
    lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
    arg4: Arg4Ty,
):
    """Call function `name` in dylib with no result and four arguments."""
    return lib.get_function[
        FourArgCallable[NoneType, Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty]
    ](name)(arg1, arg2, arg3, arg4)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
    Arg4Ty: AnyTrivialRegType,
](
    lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
    arg4: Arg4Ty,
) -> ResultTy:
    """Call function `name` in dylib with one result and four arguments."""
    return lib.get_function[
        FourArgCallable[ResultTy, Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty]
    ](name)(arg1, arg2, arg3, arg4)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyTrivialRegType,
    Arg1Ty: AnyTrivialRegType,
    Arg2Ty: AnyTrivialRegType,
    Arg3Ty: AnyTrivialRegType,
    Arg4Ty: AnyTrivialRegType,
    Arg5Ty: AnyTrivialRegType,
](
    lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
    arg4: Arg4Ty,
    arg5: Arg5Ty,
) -> ResultTy:
    """Call function `name` in dylib with one result and five arguments."""
    return lib.get_function[
        FiveArgCallable[ResultTy, Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty, Arg5Ty]
    ](name)(arg1, arg2, arg3, arg4, arg5)


struct OwningVector[T: Movable](Sized):
    var ptr: UnsafePointer[T]
    var size: Int

    alias initial_capacity = 5
    var capacity: Int

    fn __init__(inout self):
        var ptr = UnsafePointer[T].alloc(Self.initial_capacity)
        self.ptr = ptr
        self.size = 0
        self.capacity = Self.initial_capacity

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.size = existing.size
        self.capacity = existing.capacity

    fn emplace_back(inout self, owned value: T):
        if self.size < self.capacity:
            initialize_pointee_move(self.ptr + self.size, value^)
            self.size += 1
            return

        self.capacity = self.capacity * 2
        var new_ptr = UnsafePointer[T].alloc(self.capacity)
        for i in range(self.size):
            move_pointee(src=self.ptr + i, dst=new_ptr + i)
        self.ptr.free()
        self.ptr = new_ptr
        self.emplace_back(value^)

    fn get(self, idx: Int) raises -> UnsafePointer[T]:
        if idx >= self.size:
            raise "requested index(" + str(
                idx
            ) + ") exceeds size of vector(" + str(self.size) + ")"
        return self.ptr + idx

    fn __len__(self) -> Int:
        return self.size

    fn __del__(owned self):
        for i in range(self.size):
            destroy_pointee(self.ptr + i)
        self.ptr.free()


fn get_lib_path_from_cfg(
    name: StringRef, err_name: StringLiteral
) raises -> String:
    var lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", DTypePointer[DType.uint8]
    ](name)

    if not lib_path_str_ptr:
        raise "cannot get the location of " + str(
            name
        ) + " library from modular.cfg"

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getMAXConfigValue` so that it can be destroyed by Mojo.
    var lib_path = String._from_bytes(lib_path_str_ptr)

    if not Path(lib_path).exists():
        raise "AI engine library not found at " + lib_path
    return lib_path
