# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.string import StaticString, StringSlice
from os import abort
from pathlib import Path
from sys.ffi import DLHandle, c_char, external_call

from memory.unsafe_pointer import *


@value
@register_passable("trivial")
struct CString:
    """Represents `const char*` in C. Useful for binding with C APIs."""

    var ptr: UnsafePointer[c_char]

    @implicit
    fn __init__(out self, ptr: UnsafePointer[c_char]):
        """
        Construct a `CString` from a C string data pointer.

        Args:
            ptr: The string data pointer to wrap.
        """
        self.ptr = ptr.bitcast[c_char]()

    fn get_as_string_ref(self) -> StaticString:
        """
        Get the `CString` as `StringRef`. Origin is tied to C API.
        For owning version use `__str__()`.
        """
        return StaticString(unsafe_from_utf8_ptr=self.ptr)

    fn __str__(self) -> String:
        """
        Get `CString` as a owning `String`.
        """
        return String(unsafe_from_utf8_ptr=self.ptr)


@always_inline("nodebug")
fn exchange[T: AnyTrivialRegType](mut old_var: T, owned new_value: T) -> T:
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


@always_inline("nodebug")
fn call_dylib_func[
    ReturnType: AnyTrivialRegType = NoneType._mlir_type,
    *Args: AnyType,
](lib: DLHandle, name: StringSlice, *args: *Args) -> ReturnType:
    var args_pack = args.get_loaded_kgen_pack()

    var func_ptr = lib.get_function[fn (__type_of(args_pack)) -> ReturnType](
        String(name)
    )

    return func_ptr(args_pack)


struct OwningVector[T: Movable](Sized):
    var ptr: UnsafePointer[T]
    var size: Int

    alias initial_capacity = 5
    var capacity: Int

    fn __init__(out self):
        var ptr = UnsafePointer[T].alloc(Self.initial_capacity)
        self.ptr = ptr
        self.size = 0
        self.capacity = Self.initial_capacity

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = existing.ptr
        self.size = existing.size
        self.capacity = existing.capacity

    fn emplace_back(mut self, owned value: T):
        if self.size < self.capacity:
            (self.ptr + self.size).init_pointee_move(value^)
            self.size += 1
            return

        self.capacity = self.capacity * 2
        var new_ptr = UnsafePointer[T].alloc(self.capacity)
        for i in range(self.size):
            (self.ptr + i).move_pointee_into(dst=new_ptr + i)
        self.ptr.free()
        self.ptr = new_ptr
        self.emplace_back(value^)

    fn get(self, idx: Int) raises -> UnsafePointer[T]:
        if idx >= self.size:
            raise Error(
                "requested index(",
                idx,
                ") exceeds size of vector(",
                self.size,
                ")",
            )
        return self.ptr + idx

    fn __len__(self) -> Int:
        return self.size

    fn __del__(owned self):
        for i in range(self.size):
            (self.ptr + i).destroy_pointee()
        self.ptr.free()


fn get_lib_path_from_cfg(
    name: StringSlice, err_name: StaticString
) raises -> String:
    # TODO: Move KGEN_CompilerRT_getMAXConfigValue to a helper somewhere.
    var lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8]
    ](name.unsafe_ptr(), name.byte_length())

    if not lib_path_str_ptr:
        raise Error(
            "cannot get the location of ", name, " library from modular.cfg"
        )

    var lib_path = String(unsafe_from_utf8_ptr=lib_path_str_ptr)
    lib_path_str_ptr.free()

    if not Path(lib_path).exists():
        raise String(err_name) + " not found at " + lib_path
    return lib_path
