# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Foreign function interface (FFI) for calling C code and loading libraries.

This module provides tools for interfacing Mojo with C libraries and other
foreign code. It includes:

- **C type aliases**: `c_int`, `c_char`, `c_long`, `c_size_t`, etc. for
  portable type definitions that match C's type sizes on each platform.
- **Dynamic library loading**: `OwnedDLHandle` for loading shared libraries
  at runtime and calling their functions.
- **External function calls**: `external_call()` for calling C functions
  by name with compile-time resolution.
- **String interop**: `CStringSlice` for working with null-terminated C strings.

Example:

```mojo
from ffi import c_int, external_call

fn get_random() -> c_int:
    return external_call["rand", c_int]()
```

For loading dynamic libraries:

```mojo
from ffi import OwnedDLHandle

fn main() raises:
    var lib = OwnedDLHandle("libm.so")
    var sqrt = lib.get_function[fn(Float64) -> Float64]("sqrt")
    print(sqrt(4.0))  # 2.0
```
"""

from collections.string.string_slice import _get_kgen_string, get_static_string
from os import PathLike, abort
from pathlib import Path
from sys._libc import dlclose, dlerror, dlopen, dlsym
from sys._libc_errno import ErrNo, get_errno, set_errno

from memory import OwnedPointer

from sys.info import CompilationTarget, is_32bit, is_64bit
from sys.intrinsics import _mlirtype_is_eq
from .cstring import CStringSlice
from .unsafe_union import UnsafeUnion

# ===-----------------------------------------------------------------------===#
# Primitive C type aliases
# ===-----------------------------------------------------------------------===#

comptime c_char = Int8
"""C `char` type."""

comptime c_uchar = UInt8
"""C `unsigned char` type."""

comptime c_int = Int32
"""C `int` type.

The C `int` type is typically a signed 32-bit integer on commonly used targets
today.
"""

comptime c_uint = UInt32
"""C `unsigned int` type."""

comptime c_short = Int16
"""C `short` type."""

comptime c_ushort = UInt16
"""C `unsigned short` type."""

comptime c_long = Scalar[_c_long_dtype()]
"""C `long` type.

The C `long` type is typically a signed 64-bit integer on macOS and Linux, and a
32-bit integer on Windows."""

comptime c_long_long = Scalar[_c_long_long_dtype()]
"""C `long long` type.

The C `long long` type is typically a signed 64-bit integer on commonly used
targets today."""

comptime c_ulong = Scalar[_c_long_dtype[unsigned=True]()]
"""C `unsigned long` type.

The C `unsigned long` type is typically a 64-bit integer on commonly used
targets today."""

comptime c_ulong_long = Scalar[_c_long_long_dtype[unsigned=True]()]
"""C `unsigned long long` type.

The C `unsigned long long` type is typically a 64-bit integer on commonly used
targets today."""


comptime c_size_t = UInt
"""C `size_t` type."""

comptime c_ssize_t = Int
"""C `ssize_t` type."""

comptime c_float = Float32
"""C `float` type."""

comptime c_double = Float64
"""C `double` type."""

comptime c_pid_t = Int
"""C `pid_t` type."""

comptime MAX_PATH = _get_max_path()
"""Maximum path length for the current platform."""


fn _get_max_path() -> Int:
    comptime if CompilationTarget.is_linux():
        return 4096
    elif CompilationTarget.is_macos():
        return 1024
    # Default POSIX limit
    else:
        return 256


fn _c_long_dtype[unsigned: Bool = False]() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    comptime if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        # LP64: long is 64-bit on 64-bit systems (e.g. x86_64 or aarch64)
        return DType.uint64 if unsigned else DType.int64
    elif is_32bit():
        # ILP32: long is 32-bit on 32-bit systems (e.g. x86 or RISC-V 32bit)
        return DType.uint32 if unsigned else DType.int32
    else:
        comptime assert False, "size of C `long` is unknown on this target"
        abort()


fn _c_long_long_dtype[unsigned: Bool = False]() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
    # `long long` is 64 bits on all common platforms (LP64, LLP64, ILP32).

    comptime if is_64bit() or is_32bit():
        return DType.uint64 if unsigned else DType.int64
    else:
        comptime assert False, "size of C `long long` is unknown on this target"
        abort()


# ===-----------------------------------------------------------------------===#
# Dynamic Library Loading
# ===-----------------------------------------------------------------------===#


struct RTLD:
    """Enumeration of the RTLD flags used during dynamic library loading."""

    comptime LAZY = 1
    """Load library lazily (defer function resolution until needed).
    """
    comptime NOW = 2
    """Load library immediately (resolve all symbols on load)."""
    comptime LOCAL = 4
    """Make symbols not available for symbol resolution of subsequently loaded
    libraries."""
    comptime GLOBAL = 256 if CompilationTarget.is_linux() else 8
    """Make symbols available for symbol resolution of subsequently loaded
    libraries."""
    comptime NODELETE = 4096 if CompilationTarget.is_linux() else 128
    """Do not delete the library when the process exits."""


comptime DEFAULT_RTLD = RTLD.NOW | RTLD.GLOBAL
"""Default runtime linker flags for dynamic library loading."""


struct OwnedDLHandle(Movable):
    """Represents an owned handle to a dynamically linked library with RAII
    semantics.

    `OwnedDLHandle` owns the library handle and automatically calls `dlclose()`
    when the object is destroyed. This prevents resource leaks and double-free
    bugs.

    Example usage:
    ```mojo
    from ffi import OwnedDLHandle

    fn main() raises:
        var lib = OwnedDLHandle("libm.so")
        var sqrt = lib.get_function[fn(Float64) -> Float64]("sqrt")
        print(sqrt(4.0))  # Prints: 2.0
        # Library automatically closed when lib goes out of scope
    ```
    """

    var _handle: _DLHandle

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self, flags: Int = DEFAULT_RTLD) raises:
        """Initialize an owned handle to all global symbols in the current
        process.

        Args:
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(nullptr, flags)` fails.
        """
        self._handle = _DLHandle(flags)

    fn __init__[
        PathLike: os.PathLike, //
    ](out self, path: PathLike, flags: Int = DEFAULT_RTLD) raises:
        """Initialize an OwnedDLHandle by loading the dynamic library at the
        given path.

        Parameters:
            PathLike: The type conforming to the `os.PathLike` trait.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(path, flags)` fails.
        """
        self._handle = _DLHandle(path, flags)

    @doc_private
    @always_inline
    fn __init__(out self, *, unsafe_uninitialized: Bool):
        self._handle = _DLHandle({})

    fn __del__(deinit self):
        """Unload the associated dynamic library.

        This automatically calls `dlclose()` on the underlying library handle.
        """
        self._handle.close()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn borrow(self) -> _DLHandle:
        """Returns a non-owning reference to this handle.

        The returned `_DLHandle` does not own the library and should not be
        used after this `OwnedDLHandle` is destroyed.

        Returns:
            A non-owning reference to the library handle.
        """
        return self._handle

    fn __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
            `True` if the handle is not null and `False` otherwise.
        """
        return self._handle.__bool__()

    fn check_symbol(self, var name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        return self._handle.check_symbol(name)

    fn get_function[
        result_type: __TypeOfAllTypes
    ](self, var name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        return self._handle.get_function[result_type](name)

    @always_inline
    fn _get_function[
        func_name: StaticString, result_type: __TypeOfAllTypes
    ](self) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            func_name: The name of the function to get the handle for.
            result_type: The type of the function pointer to return.

        Returns:
            A handle to the function.
        """
        return self._handle._get_function[func_name, result_type]()

    @always_inline
    fn _get_function[
        result_type: __TypeOfAllTypes
    ](self, *, cstr_name: UnsafePointer[mut=False, c_char]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            cstr_name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        return self._handle._get_function[result_type](cstr_name=cstr_name)

    fn get_symbol[
        result_type: AnyType,
    ](self, name: StringSlice) -> UnsafePointer[result_type, MutAnyOrigin]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        return self._handle.get_symbol[result_type](name)

    fn get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: UnsafePointer[mut=False, Int8]) -> UnsafePointer[
        result_type, MutAnyOrigin
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        return self._handle.get_symbol[result_type](cstr_name=cstr_name)

    @always_inline
    fn call[
        name: StaticString,
        return_type: __TypeOfAllTypes = NoneType,
        *T: AnyType,
    ](self, *args: *T) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.
            T: The types of `args`.

        Args:
            args: The arguments.

        Returns:
            The result.
        """
        return self._handle.call[name, return_type](args)

    fn call[
        name: StaticString, return_type: __TypeOfAllTypes = NoneType
    ](self, args: VariadicPack[element_trait=AnyType]) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.

        Args:
            args: The arguments.

        Returns:
            The result.
        """
        return self._handle.call[name, return_type](args)


@fieldwise_init
struct _DLHandle(Boolable, Copyable, TrivialRegisterPassable):
    """Represents a non-owning reference to a dynamically linked library.

    `_DLHandle` is a lightweight, trivially copyable reference to a dynamic
    library. It does not own the library handle and multiple copies can safely
    reference the same library.

    For automatic resource management with RAII semantics, use `OwnedDLHandle`
    instead, which automatically calls `dlclose()` when destroyed.

    Notes:
        If you manually call `close()` on a `_DLHandle`, be careful not to use
        any copies of that handle afterward, as they will reference a closed
        library. For safer usage, prefer `OwnedDLHandle`.
    """

    var handle: OpaquePointer[MutExternalOrigin]
    """The handle to the dynamic library."""

    @always_inline
    fn __init__(out self, flags: Int = DEFAULT_RTLD) raises:
        """Initialize a dynamic library handle to all global symbols in the
        current process.

        Args:
            flags: The flags to load the dynamic library.

        Notes:
            On POSIX-compatible operating systems, this performs
            `dlopen(nullptr, flags)`.

        Raises:
            If `dlopen(nullptr, flags)` fails.
        """
        self = Self._dlopen(UnsafePointer[c_char, MutExternalOrigin](), flags)

    fn __init__[
        PathLike: os.PathLike, //
    ](out self, path: PathLike, flags: Int = DEFAULT_RTLD) raises:
        """Initialize a DLHandle object by loading the dynamic library at the
        given path.

        Parameters:
            PathLike: The type conforming to the `os.PathLike` trait.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(path, flags)` fails.
        """

        var fspath = path.__fspath__()
        self = Self._dlopen(fspath.as_c_string_slice().unsafe_ptr(), flags)

    @staticmethod
    fn _dlopen(
        file: UnsafePointer[mut=False, c_char], flags: Int
    ) raises -> _DLHandle:
        var handle = dlopen(file, Int32(flags))
        if not handle:
            var error_message = dlerror()
            raise Error(
                "dlopen failed: ",
                StringSlice(unsafe_from_utf8_ptr=error_message),
            )
        return _DLHandle(handle)

    fn check_symbol(self, var name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        var opaque_function_ptr = dlsym(
            self.handle,
            name.as_c_string_slice().unsafe_ptr(),
        )

        return Bool(opaque_function_ptr)

    fn close(mut self):
        """Unload the associated dynamic library.

        Warning:
            Since `DLHandle` is trivially copyable, multiple copies of this
            handle may exist. After calling `close()`, all copies will reference
            an invalid library handle. For safer resource management, prefer
            using `OwnedDLHandle` which automatically manages the library
            lifetime.
        """
        _ = dlclose(self.handle)
        self.handle = {}

    fn __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
          True if the DLHandle is not null and False otherwise.
        """
        return self.handle.__bool__()

    fn get_function[
        result_type: __TypeOfAllTypes
    ](self, var name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """

        return self._get_function[result_type](
            cstr_name=name.as_c_string_slice().unsafe_ptr()
        )

    @always_inline
    fn _get_function[
        func_name: StaticString, result_type: __TypeOfAllTypes
    ](self) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            func_name:The name of the function to get the handle for.
            result_type: The type of the function pointer to return.

        Returns:
            A handle to the function.
        """
        # Force unique the func_name so we know that it is nul-terminated.
        comptime func_name_literal = get_static_string[func_name]()
        return self._get_function[result_type](
            cstr_name=func_name_literal.unsafe_ptr().bitcast[c_char](),
        )

    @always_inline
    fn _get_function[
        result_type: __TypeOfAllTypes
    ](self, *, cstr_name: UnsafePointer[mut=False, c_char]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            cstr_name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        var opaque_function_ptr = self.get_symbol[NoneType](cstr_name=cstr_name)

        return UnsafePointer(to=opaque_function_ptr).bitcast[result_type]()[]

    fn get_symbol[
        result_type: AnyType,
    ](self, name: StringSlice) -> UnsafePointer[result_type, MutAnyOrigin]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        name_copy = String(name)
        return self.get_symbol[result_type](
            cstr_name=name_copy.as_c_string_slice().unsafe_ptr()
        )

    fn get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: UnsafePointer[mut=False, Int8]) -> UnsafePointer[
        result_type, MutAnyOrigin
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        debug_assert(
            Bool(self.handle),
            "Dylib handle is null when loading symbol: ",
            StringSlice(unsafe_from_utf8_ptr=cstr_name),
        )

        # To check for `dlsym()` results that are _validly_ NULL, we do the
        # dance described in https://man7.org/linux/man-pages/man3/dlsym.3.html:
        #
        # > In unusual cases (see NOTES) the value of the symbol could
        # > actually be NULL.  Therefore, a NULL return from dlsym() need not
        # > indicate an error.  The correct way to distinguish an error from
        # > a symbol whose value is NULL is to call dlerror(3) to clear any
        # > old error conditions, then call dlsym(), and then call dlerror(3)
        # > again, saving its return value into a variable, and check whether
        # > this saved value is not NULL.

        var res = dlsym[result_type](self.handle, cstr_name)

        if not res:
            # Clear any potential unrelated error that pre-dates the `dlsym`
            # call above.
            _ = dlerror()

            # Redo the `dlsym` call
            res = dlsym[result_type](self.handle, cstr_name)

            debug_assert(
                not res,
                (
                    "dlsym unexpectedly returned non-NULL result when loading"
                    " symbol: "
                ),
                StringSlice(unsafe_from_utf8_ptr=cstr_name),
            )

            # Check if an error occurred during the 2nd `dlsym` call.
            var err = dlerror()
            if err:
                abort(
                    String("dlsym failed: ", String(unsafe_from_utf8_ptr=err))
                )

        return res

    @always_inline
    fn call[
        name: StaticString,
        return_type: __TypeOfAllTypes = NoneType,
        *T: AnyType,
    ](self, *args: *T) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.
            T: The types of `args`.

        Args:
            args: The arguments.

        Returns:
            The result.
        """
        return self.call[name, return_type](args)

    fn call[
        name: StaticString, return_type: __TypeOfAllTypes = NoneType
    ](self, args: VariadicPack[element_trait=AnyType]) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.

        Args:
            args: The arguments.

        Returns:
            The result.
        """

        @parameter
        fn _check_symbol() -> Bool:
            return self.check_symbol(String(name))

        debug_assert[_check_symbol]("symbol not found: ", name)
        var v = args.get_loaded_kgen_pack()
        return self.get_function[fn(type_of(v)) -> return_type](String(name))(v)


@always_inline
fn _get_dylib_function[
    dylib_global: _Global[StorageType=OwnedDLHandle, ...],
    func_name: StaticString,
    result_type: __TypeOfAllTypes,
]() raises -> result_type:
    var func_cache_name = String(dylib_global.name, "/", func_name)
    var func_ptr = _get_global_or_null(func_cache_name)
    if func_ptr:
        var result = UnsafePointer(to=func_ptr).bitcast[result_type]()[]
        _ = func_ptr
        return result

    var dylib = dylib_global.get_or_create_ptr()[].borrow()
    var new_func = dylib._get_function[func_name, result_type]()

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(func_cache_name),
        UnsafePointer(to=new_func).bitcast[
            OpaquePointer[MutExternalOrigin]
        ]()[],
    )

    return new_func


fn _try_find_dylib[
    name: StaticString = ""
](paths: List[Path]) raises -> OwnedDLHandle:
    """Try to load a dynamically linked library given a list of possible paths.

    Parameters:
        name: Optional name for the library to be used in error messages.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.

    Raises:
        If the library could not be loaded from any of the provided paths.
    """
    comptime dylib_name = name if name != "" else "dynamic library"
    for path in paths:
        # If we are given a library name like libfoo.so, pass it directly to
        # dlopen(), which will invoke the system linker to find the library.
        # We can't check the existence of the path ahead of time, we have to
        # call the function and check for an error.
        try:
            return OwnedDLHandle(String(path))
        except:
            # If the call to DLOpen fails, we should just try the next path
            # in the list. It's only a fatal error if the library cannot be
            # loaded from any of the paths provided.
            pass

    raise Error("Failed to load ", dylib_name, " from ", " or ".join(paths))


fn _try_find_dylib[
    name: StaticString = ""
](*paths: Path) raises -> OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    var paths_list = List[Path](capacity=len(paths))
    for path in paths:
        paths_list.append(path)
    return _try_find_dylib[name](paths_list)


fn _find_dylib[
    name: StaticString = "", abort_on_failure: Bool = True
](paths: List[Path]) -> OwnedDLHandle:
    """Load a dynamically linked library given a list of possible paths or names.

    If the library is not found, the function will abort.

    Parameters:
        name: Optional name for the library to be used in error messages.
        abort_on_failure: If set, then the function will abort the program if
           the library is not found. Otherwise, we return a null OwnedDLHandle
           on failure.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.
    """
    try:
        return _try_find_dylib[name](paths)
    except e:
        comptime if abort_on_failure:
            abort(String(e))
        else:
            return OwnedDLHandle(unsafe_uninitialized=True)


fn _find_dylib[
    msg: fn() -> String, abort_on_failure: Bool = True
](paths: List[Path]) -> OwnedDLHandle:
    """Load a dynamically linked library given a list of possible paths or names.

    If the library is not found, the function will abort.

    Parameters:
        msg: A function that produces the error message to use if the
             library cannot be found.
        abort_on_failure: If set, then the function will abort the program if
           the library is not found. Otherwise, we return a null OwnedDLHandle
           on failure.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.
    """
    try:
        return _try_find_dylib(paths)
    except e:
        comptime if abort_on_failure:
            abort[prefix="ERROR:"](msg())
        else:
            return OwnedDLHandle(unsafe_uninitialized=True)


fn _find_dylib[name: StaticString = ""](*paths: Path) -> OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    var paths_list = List[Path]()
    for path in paths:
        paths_list.append(path)
    return _find_dylib[name](paths_list)


# ===-----------------------------------------------------------------------===#
# Globals
# ===-----------------------------------------------------------------------===#


# NOTE: This is vending shared mutable pointers to the client without locking.
# This is not guaranteeing any sort of thread safety.
struct _Global[
    StorageType: Movable,
    //,
    name: StaticString,
    init_fn: fn() -> StorageType,
    on_error_msg: Optional[fn() -> Error] = None,
](Defaultable):
    comptime ResultType = UnsafePointer[Self.StorageType, MutExternalOrigin]

    fn __init__(out self):
        pass

    @staticmethod
    fn _init_wrapper() -> OpaquePointer[MutExternalOrigin]:
        # Heap allocate space to store this "global"
        # TODO:
        #   Any way to avoid the move, e.g. by calling this function
        #   with the ABI destination result pointer already set to `ptr`?
        var ptr = OwnedPointer(Self.init_fn())

        return ptr^.steal_data().bitcast[NoneType]()

    @staticmethod
    fn _deinit_wrapper(opaque_ptr: OpaquePointer[MutExternalOrigin]):
        # Deinitialize and deallocate the storage.
        _ = OwnedPointer(
            unsafe_from_raw_pointer=opaque_ptr.bitcast[Self.StorageType]()
        )

    @staticmethod
    fn get_or_create_ptr() raises -> Self.ResultType:
        var ptr = _get_global[
            Self.name, Self._init_wrapper, Self._deinit_wrapper
        ]()

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return ptr.bitcast[Self.StorageType]()

    # Currently known values for get_or_create_indexed_ptr. See
    # NUM_INDEXED_GLOBALS in CompilerRT.
    # 0: Python runtime context
    # 1: GPU comm P2P availability cache
    # 2: Intentionally unused (reserved for prototyping / future use)
    comptime _python_idx = 0
    comptime _gpu_comm_p2p_idx = 1
    comptime _unused = 2  # Intentionally unused (enabled for prototyping).

    # This accesses a well-known global with a fixed index rather than using a
    # name to unique the value.  The index table is above.
    @staticmethod
    fn get_or_create_indexed_ptr(idx: Int) raises -> Self.ResultType:
        var ptr = external_call[
            "KGEN_CompilerRT_GetOrCreateGlobalIndexed",
            OpaquePointer[MutExternalOrigin],
        ](
            idx,
            Self._init_wrapper,
            Self._deinit_wrapper,
        )

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return ptr.bitcast[Self.StorageType]()


@always_inline
fn _get_global[
    name: StaticString,
    init_fn: fn() -> OpaquePointer[MutExternalOrigin],
    destroy_fn: fn(OpaquePointer[MutExternalOrigin]) -> None,
]() -> OpaquePointer[MutExternalOrigin]:
    return external_call[
        "KGEN_CompilerRT_GetOrCreateGlobal", OpaquePointer[MutExternalOrigin]
    ](
        name,
        init_fn,
        destroy_fn,
    )


@always_inline
fn _get_global_or_null(name: StringSlice) -> OpaquePointer[MutExternalOrigin]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer[MutExternalOrigin]
    ](name.unsafe_ptr(), name.byte_length())


# ===-----------------------------------------------------------------------===#
# external_call
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn external_call[
    callee: StaticString,
    return_type: __TypeOfAllTypes,
    *types: AnyType,
](*args: *types) -> return_type:
    """Calls an external function.

    Args:
        args: The arguments to pass to the external function.

    Parameters:
        callee: The name of the external function.
        return_type: The return type.
        types: The argument types.

    Returns:
        The external call result.
    """
    return external_call[callee, return_type](args)


@always_inline("nodebug")
fn external_call[
    callee: StaticString,
    return_type: __TypeOfAllTypes,
](args: VariadicPack[element_trait=AnyType]) -> return_type:
    """Calls an external function.

    Parameters:
        callee: The name of the external function.
        return_type: The return type.

    Args:
        args: The arguments to pass to the external function.

    Returns:
        The external call result.
    """

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()
    comptime callee_kgen_string = _get_kgen_string[callee]()

    comptime if _mlirtype_is_eq[return_type, NoneType]():
        __mlir_op.`pop.external_call`[func=callee_kgen_string, _type=None](
            loaded_pack
        )
        return rebind[return_type](None)
    else:
        return __mlir_op.`pop.external_call`[
            func=callee_kgen_string,
            _type=return_type,
        ](loaded_pack)


# ===-----------------------------------------------------------------------===#
# _external_call_const
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _external_call_const[
    callee: StaticString,
    return_type: __TypeOfAllTypes,
    *types: AnyType,
](*args: *types) -> return_type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Args:
      args: The arguments to pass to the external function.

    Parameters:
      callee: The name of the external function.
      return_type: The return type.
      types: The argument types.

    Returns:
      The external call result.
    """

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()

    return __mlir_op.`pop.external_call`[
        func = _get_kgen_string[callee](),
        resAttrs = __mlir_attr.`[{llvm.noundef}]`,
        funcAttrs = __mlir_attr.`["willreturn"]`,
        memory = __mlir_attr[
            `#llvm.memory_effects<other = none, `,
            `argMem = none, `,
            `inaccessibleMem = none, `,
            `errnoMem = none, `,
            `targetMem0 = none, `,
            `targetMem1 = none>`,
        ],
        _type=return_type,
    ](loaded_pack)
