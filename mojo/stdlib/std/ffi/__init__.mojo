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
from std.ffi import c_int, external_call

def get_random() -> c_int:
    return external_call["rand", c_int]()
```

For loading dynamic libraries:

```mojo
from std.ffi import OwnedDLHandle

def main() raises:
    var lib = OwnedDLHandle("libm.so")
    var sqrt = lib.get_function[def(Float64) thin abi("C") -> Float64](
        "sqrt"
    )
    print(sqrt(4.0))  # 2.0
```
"""

from std.collections.string.string_slice import (
    _get_kgen_string,
    get_static_string,
)
from std.os import PathLike, abort
from std.pathlib import Path
from std.sys._libc import dlclose, dlerror, dlopen, dlsym
from std.sys._libc_errno import ErrNo, get_errno, set_errno

from std.memory import OwnedPointer
from std.memory.alloc import free
from std.memory.unsafe_pointer import unsafe_cast

from std.sys.info import CompilationTarget, is_32bit, is_64bit, size_of
from std.sys.intrinsics import _type_is_eq
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


def _get_max_path() -> Int:
    comptime if CompilationTarget.is_linux():
        return 4096
    elif CompilationTarget.is_macos():
        return 1024
    # Default POSIX limit
    else:
        return 256


def _c_long_dtype[unsigned: Bool = False]() -> DType:
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


def _c_long_long_dtype[unsigned: Bool = False]() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
    # `long long` is 64 bits on all common platforms (LP64, LLP64, ILP32).

    comptime assert (
        is_64bit() or is_32bit()
    ), "size of C `long long` is unknown on this target"
    return DType.uint64 if unsigned else DType.int64


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
    comptime LOCAL = 0 if CompilationTarget.is_linux() else 4
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
    from std.ffi import OwnedDLHandle

    def main() raises:
        var lib = OwnedDLHandle("libm.so")
        var sqrt = lib.get_function[def(Float64) thin abi("C") -> Float64](
            "sqrt"
        )
        print(sqrt(4.0))  # Prints: 2.0
        # Library automatically closed when lib goes out of scope
    ```
    """

    var _handle: _DLHandle

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self, flags: Int = DEFAULT_RTLD) raises:
        """Initialize an owned handle to all global symbols in the current
        process.

        Args:
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(nullptr, flags)` fails.
        """
        self._handle = _DLHandle(flags)

    def __init__[
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

    @doc_hidden
    @always_inline
    def __init__(out self, *, unsafe_uninitialized: Bool):
        self._handle = _DLHandle({})

    def __del__(deinit self):
        """Unload the associated dynamic library.

        This automatically calls `dlclose()` on the underlying library handle.
        """
        self._handle.close()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def borrow(self) -> _DLHandle:
        """Returns a non-owning reference to this handle.

        The returned `_DLHandle` does not own the library and should not be
        used after this `OwnedDLHandle` is destroyed.

        Returns:
            A non-owning reference to the library handle.
        """
        return self._handle

    def __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
            `True` if the handle is not null and `False` otherwise.
        """
        return self._handle.__bool__()

    def check_symbol(self, var name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        return self._handle.check_symbol(name)

    def get_function[
        result_type: TrivialRegisterPassable
    ](self, var name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        `result_type` must be a C-ABI function type (using the `abi("C")` effect)
        to ensure correct argument and return-value passing. Using a plain Mojo
        function type (`fn(...) -> T`) produces silent ABI corruption for any
        struct argument or return value.

        Example:
        ```mojo
        from std.ffi import OwnedDLHandle

        var lib = OwnedDLHandle("libm.so")
        var sqrt = lib.get_function[def(Float64) thin abi("C") -> Float64]("sqrt")
        ```

        Parameters:
            result_type: The C-ABI function pointer type to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        return self._handle.get_function[result_type](name)

    @always_inline
    def _get_function[
        func_name: StaticString, result_type: TrivialRegisterPassable
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
    def _get_function[
        result_type: TrivialRegisterPassable
    ](self, *, cstr_name: UnsafePointer[mut=False, c_char, _]) -> result_type:
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

    def get_symbol[
        result_type: AnyType,
    ](self, name: StringSlice) -> Optional[
        UnsafePointer[result_type, MutAnyOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        return self._handle.get_symbol[result_type](name)

    def get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: UnsafePointer[mut=False, Int8, _]) -> Optional[
        UnsafePointer[result_type, MutAnyOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        return self._handle.get_symbol[result_type](cstr_name=cstr_name)

    @always_inline
    def call[
        name: StaticString,
        return_type: RegisterPassable = NoneType,
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
        return self._handle.call[name, return_type](*args)


def __fn_type_is_cabi[T: AnyType]() -> Bool:
    """Returns `True` if `T` is a function pointer type with the `abi("C")` effect.

    This is used to enforce that `DLHandle.get_function` is called with an
    explicit C-ABI function pointer type.  A plain Mojo function type (without
    `abi("C")`) returns `False`.

    Parameters:
        T: The type to check.

    Returns:
        `True` if `T` has the `abi("C")` effect, `False` otherwise.
    """
    return __mlir_attr[
        `#kgen.fn_type_is_cabi<`,
        T,
        `> : i1`,
    ]


@fieldwise_init
struct _DLHandle(Boolable, ImplicitlyCopyable, RegisterPassable):
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

    var handle: _CPointer[NoneType, MutExternalOrigin]
    """The handle to the dynamic library."""

    @always_inline
    def __init__(out self, flags: Int = DEFAULT_RTLD) raises:
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
        self = Self._dlopen(
            Optional[UnsafePointer[c_char, ExternalOrigin[mut=False]]](), flags
        )

    def __init__[
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
    def _dlopen(
        file: OptionalUnsafePointer[c_char, _], flags: Int
    ) raises -> _DLHandle:
        var handle = dlopen(file, Int32(flags))
        if not handle:
            var error_message = dlerror()
            var mesage = StringSlice(
                unsafe_from_utf8_ptr=error_message.value()
            ) if error_message else {}
            raise Error("dlopen failed: ", mesage)
        return _DLHandle(handle)

    def check_symbol(self, var name: String) -> Bool:
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

    def close(mut self):
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

    def __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
          True if the DLHandle is not null and False otherwise.
        """
        return Bool(self.handle)

    def get_function[
        result_type: TrivialRegisterPassable
    ](self, var name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The C-ABI function pointer type to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.

        Constraints:
            `result_type` must be a function pointer type annotated with
            `abi("C")` (e.g. `def(Float64) abi("C") -> Float64`). Using a
            plain Mojo function type causes silent ABI corruption for struct
            arguments and return values.
        """
        # TODO(MOCO-3709): Re-enable this constraint once kgen-opt passes
        # (e.g. mogg-annotate-kernels) can parse extern|cabi function types
        # in prebuilt stdlib packages without failing with "expected '->'".
        # comptime assert __fn_type_is_cabi[result_type](), (
        #     'result_type must be a C-ABI function pointer type: use abi("C") on'
        #     ' the function type, e.g. `def(Float64) abi("C") -> Float64`'
        # )

        return self._get_function[result_type](
            cstr_name=name.as_c_string_slice().unsafe_ptr()
        )

    @always_inline
    def _get_function[
        func_name: StaticString, result_type: TrivialRegisterPassable
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
    def _get_function[
        result_type: TrivialRegisterPassable
    ](self, *, cstr_name: UnsafePointer[mut=False, c_char, _]) -> result_type:
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

        if not opaque_function_ptr:
            abort(
                t"symbol not found: "
                t"{StringSlice(unsafe_from_utf8_ptr=cstr_name)}"
            )

        return UnsafePointer(to=opaque_function_ptr.value()).bitcast[
            result_type
        ]()[]

    def get_symbol[
        result_type: AnyType,
    ](self, name: StringSlice) -> Optional[
        UnsafePointer[result_type, MutAnyOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        name_copy = String(name)
        return self.get_symbol[result_type](
            cstr_name=name_copy.as_c_string_slice().unsafe_ptr()
        )

    def get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: UnsafePointer[mut=False, Int8, _]) -> Optional[
        UnsafePointer[result_type, MutAnyOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        debug_assert(
            Bool(self.handle),
            "Dylib handle is null when loading symbol: ",
            StringSlice(unsafe_from_utf8_ptr=cstr_name),
        )

        # Follow the dance described in
        # https://man7.org/linux/man-pages/man3/dlsym.3.html to distinguish
        # a symbol that was not found from a symbol whose value is NULL:
        #
        # 1. Clear any old error with dlerror()
        # 2. Call dlsym()
        # 3. Call dlerror() again — if it returns non-NULL, an error occurred

        # Clear any pre-existing error.
        _ = dlerror()

        var res = dlsym[result_type](self.handle, cstr_name)

        if not res:
            # Result is NULL — check if it's an error or a valid NULL symbol.
            var err = dlerror()
            if err:
                # Symbol lookup failed.
                return None

            # Symbol is validly NULL (unusual but possible per dlsym docs).
            # Abort rather than returning a null pointer wrapped in Some,
            # which would be misleading. Callers who need to handle NULL
            # symbols should specify a nullable pointer as the result_type.
            abort(
                t"symbol resolved to NULL: "
                t"{StringSlice(unsafe_from_utf8_ptr=cstr_name)}"
            )

        var ptr: UnsafePointer[result_type, MutAnyOrigin] = res.value()
        return ptr

    @always_inline
    def call[
        name: StaticString,
        return_type: RegisterPassable = NoneType,
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

        @parameter
        def _check_symbol() -> Bool:
            return self.check_symbol(String(name))

        debug_assert[_check_symbol]("symbol not found: ", name)
        var v = args.get_loaded_kgen_pack()
        # TODO(MOCO-3692): This uses Mojo calling convention instead of C ABI.
        # We cannot add abi("C") here because `type_of(v)` is a kgen pack type,
        # not the expanded individual argument types, and Mojo function type
        # syntax has no variadic parameter form. Safe in practice only for
        # scalar/register-passable arguments where Mojo and C conventions agree.
        return self._get_function[name, def(type_of(v)) thin -> return_type]()(
            v
        )


@always_inline
def _get_dylib_function[
    dylib_global: _Global[StorageType=OwnedDLHandle, ...],
    func_name: StaticString,
    result_type: TrivialRegisterPassable,
]() raises -> result_type:
    var func_cache_name = String(t"{dylib_global.name}/{func_name}")
    var func_ptr = _get_global_or_null(func_cache_name)
    if func_ptr:
        return UnsafePointer(to=func_ptr).bitcast[result_type]()[]

    var dylib = dylib_global.get_or_create_ptr()[].borrow()
    var new_func = dylib._get_function[func_name, result_type]()

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(func_cache_name),
        UnsafePointer(to=new_func).bitcast[
            OpaquePointer[MutExternalOrigin]
        ]()[],
    )

    return new_func


def _try_find_dylib[
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


def _try_find_dylib[
    name: StaticString = ""
](*paths: Path) raises -> OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    var paths_list = List[Path](capacity=len(paths))
    for path in paths:
        paths_list.append(path)
    return _try_find_dylib[name](paths_list)


def _find_dylib[
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


def _find_dylib[
    msg: def() thin -> String, abort_on_failure: Bool = True
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


def _find_dylib[name: StaticString = ""](*paths: Path) -> OwnedDLHandle:
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
    init_fn: def() thin -> StorageType,
    on_error_msg: Optional[def() thin -> Error] = None,
](Defaultable):
    comptime ResultType = UnsafePointer[Self.StorageType, MutExternalOrigin]

    def __init__(out self):
        pass

    @staticmethod
    def _init_wrapper() -> _CPointer[NoneType, ExternalOrigin[mut=True]]:
        # Heap allocate space to store this "global"
        # TODO:
        #   Any way to avoid the move, e.g. by calling this function
        #   with the ABI destination result pointer already set to `ptr`?
        var ptr = OwnedPointer(Self.init_fn())

        return ptr^.steal_data().bitcast[NoneType]()

    @staticmethod
    def _deinit_wrapper(
        opaque_ptr: _CPointer[NoneType, ExternalOrigin[mut=True]]
    ):
        # Deinitialize and deallocate the storage.
        if opaque_ptr:
            free(opaque_ptr.value(), {count = 1})

    @staticmethod
    def get_or_create_ptr() raises -> Self.ResultType:
        var ptr = _get_global[
            Self.name, Self._init_wrapper, Self._deinit_wrapper
        ]()

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return unsafe_cast[Type=Self.StorageType](ptr).value()

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
    def get_or_create_indexed_ptr(idx: Int) raises -> Self.ResultType:
        var ptr = external_call[
            "KGEN_CompilerRT_GetOrCreateGlobalIndexed",
            _CPointer[NoneType, ExternalOrigin[mut=True]],
        ](
            idx,
            Self._init_wrapper,
            Self._deinit_wrapper,
        )

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return unsafe_cast[Type=Self.StorageType](ptr).value()


@always_inline
def _get_global[
    name: StaticString,
    init_fn: def() thin -> _CPointer[NoneType, ExternalOrigin[mut=True]],
    destroy_fn: def(_CPointer[NoneType, ExternalOrigin[mut=True]]) thin -> None,
]() -> _CPointer[NoneType, ExternalOrigin[mut=True]]:
    return external_call[
        "KGEN_CompilerRT_GetOrCreateGlobal",
        _CPointer[NoneType, ExternalOrigin[mut=True]],
    ](
        name,
        init_fn,
        destroy_fn,
    )


@always_inline
def _get_global_or_null(
    name: StringSlice,
) -> _CPointer[NoneType, ExternalOrigin[mut=True]]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrNull",
        _CPointer[NoneType, ExternalOrigin[mut=True]],
    ](name.unsafe_ptr(), name.byte_length())


# ===-----------------------------------------------------------------------===#
# external_call
# ===-----------------------------------------------------------------------===#

comptime _CPointer[
    mut: Bool, //, T: AnyType, origin: Origin[mut=mut]
] = Optional[UnsafePointer[T, origin]]


@always_inline("nodebug")
def external_call[
    callee: StaticString,
    return_type: RegisterPassable,
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

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()
    comptime callee_kgen_string = _get_kgen_string[callee]()

    comptime if _type_is_eq[return_type, NoneType]():
        __mlir_op.`pop.external_call`[func=callee_kgen_string, _type=None](
            loaded_pack
        )
        return rebind_var[return_type](None)
    else:
        return __mlir_op.`pop.external_call`[
            func=callee_kgen_string,
            _type=return_type,
        ](loaded_pack)


# ===-----------------------------------------------------------------------===#
# _external_call_const
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _external_call_const[
    callee: StaticString,
    return_type: TrivialRegisterPassable,
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
        func=_get_kgen_string[callee](),
        resAttrs=__mlir_attr.`[{llvm.noundef}]`,
        funcAttrs=__mlir_attr.`["willreturn"]`,
        memory=__mlir_attr[
            `#llvm.memory_effects<other = none, `,
            `argMem = none, `,
            `inaccessibleMem = none, `,
            `errnoMem = none, `,
            `targetMem0 = none, `,
            `targetMem1 = none>`,
        ],
        _type=return_type,
    ](loaded_pack)
