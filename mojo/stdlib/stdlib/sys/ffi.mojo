# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements a foreign functions interface (FFI)."""

from collections.string.string_slice import _get_kgen_string, get_static_string
from os import PathLike, abort
from pathlib import DIR_SEPARATOR, Path
from sys._libc import dlclose, dlerror, dlopen, dlsym

from memory import UnsafePointer

from .info import is_64bit, os_is_linux, os_is_macos, os_is_windows
from .intrinsics import _mlirtype_is_eq

# ===-----------------------------------------------------------------------===#
# Primitive C type aliases
# ===-----------------------------------------------------------------------===#

alias c_char = Int8
"""C `char` type."""

alias c_uchar = UInt8
"""C `unsigned char` type."""

alias c_int = Int32
"""C `int` type.

The C `int` type is typically a signed 32-bit integer on commonly used targets
today.
"""

alias c_uint = UInt32
"""C `unsigned int` type."""

alias c_short = Int16
"""C `short` type."""

alias c_ushort = UInt16
"""C `unsigned short` type."""

alias c_long = Scalar[_c_long_dtype()]
"""C `long` type.

The C `long` type is typically a signed 64-bit integer on macOS and Linux, and a
32-bit integer on Windows."""

alias c_long_long = Scalar[_c_long_long_dtype()]
"""C `long long` type.

The C `long long` type is typically a signed 64-bit integer on commonly used
targets today."""

alias c_size_t = UInt
"""C `size_t` type."""

alias c_ssize_t = Int
"""C `ssize_t` type."""

alias c_float = Float32
"""C `float` type."""

alias c_double = Float64
"""C `double` type."""

alias OpaquePointer = UnsafePointer[NoneType]
"""An opaque pointer, equivalent to the C `void*` type."""


fn _c_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and (os_is_macos() or os_is_linux()):
        # LP64
        return DType.int64
    elif is_64bit() and os_is_windows():
        # LLP64
        return DType.int32
    else:
        constrained[False, "size of C `long` is unknown on this target"]()
        return abort[DType]()


fn _c_long_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and (os_is_macos() or os_is_linux() or os_is_windows()):
        # On a 64-bit CPU, `long long` is *always* 64 bits in every OS's data
        # model.
        return DType.int64
    else:
        constrained[False, "size of C `long long` is unknown on this target"]()
        return abort[DType]()


# ===-----------------------------------------------------------------------===#
# Dynamic Library Loading
# ===-----------------------------------------------------------------------===#


struct RTLD:
    """Enumeration of the RTLD flags used during dynamic library loading."""

    alias LAZY = 1
    """Load library lazily (defer function resolution until needed).
    """
    alias NOW = 2
    """Load library immediately (resolve all symbols on load)."""
    alias LOCAL = 4
    """Make symbols not available for symbol resolution of subsequently loaded
    libraries."""
    alias GLOBAL = 256 if os_is_linux() else 8
    """Make symbols available for symbol resolution of subsequently loaded
    libraries."""


alias DEFAULT_RTLD = RTLD.NOW | RTLD.GLOBAL


struct _OwnedDLHandle(Movable):
    """Represents an owned handle to a dynamically linked library that can be
    loaded and unloaded.

    This type is intended to replace `DLHandle`, by incrementally introducing
    ownership semantics to `DLHandle`.
    """

    var _handle: DLHandle

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self, path: String, flags: Int = DEFAULT_RTLD) raises:
        self._handle = DLHandle(path, flags)

    fn __moveinit__(out self, owned other: Self):
        self._handle = other._handle

    fn __del__(owned self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """
        self._handle.close()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn handle(self) -> DLHandle:
        return self._handle


@fieldwise_init
@register_passable("trivial")
struct DLHandle(Copyable, Movable, ExplicitlyCopyable, Boolable):
    """Represents a dynamically linked library that can be loaded and unloaded.

    The library is loaded on initialization and unloaded by `close`.
    """

    var handle: OpaquePointer
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
        """
        self = Self._dlopen(UnsafePointer[c_char](), flags)

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
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
        """

        var fspath = path.__fspath__()
        self = Self._dlopen(fspath.unsafe_cstr_ptr(), flags)

    @staticmethod
    fn _dlopen(file: UnsafePointer[c_char], flags: Int) raises -> DLHandle:
        @parameter
        if not os_is_windows():
            var handle = dlopen(file, flags)
            if handle == OpaquePointer():
                var error_message = dlerror()
                raise Error(
                    String(
                        "dlopen failed: ",
                        StringSlice[error_message.origin](
                            unsafe_from_utf8_ptr=error_message
                        ),
                    )
                )
            return DLHandle(handle)
        else:
            return DLHandle(OpaquePointer())

    fn copy(self) -> Self:
        """Copy the object.

        Returns:
            A copy of the value.
        """
        return self

    fn check_symbol(self, owned name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        constrained[
            not os_is_windows(),
            "Checking dynamic library symbol is not supported on Windows",
        ]()

        var opaque_function_ptr: OpaquePointer = dlsym(
            self.handle,
            name.unsafe_cstr_ptr(),
        )

        return Bool(opaque_function_ptr)

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn close(mut self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """

        @parameter
        if not os_is_windows():
            _ = dlclose(self.handle)
            self.handle = OpaquePointer()

    fn __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
          True if the DLHandle is not null and False otherwise.
        """
        return self.handle.__bool__()

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn get_function[
        result_type: AnyTrivialRegType
    ](self, owned name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """

        return self._get_function[result_type](cstr_name=name.unsafe_cstr_ptr())

    @always_inline
    fn _get_function[
        func_name: StaticString, result_type: AnyTrivialRegType
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
        alias func_name_literal = get_static_string[func_name]()
        return self._get_function[result_type](
            cstr_name=func_name_literal.unsafe_ptr().bitcast[c_char](),
        )

    @always_inline
    fn _get_function[
        result_type: AnyTrivialRegType
    ](self, *, cstr_name: UnsafePointer[c_char, **_]) -> result_type:
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
    ](self, name: StringSlice) -> UnsafePointer[result_type]:
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
            cstr_name=name_copy.unsafe_cstr_ptr()
        )

    fn get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: UnsafePointer[Int8, **_]) -> UnsafePointer[
        result_type
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
        debug_assert(self.handle, "Dylib handle is null")

        @parameter
        if os_is_windows():
            return abort[UnsafePointer[result_type]](
                "get_symbol isn't supported on windows"
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

            debug_assert(not res, "dlsym unexpectedly returned non-NULL result")

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
        return_type: AnyTrivialRegType = NoneType,
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
        name: StaticString, return_type: AnyTrivialRegType = NoneType
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
        return self.get_function[fn (__type_of(v)) -> return_type](
            String(name)
        )(v)


@always_inline
fn _get_dylib_function[
    dylib_global: _Global[_, _OwnedDLHandle, _],
    func_name: StaticString,
    result_type: AnyTrivialRegType,
]() -> result_type:
    alias func_cache_name = String(dylib_global.name) + "/" + String(func_name)
    var func_ptr = _get_global_or_null[func_cache_name]()
    if func_ptr:
        var result = UnsafePointer(to=func_ptr).bitcast[result_type]()[]
        _ = func_ptr
        return result

    var dylib = dylib_global.get_or_create_ptr()[].handle()
    var new_func = dylib._get_function[func_name, result_type]()

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(func_cache_name),
        UnsafePointer(to=new_func).bitcast[OpaquePointer]()[],
    )

    return new_func


fn _try_find_dylib[
    name: StaticString = ""
](paths: List[Path]) raises -> _OwnedDLHandle:
    """Try to load a dynamically linked library given a list of possible paths.

    Parameters:
        name: Optional name for the library to be used in error messages.

    Args:
        paths: A list of paths or library names to pass to the DLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.

    Raises:
        Error: If the library could not be loaded from any of the provided paths.
    """
    alias dylib_name = name if name != "" else "dynamic library"
    for path in paths:
        # If we are given a library name like libfoo.so, pass it directly to
        # dlopen(), which will invoke the system linker to find the library.
        # We can't check the existence of the path ahead of time, we have to
        # call the function and check for an error.
        try:
            return _OwnedDLHandle(String(path))
        except:
            # If the call to DLOpen fails, we should just try the next path
            # in the list. It's only a fatal error if the library cannot be
            # loaded from any of the paths provided.
            pass

    raise Error(
        String("Failed to load " + dylib_name + " from ")
        + String(" or ").join(paths)
    )


fn _try_find_dylib[
    name: StaticString = ""
](*paths: Path) raises -> _OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    var paths_list = List[Path]()
    for path in paths:
        paths_list.append(path)
    return _try_find_dylib[name](paths_list)


fn _find_dylib[name: StaticString = ""](paths: List[Path]) -> _OwnedDLHandle:
    """Load a dynamically linked library given a list of possible paths or names.

    If the library is not found, the function will abort.

    Parameters:
        name: Optional name for the library to be used in error messages.

    Args:
        paths: A list of paths or library names to pass to the DLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.
    """
    try:
        return _try_find_dylib[name](paths)
    except e:
        return abort[_OwnedDLHandle](String(e))


fn _find_dylib[name: StaticString = ""](*paths: Path) -> _OwnedDLHandle:
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
    name: StaticString,
    StorageType: Movable,
    init_fn: fn () -> StorageType,
](Defaultable):
    alias ResultType = UnsafePointer[StorageType]

    fn __init__(out self):
        pass

    @staticmethod
    fn _init_wrapper() -> OpaquePointer:
        # Heap allocate space to store this "global"
        var ptr = UnsafePointer[StorageType].alloc(1)

        # TODO:
        #   Any way to avoid the move, e.g. by calling this function
        #   with the ABI destination result pointer already set to `ptr`?
        ptr.init_pointee_move(init_fn())

        return ptr.bitcast[NoneType]()

    @staticmethod
    fn _deinit_wrapper(opaque_ptr: OpaquePointer):
        # Deinitialize and deallocate the storage.
        var ptr = opaque_ptr.bitcast[StorageType]()
        ptr.destroy_pointee()
        ptr.free()

    @staticmethod
    fn get_or_create_ptr() -> Self.ResultType:
        return _get_global[
            name, Self._init_wrapper, Self._deinit_wrapper
        ]().bitcast[StorageType]()

    # Currently known values for get_or_create_indexed_ptr.  See
    # NUM_INDEXED_GLOBALS in CompilerRT.
    alias _python_idx = 0
    alias _unused = 1  # Intentionally unused (enabled for prototyping).

    # This accesses a well-known global with a fixed index rather than using a
    # name to unique the value.  The index table is above.
    @staticmethod
    fn get_or_create_indexed_ptr(idx: Int) -> Self.ResultType:
        var ptr = external_call[
            "KGEN_CompilerRT_GetOrCreateGlobalIndexed", OpaquePointer
        ](
            idx,
            Self._init_wrapper,
            Self._deinit_wrapper,
        )
        return ptr.bitcast[StorageType]()


@always_inline
fn _get_global[
    name: StaticString,
    init_fn: fn () -> OpaquePointer,
    destroy_fn: fn (OpaquePointer) -> None,
]() -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetOrCreateGlobal", OpaquePointer](
        name,
        init_fn,
        destroy_fn,
    )


@always_inline
fn _get_global_or_null[name: StaticString]() -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer](
        name.unsafe_ptr(), name.byte_length()
    )


# ===-----------------------------------------------------------------------===#
# external_call
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn external_call[
    callee: StaticString,
    return_type: AnyTrivialRegType,
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
    return_type: AnyTrivialRegType,
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
    alias callee_kgen_string = _get_kgen_string[callee]()

    @parameter
    if _mlirtype_is_eq[return_type, NoneType]():
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
    return_type: AnyTrivialRegType,
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
            `inaccessibleMem = none>`,
        ],
        _type=return_type,
    ](loaded_pack)
