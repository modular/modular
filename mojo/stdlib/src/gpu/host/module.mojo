# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA module operations."""

from collections.optional import Optional
from os import abort
from pathlib import Path

from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer, Pointer

from utils import StringRef

from ._utils import _check_error, _FunctionHandle, _ModuleHandle

# ===----------------------------------------------------------------------===#
# JitOptions
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct JitOptions:
    var _value: Int32

    alias MAX_REGISTERS: JitOptions = 0
    """Max number of registers that a thread may use.
      Option type: unsigned int
      Applies to: compiler only
    """

    alias THREADS_PER_BLOCK: JitOptions = 1
    """IN: Specifies minimum number of threads per block to target compilation
    for
    OUT: Returns the number of threads the compiler actually targeted.
    This restricts the resource utilization of the compiler (e.g. max
    registers) such that a block with the given number of threads should be
    able to launch based on register limitations. Note, this option does not
    currently take into account any other resource limitations, such as
    shared memory utilization.
    Cannot be combined with ::CU_JIT_TARGET.
    Option type: unsigned int
    Applies to: compiler only
    """
    alias WALL_TIME: JitOptions = 2
    """Overwrites the option value with the total wall clock time, in
      milliseconds, spent in the compiler and linker
      Option type: float
      Applies to: compiler and linker
    """
    alias INFO_LOG_BUFFER: JitOptions = 3
    """Pointer to a buffer in which to print any log messages
      that are informational in nature (the buffer size is specified via
      option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
      Option type: char *
      Applies to: compiler and linker
    """
    alias INFO_LOG_BUFFER_SIZE_BYTES: JitOptions = 4
    """IN: Log buffer size in bytes.  Log messages will be capped at this size
      (including null terminator)
      OUT: Amount of log buffer filled with messages
      Option type: unsigned int
      Applies to: compiler and linker
    """
    alias ERROR_LOG_BUFFER: JitOptions = 5
    """Pointer to a buffer in which to print any log messages that
      reflect errors (the buffer size is specified via option
      ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
      Option type: char *
      Applies to: compiler and linker
    """
    alias ERROR_LOG_BUFFER_SIZE_BYTES: JitOptions = 6
    """IN: Log buffer size in bytes.  Log messages will be capped at this size
      (including null terminator)
      OUT: Amount of log buffer filled with messages
      Option type: unsigned int
      Applies to: compiler and linker
    """
    alias OPTIMIZATION_LEVEL: JitOptions = 7
    """Level of optimizations to apply to generated code (0 - 4), with 4
      being the default and highest level of optimizations.
      Option type: unsigned int
      Applies to: compiler only
    """
    alias TARGET_FROM_CUCONTEXT: JitOptions = 8
    """No option value required. Determines the target based on the current
      attached context (default)
      Option type: No option value needed
      Applies to: compiler and linker
    """
    alias TARGET: JitOptions = 9
    """Target is chosen based on supplied ::CUjit_target.  Cannot be
      combined with ::CU_JIT_THREADS_PER_BLOCK.
      Option type: unsigned int for enumerated type ::CUjit_target
      Applies to: compiler and linker
    """
    alias FALLBACK_STRATEGY: JitOptions = 10
    """Specifies choice of fallback strategy if matching cubin is not found.
      Choice is based on supplied ::CUjit_fallback.  This option cannot be
      used with cuLink* APIs as the linker requires exact matches.
      Option type: unsigned int for enumerated type ::CUjit_fallback
      Applies to: compiler only
    """
    alias GENERATE_DEBUG_INFO: JitOptions = 11
    """Specifies whether to create debug information in output (-g)
      (0: false, default)
      Option type: int
      Applies to: compiler and linker
    """
    alias LOG_VERBOSE: JitOptions = 12
    """Generate verbose log messages (0: false, default)
      Option type: int
      Applies to: compiler and linker
    """
    alias GENERATE_LINE_INFO: JitOptions = 13
    """Generate line number information (-lineinfo) (0: false, default)
      Option type: int
      Applies to: compiler only
    """
    alias CACHE_MODE: JitOptions = 14
    """Specifies whether to enable caching explicitly (-dlcm)
      Choice is based on supplied ::CUjit_cacheMode_enum.
      Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum
      Applies to: compiler only
    """
    alias NEW_SM3X_OPT: JitOptions = 15
    """[[depricated]]
      This jit option is deprecated and should not be used.
    """
    alias FAST_COMPILE: JitOptions = 16
    """This jit option is used for internal purpose only.
    """
    alias GLOBAL_SYMBOL_NAMES: JitOptions = 17
    """Array of device symbol names that will be relocated to the corresponding
      host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.
      Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.
      When loading a device module, driver will relocate all encountered
      unresolved symbols to the host addresses.
      It is only allowed to register symbols that correspond to unresolved
      global variables.
      It is illegal to register the same device symbol at multiple addresses.
      Option type: const char **
      Applies to: dynamic linker only
    """
    alias GLOBAL_SYMBOL_ADDRESSES: JitOptions = 18
    """Array of host addresses that will be used to relocate corresponding
      device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.
      Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.
      Option type: void **
      Applies to: dynamic linker only
    """
    alias GLOBAL_SYMBOL_COUNT: JitOptions = 19
    """Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
      ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.
      Option type: unsigned int
      Applies to: dynamic linker only
    """
    alias LTO: JitOptions = 20
    """[[depricated]]
      Enable link-time optimization (-dlto) for device code (Disabled by default).
      This option is not supported on 32-bit platforms.
      Option type: int
      Applies to: compiler and linker
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias FTZ: JitOptions = 21
    """[[depricated]]
      Control single-precision denormals (-ftz) support (0: false, default).
      1 : flushes denormal values to zero
      0 : preserves denormal values
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias PREC_DIV: JitOptions = 22
    """[[depricated]]
      Control single-precision floating-point division and reciprocals
      (-prec-div) support (1: true, default).
      1 : Enables the IEEE round-to-nearest mode
      0 : Enables the fast approximation mode
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias PREC_SQRT: JitOptions = 23
    """[[depricated]]
      Control single-precision floating-point square root
      (-prec-sqrt) support (1: true, default).
      1 : Enables the IEEE round-to-nearest mode
      0 : Enables the fast approximation mode
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias FMA: JitOptions = 24
    """[[depricated]]
      Enable/Disable the contraction of floating-point multiplies
      and adds/subtracts into floating-point multiply-add (-fma)
      operations (1: Enable, default; 0: Disable).
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_KERNEL_NAMES: JitOptions = 25
    """[[depricated]]
      Array of kernel names that should be preserved at link time while others
      can be removed.
      Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.
      Note that kernel names can be mangled by the compiler in which case the
      mangled name needs to be specified.
      Wildcard "*" can be used to represent zero or more characters instead of
      specifying the full or mangled name.
      It is important to note that the wildcard "*" is also added implicitly.
      For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
      thus preserve all kernels with those names. This can be avoided by providing
      a more specific name like "barfoobaz".
      Option type: const char **
      Applies to: dynamic linker only
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_KERNEL_COUNT: JitOptions = 26
    """[[depricated]]
      Number of entries in ::CU_JIT_REFERENCED_KERNEL_NAMES array.
      Option type: unsigned int
      Applies to: dynamic linker only
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_VARIABLE_NAMES: JitOptions = 27
    """[[depricated]]
      Array of variable names (__device__ and/or __constant__) that should be
      preserved at link time while others can be removed.
      Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.
      Note that variable names can be mangled by the compiler in which case the
      mangled name needs to be specified.
      Wildcard "*" can be used to represent zero or more characters instead of
      specifying the full or mangled name.
      It is important to note that the wildcard "*" is also added implicitly.
      For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
      thus preserve all variables with those names. This can be avoided by providing
      a more specific name like "barfoobaz".
      Option type: const char **
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias REFERENCED_VARIABLE_COUNT: JitOptions = 28
    """[[depricated]]
      Number of entries in ::CU_JIT_REFERENCED_VARIABLE_NAMES array.
      Option type: unsigned int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias OPTIMIZE_UNUSED_DEVICE_VARIABLES: JitOptions = 29
    """[[depricated]]
      This option serves as a hint to enable the JIT compiler/linker
      to remove constant (__constant__) and device (__device__) variables
      unreferenced in device code (Disabled by default).
      Note that host references to constant and device variables using APIs like
      ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
      the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.
      Option type: int
      Applies to: link-time optimization specified with CU_JIT_LTO
      *
      Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    """
    alias POSITION_INDEPENDENT_CODE: JitOptions = 30
    """Generate position independent code (0: false)
      Option type: int
      Applies to: compiler only
    """

    fn __init__(inout self):
        self._value = 0

    fn __init__(inout self, value: Int):
        self._value = value


# ===----------------------------------------------------------------------===#
# Module
# ===----------------------------------------------------------------------===#


@value
struct Module:
    var module: _ModuleHandle
    var cuda_dll: Optional[CudaDLL]

    fn __init__(inout self, ctx: Context):
        self.__init__(ctx.cuda_dll)

    fn __init__(inout self, cuda_dll: Optional[CudaDLL] = None):
        self.module = _ModuleHandle()
        self.cuda_dll = cuda_dll

    fn __init__(inout self, ctx: Context, path: Path) raises:
        self.__init__(path, ctx.cuda_dll)

    fn __init__(
        inout self, path: Path, cuda_dll: Optional[CudaDLL] = None
    ) raises:
        self.cuda_dll = cuda_dll
        var module = _ModuleHandle()
        var path_cstr = path.__str__()

        _check_error(
            cuModuleLoad.load()(
                Pointer.address_of(module),
                path_cstr.unsafe_cstr_ptr(),
            )
        )
        _ = path_cstr
        self.module = module

    fn __init__(
        inout self,
        ctx: Context,
        content: String,
        debug: Bool = False,
        verbose: Bool = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
    ) raises:
        self.__init__(
            content,
            debug,
            verbose,
            max_registers,
            threads_per_block,
            ctx.cuda_dll,
        )

    fn __init__(
        inout self,
        content: String,
        debug: Bool = False,
        verbose: Bool = False,
        max_registers: Optional[Int] = None,
        threads_per_block: Optional[Int] = None,
        cuda_dll: Optional[CudaDLL] = None,
    ) raises:
        """Loads a module in the current CUDA context by mapping PTX provided as a NULL terminated text string.
        """

        self.cuda_dll = cuda_dll
        self.module = _ModuleHandle()
        if debug or verbose:
            alias buffer_size = 4096
            alias max_num_options = 10
            var num_options = 0

            var info_buffer = stack_allocation[buffer_size, Int8]()
            var error_buffer = stack_allocation[buffer_size, Int8]()

            memset_zero(info_buffer, buffer_size)
            memset_zero(error_buffer, buffer_size)

            var opts = stack_allocation[max_num_options, JitOptions]()
            var option_vals = stack_allocation[
                max_num_options, Pointer[NoneType]
            ]()

            opts[num_options] = JitOptions.INFO_LOG_BUFFER
            option_vals[num_options] = info_buffer.bitcast[NoneType]()
            num_options += 1

            opts[num_options] = JitOptions.INFO_LOG_BUFFER_SIZE_BYTES
            option_vals[num_options] = Pointer[NoneType](address=buffer_size)
            num_options += 1

            opts[num_options] = JitOptions.ERROR_LOG_BUFFER
            option_vals[num_options] = info_buffer.bitcast[NoneType]()
            num_options += 1

            opts[num_options] = JitOptions.ERROR_LOG_BUFFER_SIZE_BYTES
            option_vals[num_options] = Pointer[NoneType](address=buffer_size)
            num_options += 1

            if debug:
                opts[num_options] = JitOptions.GENERATE_DEBUG_INFO
                option_vals[num_options] = Pointer[NoneType](address=1)
                num_options += 1

            if max_registers:
                opts[num_options] = JitOptions.MAX_REGISTERS
                option_vals[num_options] = Pointer[NoneType](
                    address=max_registers.value()
                )
                num_options += 1

            if threads_per_block:
                opts[num_options] = JitOptions.THREADS_PER_BLOCK
                option_vals[num_options] = Pointer[NoneType](
                    address=threads_per_block.value()
                )
                num_options += 1

            # Note that content has already gone through _cleanup_asm and
            # is null terminated.
            var cuModuleLoadDataEx = self.cuda_dll.value().cuModuleLoadDataEx if self.cuda_dll else cuModuleLoadDataEx.load()
            var result = cuModuleLoadDataEx(
                Pointer.address_of(self.module),
                content.unsafe_ptr(),
                UInt32(num_options),
                opts,
                option_vals,
            )

            if verbose:
                var info_buffer_str = StringRef(info_buffer)
                if info_buffer_str:
                    print(info_buffer_str)

                var error_buffer_str = StringRef(error_buffer)
                if error_buffer_str:
                    print(error_buffer_str)

            _check_error(result)
        else:
            var cuModuleLoadData = self.cuda_dll.value().cuModuleLoadData if self.cuda_dll else cuModuleLoadData.load()
            _check_error(
                cuModuleLoadData(
                    Pointer.address_of(self.module), content.unsafe_ptr()
                )
            )

    fn __del__(owned self):
        """Unloads a module ffrom the current CUDA context."""

        try:
            var cuModuleUnload = self.cuda_dll.value().cuModuleUnload if self.cuda_dll else cuModuleUnload.load()
            if self.module:
                _check_error(cuModuleUnload(self.module))
                self.module = _ModuleHandle()
        except e:
            abort(e.__str__())

    @always_inline
    fn _steal_handle(inout self) -> _ModuleHandle:
        """Steal the underlying handle from the module."""
        var res = self.module
        self.module = _ModuleHandle()
        return res

    fn load(self, name: String) raises -> _FunctionHandle:
        """Returns the handle of a function which matches the input name in
        a particular module.
        """

        var func = _FunctionHandle()
        var name_cstr = name

        var cuModuleGetFunction = self.cuda_dll.value().cuModuleGetFunction if self.cuda_dll else cuModuleGetFunction.load()
        _check_error(
            cuModuleGetFunction(
                Pointer.address_of(func),
                self.module,
                name_cstr.unsafe_cstr_ptr(),
            )
        )

        _ = name_cstr

        return func
