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
"""Implements functionality to start a mojo execution."""

from std.ffi import external_call, _CPointer, _get_global
from std.sys.compile import SanitizeAddress


def _init_global_runtime() -> _CPointer[NoneType, ExternalOrigin[mut=True]]:
    return external_call[
        "KGEN_CompilerRT_AsyncRT_GetOrCreateRuntime",
        _CPointer[NoneType, ExternalOrigin[mut=True]],
    ]()


def _destroy_global_runtime(ptr: _CPointer[NoneType, ExternalOrigin[mut=True]]):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_AsyncRT_ReleaseRuntime", NoneType](ptr)


@always_inline
def _ensure_current_or_global_runtime_init():
    var current_runtime = external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime",
        _CPointer[NoneType, ExternalOrigin[mut=True]],
    ]()
    if current_runtime:
        return
    _ = _get_global["Runtime", _init_global_runtime, _destroy_global_runtime]()


def __wrap_and_execute_main[
    main_func: def() -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for non-raising main function."""

    # Initialize the global runtime.
    _ensure_current_or_global_runtime_init()

    comptime if SanitizeAddress:
        external_call["KGEN_CompilerRT_SetAsanAllocators", NoneType]()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Initialize signal handler for SIGSEGV  SIGABRT that will print a stack
    # trace if MOJO_ENABLE_STACK_TRACE_ON_CRASH is set to non-zero or false.
    # Such functionality needs to be explicitly hidden under the env var,
    # because otherwise extra signal handler will be registered if user runs
    # code with sanitizer enabled, which will lead to extra stack trace printed.
    external_call["KGEN_CompilerRT_PrintStackTraceOnFault", NoneType]()

    # Call into the user main function.
    main_func()

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


def __wrap_and_execute_raising_main[
    main_func: def() raises -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for a raising main function."""

    # Initialize the global runtime.
    _ensure_current_or_global_runtime_init()

    comptime if SanitizeAddress:
        external_call["KGEN_CompilerRT_SetAsanAllocators", NoneType]()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Initialize signal handler for SIGSEGV  SIGABRT that will print a stack
    # trace if MOJO_ENABLE_STACK_TRACE_ON_CRASH is set to non-zero or false.
    # Such functionality needs to be explicitly hidden under the env var,
    # because otherwise extra signal handler will be registered if user runs
    # code with sanitizer enabled, which will lead to extra stack trace printed.
    external_call["KGEN_CompilerRT_PrintStackTraceOnFault", NoneType]()

    # Call into the user main function.
    try:
        main_func()
    except e:
        var stack_trace = e.get_stack_trace()
        if stack_trace:
            print(stack_trace.value())
        print("Unhandled exception caught during execution:", e)
        return 1

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


# A prototype of the main entry point, used by the compiled when synthesizing
# main.
def __mojo_main_prototype(
    argc: Int32, argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`]
) -> Int32:
    return 0
