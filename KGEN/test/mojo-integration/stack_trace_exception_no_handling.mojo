# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin

# COM: Stack traces are supported on Darwin, but result to different output.
# COM: To avoid having fragile test, mark this test as unsupported on MacOS


@no_inline
fn nested_func() raises:
    foo1()


@no_inline
fn foo1() raises:
    foo2()


@no_inline
fn foo2() raises:
    raise Error("nested gotcha!")


fn main() raises:
    nested_func()


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 %t > %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: %t > %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL-NO-STACK %s

# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
# O3-FULL-NO-STACK: Unhandled exception caught during execution: nested gotcha!

# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::error::StackTrace::__init__(::Int) open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::error::Error::__init__[__mlir_type.!kgen.string](::StringLiteral[$0])_REMOVED_ARG open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stack_trace_exception_no_handling::foo2()_REMOVED_ARG {{.*}}/stack_trace_exception_no_handling.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_raising_main[fn() raises -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception_no_handling::main()" open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} main {{.*}}open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL: Unhandled exception caught during execution: nested gotcha!
