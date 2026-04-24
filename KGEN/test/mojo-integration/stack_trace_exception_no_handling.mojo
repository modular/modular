# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin, NVIDIA-GPU, AMD-GPU

# COM: Stack traces are supported on Darwin, but result in different output.
# COM: Stack traces are disabled on GPU.
# COM: To avoid having fragile tests, mark this test as unsupported on these platforms.


@no_inline
def nested_func() raises:
    foo1()


@no_inline
def foo1() raises:
    foo2()


@no_inline
def foo2() raises:
    raise Error("nested gotcha!")


def main() raises:
    nested_func()


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: MODULAR_DEBUG=stack-trace-on-error %t > %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: %t > %t.log || true
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL-NO-STACK %s

# O3-FULL-NO-STACK: Unhandled exception caught during execution: nested gotcha!

# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} std::builtin::error::StackTrace::collect_if_enabled(::Int)
# O3-FULL-NEXT: #{{.*}} stack_trace_exception_no_handling::foo2()_REMOVED_ARG {{.*}}/stack_trace_exception_no_handling.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} std::builtin::_startup::__wrap_and_execute_raising_main
# O3-FULL-NEXT: #{{.*}} main {{.*}}oss/modular/mojo/stdlib/std/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL: Unhandled exception caught during execution: nested gotcha!
