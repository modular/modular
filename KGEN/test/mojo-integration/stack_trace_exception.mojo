# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin

# COM: Stack traces are supported on Darwin, but result to different output.
# COM: To avoid having fragile test, mark this test as unsupported on MacOS


fn foo() raises:
    raise Error("gotcha!")


@no_inline
fn nested_func() raises:
    foo1()


@no_inline
fn foo1() raises:
    foo2()


@no_inline
fn foo2() raises:
    raise Error("nested gotcha!")


fn main():
    try:
        foo()
    except e:
        print("stack trace of", e)
        print(String(e.get_stack_trace()))

    try:
        nested_func()
    except e:
        print("stack trace of", e)
        print(String(e.get_stack_trace()))


# RUN: %mojo-build-no-debug-no-assert %s --debug-level full -o %t 2>&1
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL %s
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=0 %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-FULL-NO-STACK %s

# RUN: %mojo-build-no-debug-no-assert %s --debug-level none -o %t 2>&1
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O3-NONE %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level full -o %t 2>&1
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O0-FULL %s

# RUN: %mojo-build-no-debug-no-assert %s -O0 --debug-level none -o %t 2>&1
# RUN: MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 %t > %t.log
# RUN: cat %t.log | FileCheck --check-prefix=O0-NONE %s

# O3-FULL-LABEL: stack trace of gotcha!
# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::error::StackTrace::__init__(::Int) open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O3-FULL:      #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} main open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}

# O3-FULL-LABEL: stack trace of nested gotcha!
# O3-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::error::StackTrace::__init__(::Int) open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::error::Error::__init__[__mlir_type.!kgen.string](::StringLiteral[$0])_REMOVED_ARG open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stack_trace_exception::foo2()_REMOVED_ARG {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O3-FULL-NEXT: #{{.*}} main {{.*}}open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}

# O3-FULL-NO-STACK-LABEL: stack trace of gotcha!
# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`

# O3-FULL-NO-STACK-LABEL: stack trace of nested gotcha!
# O3-FULL-NO-STACK: stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`

# O3-NONE-LABEL: stack trace of gotcha!
# O3-NONE: #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-NONE: #{{.*}} main

# O3-NONE-LABEL: stack trace of nested gotcha!
# O3-NONE:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O3-NONE:      #{{.*}} stack_trace_exception::foo2()_REMOVED_ARG stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-NONE-NEXT: #{{.*}} stack_trace_exception::foo1() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O3-NONE-NEXT: #{{.*}} main


# O0-FULL-LABEL: stack trace of gotcha!
# O0-FULL:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-FULL-NEXT: #{{.*}} stdlib::builtin::error::StackTrace::__init__(::Int) open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stdlib::builtin::error::Error::__init__[__mlir_type.!kgen.string](::StringLiteral[$0]),value`2x="gotcha!" open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stack_trace_exception::foo() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} main {{.*}}open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}

# O0-FULL-LABEL: stack trace of nested gotcha!
# O0-FULL:       #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-FULL-NEXT: #{{.*}} stdlib::builtin::error::StackTrace::__init__(::Int) open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stdlib::builtin::error::Error::__init__[__mlir_type.!kgen.string](::StringLiteral[$0]),value`2x="nested gotcha!" open-source/max/mojo/stdlib/stdlib/builtin/error.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::foo2() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::foo1() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::nested_func() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stack_trace_exception::main() {{.*}}/stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT:  #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}
# O0-FULL-NEXT: #{{.*}} main {{.*}}open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:{{.*}}:{{.*}}

# O0-NONE-LABEL: stack trace of gotcha!
# O0-NONE:      #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-NONE:      #{{.*}} stack_trace_exception::foo() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT: #{{.*}} stack_trace_exception::main() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT: #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" {{.*}}stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT: #{{.*}} main

# O0-NONE-LABEL: stack trace of nested gotcha!
# O0-NONE:       #{{.*}} KGEN_CompilerRT_GetStackTrace
# O0-NONE:       #{{.*}} stack_trace_exception::foo2() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::foo1() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::nested_func() {{.*}}stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stack_trace_exception::main() stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} stdlib::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>),main_func="stack_trace_exception::main()" {{.*}}stack_trace_exception.mojo:{{.*}}:{{.*}}
# O0-NONE-NEXT:  #{{.*}} main
