# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate %s --verify-diagnostics
# RUN: not kgen -elaborate %s --elaboration-error-include-prelude 2>&1 | FileCheck %s --check-prefix=CHECK-PRELUDE
# RUN: not mojo --elaboration-error-include-prelude %s 2>&1 | FileCheck %s --check-prefix=CHECK-PRELUDE
# RUN: not mojo --elaboration-error-include-prelude --elaboration-error-verbose %s 2>&1 | FileCheck %s --check-prefix=CHECK-VERBOSE

from collections.string.string_slice import _get_kgen_string

# CHECK-PRELUDE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: error: function instantiation failed
# CHECK-VERBOSE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-VERBOSE-SAME: error: function instantiation failed
# CHECK-PRELUDE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: call expansion failed with parameter value(s): (...)
# CHECK-VERBOSE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-VERBOSE-SAME: note: call expansion failed with parameter value(s): ("main_func":
# CHECK-PRELUDE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: function instantiation failed
# CHECK-PRELUDE: {{.*}}stdlib/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: call expansion failed


@always_inline("nodebug")
fn my_constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(),
        message = _get_kgen_string[msg, extra](),
    ]()  # expected-note {{constraint failed}}


# expected-note @below {{function instantiation failed}}
fn my_func():
    # expected-note @below {{call expansion failed}}
    my_constrained[False, "foo"]()


# expected-error @below {{function instantiation failed}}
fn main():
    # expected-note @below {{call expansion failed}}
    my_func()
