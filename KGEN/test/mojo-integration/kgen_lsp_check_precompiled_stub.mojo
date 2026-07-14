# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# Regression test for cloneDeclModuleForCompilation giving imported-but-unused
# stdlib functions a valid stub body. `std` is precompiled for this
# directory's mojo_deps (see BUILD.bazel), so checking any file here clones
# the whole precompiled std module, including functions this file never
# calls, like `__MLIRType.copy` -- those are left as signature-only `external` stubs.
# FileCheck the resulting stub directly: it must have a real block argument
# for its `ptr` parameter and a single block terminated by `lit.end_fn
# unresolved`, rather than crashing (empty argument list) or failing
# verification (0-block region).
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -lsp %s | FileCheck %s


def main():
    pass


# CHECK: lit.fn @"copy(::__MLIRType{{.*}}(%self: {{.*}}%__result__: {{.*}}attributes {external
# CHECK-SAME: inheritedFrom = @std::@builtin::@value::@Copyable::@"copy($0)"
# CHECK-NEXT: lit.end_fn unresolved
# CHECK-NEXT: }
