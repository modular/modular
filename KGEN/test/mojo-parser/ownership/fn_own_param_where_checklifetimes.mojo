# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

# A function's own trailing where-clause about an associated type of its own
# generic parameter must be a known assumption when reasoning about values of
# that associated type in the function's body. This previously failed because
# the function's own body constraints reference its own parameters by index,
# while types elsewhere in the body reference them by name.

trait MyIterator(ImplicitlyDeletable):
    comptime Element: Movable

    def next(mut self) -> Self.Element:
        ...


# CHECK-LABEL: lit.fn @"take_and_peek
# CHECK: lit.var.decl "x" var
# CHECK: lit.call{{.*}}"next
# CHECK: get_witness<:!AnyType_Movable #kgen.get_witness<{{.*}}"Element">, "std::builtin::stubs::ImplicitlyDeletable", "__deinit__
def take_and_peek[Iter: MyIterator](var it: Iter) where conforms_to(
    Iter.Element, ImplicitlyDeletable
):
    var x = it.next()
    _ = x
