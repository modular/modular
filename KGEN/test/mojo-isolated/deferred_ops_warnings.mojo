# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


@export
fn use_mlir_attr_warning_subscript(a: Int, b: Int) -> Bool:
    # expected-warning @+1 {{trivially constructable attribute. Use `__mlir_attr` instead.}}
    var res = __mlir_op.`index.cmp`[pred=__mlir_deferred_attr[`#index<cmp_predicate sle>`]](a, b)
    return res


@export
fn use_mlir_attr_warning_backticks(a: Int, b: Int) -> Bool:
    # expected-warning @+1 {{trivially constructable attribute. Use `__mlir_attr` instead}}
    var res = __mlir_op.`index.cmp`[pred=__mlir_deferred_attr.`#index<cmp_predicate sle>`](a, b)
    return res
