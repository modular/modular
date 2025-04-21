# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -verify-diagnostics


@export
def wrong_attribute_name(a: Int, b: Int) -> Bool:
    alias pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    # expected-error @below {{MLIR verification error: 'index.cmp' op requires attribute 'pred'}}
    var res = __mlir_op.`index.cmp`[foobar=pred_attr](a, b)
    return res


@export
def extra_attribute(a: Int, b: Int) -> Bool:
    alias pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    # expected-error @below {{unexpected attribute 'foobar' on operation}}
    var res = __mlir_op.`index.cmp`[pred=pred_attr, foobar=pred_attr](a, b)
    return res
