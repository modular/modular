# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"test0(::Int,::Int)"
def test0(a: Int, b: Int) -> Bool:
    alias pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    # CHECK: kgen.deferred "index.cmp"(%{{.*}}, %{{.*}} : !Int, !Int) {pred = #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred = pred_attr](a, b)
    return res

# CHECK-LABEL: lit.fn @"test1[::Bool](::Int,::Int)"
def test1[cmp: Bool](a: Int, b: Int) -> Bool:
    # CHECK: lit.fn *"select_pred[::Bool]()"<*"cmp`2x": !Bool>() -> !kgen.deferred
    fn select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        @parameter
        if cmp:
            return __mlir_attr.`#index<cmp_predicate sle>`
        else:
            return __mlir_attr.`#index<cmp_predicate sgt>`
    alias pred_attr = select_pred[cmp]()

    # CHECK: kgen.deferred "index.cmp"(%{{.}}, %{{.*}} : !Int, !Int) {pred = #kgen.param.expr<apply, #kgen.bind_params<:!lit.generator<<"cmp": !Bool>() -> !kgen.deferred> *"select_pred[::Bool]()", cmp> : !kgen.generator<!lit.generator<() -> !kgen.deferred>>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred = pred_attr](a, b)
    return res
