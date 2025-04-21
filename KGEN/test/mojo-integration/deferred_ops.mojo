# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


def test0(a: Int, b: Int) -> Bool:
    alias pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


def test1[cmp: Bool](a: Int, b: Int) -> Bool:
    fn select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        @parameter
        if cmp:
            return __mlir_attr.`#index<cmp_predicate sle>`
        else:
            return __mlir_attr.`#index<cmp_predicate sgt>`

    alias pred_attr = select_pred[cmp]()

    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


def main():
    # CHECK: test0 = True
    print("test0 = ", test0(1, 2))

    # CHECK: test1[True] = True
    print("test1[True] = ", test1[True](1, 2))

    # CHECK: test1[False] = False
    print("test1[False] = ", test1[False](1, 2))
