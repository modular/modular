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


@always_inline("nodebug")
fn to_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return to_string[string, extra]()


@always_inline("nodebug")
fn to_string[
    string: StaticString, extra: VariadicList[StaticString]
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.value,
        `> : !kgen.string`,
    ]


fn test2[pred: StaticString](x: Int, y: Int) -> Bool:
    fn get_pred[pred: StaticString]() -> __mlir_type.`!kgen.deferred`:
        return __mlir_deferred_attr[
            `#index<cmp_predicate `, +to_string[pred](), `>`
        ]

    var z = __mlir_op.`index.cmp`[pred = get_pred[pred]()](x, y)

    return z


def main():
    # CHECK: test0 = True
    print("test0 = ", test0(1, 2))

    # CHECK: test1[True] = True
    print("test1[True] = ", test1[True](1, 2))

    # CHECK: test1[False] = False
    print("test1[False] = ", test1[False](1, 2))

    # CHECK: test2["sle"] = True
    print('test2["sle"] = ', test2["sle"](1, 2))

    # CHECK: test2["sge"] = False
    print('test2["sge"] = ', test2["sge"](1, 2))
