# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


def test1():
    # expected-error @below {{invalid MLIR attribute: `#kgen.deferred` can only be used for non-typed attributes}}
    # expected-note @below {{attempting to parse: '#kgen.deferred 0 : index'}}
    _ = __mlir_attr.`#kgen.deferred 0 : index`


struct DType:
    comptime type = __mlir_type.`!kgen.dtype`
    var value: Self.type


def test3[
    n: Int, dtype: DType
](
    x: __mlir_type[
        `!kgen.struct<(`,
        __mlir_type[
            `!kgen.param_list_splat<`,
            __mlir_type[`!pop.scalar<`, dtype.value, `>`],
            `, `,
            n._mlir_value,
            `>`,
        ],
        `)>`,
    ]
):
    # expected-error @below {{unable to infer result type from MLIR operation 'kgen.struct.extract'}}
    # expected-error @below {{expected an index attribute}}
    _ = __mlir_op.`kgen.struct.extract`[index=__mlir_attr.`1:index`](x)
