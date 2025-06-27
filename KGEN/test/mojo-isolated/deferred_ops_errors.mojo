# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @below {{invalid MLIR attribute: `#kgen.deferred` can only be used for non-typed attributes}}
# expected-note @below {{attempting to parse: '#kgen.deferred 0 : index'}}
_ = __mlir_attr.`#kgen.deferred 0 : index`

struct DType:
    alias type = __mlir_type.`!kgen.dtype`
    var value: Self.type

fn test3[n: Int, dtype: DType](x: __mlir_type[`!kgen.struct<(`, __mlir_type[`!kgen.variadic_splat<`, __mlir_type[`!pop.scalar<`, dtype.value, `>`], `, `, n.value, `>`] , `)>`]):
    # expected-error @below {{unable to infer result type from MLIR operation 'kgen.struct.extract'}}
    # expected-error @below {{expected an integer index attribute}}
    _ = __mlir_op.`kgen.struct.extract`[index = __mlir_attr.`1:index`](x)
