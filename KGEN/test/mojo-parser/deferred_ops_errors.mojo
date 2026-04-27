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


def test3(x: __mlir_type.`!kgen.struct<(!pop.scalar<f32>)>`):
    # expected-error @+1 {{element index 1 out of bounds (>=1)}}
    _ = __mlir_op.`kgen.struct.extract`[_type = __mlir_type.`!pop.scalar<f32>`, index=__mlir_attr.`1:index`](x)
