# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct StaticTensorSpec:
    # Represents the DimList type (not accessible from KGEN tests).
    alias _dims_type = __mlir_type[`!kgen.variadic<index>`]
    var shape: Self._dims_type
    var strides: Self._dims_type

    fn __init__(inout self, shape: Self._dims_type, strides: Self._dims_type):
        self.shape = shape
        self.strides = strides

    fn __init__(inout self):
        var shape = __mlir_op.`pop.variadic.create`[_type = Self._dims_type,]()
        var strides = __mlir_op.`pop.variadic.create`[
            _type = Self._dims_type,
        ]()
        self = Self(shape, strides)
