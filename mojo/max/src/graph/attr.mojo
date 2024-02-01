# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import mlir


@value
struct AttrMap(Sized):
    var attrs: DynamicVector[mlir.NamedAttribute]

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: mlir.NamedAttribute):
        self.attrs = DynamicVector[mlir.NamedAttribute]()
        for attr in attrs:
            self.attrs.append(attr[])

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return len(self.attrs)
