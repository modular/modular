# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import _mlir


@value
struct AttrMap(Sized):
    var attrs: DynamicVector[_mlir.NamedAttribute]

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: _mlir.NamedAttribute):
        self.attrs = DynamicVector[_mlir.NamedAttribute]()
        for attr in attrs:
            self.attrs.append(attr[])

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return len(self.attrs)
