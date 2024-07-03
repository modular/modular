# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .int_tuple import IntTuple
from .layout import Layout
from .runtime_tuple import RuntimeTuple

# A `Layout` like type that uses RuntimeTuple as its storage instead of
# IntTuple.


struct RuntimeLayout[layout: Layout](Stringable, Formattable):
    var shape: RuntimeTuple[layout.shape]
    var stride: RuntimeTuple[layout.stride]

    fn __init__(inout self):
        constrained[
            layout.all_dims_known(), "Static layout with known dims is required"
        ]()
        self.shape = RuntimeTuple[layout.shape]()
        self.stride = RuntimeTuple[layout.stride]()

    fn __init__(
        inout self,
        shape: RuntimeTuple[layout.shape],
        stride: RuntimeTuple[layout.stride],
    ):
        self.shape = shape
        self.stride = stride

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout f: Formatter):
        f.write_str["("]()
        self.shape.format_to(f)
        f.write_str[":"]()
        self.stride.format_to(f)
        f.write_str[")"]()
