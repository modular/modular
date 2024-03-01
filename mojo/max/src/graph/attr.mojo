# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Attribute primitives.

`Attribute`s are key-value pairs that can be attached to a `Node`, `Graph` and
other elements. Attributes are similar to inputs, except they are constant -
their value doesn't change at runtime. The attribute name is always a string.

For exmple, `mo.constant` has a `value` attribute, representing the value
of the constant it holds.

`Attribute`s can hold various types of values, including primitive values,
lists, tensors, etc.
"""

import _mlir


@value
struct AttrMap(Sized):
    """Holds a set of attributes."""

    var attrs: List[_mlir.NamedAttribute]
    """The list of attributes held by this map.

    The list values are an opaque handle to an `Attribute`.
    """

    # ===------------------------------------------------------------------=== #
    # Basic constructors and accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: _mlir.NamedAttribute):
        """Constructs an `AttrMap` from a set of opaque `Attribute` handles."""
        self.attrs = List[_mlir.NamedAttribute]()
        for attr in attrs:
            self.attrs.append(attr[])

    fn __len__(self) -> Int:
        """Returns the length of this map."""
        return len(self.attrs)
