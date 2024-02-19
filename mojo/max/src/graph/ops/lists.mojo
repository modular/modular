# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""List operations.

Note: All the helpers in this module are documented as "Creates foo". This is
a shorthand notation for "Adds a node representing an op that returns foo".
"""

from collections import Optional


fn list(elements: SymbolTuple) raises -> Symbol:
    """Creates a new list of `MOTensor` elements.

    This uses the `mo.list.create` operation.

    Args:
        elements: The list's elements.
    """
    if len(elements) == 0:
        raise "`elements` cannot be empty"

    let g = elements[0].graph()
    let m = g.module()
    let type = elements[0].tensor_type()

    for i in range(1, len(elements)):
        let elt_type = elements[i].tensor_type()
        if not elt_type == type:
            raise "elements must all have the same type " + type.to_string(
                m
            ) + ", got " + elt_type.to_string(m) + " at position " + str(i)

    return g.op("mo.list.create", elements, MOList(type))


fn list(type: MOTensor, g: Graph) raises -> Symbol:
    """Creates a new empty list of `MOTensor` elements.

    This uses the `mo.list.create` operation.

    Args:
        type: The list's element type.
        g: The `Graph` to add nodes to.
    """
    return g.op("mo.list.create", MOList(type))
