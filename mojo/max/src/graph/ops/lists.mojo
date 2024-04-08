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
    """Creates a new list and fills it with elements.

    This uses the `mo.list.create` operation. The elements must have the same
    type.

    Args:
        elements: The list's elements.

    Returns:
        The list filled with `elements`. It's type will be `MOList`.
    """
    if len(elements) == 0:
        raise "`elements` cannot be empty"

    var g = elements[0].graph()
    var m = g.module()
    var ctx = g._op.context()
    var type = elements[0].tensor_type()

    for i in range(1, len(elements)):
        var elt_type = elements[i].tensor_type()
        if not elt_type == type:
            raise (
                "elements must all have the same type "
                + str(type.to_mlir(ctx))
                + ", got "
                + str(elt_type.to_mlir(ctx))
                + " at position "
                + str(i)
            )

    return g.op("mo.list.create", elements, MOList(type))


fn list(type: MOTensor, g: Graph) raises -> Symbol:
    """Creates a new empty list of `MOTensor` elements.

    This uses the `mo.list.create` operation.

    Args:
        type: The list's element type.
        g: The `Graph` to add nodes to.

    Returns:
        A new empty list. It's type will be `MOList`.
    """
    return g.op("mo.list.create", MOList(type))
