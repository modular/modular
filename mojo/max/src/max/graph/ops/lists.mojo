# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Ops that create lists."""

from collections import List, Optional

from ..error import error


fn list(elements: List[Symbol]) raises -> Symbol:
    """Creates a new list and fills it with elements.

    This uses the `mo.list.create` operation. The elements must have the same
    type.

    Args:
        elements: The list's elements.

    Returns:
        The list filled with `elements`. Its type will be `ListType`.
    """
    if len(elements) == 0:
        # Unfortunately no way to get a graph here :(
        raise error(None, "`elements` cannot be empty")

    var g = elements[0].graph()
    var ctx = g._context()
    var type = elements[0].tensor_type()

    for i in range(1, len(elements)):
        var elt_type = elements[i].tensor_type()
        if not elt_type == type:
            raise error(
                g,
                "elements must all have the same type ",
                type.to_mlir(ctx),
                ", got ",
                elt_type.to_mlir(ctx),
                " at position ",
                i,
            )

    return g.op("mo.list.create", elements, ListType(type))


fn list(type: TensorType, g: Graph) raises -> Symbol:
    """Creates a new empty list of `TensorType` elements.

    This uses the `mo.list.create` operation.

    Args:
        type: The list's element type.
        g: The `Graph` to add nodes to.

    Returns:
        A new empty list. Its type will be `ListType`.
    """
    return g.op("mo.list.create", ListType(type))
