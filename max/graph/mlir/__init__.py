# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._mlir_libs._mlir import *
from ._mlir_libs._mlir.ir import *
from ._mlir_libs._mlir.ir import (
    _BaseContext as Context,
    Attribute,
    Block,
    FunctionType,
    Location,
    Type,
    TypeAttr,
)
from .dialects import _ods_common, mo


@_ods_common._cext.register_operation(mo._Dialect, replace=True)
class GraphOp(mo.GraphOp):
    """Extends mo.graph op with simpler builders."""

    def __init__(self, name: str, input_types: list, output_types: list):
        function_type = FunctionType.get(input_types, output_types)
        signature = Type.parse(f"!kgen.signature<{function_type}>")
        params = Attribute.parse("#kgen<param.decls[]>")
        super().__init__(
            name,
            TypeAttr.get(signature),
            TypeAttr.get(function_type),
            params,
            params,
            counter=0,
        )
        body = Block.create_at_start(self.regions[0])
        for input_type in input_types:
            body.add_argument(input_type, Location.current)


mo.GraphOp = GraphOp
