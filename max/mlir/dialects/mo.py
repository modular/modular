# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._mlir.dialects import _ods_common
from max._mlir.dialects.mo import *
from max._mlir.dialects.mo import GraphOp, _Dialect

from .. import Attribute, Block, FunctionType, Type, TypeAttr


@_ods_common._cext.register_operation(_Dialect, replace=True)  # type:ignore
class GraphOp(GraphOp):  # type: ignore[no-redef]
    """Extends mo.graph op with simpler builders."""

    def __init__(
        self, name: str, input_types: list[Type], output_types: list[Type]
    ):
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
        Block.create_at_start(self.regions[0], input_types)
