# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Iterable

from max._mlir.dialects import _ods_common
from max._mlir.dialects.mo import *
from max._mlir.dialects.mo import GraphOp, IfOp, _Dialect

from .. import Attribute, Block, FunctionType, Type, TypeAttr


@_ods_common._cext.register_operation(_Dialect, replace=True)
class GraphOp(GraphOp):  # type: ignore[no-redef]
    """Extends mo.graph op with simpler builders."""

    def __init__(
        self, name: str, input_types: list[Type], output_types: list[Type]
    ):
        function_type = FunctionType.get(input_types, output_types)
        signature = Type.parse(f"!kgen.generator<{function_type}>")
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


@_ods_common._cext.register_operation(_Dialect, replace=True)
class IfOp(IfOp):  # type: ignore[no-redef]
    """Extends mo.if op with simpler builders."""

    def __init__(
        self,
        pred,
        out_types: Iterable[Type] | None,
        loc=None,
        ip=None,
    ):
        if out_types is None:
            out_types = []
        super().__init__(results_=out_types, cond=pred)
        Block.create_at_start(self.thenRegion, [])
        Block.create_at_start(self.elseRegion, [])


def if_(  # type: ignore[no-redef]
    pred,
    out_types,
    loc=None,
    ip=None,
) -> _ods_common.VariadicResultValueT:
    return _ods_common.get_op_result_or_op_results(
        # mypy doesn't see the IfOp definition above, but the one that is replaced
        IfOp(pred=pred, out_types=out_types, loc=loc, ip=ip)  # type: ignore[call-arg]
    )
