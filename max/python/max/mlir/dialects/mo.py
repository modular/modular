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

from __future__ import annotations

from collections.abc import Iterable

from max._mlir.dialects import _ods_common
from max._mlir.dialects.mo import *
from max._mlir.dialects.mo import GraphOp, IfOp, _Dialect

from .. import Attribute, Block, FunctionType, Type, TypeAttr


@_ods_common._cext.register_operation(_Dialect, replace=True)
class GraphOp(GraphOp):  # type: ignore[no-redef, operator]
    """Extends mo.graph op with simpler builders."""

    def __init__(
        self, name: str, input_types: list[Type], output_types: list[Type]
    ) -> None:
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
class IfOp(IfOp):  # type: ignore[no-redef, operator]
    """Extends mo.if op with simpler builders."""

    def __init__(
        self,
        pred,  # noqa: ANN001
        out_types: Iterable[Type] | None,
        loc=None,  # noqa: ANN001
        ip=None,  # noqa: ANN001
    ) -> None:
        if out_types is None:
            out_types = []
        super().__init__(results_=out_types, cond=pred)
        Block.create_at_start(self.thenRegion, [])
        Block.create_at_start(self.elseRegion, [])


def if_(  # type: ignore[no-redef]
    pred,  # noqa: ANN001
    out_types,  # noqa: ANN001
    loc=None,  # noqa: ANN001
    ip=None,  # noqa: ANN001
) -> _ods_common.VariadicResultValueT:
    return _ods_common.get_op_result_or_op_results(
        # mypy doesn't see the IfOp definition above, but the one that is replaced
        IfOp(pred=pred, out_types=out_types, loc=loc, ip=ip)  # type: ignore[call-arg]
    )


@_ods_common._cext.register_operation(_Dialect, replace=True)
class WhileOp(WhileOp):  # type: ignore[no-redef, operator]
    """Extends mo.while op with simpler builders."""

    def __init__(self, results_, inputs, *, loc=None, ip=None) -> None:  # noqa: ANN001
        if results_ is None:
            results_ = []
        super().__init__(results_=results_, inputs=inputs, loc=loc, ip=ip)
        Block.create_at_start(self.condRegion, results_)
        Block.create_at_start(self.bodyRegion, results_)


def while_(  # type: ignore[no-redef]
    results_,  # noqa: ANN001
    inputs,  # noqa: ANN001
    *,
    loc=None,  # noqa: ANN001
    ip=None,  # noqa: ANN001
) -> _ods_common.VariadicResultValueT:
    return _ods_common.get_op_result_or_op_results(
        WhileOp(results_=results_, inputs=inputs, loc=loc, ip=ip)
    )


def call_(
    symbol,  # noqa: ANN001
    results,  # noqa: ANN001
    operands,  # noqa: ANN001
    *,
    loc=None,  # noqa: ANN001
    ip=None,  # noqa: ANN001
    prefix: str = "",
) -> _ods_common.VariadicResultValueT:
    if results is None:
        results = []
    call_op = CallOp(
        symbol=symbol,
        results_=results,
        operands_=operands,
        loc=loc,
        prefix=prefix,
        ip=ip,
    )

    return _ods_common.get_op_result_or_op_results(call_op)
