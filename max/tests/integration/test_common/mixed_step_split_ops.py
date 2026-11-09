# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Finds the mixed-step split record operands of staged graph ops."""

from __future__ import annotations

from max import mlir

# No other operand of the ops that take the record is a uint32[7].
_RECORD_TYPE = "[7], ui32"


def record_operands(
    op: mlir.Operation | mlir.OpView,
) -> list[mlir.Value[mlir.Type]]:
    """Returns the operands of ``op`` that are a mixed-step split record."""
    return [v for v in op.operands if _RECORD_TYPE in str(v.type)]


def record_source(op: mlir.Operation | mlir.OpView) -> str | None:
    """Returns the op that makes ``op``'s record, ``argument`` for a graph or
    subgraph argument, or ``None`` when ``op`` takes no record."""
    records = record_operands(op)
    if not records:
        return None
    (record,) = records
    return getattr(record.owner, "name", None) or "argument"


def record_sources(op: mlir.Operation | mlir.OpView) -> list[tuple[str, str]]:
    """Returns ``(symbol, source)`` for each custom op at or under ``op`` that
    takes a record, in program order; see :func:`record_source`."""
    found = []
    if op.name == "mo.custom" and "symbol" in op.attributes:
        source = record_source(op)
        if source is not None:
            symbol = mlir.StringAttr(op.attributes["symbol"]).value
            found.append((symbol, source))
    return found + [
        source
        for region in op.regions
        for block in region.blocks
        for child in block.operations
        for source in record_sources(child)
    ]
