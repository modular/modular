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
"""The surface of the two fused wrappers, pinned without a GPU.

Each wrapper emits one op of its own name and none of the ops it replaces.
That the drafter emits them is pinned by the architecture's fused-graph CPU
test.

The cache is also a mutation chain, and that part is pinned here too: the
graph's consumer of the cache has to read the chain the fused store left.
"""

from __future__ import annotations

import pytest
from max import mlir
from max.graph import Graph
from rope_qkv_store_testlib import (
    FUSED_KV,
    FUSED_QKV,
    SHAPES,
    build_graph,
    params_of,
)

_SHAPE = SHAPES[0]

STORE_OP = {
    FUSED_QKV: "kv_cache.rope_qkv_store.ragged.paged",
    FUSED_KV: "kv_cache.k_rope_v_store.ragged.paged",
}
"""The symbol each wrapper registers its op under."""


def _nested_ops(
    op: mlir.Operation | mlir.OpView,
) -> list[mlir.Operation | mlir.OpView]:
    """Returns ``op`` and every op nested under it, in program order."""
    return [op] + [
        inner
        for region in op.regions
        for block in region.blocks
        for child in block.operations
        for inner in _nested_ops(child)
    ]


def _symbol(op: mlir.Operation | mlir.OpView) -> str:
    """The registered symbol of a custom op, or the empty string."""
    if op.name != "mo.custom":
        return ""
    return mlir.StringAttr(op.attributes["symbol"]).value


def _write_then_read(graph: Graph, store_symbol: str) -> tuple[object, object]:
    """The chain the fused store returns, and the one the cache load reads.

    The cache's consumer is a ``mo.mutable.load`` of its blocks buffer, which
    takes the device chain as its second operand: whatever the write left.
    """
    ops = _nested_ops(graph._mlir_op)
    (store,) = [op for op in ops if _symbol(op) == store_symbol]
    (load,) = [op for op in ops if op.name == "mo.mutable.load"]
    return store.results[-1], load.operands[1]


def test_the_query_wrapper_emits_one_op() -> None:
    fused = str(build_graph(params_of(_SHAPE), _SHAPE, 0.1, mode=FUSED_QKV))
    assert fused.count(STORE_OP[FUSED_QKV]) == 1
    assert "kv_cache.store.paged.ragged" not in fused
    assert "kv_cache.fused_qk_rope.ragged.paged" not in fused


def test_the_queryless_wrapper_emits_one_op() -> None:
    fused = str(build_graph(params_of(_SHAPE), _SHAPE, 0.1, mode=FUSED_KV))
    assert fused.count(STORE_OP[FUSED_KV]) == 1
    assert "kv_cache.store.paged.ragged" not in fused
    assert "kv_cache.fused_qk_rope.ragged.paged" not in fused


@pytest.mark.parametrize("mode", [FUSED_QKV, FUSED_KV])
def test_the_cache_load_reads_the_fused_store(mode: str) -> None:
    """The graph's consumer of the cache is chained after the fused write.

    The store and the load run as one mutation chain on the device, so the
    consumer sees the writes. That is what the GPU parity test next door grades
    against the cache's bytes; on the CPU lane the chain itself is what can be
    read, and a store emitted after the load -- or a load taken from the graph's
    opening chain -- leaves the consumer reading the cache as it was before the
    write.
    """
    graph = build_graph(params_of(_SHAPE), _SHAPE, 0.1, mode=mode)
    written, read = _write_then_read(graph, STORE_OP[mode])
    assert read == written, (
        f"{STORE_OP[mode]} does not precede the cache load in the mutation"
        f" chain: the consumer reads the cache before the write.\n{graph}"
    )
