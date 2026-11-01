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
"""Tests that the target's fused QKV preparation reaches the spec graph.

The graphs are built on virtual B200s, never compiled, so the test needs no
GPU.
"""

from __future__ import annotations

import pytest
from max import mlir
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.graph import DeviceRef
from mimo_dflash_harness import fused_graph, tiny_model
from tiny_mimo_dflash import Model

FUSED_QKV = "mo.chunked_qkv_rope_store.ragged.paged"


@pytest.fixture(scope="module")
def model() -> Model:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(4)
    # The production drafter's eight KV heads, so the drafter shards four
    # ways.
    return tiny_model(draft_kv_heads=8)


def _count_fused(op: mlir.Operation | mlir.OpView) -> int:
    count = int(
        op.name == "mo.custom"
        and "symbol" in op.attributes
        and mlir.StringAttr(op.attributes["symbol"]).value == FUSED_QKV
    )
    return count + sum(
        _count_fused(child)
        for region in op.regions
        for block in region.blocks
        for child in block.operations
    )


@pytest.mark.parametrize("num_devices", [2, 4])
@pytest.mark.parametrize(
    ("sliding", "full"),
    [(False, False), (True, False), (True, True)],
    ids=["off", "sliding", "sliding-full"],
)
def test_qkv_fusion_reaches_the_spec_graph(
    model: Model, num_devices: int, sliding: bool, full: bool
) -> None:
    graph = fused_graph(
        model,
        [DeviceRef.GPU(i) for i in range(num_devices)],
        7,
        fused_sliding_qkv=sliding,
        fused_full_qkv=full,
    ).graph
    operation = mlir.Operation._CAPICreate(graph.module.mlir_module._CAPIPtr)
    # The tiny target has one sliding layer and two full layers.
    assert _count_fused(operation) == (
        num_devices * (int(sliding) + 2 * int(full))
    )
    assert len(graph.output_types) == 3
