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
"""Test the max.graph Python bindings."""

import re

import pytest
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)


def test_add_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        w2 = Weight(
            "scalar_float",
            dtype=DType.float32,
            shape=[1],
            device=DeviceRef.CPU(),
        )

        graph.output(
            graph.add_weight(w),
            graph.add_weight(w2),
        )
        gen_mlir = str(graph._mlir_op).splitlines()
        # Most recent weight is at the top
        assert re.search(
            r'mo.constant.external.*name = "scalar_float".*!mo.tensor<\[1\], f32',
            gen_mlir[1],
        )
        assert re.search(
            r'mo.constant.external.*name = "random_weight".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[2],
        )


def test_add_weights_with_sum() -> None:
    """Tests adding weights with a sum and ensuring that weights are added to the top of the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w1 = Weight(
            "random_weight1",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        w2 = Weight(
            "random_weight2",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        graph.output(
            graph.add_weight(w1) + graph.add_weight(w2),
        )
        gen_mlir = str(graph._mlir_op).splitlines()
        # Most recent weight is at the top
        assert re.search(
            r'mo.constant.external.*name = "random_weight2".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[1],
        )
        assert re.search(
            r'mo.constant.external.*name = "random_weight1".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[2],
        )


def test_add_same_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
        value = graph.add_weight(w)

        # TODO(...): Make it return the exact same value
        # Adding the same Weight is fine, and should return a similar value
        value2 = graph.add_weight(w)
        assert value.type == value2.type

        # Test that adding a different Weight with the same name fails.
        w2 = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())

        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(w2)


def test_weight_is_value_like() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
        constant = ops.constant(1, DType.float32, device=DeviceRef.CPU())
        graph.output(constant + w)
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*!mo.tensor<\[\], f32", gen_mlir
        )


def test_weight_outside_graph_error() -> None:
    w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
    with pytest.raises(ValueError, match="no parent graph"):
        _ = w * 5

    with pytest.raises(ValueError, match="no parent graph"):
        _ = ops.cast(w, DType.float64)


def test_weight_is_placeholder() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "w",
            dtype=DType.float32,
            shape=[],
            device=DeviceRef.CPU(),
            _placeholder=True,
        )
        graph.output(w)
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*isPlaceholder = true.*!mo.tensor<\[\], f32",
            gen_mlir,
        )


def test_weight_has_alias() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "w",
            dtype=DType.float32,
            shape=[],
            device=DeviceRef.CPU(),
            _has_alias=True,
        )
        graph.output(w)
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*hasAlias = true.*!mo.tensor<\[\], f32",
            gen_mlir,
        )


def _replicated_shard(name: str = "w") -> Weight:
    w = Weight(name, dtype=DType.float32, shape=[8, 8], device=DeviceRef.CPU())
    w.sharding_strategy = ShardingStrategy.replicate(1)
    return w.shard([DeviceRef.CPU()])[0]


def _matmul_on_side_stream(x: TensorValue, shard: Weight) -> TensorValue:
    return ops.side_stream(
        [x], lambda a: [a @ shard], result_types=[x.type], stream_id=1
    )[0]


def test_weight_shard_staged_once_per_block() -> None:
    """A shard reused within one block yields one value, staging no more ops."""
    with Graph("shard_reuse", input_types=()) as graph:
        shard = _replicated_shard()
        first = shard._mlir_value
        num_ops = len(list(graph._current_block.operations))

        assert all(shard._mlir_value == first for _ in range(10))
        assert len(list(graph._current_block.operations)) == num_ops

        graph.output(shard)


def test_weight_shard_restaged_inside_region() -> None:
    """A shard staged in the enclosing block is restaged inside a region.

    The outer value would have dominated the region, but a ``Weight`` can only
    compare blocks, not walk to a block's parent, so it cannot tell.
    """
    input_type = TensorType(DType.float32, [4, 8], device=DeviceRef.CPU())
    with Graph("shard_capture", input_types=(input_type,)) as graph:
        (x,) = graph.inputs
        shard = _replicated_shard()
        outer = shard._mlir_value

        in_region = []

        def body(a: TensorValue) -> list[TensorValue]:
            in_region.append(shard._mlir_value)
            return [a @ shard]

        ops.side_stream(
            [x.tensor], body, result_types=[input_type], stream_id=1
        )

        assert in_region != [outer]
        # The enclosing block keeps its own value.
        assert shard._mlir_value == outer

        graph.output()


def test_weight_shard_restaged_per_region() -> None:
    """A shard first used inside a region is restaged for every later use.

    Handing a sibling region the value staged inside the first one would fail
    verification, since it dominates neither.
    """
    input_type = TensorType(DType.float32, [4, 8], device=DeviceRef.CPU())
    with Graph("shard_per_region", input_types=(input_type,)) as graph:
        (x,) = graph.inputs
        shard = _replicated_shard()

        y = _matmul_on_side_stream(x.tensor, shard)
        graph.output(_matmul_on_side_stream(y, shard))

    assert str(graph._mlir_op).count("rmo.slice") == 2


def test_weight_reuse_does_not_restage_transfer() -> None:
    """Reusing a device weight must not stage a host-to-device copy per access.

    ``Graph.add_weight`` re-runs ``.to(device)`` on every cache hit, so without
    a cached value the graph would grow with each use.
    """
    with Graph("weight_reuse", input_types=()) as graph:
        w = Weight(
            "w", dtype=DType.float32, shape=[8, 8], device=DeviceRef.GPU(0)
        )
        _ = w._mlir_value
        num_ops = len(list(graph._current_block.operations))

        for _ in range(10):
            _ = w._mlir_value
        assert len(list(graph._current_block.operations)) == num_ops

        graph.output(w)


def test_unsharded_weight_shared_across_regions() -> None:
    """An unsharded weight is safe in any region, and needs no restaging.

    ``Graph.add_weight`` inserts the ``mo.constant.external`` at the start of
    the graph body and moves the device transfer next to it, so both sit in the
    block that encloses every region and one value dominates every use.
    """
    input_type = TensorType(DType.float32, [8, 8], device=DeviceRef.GPU(0))
    with Graph("weight_regions", input_types=(input_type,)) as graph:
        (x,) = graph.inputs
        w = Weight(
            "w", dtype=DType.float32, shape=[8, 8], device=DeviceRef.GPU(0)
        )

        y = _matmul_on_side_stream(x.tensor, w)
        graph.output(_matmul_on_side_stream(y, w))

    gen_mlir = str(graph._mlir_op)
    assert gen_mlir.count("mo.constant.external") == 1
    assert gen_mlir.count("mo.transfer") == 1
