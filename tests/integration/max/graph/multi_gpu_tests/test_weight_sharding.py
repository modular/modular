# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests weight sharding API."""

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Weight


def create_sharded_weight_graph() -> Graph:
    sharding_strategy = lambda weight, i: weight[5 * i : 5 * (i + 1)]

    # The Weight is sharded evenly across CPU, GPU 0, and GPU 1.
    # Assuming weight=np.arange(15, dtype=np.float32):
    # Shard 0: [0, 1, 2, 3, 4]
    # Shard 1: [5, 6, 7, 8, 9]
    # Shard 2: [10, 11, 12, 13, 14]
    weight = Weight(
        "weight",
        DType.float32,
        [15],
        DeviceRef.CPU(),
        sharding_strategy=sharding_strategy,
    )
    # Shards must be able to be created outside the graph.
    shard_0 = weight.shard(0, DeviceRef.CPU())  # Keep on CPU.
    assert isinstance(shard_0, Weight)
    shard_1 = weight.shard(1, DeviceRef.GPU(0))  # Move to GPU 0.
    shard_2 = weight.shard(2, DeviceRef.GPU(1))  # Move to GPU 1.

    with Graph("shard_weight", input_types=()) as graph:
        assert shard_0.device == DeviceRef.CPU()
        assert shard_1.device == DeviceRef.GPU(0)
        assert shard_2.device == DeviceRef.GPU(1)
        new_shard_1 = shard_1 * 2
        new_shard_2 = shard_2 + 1

        # Add the modified shards (copy them all to CPU)

        # Shard 0 = [0, 1, 2, 3, 4]
        # Shard 1 = [10, 12, 14, 16, 18]
        # Shard 2 = [11, 12, 13, 14, 15]
        # Expected total should be [21, 25, 29, 33, 37]
        total = (
            shard_0
            + new_shard_1.to(DeviceRef.CPU())
            + new_shard_2.to(DeviceRef.CPU())
        )
        graph.output(total)
    return graph


def test_weight_sharding() -> None:
    """Tests weight sharding onto different devices.."""
    graph = create_sharded_weight_graph()
    # Check built graph
    assert str(DeviceRef.GPU(0)) in str(graph)
    assert str(DeviceRef.GPU(1)) in str(graph)
    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)
    session = InferenceSession(devices=[host, device0, device1])
    compiled = session.load(
        graph, weights_registry={"weight": np.arange(15, dtype=np.float32)}
    )
    # Check Compiled Graph
    assert str(host) == str(compiled.devices[0])
    assert str(device0) == str(compiled.devices[1])
    assert str(device1) == str(compiled.devices[2])

    output = compiled.execute()
    assert isinstance(output[0], Tensor)
    # Check Executed Graph
    assert np.allclose([21, 25, 29, 33, 37], output[0].to_numpy())
