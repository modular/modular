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
"""Tests weight sharding API."""

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, Weight


def create_sharded_weight_graph() -> Graph:
    num_devices = 5
    sharding_strategy = lambda weight, i, n: weight[n * i : n * (i + 1)]

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
        sharding_strategy=ShardingStrategy(num_devices, sharding_strategy),
    )

    devices = [DeviceRef.CPU(), DeviceRef.GPU(0), DeviceRef.GPU(1)]

    # Shards must be able to be created outside the graph.
    shards = weight.shard(devices)
    shard_0, shard_1, shard_2 = shards
    assert isinstance(shards[0], Weight)
    assert isinstance(shards[1], Weight)
    assert isinstance(shards[2], Weight)

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
