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
"""Tests which values each device receives from call_on_mesh, on a simulated CPU mesh."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from max.driver import CPU
from max.experimental import functional as F
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    ShardingError,
)
from max.experimental.sharding.action import PerShard
from max.experimental.tensor import Tensor
from max.graph import Shape

MESH_2 = DeviceMesh(devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",))
MESH_2X2 = DeviceMesh(
    devices=(CPU(), CPU(), CPU(), CPU()),
    mesh_shape=(2, 2),
    axis_names=("dp", "tp"),
)


def _sharded_rows(rows: int, mesh: DeviceMesh = MESH_2) -> Tensor:
    data = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    return transfer_to(Tensor(data), DeviceMapping(mesh, (Sharded(0),)))


def test_passes_each_device_its_values_anywhere_in_the_arguments() -> None:
    x = _sharded_rows(4)
    host = Tensor(np.zeros(1, dtype=np.float32))
    seen: list[dict[str, Any]] = []

    def record(pair: tuple[Tensor, str], *, host: Tensor, scale: int) -> Tensor:
        seen.append({"shard": pair[0], "label": pair[1], "host": host})
        assert scale == 3
        return pair[0]

    out = F.call_on_mesh(
        record, MESH_2, out_specs=DeviceMapping(MESH_2, (Sharded(0),))
    )((x, PerShard(["first", "second"])), host=host, scale=3)

    assert [entry["label"] for entry in seen] == ["first", "second"]
    for entry, shard in zip(seen, x.local_shards, strict=True):
        np.testing.assert_array_equal(
            entry["shard"].to_numpy(), shard.to_numpy()
        )
        assert entry["host"] is host
    assert out.placements == (Sharded(0),)
    np.testing.assert_array_equal(out.to_numpy(), x.to_numpy())


def test_passes_each_device_its_own_shape() -> None:
    x = _sharded_rows(3)
    local_shapes: list[Shape] = []

    def record(shard: Tensor, shape: Shape) -> Tensor:
        local_shapes.append(shape)
        return shard

    F.call_on_mesh(record, MESH_2)(x, x.shape)

    # Three rows split unevenly, so the devices see different sizes.
    assert local_shapes[0] != local_shapes[1]
    assert local_shapes == [Shape(s.shape) for s in x.local_shards]


@pytest.mark.parametrize(
    ("mesh_axes", "expected_groups"),
    [
        (("tp",), [[0.0, 1.0], [2.0, 3.0]]),
        (("dp",), [[0.0, 2.0], [1.0, 3.0]]),
        (("dp", "tp"), [[0.0, 1.0, 2.0, 3.0]]),
    ],
)
def test_groups_the_devices_along_the_mesh_axes(
    mesh_axes: tuple[str, ...], expected_groups: list[list[float]]
) -> None:
    # Device (i, j) holds the element at row i, column j.
    x = transfer_to(
        Tensor(np.arange(4, dtype=np.float32).reshape(2, 2)),
        DeviceMapping(MESH_2X2, (Sharded(0), Sharded(1))),
    )
    groups: list[list[float]] = []

    def record(shards: list[Tensor]) -> list[Tensor]:
        groups.append([float(s.to_numpy().item()) for s in shards])
        return shards

    out = F.call_on_mesh(record, MESH_2X2, mesh_axes, out_specs=x.mapping)(x)

    assert groups == expected_groups
    np.testing.assert_array_equal(out.to_numpy(), x.to_numpy())


def test_passes_one_tensor_given_twice_as_the_same_shards() -> None:
    x = _sharded_rows(4)

    def same(first: Tensor, second: Tensor) -> Tensor:
        assert first is second
        return first

    F.call_on_mesh(same, MESH_2)(x, x)


def test_rejects_a_group_returning_too_few_results() -> None:
    x = _sharded_rows(4)
    with pytest.raises(ShardingError, match="returned 1 results"):
        F.call_on_mesh(lambda shards: shards[:1], MESH_2, ("tp",))(x)


def test_rejects_a_per_device_value_of_another_length() -> None:
    with pytest.raises(ShardingError, match="holds 3 values"):
        F.call_on_mesh(lambda label: None, MESH_2)(PerShard(["a", "b", "c"]))


@pytest.mark.parametrize(
    "mesh",
    [
        MESH_2X2,
        DeviceMesh(devices=(CPU(),), mesh_shape=(1,), axis_names=("tp",)),
    ],
    ids=["other_shape", "one_device"],
)
def test_rejects_a_distributed_input_on_another_mesh_shape(
    mesh: DeviceMesh,
) -> None:
    x = _sharded_rows(4)
    with pytest.raises(ShardingError, match="meshes of one shape"):
        F.call_on_mesh(lambda shard: shard, mesh)(x)


def test_rejects_out_specs_that_do_not_match_the_results() -> None:
    x = _sharded_rows(4)
    replicated = DeviceMapping(MESH_2, (Replicated(),))
    with pytest.raises(ShardingError, match="2 mappings for 1 tensor results"):
        F.call_on_mesh(
            lambda shard: shard, MESH_2, out_specs=[replicated, replicated]
        )(x)
