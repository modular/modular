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
"""Tests how a data-parallel batch splits among its replicas."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.pipelines.lib.utils import split_replicated_batch
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated, Sharded


def _split(
    row_offsets: list[int], splits: list[int], devices: int, group_size: int = 1
) -> tuple[Tensor, Tensor]:
    """Splits a real batch whose tokens are 0, 1, 2 and so on."""
    mesh = DeviceMesh(tuple(CPU() for _ in range(devices)), (devices,), ("dp",))

    def replicated(array: np.ndarray) -> Tensor:
        return Tensor._from_shards(
            tuple(Buffer.from_numpy(array.copy()) for _ in range(devices)),
            DeviceMapping(mesh, (Replicated(),)),
        )

    offsets = np.array(row_offsets, dtype=np.uint32)
    return split_replicated_batch(
        replicated(np.arange(row_offsets[-1], dtype=np.int64)),
        replicated(offsets),
        Tensor(storage=Buffer.from_numpy(offsets.astype(np.int64))),
        Tensor(storage=Buffer.from_numpy(np.array(splits, dtype=np.int64))),
        DeviceMapping(mesh, (Sharded(0),)),
        group_size=group_size,
    )


def _shards(t: Tensor) -> list[list[int]]:
    return [shard.to_numpy().tolist() for shard in t.local_shards]


def test_each_replica_gets_its_rows() -> None:
    tokens, offsets = _split([0, 2, 5], [0, 1, 2], devices=2)

    assert tokens.real and offsets.real
    assert _shards(tokens) == [[0, 1], [2, 3, 4]]
    assert _shards(offsets) == [[0, 2], [0, 3]]
    assert offsets.local_shards[0].to_numpy().dtype == np.uint32


def test_a_replica_with_no_sequences_gets_no_rows() -> None:
    tokens, offsets = _split([0, 2, 5], [0, 0, 2], devices=2)

    assert _shards(tokens) == [[], [0, 1, 2, 3, 4]]
    assert _shards(offsets) == [[0], [0, 2, 5]]


def test_a_replica_spans_its_group_of_devices() -> None:
    tokens, offsets = _split([0, 2, 5], [0, 1, 2], devices=4, group_size=2)

    assert _shards(tokens) == [[0, 1], [0, 1], [2, 3, 4], [2, 3, 4]]
    assert _shards(offsets) == [[0, 2], [0, 2], [0, 3], [0, 3]]
