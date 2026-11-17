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
"""Splitting a replicated batch into data-parallel shards."""

from __future__ import annotations

from max.graph import DeviceRef, TensorValue
from max.nn._graph_ops.data_parallel import split_batch_replicated
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping


def split_replicated_batch(
    h: Tensor,
    input_row_offsets: Tensor,
    input_row_offsets_i64: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
    group_size: int = 1,
) -> tuple[Tensor, Tensor]:
    """Splits a replicated batch into data-parallel shards.

    ``group_size`` consecutive devices form one replica and take the same
    split: the ``tp`` axis of a ``(dp, tp)`` mesh.
    """
    devices = mapping.mesh.devices

    h_shards, input_row_offsets_shards = split_batch_replicated(
        [DeviceRef.from_device(d) for d in devices],
        [TensorValue(shard) for shard in h.local_shards],
        [TensorValue(shard) for shard in input_row_offsets.local_shards],
        TensorValue(input_row_offsets_i64),
        TensorValue(data_parallel_splits),
        group_size=group_size,
    )
    return (
        Tensor.from_shard_values(h_shards, mapping),
        Tensor.from_shard_values(input_row_offsets_shards, mapping),
    )
