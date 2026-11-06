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

"""Helpers that build and reassemble per-device weight shards."""

from collections.abc import Sequence

from max import functional as F
from max.graph import TensorValue, ops
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

shard_and_stack = F.functional(ops.shard_and_stack)


def stack_device_shards(
    shards: Sequence[Tensor], axis: int, mesh: DeviceMesh
) -> Tensor:
    """Reassembles a per-device weight-shard bundle into one ``Sharded`` tensor."""
    if len(shards) == 1:
        return shards[0]
    mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
    return Tensor.from_shard_values([TensorValue(s) for s in shards], mapping)
