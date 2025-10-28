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
from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.norm.rms_norm import RMSNorm


class Gemma3RMSNorm(RMSNorm):
    def __init__(self, dim: int, dtype: DType, eps: float = 1e-6) -> None:
        # Gemma3 uses (1.0 + weight) as the scale factor
        super().__init__(dim=dim, dtype=dtype, eps=eps, weight_offset=1)

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Gemma3RMSNorm]:
        """Creates sharded views of this RMSNorm across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded RMSNorm instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded weights
        weight_shards = self.weight.shard(devices)

        shards = []
        for weight_shard in weight_shards:
            # Create new RMSNorm instance with the same configuration
            sharded = Gemma3RMSNorm(
                dim=self.dim,
                dtype=self.dtype,
                eps=self.eps,
            )

            # Assign the sharded weight
            sharded.weight = weight_shard

            shards.append(sharded)

        return shards
