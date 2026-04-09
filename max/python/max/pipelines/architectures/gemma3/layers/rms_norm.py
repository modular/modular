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
from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
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


def gemma3_rms_norm_fused_residual_add(
    x: TensorValue,
    residual: TensorValue,
    norm1: Gemma3RMSNorm,
    norm2: Gemma3RMSNorm,
) -> tuple[TensorValue, TensorValue]:
    """Compute norm2(norm1(x) + residual) and return both outputs."""
    input_last_dim = x.shape[-1]

    if input_last_dim != norm1.weight.shape[0]:
        raise ValueError(
            "First RMSNorm weight dimension "
            f"({norm1.weight.shape[0]}) must match the input's last dimension "
            f"({input_last_dim})"
        )
    if input_last_dim != norm2.weight.shape[0]:
        raise ValueError(
            "Second RMSNorm weight dimension "
            f"({norm2.weight.shape[0]}) must match the input's last dimension "
            f"({input_last_dim})"
        )
    if norm1.multiply_before_cast != norm2.multiply_before_cast:
        raise ValueError(
            "Fused Gemma3 RMSNorm requires both norms to share the same "
            "multiply_before_cast setting"
        )

    gamma1: TensorValue = norm1.weight.cast(x.dtype)
    gamma2: TensorValue = norm2.weight.cast(x.dtype)
    if x.device:
        gamma1 = gamma1.to(x.device)
        gamma2 = gamma2.to(x.device)

    results = ops.custom(
        "rms_norm_fused_residual_add",
        x.device,
        [
            x,
            residual,
            gamma1,
            gamma2,
            ops.constant(norm1.eps, dtype=x.dtype, device=DeviceRef.CPU()),
            ops.constant(norm2.eps, dtype=x.dtype, device=DeviceRef.CPU()),
            ops.constant(
                norm1.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
            ),
            ops.constant(
                norm2.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
            ),
        ],
        [
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
        ],
        parameters={
            "multiply_before_cast": norm1.multiply_before_cast,
        },
    )

    return results[0].tensor, results[1].tensor
