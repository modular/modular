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
"""Inkling short convolution: depthwise causal conv1d with a residual.

Semantics ported from the vLLM reference's ``fused_sconv(activation=None,
use_residual=True)``: depthwise, causal, width 4, no bias, ``x + conv(x)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.layer import Module, Shardable
from max.nn.state_space import causal_conv1d_varlen_fwd


class ShortConvolution(Module, Shardable):
    """Depthwise causal conv1d with a residual, stateful across decode
    steps via a caller-owned conv state pool."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self._sharding_strategy: ShardingStrategy | None = None
        self.weight = Weight(
            "weight", dtype, [channels, 1, kernel_size], device=device
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the channels across devices, or replicates them.

        Replicate suits a caller that convolves per-rank partial sums and
        reduces afterwards: the convolution is linear, so the result matches
        convolving the reduced value.
        """
        if strategy.is_replicate:
            self.weight.sharding_strategy = strategy
            self._sharding_strategy = strategy
            return
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "ShortConvolution supports the tensor parallel and replicate "
                "sharding strategies."
            )
        if self.channels % strategy.num_devices:
            raise ValueError(
                f"{self.channels} convolution channels do not divide over "
                f"{strategy.num_devices} devices"
            )
        self.weight.sharding_strategy = ShardingStrategy.rowwise(
            strategy.num_devices
        )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[ShortConvolution]:
        """Creates one per-device view of this convolution."""
        if self._sharding_strategy is None:
            raise ValueError(
                "ShortConvolution cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        replicated = self._sharding_strategy.is_replicate
        shards = []
        for device, weight in zip(
            devices, self.weight.shard(devices), strict=True
        ):
            sharded = ShortConvolution(
                channels=(
                    self.channels
                    if replicated
                    else self.channels // len(devices)
                ),
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=device,
            )
            sharded.weight = weight
            shards.append(sharded)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_state_pool: BufferValue,
        conv_row: TensorValue,
        input_row_offsets: TensorValue,
        has_initial_state: TensorValue,
    ) -> TensorValue:
        """Returns ``x + conv(x)``; updates ``conv_state_pool`` in place.

        ``has_initial_state`` says whether to read the row's stored history.
        The cache group wipes a fresh request's row and copies a resumed one
        into it before the forward runs, so reading a wiped row is the zero
        padding a first chunk wants.
        """
        device = x.device
        channels, _, kernel_size = self.weight.shape

        # The kernel adds the residual and widens to its own accumulator, so
        # x goes in and comes back at the model dtype.
        return causal_conv1d_varlen_fwd(
            x,
            self.weight.reshape([channels, kernel_size]),
            # No bias tensor at any site; the kernel wants one anyway.
            ops.broadcast_to(
                ops.constant(0.0, x.dtype, device=device), [channels]
            ),
            conv_state_pool,
            ops.cast(input_row_offsets, DType.int32),
            conv_row,
            has_initial_state,
            activation="none",
            channels_last=True,
            use_residual=True,
        )
