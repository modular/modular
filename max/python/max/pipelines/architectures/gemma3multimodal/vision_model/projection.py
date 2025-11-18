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
from typing import Iterable

from max.driver import Tensor
from max.experimental.functional import matmul
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.ops import avg_pool2d
from max.nn import (
    Linear,
    Module,
)
from max.nn.layer import Shardable
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm

from ..model_config import Gemma3ForConditionalGenerationConfig


# ✅ working, based on HF and MLX-VLM
class Gemma3MultiModalProjector(Module, Shardable):
    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None
    ):
        super().__init__()

        self.config = config
        self.device = device if device is not None else config.devices[0]

        self.mm_input_projection_weight = Weight(
            "mm_input_projection_weight",
            dtype=config.dtype,
            shape=(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            device=self.device,
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
            dtype=config.dtype,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )  # 64

        self.tokens_per_side = int(
            config.mm_tokens_per_image**0.5
        )  # 256 ** 05 = 16
        self.kernel_size = (
            self.patches_per_image // self.tokens_per_side
        )  # 64 / 16 = 4

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.mm_input_projection_weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.mm_input_projection_weight.sharding_strategy = strategy
        self.mm_soft_emb_norm.weight.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3MultiModalProjector]:
        assert self.sharding_strategy

        projection_weight_shards = self.mm_input_projection_weight.shard(devices)
        norm_weight_shards = self.mm_soft_emb_norm.weight.shard(devices)

        shards = []
        for device, proj_weight_shard, norm_weight_shard in zip(
            devices,
            projection_weight_shards,
            norm_weight_shards,
            strict=True,
        ):
            sharded = Gemma3MultiModalProjector(self.config, device)

            sharded.mm_input_projection_weight = proj_weight_shard
            sharded.mm_soft_emb_norm.weight = norm_weight_shard

            shards.append(sharded)

        return shards

    def __call__(self, vision_outputs: Tensor) -> TensorValue:
        batch_size, _, seq_length = (
            vision_outputs.shape
        )  # TensorValue shape: ['batch_size', 4096, 1152]

        transposed_vision_outputs = vision_outputs.transpose(1, 2)

        reshaped_vision_outputs = ops.reshape(
            transposed_vision_outputs,
            [
                batch_size,
                seq_length,
                self.patches_per_image,
                self.patches_per_image,
            ]
        )

        # reshape to 0 2 3 1 HWBC
        reshaped_vision_outputs = ops.permute(reshaped_vision_outputs, [0, 2, 3, 1])
        # avg pool
        pooled_vision_outputs = avg_pool2d(
            input=reshaped_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),  # (4,4)
            stride=self.kernel_size,  # 4
        )
        # permute to 0 3 1 2 then flatten from dim 2
        pooled_vision_outputs = ops.permute(pooled_vision_outputs, [0, 3, 1, 2])
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        # transpose 0 2 1
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )

        # Reshape to flatten batch and sequence dimensions for language model
        # From [batch, seq, hidden] to [batch*seq, hidden]
        #image_hidden_states = ops.reshape(
        #    projected_vision_outputs, (-1, projected_vision_outputs.shape[-1])
        #)
        image_hidden_states = ops.flatten(projected_vision_outputs, start_dim=0, end_dim=1) # TODO the only thing different to MLX-VLM/HF

        return image_hidden_states


# ✅ working, based on HF and MLX-VLM
class Gemma3VisionMLP(Module):
    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.vision_config.hidden_size
        self.intermediate_size = config.vision_config.intermediate_size
        self.device = device if device is not None else config.devices[0]

        self.fc1 = Linear(
            self.hidden_size,
            self.intermediate_size,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

        self.fc2 = Linear(
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.fc1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.fc1.sharding_strategy = strategy
        self.fc2.sharding_strategy = strategy

    # ⚠️ from Claude.  looks logical but who would know?
    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionMLP]:
        assert self.sharding_strategy

        fc1_shards = self.fc1.shard(devices)
        fc2_shards = self.fc2.shard(devices)

        shards = []
        for device, fc1_shard, fc2_shard in zip(
            devices,
            fc1_shards,
            fc2_shards,
            strict=True,
        ):
            sharded = Gemma3VisionMLP(
                self.config,
                device
            )

            sharded.fc1 = fc1_shard
            sharded.fc2 = fc2_shard

            shards.append(sharded)

        return shards

    def __call__(self, x: TensorValue):
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x
