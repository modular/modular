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

from collections.abc import Sequence

from max.dtype.dtype import DType
from max.graph import TensorValue
from max.graph.type import DeviceRef
from max.graph.weight import Weight
from max.nn import Conv2d, Module

from ..model_config import VisionConfig


class PositionEmbeddingModule(Module):
    """Simple module to hold position embedding weight."""

    def __init__(
        self,
        num_positions: int,
        embed_dim: int,
        dtype: DType,
        device: DeviceRef,
    ):
        super().__init__()
        self.weight = Weight(
            name="weight",
            dtype=dtype,
            shape=[num_positions, embed_dim],
            device=device,
        )

    def __call__(self, *args, **kwargs) -> None:
        """Not meant to be called."""
        raise NotImplementedError(
            "PositionEmbeddingModule is a container, use .weight directly"
        )


class VisionEmbeddings(Module):
    """Vision embeddings with patch embedding and position embeddings."""

    def __init__(
        self,
        config: VisionConfig,
        devices: Sequence[DeviceRef],
    ):
        """Initialize vision embeddings.

        Args:
            config: Vision configuration.
            devices: Devices to place the weights on.
        """
        super().__init__()
        self.config = config
        self.devices = devices
        self.device = devices[0] if devices else DeviceRef.CPU()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = Conv2d(
            kernel_size=config.patch_size,
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            stride=config.patch_size,
            padding=0,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
            permute=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = PositionEmbeddingModule(
            num_positions=self.num_positions,
            embed_dim=self.embed_dim,
            dtype=config.dtype,
            device=self.device,
        )

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Forward pass for vision embeddings.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        patch_embeds = self.patch_embedding(pixel_values)

        batch_size = patch_embeds.shape[0]
        embed_dim = patch_embeds.shape[1]
        num_patches_h = patch_embeds.shape[2]
        num_patches_w = patch_embeds.shape[3]

        patch_embeds = patch_embeds.reshape(
            [batch_size, embed_dim, num_patches_h * num_patches_w]
        )

        patch_embeds = patch_embeds.permute([0, 2, 1])
        embeddings = patch_embeds + self.position_embedding.weight

        return embeddings
