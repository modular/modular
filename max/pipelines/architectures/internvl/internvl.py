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
"""Implements the InternVL multimodal model."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import DeviceRef, TensorValue, Weight, ops
from max.graph.ops.resize import InterpolationMode
from max.graph.type import Dim, StaticDim
from max.nn import Conv2D, Module
from max.pipelines.architectures.llama3.distributed_llama import (
    DistributedLlama3,
)

from .model_config import InternVLConfig


class InternVLLanguageModel(DistributedLlama3):
    """The InternVL language model for text generation with image embeddings.

    The model is actually Qwen 2, which in turn has the same architecture as
    Llama 3, but with `attention_bias=True`.
    That config is handled at the callsite in the InternVLPipelineModel.
    """


class InternVisionEmbeddings(Module):
    def __init__(self, config: InternVLConfig, device: DeviceRef) -> None:
        self.device = device
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.dtype = config.vision_config.dtype

        self.patch_embedding = Conv2D(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            dtype=self.dtype,
            device=self.device,
            permute=True,  # Convert from PyTorch weight format
            has_bias=True,  # PyTorch Conv2d has bias by default
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = Weight(
            "class_embedding",
            dtype=self.dtype,
            shape=(1, 1, self.embed_dim),
            device=device,
        )

        self.position_embedding = Weight(
            "position_embedding",
            dtype=self.dtype,
            shape=(1, self.num_positions, self.embed_dim),
            device=device,
        )

    def _get_position_embedding(self, H: Dim, W: Dim) -> TensorValue:
        """Get position embeddings, interpolating if needed for different resolutions.

        Args:
            H: Height in patches (can be int or symbolic Dim)
            W: Width in patches (can be int or symbolic Dim)
        """
        # For static dimensions, check if we need interpolation.
        if isinstance(H, StaticDim) and isinstance(W, StaticDim):
            h_int = int(H)
            w_int = int(W)
            if self.num_patches == h_int * w_int:
                return self.position_embedding

        # Otherwise, interpolate position embeddings.
        # Split class token and patch position embeddings.
        class_pos_embed = self.position_embedding[:, :1, :]
        patch_pos_embed = self.position_embedding[:, 1:, :]

        # Reshape patch position embeddings to spatial layout.
        orig_size = int(self.num_patches**0.5)
        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, orig_size, orig_size, self.embed_dim]
        )

        # Permute to NCHW format for interpolation.
        patch_pos_embed = ops.permute(patch_pos_embed, [0, 3, 1, 2])

        # Interpolate using bicubic.
        # resize expects full shape (N, C, H, W).
        patch_pos_embed = ops.resize(
            patch_pos_embed,
            shape=[1, self.embed_dim, H, W],
            interpolation=InterpolationMode.BICUBIC,
        )

        # Permute back to NHWC and reshape.
        patch_pos_embed = ops.permute(patch_pos_embed, [0, 2, 3, 1])
        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, H * W, self.embed_dim]
        )

        # Concatenate class token and interpolated patch embeddings
        return ops.concat([class_pos_embed, patch_pos_embed], axis=1)

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Compute embeddings for input pixel values.

        Args:
            pixel_values: Input image tensor of shape (batch, channels, height, width).

        Returns:
            Embeddings tensor of shape (batch, num_positions, embed_dim).
        """
        # 1. Apply patch embedding convolution
        pixel_values = pixel_values.cast(self.patch_embedding.filter.dtype)
        patch_embeds = self.patch_embedding(pixel_values)

        # patch_embeds is now (B, C, H, W) where C=embed_dim, H=W=num_patches_per_side
        batch_size = patch_embeds.shape[0]
        height = patch_embeds.shape[2]
        width = patch_embeds.shape[3]

        # 2. Reshape from (B, C, H, W) to (B, H*W, C)
        # First permute to (B, H, W, C)
        patch_embeds = ops.permute(patch_embeds, [0, 2, 3, 1])
        # Then reshape to (B, H*W, C)
        patch_embeds = ops.reshape(
            patch_embeds, [batch_size, height * width, self.embed_dim]
        )

        # 3. Add class token
        class_embeds = self.class_embedding.broadcast_to(
            (batch_size, 1, self.embed_dim)
        )
        embeddings = ops.concat([class_embeds, patch_embeds], axis=1)

        # 4. Add position embeddings
        # Handle both static and symbolic dimensions
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding

        return embeddings


class InternVLVisionModel(Module):
    """The InternVL vision model for processing images."""

    def __init__(self, config: InternVLConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = [
            InternVisionEmbeddings(config, dev) for dev in config.devices
        ]

    def __call__(
        self, pixel_values: Sequence[TensorValue]
    ) -> Sequence[TensorValue]:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values tensor.

        Returns:
            Image embeddings tensor.
        """
        # TODO: need Shardable to enable this.
        # hidden_states = [
        #     # Call one forward per device -- embeddings are replicated.
        #     embed(pixels)
        #     for embed, pixels in zip(self.embeddings, pixel_values)
        # ]
        # return hidden_states

        # TODO: Implement vision encoder
        # 1. Process pixel values through InternViT encoder
        # 2. Apply multimodal projector

        return tuple(
            ops.constant(
                0.0, self.config.llm_config.dtype, device
            ).broadcast_to(
                shape=(
                    pixel_values[0].shape[0],
                    self.config.vision_config.hidden_size,
                )
            )
            for device in self.config.llm_config.devices
        )
