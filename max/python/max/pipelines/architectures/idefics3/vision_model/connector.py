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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Linear, Module

from ..model_config import Idefics3VisionConfig


class Idefics3SimpleMLP(Module):
    """Simple MLP for modality projection in Idefics3 Connector.

    This implements a basic linear projection from vision embedding space
    to text embedding space, without bias terms.
    """

    def __init__(
        self,
        config: Idefics3VisionConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize the simple MLP.

        Args:
            config: Vision configuration object containing scale_factor and text_config_hidden_size.
            dtype: Data type for the weights.
            device: Device to place the weights on.
        """
        super().__init__()
        self.input_size = config.hidden_size * (config.scale_factor**2)
        self.output_size = config.text_config_hidden_size

        self.proj = Linear(
            in_dim=self.input_size,
            out_dim=self.output_size,
            dtype=dtype,
            device=device,
            has_bias=False,  # No bias in the original implementation
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of the simple MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size].

        Returns:
            Output tensor of shape [batch_size, seq_len, output_size].
        """
        return self.proj(x)


class Idefics3Connector(Module):
    """Connector module for Idefics3 that bridges vision and text modalities.

    This module performs two key operations:
    1. Pixel shuffle: Reduces spatial resolution while increasing embedding dimension
    2. Modality projection: Projects vision embeddings to text embedding space

    The pixel shuffle operation reduces the number of image tokens by scale_factor^2
    while preserving information by increasing the embedding dimension proportionally.
    """

    def __init__(
        self,
        config: Idefics3VisionConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize the Idefics3 Connector.

        Args:
            config: Vision configuration object containing scale_factor and text_config_hidden_size.
            dtype: Data type for the weights.
            device: Device to place the weights on.
        """
        super().__init__()
        self.scale_factor = config.scale_factor

        self.modality_projection = Idefics3SimpleMLP(
            config=config,
            dtype=dtype,
            device=device,
        )

    def pixel_shuffle(self, x: TensorValue, scale_factor: int) -> TensorValue:
        """Perform pixel shuffle operation to reduce spatial resolution.

        This operation takes a tensor with shape [batch_size, seq_len, embed_dim]
        where seq_len represents flattened image patches (height * width), and
        reduces the spatial dimensions by scale_factor while increasing the
        embedding dimension by scale_factor^2.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim].
            scale_factor: Factor by which to reduce spatial dimensions.

        Returns:
            Output tensor of shape [batch_size, seq_len/(scale_factor^2), embed_dim*(scale_factor^2)].
        """
        # Get tensor dimensions - extract integer values from Dim objects
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]

        # Assume seq_len is a perfect square (height * width)
        # Use double int conversion to handle StaticDim objects
        # TODO: Double check if this is correct
        height = width = int(int(seq_len) ** 0.5)

        # Reshape to [batch_size, height, width, embed_dim]
        x = ops.reshape(x, [batch_size, height, width, embed_dim])

        # Reshape to [batch_size, height, width/scale_factor, embed_dim * scale_factor]
        new_width = width // scale_factor
        new_embed_dim_1 = embed_dim * scale_factor
        x = ops.reshape(x, [batch_size, height, new_width, new_embed_dim_1])

        # Permute dimensions: [batch_size, width/scale_factor, height, embed_dim * scale_factor]
        # Original: [0, 1, 2, 3] -> [0, 2, 1, 3] means swap axes 1 and 2
        x = ops.transpose(x, 1, 2)

        # Reshape to [batch_size, width/scale_factor, height/scale_factor, embed_dim * scale_factor^2]
        new_height = height // scale_factor
        final_embed_dim = embed_dim * (scale_factor * scale_factor)
        x = ops.reshape(x, [batch_size, new_width, new_height, final_embed_dim])

        # Permute back: [batch_size, height/scale_factor, width/scale_factor, embed_dim * scale_factor^2]
        # Original: [0, 1, 2, 3] -> [0, 2, 1, 3] means swap axes 1 and 2
        x = ops.transpose(x, 1, 2)

        # Reshape to final output: [batch_size, seq_len/(scale_factor^2), embed_dim * scale_factor^2]
        final_seq_len = seq_len // (scale_factor * scale_factor)
        x = ops.reshape(x, [batch_size, final_seq_len, final_embed_dim])

        return x

    def __call__(self, image_hidden_states: TensorValue) -> TensorValue:
        """Forward pass of the Idefics3 Connector.

        Args:
            image_hidden_states: Input tensor from vision encoder with shape
                                [batch_size, seq_len, vision_hidden_size].

        Returns:
            Output tensor with shape [batch_size, seq_len/(scale_factor^2), text_hidden_size].
        """
        # Apply pixel shuffle to reduce spatial resolution and increase embedding dimension
        image_hidden_states = self.pixel_shuffle(
            image_hidden_states, self.scale_factor
        )

        # Project from vision embedding space to text embedding space
        image_hidden_states = self.modality_projection(image_hidden_states)

        return image_hidden_states
