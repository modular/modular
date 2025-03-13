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

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, dtype_promotion, ops
from max.nn import Conv3D
from max.nn.layer import Module


@dataclass
class VisionPatchEmbed(Module):
    proj: Conv3D
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    embed_dim: int = 1152

    def __post_init__(self):
        super().__init__()

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """
        Reshapes input to (batch_size, in_channels, depth, height, width).
        Permutes it to be compatible with max.pipelines.nn.Conv3D input tensor.
        Permutes the output then flatten it.

        Args:
            x: tensor representing pixel values of shape [resized_height, resized_width].

        Returns:
            a tensor of size (seq_len, hidden_size = embed_dim)
        """
        x, filter = dtype_promotion._promote_weak_dtypes(x, self.proj.filter)
        x = x.cast(filter.dtype)
        x = x.reshape(
            (
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
        )
        # Permute (batch_size, in_channels, depth, height, width) inputs to (batch_size, depth, height, width, in_channels) for our Graph API.
        x = x.permute([0, 2, 3, 4, 1])
        x = self.proj(x)
        # Permute max output from (batch_size, depth, height, width, out_channels) to (batch_size, out_channels, depth, height, width)
        x = x.permute([0, 2, 3, 4, 1])
        return x.reshape((-1, self.embed_dim))


@dataclass
class VisionTransformer(Module):
    """The bare Qwen2.5VL Vision Transformer (a redesigned Vision Transformer (ViT))
    outputting raw hidden-states without any specific head on top.

    Its a native dynamic-resolution Vision Transformer that incorporates Window Attention.
    It reduces computational overhead while maintaining native resolution. It incorporates
    2D-RoPE and window attention to support native input resolutions while accelerating
    the computation of the entire visual encoder.

    The height and width of the input images are resized to multiples of 28 before being fed into the ViT.

    This module processes images by splitting them into patches with a stride of 14,
    generating a set of image features (embeddings).

    Then, it groups spatially adjacent sets of four patch features, then concatenate
    and pass through a two-layer (MLP) instead of directly using the raw patch embeds.
    The output of ViT is passed through a spatial merging operation that reduces the
    spatial dimensions of the output, controlled by the spatial_merge_size parameter.
    After that, A linear layer projects the merged features into a space compatible with
    the language model's embeddings.

    Qwen2.5 ViT efficiently process visual information and seamlessly integrate it with
    language models, enabling a wide range of multimodal tasks.
    """

    patch_embed: VisionPatchEmbed

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        x: TensorValue,
        grid_thw: TensorValue,
    ):
        """Outputs raw hidden states of the transformer model on input `x`.

        1. Patch Embedding: Converts raw input into patches and embeds them.
        2. Rotary Positional Embeddings: Computes and applies rotary positional encodings to the patches.
        3. Windowing: Divides the sequence into windows and performs attention within those windows using the window_index.
        4. Transformer Processing: Processes the sequence through multiple transformer blocks, with attention to cumulative sequence lengths and positional encodings.
        5. Merging and Sorting: After transformer processing, the results are merged and sorted to restore the original sequence order.
        6. Final Output: The processed hidden_states are returned as the model's output.

        Args:
            x: Tensor of images of shape (seq_len=n_patches, in_channels * temporal_patch_size * patch_size * patch_size)
            seq_len depends on the spatial dims of the image or video and second dim of x for Qwen2.5VL is 1176.
            Qwen2.5VL processor that handles multiple images of different shapes by flattening all dims and returning
            a 2D tensor of all patches in all images + a grid_thw representing the coords of patches.

            grid_thw: a tensor of temporal, height and width of feature shape of each image in LLM.
                Its the represent the dimensions of patches in images. Shape = (num_images_or_videos, 3).

        Returns:
            TensorValue : hidden_states.
        """
        # The input image is passed through the ViT to obtain patch embeddings.
        h = self.patch_embed(x)  # Shape = [seq_len, hidden_size]

        casted_output = ops.cast(h, DType.float32)
