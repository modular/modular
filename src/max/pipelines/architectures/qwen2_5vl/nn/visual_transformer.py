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
from functools import cached_property
from typing import Optional

from max.dtype import DType
from max.graph import Dim, TensorValue, TensorValueLike, dtype_promotion, ops
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
class VisionRotaryEmbedding(Module):
    """Rotary embedding layer for the qwen2 vision model.

    Differences compared to `max.nn.RotaryEmbedding`:

    - In _compute_inv_freqs, the head dimension (n) is divided by 2.
    - inv_freqs is cached rather than freqs. generate_rot_pos_embeddings takes seq_len as input because it depends on the image and video inputs
        rather than using a fixed value for the model.
        TODO: We might change this depending on how our infra handles seq_length now.
    """

    dim: int
    n_heads: int
    theta: float
    """The maximum sequence length for model's input."""
    _inv_freqs: Optional[TensorValueLike] = None

    def __post_init__(self):
        super().__init__()

    def _compute_inv_freqs(self) -> TensorValue:
        if self._inv_freqs is None:
            n = (self.dim // self.n_heads) // 2
            # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
            iota = ops.range(
                ops.constant(0, DType.float64),
                ops.constant(n - 1, DType.float64),
                ops.constant(2, DType.float64),
                out_dim=n // 2,
            )
            inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
            self._inv_freqs = inv_freq
        return TensorValue(self._inv_freqs)

    @cached_property
    def inv_freqs(self) -> TensorValue:
        self._inv_freqs = self._compute_inv_freqs()
        return self._inv_freqs

    def generate_rot_pos_embeddings(
        self,
        rot_pos_ids: TensorValue,
        window_index: TensorValue,
        spatial_merge_unit: int,
        max_grid_size: int,
        seq_len: Dim,
    ) -> tuple[TensorValue, TensorValue]:
        # Generate rot_embs assuming full grid.
        t = ops.range(
            ops.constant(0, DType.float64),
            ops.constant(max_grid_size, DType.float64),
            ops.constant(1, DType.float64),
            out_dim=max_grid_size,
        )
        rotary_pos_emb_full = ops.outer(t, self.inv_freqs)
        # Retrieve position embeddings for ieach patch in nput images or videos.
        rotary_pos_emb = ops.gather(rotary_pos_emb_full, rot_pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            [seq_len // spatial_merge_unit, spatial_merge_unit, -1]
        )

        # Reorders rot position embeddings according to window_index indices.
        rotary_pos_emb = ops.gather(
            rotary_pos_emb, window_index, axis=0
        ).reshape([seq_len, -1])
        # Generates a cos and a sin of rotary posiiton embeddings which will be applied later.
        rotary_pos_emb = ops.concat(
            (rotary_pos_emb, rotary_pos_emb), -1
        )  # (seq_len, 2 * hidden_size).
        position_embeddings = (ops.cos(rotary_pos_emb), ops.sin(rotary_pos_emb))
        return position_embeddings

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        raise NotImplementedError


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
    rotary_pos_emb: VisionRotaryEmbedding
    spatial_merge_unit: int

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        x: TensorValue,
        grid_thw: TensorValue,
        rot_pos_ids: TensorValue,
        max_grid_size: int,
        window_index: TensorValue,
        cu_window_seqlens: TensorValue,
    ) -> TensorValue:
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
            rotary_pos_ids: Tensor of shape (seq_len, 2) generated by data_processing.mrope_pos_ids_3d(grid_thw, spatial_merge_size).
            max_grid_size: int representing the maximum spatial dimension in grid_thw
            window_index:  1D TensorValue generated by data_processing.get_window_index(grid_thw, window_size, spatial_merge_size, patch_size, spatial_merge_unit).
            cu_window_seqlens: 1D TensorValue generated by data_processing.get_window_index(grid_thw, window_size, spatial_merge_size, patch_size, spatial_merge_unit).

        Returns:
            TensorValue : hidden_states.
        """
        # The input image is passed through the ViT to obtain patch embeddings.
        h = self.patch_embed(x)  # Shape = [seq_len, hidden_size]
        seq_len = h.shape[0]
        # Reshape into a 3D tensor of blocks.
        h = h.reshape(
            [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1]
        )
        # Reorders patch_embeddings according to window_index indices.
        h = ops.gather(h, window_index, axis=0).reshape([seq_len, -1])

        # Compute rotary positional encodings to the input patches.
        position_embeddings = self.rotary_pos_emb.generate_rot_pos_embeddings(
            rot_pos_ids,
            window_index,
            self.spatial_merge_unit,
            max_grid_size,
            seq_len,
        )

        # Calculate the Cumulative Sequence Lengths (number of tokens in each grid for each dim).
        # TODO(GEX-1938): Implement ops.repeat_interleave with a vector of repeats
        # cu_seqlens = ops.cumsum(
        #     ops.repeat_interleave(
        #         grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        #     ),
        #     0,
        # )
        # TODO(GEX-1861): Implement padding
        # cu_seqlens = ops.pad(cu_seqlens, (1, 0), value=0)

        cu_window_seqlens = ops.cast(cu_window_seqlens, grid_thw.dtype)
        # TODO(GEX-1923): Implement ops.unique_consecutive and uncomment this
        # cu_window_seqlens = ops.unique_consecutive(cu_window_seqlens)

        return position_embeddings[0]
