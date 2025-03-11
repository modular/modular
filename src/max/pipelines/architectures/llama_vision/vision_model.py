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

"""Llama 3.2 Transformer Vision Model."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import Dim, StaticDim, TensorValue, TensorValueLike, ops
from max.graph.weights import Weights
from max.nn import Conv2D, Embedding, LayerNorm, Linear
from max.nn.layer import Layer

from .attention import Attention
from .encoder import VisionEncoder, VisionEncoderLayer
from .mlp import MLP
from .positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)


# TODO(MAXCORE-170): We should clean this up. This is just a RSCF layout permutation so it
# conforms with our conv op API.
@dataclass
class VisionConv2D(Conv2D):
    def __call__(self, x: TensorValue) -> TensorValue:
        # Permute first before calling the parent forward pass.
        self.filter = ops.permute(self.filter, [2, 3, 1, 0])
        return super().__call__(x=x)


@dataclass
class VisionModel(Layer):
    """
    Llama 3.2 vision model responsible for encoding images. It consists of two
    vision encoders.

    This model is designed to process input images through a combination of convolutional
    layers and transformer-based encoders. It utilizes gated and precomputed positional
    embeddings to handle spatial information effectively, and supports multi-aspect ratio inputs.

    Args:
        params : Hyperparameters that define the architecture and training behavior of the vision model.
        gated_positional_embedding: Precomputed positional embeddings that are gated for enhanced spatial encoding.
        pre_tile_positional_embedding: Precomputed aspect ratio positional embeddings applied before tiling the input patches.
        post_tile_positional_embedding: Precomputed aspect ratio positional embeddings applied after tiling the input patches.
        patch_embedding: Convolutional layer that extracts features from input image patches.
        class_embedding: Embedding that is concatenated to the sequence for classification tasks.
        layernorm_pre: Layer normalization applied before feeding inputs into the transformer encoders.
        layernorm_post: Layer normalization applied after processing through the transformer layers.
        transformer: Transformer responsible for capturing local spatial relationships in the image.
        global_transformer: Transformer focused on global context and capturing long-range dependencies within the image.
    """

    gated_positional_embedding: PrecomputedPositionEmbedding
    pre_tile_positional_embedding: PrecomputedAspectRatioEmbedding
    post_tile_positional_embedding: PrecomputedAspectRatioEmbedding
    patch_embedding: VisionConv2D
    class_embedding: TensorValueLike
    layernorm_pre: LayerNorm
    layernorm_post: LayerNorm
    transformer: VisionEncoder
    global_transformer: VisionEncoder
    dtype: DType
    intermediate_layers_indices: list[int]
    num_patches: int

    def apply_class_embedding(self, hidden_state: TensorValue) -> TensorValue:
        """
        Adds a learnable class token embedding to the sequence of patch embeddings for a vision transformer.

        This function is responsible for prepending a class token to the sequence of image patch embeddings.
        The class token is a learnable parameter that captures global information from the image through
        the self-attention mechanism. After processing through the transformer layers, the class token
        serves as a summary representation of the entire image, typically used for classification tasks.

        Args:
            embedding_sequence (TensorValue): A tensor representing the sequence of embedded image patches.
                Shape: (batch_size, num_patches, embedding_dim)

        Returns:
            TensorValue: A tensor with the class token prepended to the sequence of patch embeddings.
                Shape: (batch_size, num_patches + 1, embedding_dim)
                The first token in the sequence is the class token, followed by the image patch embeddings.

        Example:
            >>> class_token = model.apply_class_embedding(patch_embeddings)
            >>> # class_token now holds the class embedding prepended to the patch embeddings
        """
        batch_size, _, hidden_size = hidden_state.shape
        # This was a reshape in torch reference implementation but we need to
        # broadcast this into the right shapes.
        class_embedding = TensorValue(self.class_embedding)

        class_embedding = class_embedding.broadcast_to(
            (batch_size, 1, hidden_size)
        )
        return ops.concat((class_embedding, hidden_state), axis=1)

    def _prepare_aspect_ratio_attention_mask(
        self,
        aspect_ratio_mask: TensorValue,
        num_patches: int,
        target_length: int,
        dtype: DType,
    ) -> TensorValue:
        # Expand aspect ratio mask to target_length
        batch_size, max_num_tiles = aspect_ratio_mask.shape
        attention_mask = aspect_ratio_mask.reshape(
            (batch_size, max_num_tiles, 1, 1)
        ).cast(dtype)  # (1, 4, 1, 1)
        # attention_shape (1, 4, 1, 1) -> (1, 4, 1032, 1)
        attention_mask = ops.tile(attention_mask, (1, 1, target_length, 1))

        # Mask padding patches
        pad_patches = target_length - num_patches

        # The snippet below is a workaround for
        # attention_mask[:, :, 0 - pad_patches :] = 0
        valid_mask = attention_mask[:, :, :-pad_patches, :]
        zero_pad = ops.constant(0, dtype).broadcast_to(
            (batch_size, max_num_tiles, pad_patches, attention_mask.shape[-1])
        )
        attention_mask = ops.concat((valid_mask, zero_pad), axis=2)

        # Invert the mask (0 -> 1, 1 -> 0)
        attention_mask = 1 - attention_mask

        # Reshape to 2D and create 4D attention mask
        # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
        attention_mask = attention_mask.reshape(
            (batch_size, max_num_tiles * target_length, 1)
        )

        # TODO: Hardcoded for now. Reference implementation uses torch.finfo(torch.bfloat16).min
        bfloat_dtype_min_val = -3.3895313892515355e38
        # Perform outer product by broadcasting elementwise multiplication.
        attention_mask = (
            attention_mask
            * attention_mask.reshape(
                (batch_size, 1, max_num_tiles * target_length)
            )
        ) * bfloat_dtype_min_val

        # before unsqueeze: attention_mask shape: (1, 4128, 4128)
        return ops.unsqueeze(attention_mask, axis=1)

    def _manual_constant_pad_4d(
        self,
        dtype: DType,
        input_tensor,
        pad: tuple[int, int, int, int],
        value=0,
    ) -> TensorValue:
        """
        Manually pads a 4D tensor (batch of images) with constant values.

        Args:
            input_tensor (TensorValue): The input 4D tensor (batch_size, channels, height, width).
            pad (tuple): Tuple of the form (left, right, top, bottom) specifying padding sizes.
            value (float): The value to pad with.

        Returns:
            TensorValue: Padded tensor.
        """
        left, right, top, bottom = pad
        batch_size, channels, height, width = input_tensor.shape

        # Compute new height and width after padding
        new_height = height + top + bottom
        new_width = width + left + right

        padded_tensor = ops.constant(value, dtype).broadcast_to(
            (batch_size, channels, new_height, new_width)
        )

        # Insert the original tensor into the center of the padded tensor
        # The code snippet below is a workaround for:
        # padded_tensor[
        #     :, :, top : top + height, left : left + width
        # ] = input_tensor

        # Slice regions along height (dim=2)
        # Unchanged region above
        top_region = padded_tensor[:, :, :top, :]
        # Unchanged region below
        bottom_region = padded_tensor[:, :, top + height.dim :, :]

        # Slice regions along width (dim=3)
        # Unchanged region to the left
        left_region = padded_tensor[:, :, top : top + height.dim, :left]
        width_tuple: tuple[TensorValue, ...] = (left_region, input_tensor)
        if left > 0:
            # Unchanged region to the right
            right_region = padded_tensor[
                :, :, top : top + height.dim, left + width.dim :
            ]
            width_tuple += (right_region,)

        # Concatenate along width (axis=3)
        middle_region = ops.concat(width_tuple, axis=3)

        # Concatenate along height (axis=2)
        updated_padded_tensor = ops.concat(
            (top_region, middle_region, bottom_region), axis=2
        )

        return updated_padded_tensor

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
    ) -> tuple[TensorValue, TensorValue | None, TensorValue | None]:
        (
            batch_size,
            num_concurrent_media,
            num_tiles,
            height,
            width,
            num_channels,
        ) = pixel_values.shape

        # Pixel values isn't always guaranteed to be the same dtype as configured
        # at the pipeline config level. We insert an explicit cast here to account
        # for it.
        pixel_values = ops.cast(pixel_values, self.dtype)

        pixel_values = pixel_values.reshape(
            (
                batch_size * num_concurrent_media * num_tiles,
                height,
                width,
                num_channels,
            )
        )

        aspect_ratio_ids = aspect_ratio_ids.reshape(
            (batch_size * num_concurrent_media, -1)
        )

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)

        # Permute it back to original dim of (4, 1280, 32, 32)
        patch_embeds = patch_embeds.permute([0, 3, 1, 2])

        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape

        hidden_state = hidden_state.reshape(
            (batch_size * num_concurrent_media, num_tiles, -1, dim)
        )

        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )

        # Add cls token
        hidden_state = hidden_state.reshape(
            (batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        )
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            (batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        )
        hidden_state = self.gated_positional_embedding(
            hidden_state, aspect_ratio_ids
        )

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        curr_num_patches = StaticDim(hidden_state.shape[-2]).dim
        num_padding_patches = (8 - (curr_num_patches % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = self._manual_constant_pad_4d(
            dtype=self.dtype,
            input_tensor=hidden_state,
            pad=padding,
            value=0,
        )

        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(
            (batch_size * num_concurrent_media, -1)
        )  # (1, 4)

        attention_mask = self._prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=StaticDim(hidden_state.shape[2]).dim,
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.reshape(
            (batch_size * num_concurrent_media, -1, dim)
        )

        # hidden_state: 1, 4128, 1280
        # attention_mask: 1, 1, 4128, 4128

        hidden_state, all_intermediate_hidden_states = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        assert all_intermediate_hidden_states is not None, (
            "expect intermediate hidden states output"
        )

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            (
                batch_size * num_concurrent_media,  # 1
                num_tiles,  # 4
                num_patches + num_padding_patches,  # 1025 + 7 = 1032
                dim,
            )
        )
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )
        hidden_state = hidden_state.reshape(
            (
                batch_size * num_concurrent_media,
                num_tiles * (num_patches + num_padding_patches),
                dim,
            )
        )

        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_state = global_output[0]

        # Remove padding from hidden state.
        hidden_state = hidden_state.reshape(
            (
                batch_size * num_concurrent_media,  # 1
                num_tiles,  # 4
                num_patches + num_padding_patches,  # 1025 + 7 = 1032
                dim,
            )
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(
            (batch_size, num_concurrent_media, num_tiles, num_patches, dim)
        )

        # Collect intermediate layer outputs from encoder output.
        intermediate_hidden_states = ops.stack(
            all_intermediate_hidden_states, axis=-1
        )

        # These two operations are similar to:
        # `intermediate_hidden_states
        # = intermediate_hidden_states[..., self.intermediate_layers_indices]`
        # We don't currently support slicing based on a provided list of indices
        # yet.
        selected_hidden_states_list = [
            intermediate_hidden_states[:, :, :, idx]
            for idx in self.intermediate_layers_indices
        ]
        intermediate_hidden_states = ops.stack(
            selected_hidden_states_list, axis=-1
        )

        # Remove padding from intermediate hidden states.
        # ('batch_size' * 'num_concurrent_media', 4128, 1280, 5)
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            (
                Dim(batch_size) * num_concurrent_media,  # 1
                num_tiles,  # 4
                num_patches + num_padding_patches,  # 1025 + 7 = 1032
                dim * len(self.intermediate_layers_indices),
            )
        )

        # (1, 4, 1032, 6400) -> (1, 4, 1025, 6400)
        intermediate_hidden_states = intermediate_hidden_states[
            :, :, :slice_index
        ]

        intermediate_hidden_states = intermediate_hidden_states.reshape(
            (
                batch_size,
                num_concurrent_media,
                num_tiles,
                num_patches,
                dim * len(self.intermediate_layers_indices),
            )
        )

        # Concatenate final hidden state and intermediate hidden states.
        hidden_state = ops.concat(
            (hidden_state, intermediate_hidden_states), axis=-1
        )

        # output_attentions: False, output_hidden_states: False in reference
        # implementation, so these are just returned as `None`s.
        return (
            hidden_state,  # "last_hidden_state"
            None,  # "hidden_states"
            None,  # "attentions"
        )


def instantiate_vision_model(
    dtype: DType,
    image_size: int,
    patch_size: int,
    supported_aspect_ratios: list[list[int]],
    hidden_size: int,
    max_num_tiles: int,
    num_channels: int,
    norm_eps: float,
    attention_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    num_global_layers: int,
    intermediate_layers_indices: list[int],
    weights: Weights,
) -> VisionModel:
    # Shared variables.
    num_patches = (image_size // patch_size) ** 2 + 1
    max_aspect_ratio_id = (len(supported_aspect_ratios)) + 1

    gated_positional_embedding = PrecomputedPositionEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        max_num_tiles=max_num_tiles,
        gate=weights.vision_model.gated_positional_embedding.gate.allocate(
            dtype, [1]
        ),
        embedding=weights.vision_model.gated_positional_embedding.embedding.allocate(
            dtype,
            [
                num_patches,
                hidden_size,
            ],
        ),
        tile_embedding=Embedding(
            weights.vision_model.gated_positional_embedding.tile_embedding.weight.allocate(
                dtype,
                [
                    max_aspect_ratio_id,
                    max_num_tiles * num_patches * hidden_size,
                ],
            ),
        ),
    )

    pre_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
        max_num_tiles=max_num_tiles,
        hidden_size=hidden_size,
        gate=weights.vision_model.pre_tile_positional_embedding.gate.allocate(
            dtype, [1]
        ),
        embedding=Embedding(
            weights.vision_model.pre_tile_positional_embedding.embedding.weight.allocate(
                dtype,
                [max_aspect_ratio_id, max_num_tiles * hidden_size],
            ),
        ),
        is_gated=True,
    )

    post_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
        max_num_tiles=max_num_tiles,
        hidden_size=hidden_size,
        gate=weights.vision_model.post_tile_positional_embedding.gate.allocate(
            dtype, [1]
        ),
        embedding=Embedding(
            weights.vision_model.post_tile_positional_embedding.embedding.weight.allocate(
                dtype,
                [
                    max_aspect_ratio_id,
                    max_num_tiles * hidden_size,
                ],
            ),
        ),
        is_gated=True,
    )

    # patch_embedding filter has a shape of (1280, 3, 14, 14).
    patch_embedding = VisionConv2D(
        filter=weights.vision_model.patch_embedding.weight.allocate(
            dtype,
            [hidden_size, num_channels, patch_size, patch_size],
        ),
        stride=patch_size,
        padding=(0, 0, 0, 0),
        bias=None,
    )

    class_embedding = weights.vision_model.class_embedding.allocate(
        dtype,
        [hidden_size],
    )

    layernorm_pre = LayerNorm(
        weights.vision_model.layernorm_pre.weight.allocate(
            dtype, [hidden_size]
        ),
        bias=weights.vision_model.layernorm_pre.bias.allocate(
            dtype, [hidden_size]
        ),
        eps=norm_eps,
    )

    layernorm_post = LayerNorm(
        weights.vision_model.layernorm_post.weight.allocate(
            dtype, [hidden_size]
        ),
        bias=weights.vision_model.layernorm_post.bias.allocate(
            dtype, [hidden_size]
        ),
        eps=norm_eps,
    )

    transformer_encoder_layers: list[VisionEncoderLayer] = []

    head_dim = hidden_size // attention_heads

    for index in range(num_hidden_layers):
        curr_layer_weight = weights.vision_model.transformer.layers[index]
        transformer_encoder_layers.append(
            VisionEncoderLayer(
                mlp=MLP(
                    Linear(
                        curr_layer_weight.mlp.fc1.weight.allocate(
                            dtype,
                            [intermediate_size, hidden_size],
                        ),
                        bias=curr_layer_weight.mlp.fc1.bias.allocate(
                            dtype, [intermediate_size]
                        ),
                    ),
                    Linear(
                        curr_layer_weight.mlp.fc2.weight.allocate(
                            dtype,
                            [hidden_size, intermediate_size],
                        ),
                        bias=curr_layer_weight.mlp.fc2.bias.allocate(
                            dtype, [hidden_size]
                        ),
                    ),
                ),
                input_layernorm=LayerNorm(
                    curr_layer_weight.input_layernorm.weight.allocate(
                        dtype, [hidden_size]
                    ),
                    bias=curr_layer_weight.input_layernorm.bias.allocate(
                        dtype, [hidden_size]
                    ),
                    eps=norm_eps,
                ),
                post_attention_layernorm=LayerNorm(
                    curr_layer_weight.post_attention_layernorm.weight.allocate(
                        dtype, [hidden_size]
                    ),
                    bias=curr_layer_weight.post_attention_layernorm.bias.allocate(
                        dtype, [hidden_size]
                    ),
                    eps=norm_eps,
                ),
                self_attn=Attention(
                    n_heads=attention_heads,
                    head_dim=head_dim,
                    wk=Linear(
                        curr_layer_weight.self_attn.k_proj.weight.allocate(
                            dtype,
                            [attention_heads * head_dim, hidden_size],
                        ),
                        bias=None,
                    ),
                    wv=Linear(
                        curr_layer_weight.self_attn.v_proj.weight.allocate(
                            dtype,
                            [attention_heads * head_dim, hidden_size],
                        ),
                        bias=None,
                    ),
                    wq=Linear(
                        curr_layer_weight.self_attn.q_proj.weight.allocate(
                            dtype,
                            [attention_heads * head_dim, hidden_size],
                        ),
                        bias=None,
                    ),
                    wo=Linear(
                        curr_layer_weight.self_attn.o_proj.weight.allocate(
                            dtype,
                            [hidden_size, attention_heads * head_dim],
                        ),
                        bias=None,
                    ),
                ),
                is_gated=False,
                gate_attn=None,
                gate_ffn=None,
            )
        )
    transformer = VisionEncoder(transformer_encoder_layers)

    global_transformer_layers: list[VisionEncoderLayer] = []

    for index in range(num_global_layers):
        curr_layer_weight = weights.vision_model.global_transformer.layers[
            index
        ]

        global_transformer_layers.append(
            VisionEncoderLayer(
                mlp=MLP(
                    Linear(
                        curr_layer_weight.mlp.fc1.weight.allocate(
                            dtype,
                            [intermediate_size, hidden_size],
                        ),
                        bias=curr_layer_weight.mlp.fc1.bias.allocate(
                            dtype, [intermediate_size]
                        ),
                    ),
                    Linear(
                        curr_layer_weight.mlp.fc2.weight.allocate(
                            dtype,
                            [hidden_size, intermediate_size],
                        ),
                        bias=curr_layer_weight.mlp.fc2.bias.allocate(
                            dtype, [hidden_size]
                        ),
                    ),
                ),
                input_layernorm=LayerNorm(
                    curr_layer_weight.input_layernorm.weight.allocate(
                        dtype, [hidden_size]
                    ),
                    bias=curr_layer_weight.input_layernorm.bias.allocate(
                        dtype, [hidden_size]
                    ),
                    eps=norm_eps,
                ),
                post_attention_layernorm=LayerNorm(
                    curr_layer_weight.post_attention_layernorm.weight.allocate(
                        dtype, [hidden_size]
                    ),
                    curr_layer_weight.post_attention_layernorm.bias.allocate(
                        dtype, [hidden_size]
                    ),
                    eps=norm_eps,
                ),
                self_attn=Attention(
                    n_heads=attention_heads,
                    head_dim=head_dim,
                    wk=Linear(
                        curr_layer_weight.self_attn.k_proj.weight.allocate(
                            dtype,
                            [hidden_size, attention_heads * head_dim],
                        ),
                        bias=None,
                    ),
                    wv=Linear(
                        curr_layer_weight.self_attn.v_proj.weight.allocate(
                            dtype,
                            [hidden_size, attention_heads * head_dim],
                        ),
                        bias=None,
                    ),
                    wq=Linear(
                        curr_layer_weight.self_attn.q_proj.weight.allocate(
                            dtype,
                            [hidden_size, attention_heads * head_dim],
                        ),
                        bias=None,
                    ),
                    wo=Linear(
                        curr_layer_weight.self_attn.o_proj.weight.allocate(
                            dtype,
                            [attention_heads * head_dim, hidden_size],
                        ),
                        bias=None,
                    ),
                ),
                is_gated=True,
                gate_attn=curr_layer_weight.gate_attn.allocate(dtype, [1]),
                gate_ffn=curr_layer_weight.gate_ffn.allocate(dtype, [1]),
            )
        )
    global_transformer = VisionEncoder(global_transformer_layers)

    return VisionModel(
        gated_positional_embedding=gated_positional_embedding,
        pre_tile_positional_embedding=pre_tile_positional_embedding,
        post_tile_positional_embedding=post_tile_positional_embedding,
        patch_embedding=patch_embedding,
        class_embedding=class_embedding,
        layernorm_pre=layernorm_pre,
        layernorm_post=layernorm_post,
        transformer=transformer,
        global_transformer=global_transformer,
        dtype=dtype,
        intermediate_layers_indices=intermediate_layers_indices,
        num_patches=(image_size // patch_size) ** 2 + 1,
    )
