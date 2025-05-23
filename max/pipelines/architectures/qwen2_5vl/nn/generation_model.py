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
from typing import Callable, Union, cast

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn import Linear, ReturnLogits
from max.nn.layer import Module

from .decoder import Qwen2_5VLDecoderTransformer
from .visual_transformer import VisionTransformer


class GenerationModel(Module):
    visual: VisionTransformer
    model: Qwen2_5VLDecoderTransformer
    lm_head: Linear
    dim: int
    n_heads: int
    image_token_id: int
    video_token_id: int
    return_n_logits: int
    embedding_multiplier: float
    logits_postprocessor: Union[Callable[[TensorValue], TensorValue], None]

    def __init__(
        self,
        visual: VisionTransformer,
        model: Qwen2_5VLDecoderTransformer,
        lm_head: Linear,
        dim: int,
        n_heads: int,
        image_token_id: int,
        video_token_id: int,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        embedding_multiplier: float = 1.0,
        logits_postprocessor: Union[
            Callable[[TensorValue], TensorValue], None
        ] = None,
    ):
        super().__init__()
        self.visual = visual
        self.model = model
        self.lm_head = lm_head
        self.dim = dim
        self.n_heads = n_heads
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.return_logits = return_logits
        self.embedding_multiplier = embedding_multiplier
        self.logits_postprocessor = logits_postprocessor

    # copied from nn.Transformer
    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue, ...]
    ) -> tuple[TensorValue, ...]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)

    def __call__(
        self,
        input_ids: TensorValue,
        # vision inputs:
        image_pixel_values: TensorValue,
        image_rot_pos_ids: TensorValue,
        image_window_index: TensorValue,
        image_attention_mask_window: TensorValue,
        image_attention_mask_full: TensorValue,
        image_max_grid_size: int,
        video_pixel_values: TensorValue,
        video_rot_pos_ids: TensorValue,
        video_window_index: TensorValue,
        video_attention_mask_window: TensorValue,
        video_attention_mask_full: TensorValue,
        video_max_grid_size: int,
        # decoder model inputs
        position_ids: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """Generates embeddings for text and vision tokesn separately and passes them for transformer for text-generation.

        Args:
            input_ids: ragged `TensorValue of token ids.
            input_row_offsets: offsets of the tokens in input_ids.
            image_pixel_values: `TensorValue` of all images input to the model split into patches.
            image_rot_pos_ids: `TensorValue` of position ids for patches in image_pixel_values. Used to generate rotary position embeddings in the vision encoder.
            image_attention_mask_window: Attention mask for Sliding Window Attention in the vision encoder to ensure that only patches from the same image in the same window attend to each other.
            image_attention_mask_full: Attention mask for Full Attention in the vision encoder to ensure that only patches from the same image attend to each other.
            image_max_grid_size: int representing max no. of patches in any image along one spatial dimension. Its the max positional embeddings length required.
            video_pixel_values: `TensorValue` of all video input to the model split into patches.
            video_rot_pos_ids: `TensorValue` of position ids for patches in video_pixel_values. Used to generate rotary position embeddings in the vision encoder.
            video_attention_mask_window: Attention mask for Sliding Window Attention in the vision encoder to ensure that only patches from the same frame in the same window attend to each other.
            video_attention_mask_full: Attention mask for Full Attention in the vision encoder to ensure that only patches from the same frame attend to each other.
            video_max_grid_size: int representing max no. of patches in any frame along one spatial dimension. Its the max positional embeddings length required.

        Shapes:
            Inputs:
                input_ids: [total_seq_len]
                image_pixel_values: [n_img_patches, flattened_patch_length]
                image_rot_pos_ids: [n_img_patches, 2]
                image_attention_mask_window: [1, n_img_patches, n_img_patches]
                image_attention_mask_full: [1, n_img_patches, n_img_patches]
                video_pixel_values: [n_vid_patches, flattened_patch_length]
                video_rot_pos_ids: [n_vid_patches, 2]
                video_attention_mask_window: [1, n_vid_patches, n_vid_patches]
                video_attention_mask_full: [1, n_vid_patches, n_vid_patches]
        """
        # Retrieve token embedings for text tokens
        inputs_embeds: TensorValue = self.model.embed_tokens(input_ids)

        # If image tokens exist, replace place-holder image tokens by patch embeddings from vision encoder
        def img_then_fn():
            image_embeds = self.visual(
                pixel_values=image_pixel_values,
                rot_pos_ids=image_rot_pos_ids,
                window_index=image_window_index,
                attention_mask_window=image_attention_mask_window,
                attention_mask_full=image_attention_mask_full,
                max_grid_size=image_max_grid_size,
            )
            image_embeds = ops.cast(image_embeds, inputs_embeds.dtype)

            image_mask = ops.broadcast_to(
                ops.unsqueeze((input_ids == self.image_token_id), -1),
                inputs_embeds.shape,
            )
            return ops.masked_scatter(
                inputs_embeds,
                image_mask,
                image_embeds,
                out_dim="unmasked_inputs",
            )

        def else_fn():
            return inputs_embeds

        inputs_embeds = ops.cond(
            TensorValue(image_pixel_values.shape[0]) > 0,
            [
                TensorType(
                    inputs_embeds.dtype,
                    inputs_embeds.shape,
                    device=inputs_embeds.device,
                )
            ],
            img_then_fn,
            else_fn,
        )[0]

        # If video tokens exist, replace place-holder video tokens by patch embeddings from vision encoder
        def video_then_fn():
            video_embeds = self.visual(
                pixel_values=video_pixel_values,
                rot_pos_ids=video_rot_pos_ids,
                window_index=video_window_index,
                attention_mask_window=video_attention_mask_window,
                attention_mask_full=video_attention_mask_full,
                max_grid_size=video_max_grid_size,
            )
            video_embeds = ops.cast(video_embeds, inputs_embeds.dtype)
            video_mask = ops.broadcast_to(
                ops.unsqueeze((input_ids == self.video_token_id), -1),
                inputs_embeds.shape,
            )
            return ops.masked_scatter(
                inputs_embeds,
                video_mask,
                video_embeds,
                out_dim="unmasked_inputs",
            )

        inputs_embeds = ops.cond(
            TensorValue(video_pixel_values.shape[0]) > 0,
            [
                TensorType(
                    inputs_embeds.dtype,
                    inputs_embeds.shape,
                    device=inputs_embeds.device,
                )
            ],
            video_then_fn,
            else_fn,
        )[0]

        # Pass all token's embeddings through the decoder transformer to generate text.
        h = self.model(inputs_embeds, position_ids, kv_cache_inputs, **kwargs)

        input_row_offsets = kwargs["input_row_offsets"]

        # The rest is copied from nn.Transformer. The only change is lm_head(norm(x)) is lm_head(x) here.
        # Retrieve a variable number of tokens
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(last_h), DType.float32)
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=inputs_embeds.device,
                dtype=DType.int64,
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            last_tokens = ops.gather(h, last_indices, axis=0)
            logits = ops.cast(self.lm_head(last_tokens), DType.float32)
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=inputs_embeds.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(h), DType.float32)
            offsets = cast(TensorValue, kwargs["input_row_offsets"])

        if logits:
            last_logits, logits = self._apply_logits_postprocessor(
                (
                    last_logits,
                    logits,
                )
            )
        else:
            last_logits = self._apply_logits_postprocessor((last_logits,))[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
