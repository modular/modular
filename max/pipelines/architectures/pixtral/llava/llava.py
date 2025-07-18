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
from dataclasses import dataclass

from max.graph import TensorType, TensorValue, TensorValueLike, ops
from max.nn.layer import Layer, Module
from max.pipelines.architectures.pixtral.vision_encoder.vision_encoder import (
    VisionEncoder,
)

from .llava_decoder import Transformer
from .llava_projector import LlavaMultiModalConnector


class LlavaConditionalGeneration(Module):
    """The LLAVA model which consists of a vision encoder and a language model.

    image_token_index: a specific token index used to denote images in input_ids.
    """

    vision_encoder: VisionEncoder
    multi_modal_projector: LlavaMultiModalConnector
    language_model: Transformer
    image_token_index: int = 10

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        multi_modal_projector: LlavaMultiModalConnector,
        language_model: Transformer,
        image_token_index: int,
    ) -> None:
        self.vision_encoder = vision_encoder
        self.multi_modal_projector = multi_modal_projector
        self.language_model = language_model
        self.image_token_index = image_token_index

    # TODO: change pixel_values type to List[TensorValue] to support multiple images.
    def __call__(
        self,
        input_ids: TensorValue,
        pixel_values: TensorValue,
        attention_mask: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """
        Args:
            input_ids (ragged `TensorValue` of shape `(batch_size * sequence_length for each batch)`):
                Indices of input sequence tokens in the vocabulary. Can be obtained from language model tokenizer.
                input_ids[i] is a sequence of token ids (indices) in sequence i. Expanding inputs for
                image tokens in LLaVa should be done in processing. Each image is represented in the
                input_ids sequence by a sequence of patches that have index(id) = self.image_token_index.
                The maximum number of image tokens in one sequence (prompt) =
                    (input_ids == self.image_token_index).sum(1).max())
                Padding will be ignored by default should you provide it.
            pixel_values (list`TensorValue` of length batch_size
                The tensors corresponding to the input image are of shape `(image_height, image_width, num_channels)):
                Pixel values can be obtained using ImageProcessor.
        """
        # inputs_embeds shape (total_sequence_length=text_and_image_tokens_length for all seqs,
        #   language_model_hidden_dim)
        inputs_embeds: TensorValue = self.language_model.embed_tokens(input_ids)

        # If image tokens exist, replace place-holder image tokens by patch embeddings from vision encoder
        def img_then_fn():
            # Replace image place-holders in inputs_embeds with image embeddings.
            # Obtain image embeddings from the vision encoder and project it to text embeddings space.
            # Output shape = (num_images, num_patches_in_image, language_model_hidden_dim)
            # TODO(AIPIPE-320): If input pixel_values is a list of images, don't wrap it in a list.
            image_embeds = self.multi_modal_projector(
                self.vision_encoder(
                    imgs=[pixel_values], attention_mask=attention_mask
                )
            )
            image_embeds = ops.cast(image_embeds, inputs_embeds.dtype)
            special_image_mask = ops.broadcast_to(
                ops.unsqueeze((input_ids == self.image_token_index), -1),
                inputs_embeds.shape,
            )
            return ops.masked_scatter(
                inputs_embeds,
                special_image_mask,
                image_embeds,
                out_dim="unmasked_inputs",
            )

        def else_fn():
            return inputs_embeds

        # Create a runtime condition by computing the total number of elements in pixel_values
        # This ensures the condition is evaluated at execution time, not compilation time
        inputs_embeds = ops.cond(
            TensorValue(pixel_values.shape[0]) > 0,
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

        return self.language_model(
            embeds=inputs_embeds,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            input_row_offsets=input_row_offsets,
        )


@dataclass
class LlavaVisionEncoder(Layer):
    """The LLAVA model which consists of a vision encoder and a language model.

    image_token_index: a specific token index used to denote images
    """

    vision_encoder: VisionEncoder
    multi_modal_projector: LlavaMultiModalConnector

    # TODO: change pixel_values type to List[TensorValue] to support multiple images.
    def __call__(
        self,
        pixel_values: TensorValueLike,
        attention_mask: TensorValueLike,
        **kwargs,
    ) -> TensorValue:
        """
        Args:
            pixel_values (list`TensorValue` of length batch_size
                The tensors corresponding to the input image are of shape `(image_height, image_width, num_channels)):
                Pixel values can be obtained using ImageProcessor.
        """
        # Replace image place-holders in inputs_embeds with image embeddings.
        # Obtain image embeddings from the vision encoder and project it to text embeddings space.
        # Output shape = (num_images, num_patches_in_image, language_model_hidden_dim)
        # TODO(AIPIPE-320): If input pixel_values is a list of images, don't wrap it in a list.
        return self.multi_modal_projector(
            self.vision_encoder(
                imgs=[pixel_values], attention_mask=attention_mask
            )
        )


@dataclass
class LlavaConditionalGenerationTextOnly(Layer):
    """The LLAVA model which consists of a vision encoder and a language model.

    Because flow control is not added to the graph API yet. We have to create 2 llava models.

    image_token_index: a specific token index used to denote images
    """

    language_model: Transformer
    vocab_size: int
    image_token_index: int = 10

    def __call__(
        self,
        input_ids: TensorValueLike,
        image_embeds: TensorValue,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """
        Args:
            input_ids (ragged `TensorValue` of shape `(batch_size * sequence_length for each batch)`):
                Indices of input sequence tokens in the vocabulary. Can be obtained from language model tokenizer.
                input_ids[i] is a sequence of token ids (indices) in sequence i. Expanding inputs for
                image tokens in LLaVa should be done in processing. Each image is represented in the
                input_ids sequence by a sequence of patches that have index(id) = self.image_token_index.
                The maximum number of image tokens in one sequence (prompt) =
                    (input_ids == self.image_token_index).sum(1).max())
                Padding will be ignored by default should you provide it.
            inputs_embeds:
                Embeddings of pixel_values generated by the vision encoder and projected to the
                embedding space of the language model.
        """
        # inputs_embeds shape (total_sequence_length=text_and_image_tokens_length for all seqs,
        #  language_model_hidden_dim)
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_embeds = ops.cast(image_embeds, inputs_embeds.dtype)
        special_image_mask = ops.broadcast_to(
            ops.unsqueeze((input_ids == self.image_token_index), -1),
            inputs_embeds.shape,
        )
        inputs_embeds = ops.masked_scatter(
            inputs_embeds,
            special_image_mask,
            image_embeds,
            out_dim="unmasked_inputs",
        )
        logits = self.language_model(inputs_embeds, kv_cache_inputs, **kwargs)

        return logits
