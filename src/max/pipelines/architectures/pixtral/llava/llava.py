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

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops
from max.nn.layer import Layer
from max.pipelines.architectures.pixtral.vision_encoder.vision_encoder import (
    VisionEncoder,
)

from .llava_decoder import Transformer
from .llava_projector import LlavaMultiModalConnector


@dataclass
class LlavaConditionalGeneration(Layer):
    """The LLAVA model which consists of a vision encoder and a language model.

    image_token_index: a specific token index used to denote images in input_ids.
    """

    vision_encoder: VisionEncoder
    multi_modal_projector: LlavaMultiModalConnector
    language_model: Transformer
    vocab_size: int
    image_token_index: int = 10
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"
    image_seq_length: int = 1

    # TODO: change pixel_values type to List[TensorValue] to support multiple images.
    def __call__(
        self,
        input_ids: TensorValueLike,
        pixel_values: TensorValueLike,
        attention_mask: TensorValueLike,
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
            pixel_values (list`TensorValue` of length batch_size
                The tensors corresponding to the input image are of shape `(image_height, image_width, num_channels)):
                Pixel values can be obtained using ImageProcessor.
        """
        # inputs_embeds shape (total_sequence_length=text_and_image_tokens_length for all seqs,
        #   language_model_hidden_dim)
        inputs_embeds = self.language_model.embedding(input_ids)

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
        inputs_embeds = ops.masked_scatter(
            inputs_embeds, special_image_mask, image_embeds
        )

        logits = self.language_model(inputs_embeds, kv_cache_inputs, **kwargs)

        return logits


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
        inputs_embeds = self.language_model.embedding(input_ids)

        image_embeds = ops.cast(image_embeds, inputs_embeds.dtype)
        special_image_mask = ops.broadcast_to(
            ops.unsqueeze((input_ids == self.image_token_index), -1),
            inputs_embeds.shape,
        )
        inputs_embeds = ops.masked_scatter(
            inputs_embeds, special_image_mask, image_embeds
        )
        logits = self.language_model(inputs_embeds, kv_cache_inputs, **kwargs)

        return logits
