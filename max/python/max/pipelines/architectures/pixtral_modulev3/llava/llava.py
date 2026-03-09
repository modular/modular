# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue, ops
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.kv_cache.input_types import unflatten_ragged_attention_inputs
from max.pipelines.architectures.pixtral_modulev3.vision_encoder.vision_encoder import (
    VisionEncoder,
)

from .llava_decoder import Transformer
from .llava_projector import LlavaMultiModalConnector


class LlavaConditionalGeneration(Module[..., tuple[Tensor, ...]]):
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
        kv_params: KVCacheParamInterface | None = None,
    ) -> None:
        self.vision_encoder = vision_encoder
        self.multi_modal_projector = multi_modal_projector
        self.language_model = language_model
        self.image_token_index = image_token_index
        self.kv_params = kv_params

    # TODO: change pixel_values type to List[Tensor] to support multiple images.
    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        attention_mask: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        """
        Args:
            input_ids: Ragged tensor of input token IDs.
            pixel_values: Image tensor (CHW format).
            attention_mask: Vision attention mask.
            input_row_offsets: Row offsets for ragged batching.
            return_n_logits: Number of logits to return.
            *variadic_args: Flattened KV cache inputs.
        """
        # Unflatten KV cache inputs.
        # Convert Tensors to graph Values for the unflatten utility.
        assert self.kv_params is not None
        kv_collection = unflatten_ragged_attention_inputs(
            [t._graph_value for t in variadic_args],
            n_devices=self.kv_params.n_devices,
        )

        # inputs_embeds shape (total_sequence_length=text_and_image_tokens_length for all seqs,
        #   language_model_hidden_dim)
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # If image tokens exist, replace place-holder image tokens by patch embeddings from vision encoder
        def img_then_fn() -> TensorValue:
            # Replace image place-holders in inputs_embeds with image embeddings.
            image_embeds = self.multi_modal_projector(
                self.vision_encoder(
                    imgs=[pixel_values], attention_mask=attention_mask
                )
            )
            image_embeds_cast = F.cast(image_embeds, inputs_embeds.dtype)
            special_image_mask = F.broadcast_to(
                F.unsqueeze((input_ids == self.image_token_index), -1),
                inputs_embeds.shape,
            )
            return ops.masked_scatter(
                inputs_embeds,
                special_image_mask,
                image_embeds_cast,
                out_dim="unmasked_inputs",
            )

        def else_fn() -> TensorValue:
            return TensorValue(inputs_embeds)

        # Create a runtime condition by computing the total number of elements in pixel_values
        # This ensures the condition is evaluated at execution time, not compilation time
        inputs_embeds = ops.cond(  # type: ignore[assignment]
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
            inputs_embeds,
            kv_collection[0],
            return_n_logits,
            input_row_offsets,
        )
