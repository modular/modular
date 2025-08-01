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

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.llama3.model_config import Llama3Config


class Idefics3LanguageModel(Llama3):
    """Implements the language model component of Idefics3.

    This model is based on Llama3 and is designed to process
    text tokens and generate language outputs. It supports multimodal inputs
    through embedding merging before passing to the standard Llama3 forward pass.

    The model consists of:
    - Standard Llama3 architecture (inherited)
    - Multimodal embedding merging capability
    - Image token processing
    """

    def __init__(
        self,
        config: Llama3Config,
        image_token_id: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initializes the Idefics3 language model.

        Args:
            config: The Llama3Config containing model parameters for the text component.
            image_token_id: Token ID used to represent image tokens in the text sequence.
            dtype: Data type for the model weights.
            device: Device for the model computation.
        """
        # Initialize the parent Llama3 model with the text config
        super().__init__(config)

        self.config = config
        self.device = device
        self.dtype = dtype

        # Store image token ID for multimodal processing
        self.image_token_id = image_token_id

    def __call__(  # type: ignore[override]
        self,
        tokens: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: TensorValue,
        image_token_indices: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Executes the language model forward pass with multimodal embedding merging.

        Args:
            tokens: Input token IDs to process.
            kv_cache_inputs: KV cache inputs for each device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Offsets for flattened input sequences.
            image_embeddings: Image embeddings to merge into text embeddings,
                one per device. Can be empty tensors for text-only inputs.
            image_token_indices: Indices where image tokens should be placed.

        Returns:
            A tuple containing the output logits.
        """
        # Get text embeddings using the inherited embedding layer
        h = self.embed_tokens(tokens)

        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )
        # h = distribute_value(h0_merged, self.config.text_config.devices)

        # Create KV cache collections using inherited constructor
        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        # Run through decoder layers using inherited layers
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=self.rope.freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(self.norm(last_h)), DType.float32)

        return (last_logits,)
