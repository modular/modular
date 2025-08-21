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
"""Base encoder model for transformer architectures."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.weights import Weights
from max.nn import Layer
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from .embeddings import EmbeddingLayer, PoolingLayer
from .transformer import TransformerEncoder


class BaseEncoderModel(Layer):
    """Base encoder model that can be extended by BERT, RoBERTa, etc."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
        include_pooler: bool = True,
    ) -> None:
        self.embeddings = EmbeddingLayer(
            pipeline_config,
            weights.embeddings,
            huggingface_config,
            dtype,
            device,
        )
        self.encoder = TransformerEncoder(
            pipeline_config, weights.encoder, huggingface_config, dtype
        )

        # Some models don't have a pooler layer
        self.pooler = None
        if include_pooler and hasattr(weights, "pooler"):
            self.pooler = PoolingLayer(
                pipeline_config, weights.pooler, huggingface_config, dtype
            )

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
        token_type_ids: TensorValue | None = None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Forward pass through the encoder model."""
        # Convert attention mask to the format expected by attention layers
        # (1.0 for tokens to attend to, large negative value for masked tokens)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.unsqueeze(
            1
        ).unsqueeze(2)

        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Pass through encoder
        sequence_output = self.encoder(
            embedding_output, extended_attention_mask
        )

        # Apply pooler if present
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output
        else:
            return sequence_output
