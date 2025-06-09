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
"""Implements the InternVL multimodal model."""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn import Module
from max.pipelines.architectures.llama3.distributed_llama import (
    DistributedLlama3,
)

from .model_config import InternVLConfig


class InternVLLanguageModel(DistributedLlama3):
    """The InternVL language model for text generation with image embeddings.

    The model is actually Qwen 2, which in turn has the same architecture as
    Llama 3, but with `attention_bias=True`.
    That config is handled at the callsite in the InternVLPipelineModel.
    """


class InternVLVisionModel(Module):
    """The InternVL vision model for processing images."""

    def __init__(self, config: InternVLConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: Initialize vision encoder components

    def __call__(self, pixel_values: TensorValue) -> tuple[TensorValue, ...]:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values tensor.

        Returns:
            Image embeddings tensor.
        """
        # TODO: Implement vision processing
        # 1. Process pixel values through InternViT encoder
        # 2. Apply multimodal projector
        # 3. Return image embeddings
        return tuple(
            ops.constant(
                0.0, self.config.llm_config.dtype, device
            ).broadcast_to(
                shape=(pixel_values.shape[0], self.config.vision_hidden_size)
            )
            for device in self.config.llm_config.devices
        )
