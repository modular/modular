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

import logging
from typing import Any, Callable, Literal, Optional

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorValue
from max.graph.weights import Weights, WeightsAdapter
from max.nn import Module, ReturnLogits, Signals
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
)
from transformers.models.auto.configuration_auto import AutoConfig

from ..llama3.model import LlamaModelBase
from .model_config import Qwen3Config
from .qwen3 import Qwen3

logger = logging.getLogger("max.pipelines")


class Qwen3Model(LlamaModelBase):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    """Postprocessor for the logits."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

        # Initialize state needed for communication collectives.
        # Contents of signal buffer should be filled with zeros.
        self.signal_buffers = (
            [
                Tensor.zeros(
                    shape=(Signals.NUM_BYTES,),
                    dtype=DType.uint8,
                    device=dev,
                )
                for dev in self.devices
            ]
            if len(self.devices) > 1
            # Skip creating buffers for single-device, where communication
            # collectives shouldn't be called.
            else []
        )

    def _build_graph(
        self, weights: Weights, adapter: Optional[WeightsAdapter] = None
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Qwen3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=self.logits_postprocessor,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph
        nn_model: Module
        nn_model = Qwen3(model_config)

        # Load weights.
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            # Stops strict from raising error when sharing LM head weights
            # (as LM head is never technically loaded from the state dict)
            strict=(not self.huggingface_config.tie_word_embeddings),
        )

        self.state_dict = nn_model.state_dict()

        with Graph(
            "qwen3",
            input_types=graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                inp.tensor for inp in graph.inputs
            )
            outputs = nn_model(
                tokens, kv_cache_inputs, return_n_logits, input_row_offsets
            )
            graph.output(*outputs)
            return graph
