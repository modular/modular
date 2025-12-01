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
from typing import Any, Literal

from max._core.engine import Model
from max.driver import Tensor
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module

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

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Qwen3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Build Graph
        nn_model: Module
        nn_model = Qwen3(model_config)

        # Get Graph Inputs
        graph_inputs = nn_model.input_types(self.kv_manager)

        # Load weights.
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            # Stops strict from raising error when sharing LM head weights
            # (as LM head is never technically loaded from the state dict)
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )

        self.state_dict = nn_model.state_dict()

        with Graph("qwen3", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph
