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

from __future__ import annotations

import logging
from typing import Any, Literal

from max._core.engine import Model
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin

from ..llama3_legacy.model import LlamaModelBase
from .model_config import Qwen3Config
from .qwen3 import Qwen3

logger = logging.getLogger("max.pipelines")


class Qwen3Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3 pipeline model implementation.

    Supports both single and multi-GPU inference through the unified Qwen3 class.
    Uses AlwaysSignalBuffersMixin since it uses VocabParallelEmbedding and
    ColumnParallelLinear which always require signal buffers for allreduce.
    """

    model: Model
    """Compiled and initialized model ready for inference."""

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
        model_config = Qwen3Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # Build the unified Qwen3 model (works for single and multi-GPU)
        nn_model = Qwen3(model_config)

        # Get Graph Inputs
        graph_inputs = nn_model.input_types(self.kv_params)

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

        num_devices = len(self.devices)

        with Graph("qwen3", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            # Extract signal buffers (always present, even for single GPU)
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]

            # Unmarshal KV cache inputs for each device
            kv_cache_inputs = variadic_args[num_devices:]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
            )

            graph.output(*outputs)
            return graph
