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
from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from .batch_processor import Gemma3BatchProcessor
from .gemma3 import Gemma3
from .model_config import Gemma3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Gemma3Inputs(ModelInputs):
    """A class representing inputs for the Gemma3 model.

    This class encapsulates the input tensors required for the Gemma3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return."""


class Gemma3Model(
    LogProbabilitiesMixin,
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[TextContext, Gemma3Config, Model],
):
    """A Gemma 3 pipeline model for text generation.

    This class integrates the Gemma 3 architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma3Config
    batch_processor_cls: ClassVar[type[Gemma3BatchProcessor]] = (
        Gemma3BatchProcessor
    )

    _language_weights_dict: dict[str, Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            return_logits: The number of top logits to return from the model
                execution.
        """
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        # Detect multimodal models by presence of text_config
        self._is_multimodal = hasattr(self.huggingface_config, "text_config")

        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled and initialized MAX Engine model ready for inference."""
        return self._models[GraphRole.DECODER]

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration.

        Delegates to the :obj:`Gemma3Config.get_num_layers` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers.
        """
        return Gemma3Config.get_num_layers(huggingface_config)

    _strict_state_dict_loading = True

    def _hf_config_for_weights(self) -> Any:
        if hasattr(self.huggingface_config, "text_config"):
            return self.huggingface_config.text_config
        return self.huggingface_config

    def _load_state_dict(self) -> dict[str, Any]:
        text_config = self._hf_config_for_weights()
        if self.adapter:
            self._language_weights_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=text_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            self._language_weights_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        return self._language_weights_dict

    def _create_model_config(self, state_dict: dict[str, Any]) -> Gemma3Config:
        text_config = self._hf_config_for_weights()
        state_dict_prefix = (
            "language_model."
            if hasattr(self.huggingface_config, "text_config")
            else ""
        )
        quant_config = parse_quant_config(
            text_config,
            state_dict,
            self.dtype,
            state_dict_name_prefix=state_dict_prefix,
            ignored_modules_prefix=state_dict_prefix or "model.",
        )
        model_config = Gemma3Config.initialize_from_config(
            self.pipeline_config, text_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=text_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            quant_config=quant_config,
        )
        return model_config

    def graph_specs(self, model_config: Gemma3Config) -> dict[str, GraphSpec]:
        """Returns the decoder graph."""
        nn_model = Gemma3(model_config)
        nn_model.load_state_dict(
            self._language_weights_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        # The graph's weights registry holds what it binds; drop the rest.
        self._language_weights_dict = {}
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                getattr(self._hf_config_for_weights(), "model_type", "Gemma3"),
                self._bind_graph_inputs(nn_model.input_tree()),
                weights=nn_model.state_dict(auto_initialize=False),
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma 3 model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        assert isinstance(model_inputs, Gemma3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        input_row_offsets_per_dev = [
            model_inputs.input_row_offsets.to(d) for d in self.devices
        ]

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets_per_dev,
            *model_inputs.signal_buffers,
            *tree.leaves(curr_kv_cache_inputs),
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )
