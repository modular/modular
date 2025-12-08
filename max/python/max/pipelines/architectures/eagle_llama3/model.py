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

from typing import Literal

from max.driver import Device, Tensor
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn import (
    ReturnHiddenStates,
    ReturnLogits,
)
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    SupportedEncoding,
)
from transformers import AutoConfig

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .eagle_llama3 import EagleLlama3
from .model_config import Llama3Config


class EagleLlama3Model(LlamaModelBase):
    """EAGLE Llama3 draft model pipeline implementation."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.LAST,
    ) -> None:
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
            return_hidden_states,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        assert isinstance(model_inputs, Llama3Inputs)

        # EAGLE models require hidden states
        assert (
            hasattr(model_inputs, "hidden_states")
            and model_inputs.hidden_states is not None
        ), "EAGLE model requires hidden_states in model_inputs"

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.hidden_states,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )

        assert len(model_outputs) == 2
        assert isinstance(model_outputs[0], Tensor)
        assert isinstance(model_outputs[1], Tensor)

        return ModelOutputs(
            logits=model_outputs[0],
            hidden_states=model_outputs[1],
        )

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Llama3Config.generate(
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
            return_hidden_states=self.return_hidden_states,
            data_parallel_degree=self.pipeline_config.model_config.data_parallel_degree,
        )

        assert len(self.devices) == 1, "EAGLE only supports single device"

        single_model: EagleLlama3 = EagleLlama3(model_config)

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,  # We don't use the input layer norm and output layer norm
        )
        self.state_dict = single_model.state_dict()

        with Graph(
            "eagle_llama3",
            input_types=single_model.input_types(
                self.kv_manager,
                self._lora_manager,
                needs_hidden_state_input=True,
            ),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                hidden_states,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )

            outputs = single_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                hidden_states.tensor,
            )
            graph.output(*outputs)

            return graph
