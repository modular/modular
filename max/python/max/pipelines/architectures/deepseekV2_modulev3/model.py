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
"""Implements the DeepseekV2 nn.model (ModuleV3)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheParamInterface,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    upper_bounded_default,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .batch_processor import DeepseekV2ModuleV3BatchProcessor
from .deepseekV2 import DeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV2Inputs(ModelInputs):
    """Inputs for the DeepseekV2 model."""

    tokens: Buffer
    input_row_offsets: Buffer

    return_n_logits: Buffer = field(kw_only=True)


class DeepseekV2Model(
    LogProbabilitiesMixin, PipelineModelWithKVCache[TextContext]
):
    model_config_cls: ClassVar[type[Any]] = DeepseekV2Config
    batch_processor_cls: ClassVar[type[DeepseekV2ModuleV3BatchProcessor]] = (
        DeepseekV2ModuleV3BatchProcessor
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")

        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )

        self.model = self.load_model()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *curr_kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        return ModelOutputs(
            logits=cast(Buffer, model_outputs[0].driver_tensor),
            next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return DeepseekV2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for DeepseekV2, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def load_model(self) -> Callable[..., Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV2."
            )

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        model_config = DeepseekV2Config.initialize(self.pipeline_config)
        model_config.max_batch_context_length = (
            self.pipeline_config.runtime.max_batch_total_tokens
            or model_config.max_batch_context_length
        )

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        with F.lazy(), default_dtype(model_config.dtype):
            nn_model = DeepseekV2(model_config, self.kv_params)
            nn_model.to(self.devices[0])

        assert self.batch_processor is not None
        compile_input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params,
            device_refs=[device_ref],
        )

        return nn_model.compile(
            *compile_input_types,
            weights=state_dict,
        )
