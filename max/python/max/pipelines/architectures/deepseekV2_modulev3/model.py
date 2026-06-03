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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV2Inputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return DeepseekV2Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
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
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

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
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        with F.lazy(), default_dtype(model_config.dtype):
            nn_model = DeepseekV2(model_config, self.kv_params)
            nn_model.to(self.devices[0])

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        return nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *flattened_kv_types,
            weights=state_dict,
        )
