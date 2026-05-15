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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMesh,
)
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .gemma3 import Gemma3
from .model_config import Gemma3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Gemma3Inputs(ModelInputs):
    """A class representing inputs for the Gemma3 model (ModuleV3).

    This class encapsulates the input tensors required for the Gemma3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    return_n_logits: Buffer
    """Number of logits to return."""


class Gemma3Model(
    LogProbabilitiesMixin,
    PipelineModelWithKVCache[TextContext],
):
    """A Gemma3 pipeline model for text generation using the ModuleV3 API.

    This class integrates the Gemma3 architecture with the MAX Engine pipeline
    infrastructure using the V3 eager compilation API.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        # Detect multimodal models by presence of text_config
        self._is_multimodal = hasattr(self.huggingface_config, "text_config")

        self.model = self.load_model()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Gemma3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3Config.get_num_layers(huggingface_config)

    def load_model(self) -> Callable[..., Any]:
        """Loads the compiled Gemma3 model using the ModuleV3 API."""
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=self.devices[0]
        )

        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=self.devices[0],
        )

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        text_config = (
            self.huggingface_config.text_config
            if self._is_multimodal
            else self.huggingface_config
        )

        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=text_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        model_config = Gemma3Config.initialize_from_config(
            self.pipeline_config, text_config
        )
        model_config.finalize(
            huggingface_config=text_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )

        with F.lazy():
            nn_model = Gemma3(model_config, self.kv_params)
            nn_model.to(mesh)

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        compiled_model = nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *flattened_kv_types,
            weights=state_dict,
        )

        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma3 model with the prepared inputs."""
        assert isinstance(model_inputs, Gemma3Inputs)
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
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return Gemma3Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        assert isinstance(prev_model_inputs, Gemma3Inputs)

        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        return Gemma3Inputs(
            tokens=next_tokens,
            input_row_offsets=self._input_row_offsets_prealloc[
                :row_offsets_size
            ],
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
