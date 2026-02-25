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
from typing import Any, Literal, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from max.profiler import traced
from transformers import AutoConfig

from .llama3 import Llama3
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        kv_cache_inputs = tuple(self.kv_cache_inputs or ())
        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *kv_cache_inputs,
        )


class Llama3Model(PipelineModelWithKVCache[TextContext]):
    """Llama3 pipeline model using the ModuleV3 API."""

    config_class: type[Llama3Config] = Llama3Config
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
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
        self.logprobs_device = devices[0]
        self.logprobs_model = self._load_logprobs_model(session)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Llama3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self) -> Callable[..., Any]:
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        timer = CompilationTimer("model")
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device0,
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
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
        model_config = self.config_class.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        with F.lazy(), default_dtype(model_config.dtype):
            nn_model = Llama3(model_config, self.kv_params)
            nn_model.to(self.devices[0])

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        timer.mark_build_complete()
        compiled_model = nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *flattened_kv_types,
            weights=state_dict,
        )
        timer.done()

        return compiled_model

    @traced
    def _load_logprobs_model(self, session: InferenceSession) -> Model:
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(Llama3Inputs, model_inputs)
        model_outputs = self.model(*model_inputs.buffers)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[3].driver_tensor),
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[1].driver_tensor),
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        input_row_offsets_tensor = Buffer.from_numpy(input_row_offsets).to(
            self.devices[0]
        )

        return Llama3Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensor,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        prev_model_inputs = cast(Llama3Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        next_row_offsets = self._input_row_offsets_prealloc[
            :row_offsets_size
        ].to(self.devices[0])

        return Llama3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, Llama3Inputs)
        llama3_inputs: Llama3Inputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = llama3_inputs.tokens.to_numpy()
        input_row_offsets = llama3_inputs.input_row_offsets.to_numpy()

        has_full_logits = self.return_logits in (
            ReturnLogits.ALL,
            ReturnLogits.VARIABLE,
        )

        if any(batch_echo) and not has_full_logits:
            raise ValueError(
                "Log probabilities with echo=true requires enable_echo=true "
                "in the pipeline configuration to return logits for all tokens."
            )

        logits = model_outputs.logits if has_full_logits else None

        return compute_log_probabilities_ragged(
            self.logprobs_device,
            self.logprobs_model,
            input_row_offsets=input_row_offsets,
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,
        )
