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
import math
import time
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from max.pipelines.core import TextContext
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from transformers import AutoConfig

from .mistral import Mistral
from .model_config import MistralConfig

logger = logging.getLogger("max.pipelines")


class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - input_tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of expected token logits.
    """

    input_tokens: Tensor
    input_row_offsets: Tensor
    return_n_logits: Tensor

    def __init__(
        self,
        input_tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs


class MistralModel(PipelineModel[TextContext]):
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
        return_n_logits: int = 1,
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
            return_n_logits,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the graph."""
        model_inputs = cast(MistralInputs, model_inputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Mistral has KV cache inputs, but none were provided"
        )
        model_outputs = self.model.execute(
            model_inputs.input_tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.kv_cache_inputs,
            copy_inputs_to_device=False,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[1]),
                logit_offsets=cast(Tensor, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> MistralInputs:
        # Get tokens and seq ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = np.concatenate(tokens)
        next_tokens_batch = Tensor.from_numpy(next_tokens_batch).to(
            self.devices[0]
        )

        return MistralInputs(
            input_tokens=next_tokens_batch,
            input_row_offsets=input_row_offsets,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ).to(self.devices[0]),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> MistralInputs:
        prev_model_inputs = cast(MistralInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        return MistralInputs(
            input_tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return MistralConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return MistralConfig.get_num_layers(huggingface_config)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Mistral, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        assert self.devices, "devices must be provided to load kv manager."
        return load_kv_manager(
            params=MistralConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.huggingface_config.num_hidden_layers,
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        assert devices, "devices must be provided to estimate kv cache size."
        return estimate_kv_cache_size(
            params=MistralConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if self.pipeline_config.enable_echo:
            msg = "Mistral model does not currently implement enable echo."
            raise ValueError(msg)

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            msg = "only safetensors weights are currently supported in Mistral models."
            raise ValueError(msg)

        logger.info("Building and compiling model...")
        before = time.perf_counter()

        pipeline_config = self.pipeline_config
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

        kv_params = MistralConfig.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]
        model_config = MistralConfig(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=kv_params.n_kv_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            vocab_size=huggingface_config.vocab_size,
            dtype=self.dtype,
            kv_params=kv_params,
            return_n_logits=-1 if pipeline_config.enable_echo else 1,
            attention_multiplier=math.sqrt(1 / kv_params.head_dim),
            head_dim=huggingface_config.head_dim,
            rope_theta=huggingface_config.rope_theta,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config,
                huggingface_config=self.huggingface_config,
            ),
            rms_norm_eps=huggingface_config.rms_norm_eps,
            feed_forward_length=huggingface_config.intermediate_size,
            devices=device_refs,
        )
        nn_model = Mistral(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1)

        tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"]
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"]
        )

        kv_cache_types = self.kv_manager.input_symbols()[0]
        with Graph(
            "mistral",
            input_types=[
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *kv_cache_types,
            ],
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            # This is just needed for type checking.
            kv_cache_tensors = [v.tensor for v in kv_cache_inputs]
            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets,
                return_n_logits=return_n_logits.tensor,
                kv_cache_inputs=kv_cache_tensors,
            )
            graph.output(*outputs)

        model = session.load(graph, weights_registry=nn_model.state_dict())
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model
