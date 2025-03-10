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
import time
from typing import Sequence, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
    upper_bounded_default,
)
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from transformers import AutoConfig

from .graph import _build_graph

logger = logging.getLogger("max.pipelines")


class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - input_tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    """

    input_tokens: Tensor
    input_row_offsets: Tensor

    def __init__(
        self,
        input_tokens: Tensor,
        input_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.input_row_offsets = input_row_offsets
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
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
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
            *model_inputs.kv_cache_inputs,
            copy_inputs_to_device=False,
        )
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
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
        return KVCacheParams(
            page_size=kv_cache_config.kv_cache_page_size,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            n_devices=n_devices,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

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
            params=self.get_kv_params(
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
            params=cls.get_kv_params(
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

    def load_model(self, session: InferenceSession) -> Model:
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

        self._weights = self.pipeline_config.load_weights()

        if not isinstance(self._weights, SafetensorWeights):
            msg = "only safetensors weights are currently supported in Mistral models."
            raise ValueError(msg)

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()
            logger.info(
                "Loading serialized model from ", serialized_path, "..."
            )
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            logger.info("Building and compiling model...")
            before = time.perf_counter()
            graph = _build_graph(
                pipeline_config=self.pipeline_config,
                weights=self._weights,
                max_seq_len=self.calculate_max_seq_len(
                    self.pipeline_config,
                    huggingface_config=self.huggingface_config,
                ),
                kv_params=self.get_kv_params(
                    huggingface_config=self.huggingface_config,
                    n_devices=len(self.devices),
                    kv_cache_config=self.kv_cache_config,
                    cache_dtype=self.encoding.cache_dtype,
                ),
                kv_manager=self.kv_manager,
                huggingface_config=self.huggingface_config,
                dtype=self.dtype,
            )
            model = session.load(
                graph, weights_registry=self._weights.allocated_weights
            )
            after = time.perf_counter()
            logger.info(
                f"Building and compiling model took {after - before:.6f} seconds"
            )
            if (
                export_path
                := self.pipeline_config.save_to_serialized_model_path
            ):
                logger.info("Exporting serialized model to %s", export_path)
                model._export_mef(export_path)
            return model
