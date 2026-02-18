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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines.lib.float8 import parse_float8_config
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from transformers import AutoConfig

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

    input_row_offsets: list[Buffer]
    """List of tensors containing the offsets for each row in the ragged input
    sequence, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return."""


class Gemma3Model(
    AlwaysSignalBuffersMixin, PipelineModel[TextContext], KVCacheMixin
):
    """A Gemma 3 pipeline model for text generation.

    This class integrates the Gemma 3 architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

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
        text_huggingface_config: AutoConfig | None = None,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            huggingface_config: The configuration loaded from HuggingFace
                (:obj:`transformers.AutoConfig`).
            encoding: The quantization and data type encoding used for the model
                (:obj:`max.pipelines.config_enums.SupportedEncoding`).
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            text_huggingface_config: The text configuration loaded from HuggingFace
                if it differs from the base huggingface_config (:obj:`transformers.AutoConfig`).
            return_logits: The number of top logits to return from the model
                execution.
        """
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
        )
        self._is_multimodal = text_huggingface_config is not None
        if self._is_multimodal:
            self.huggingface_config = text_huggingface_config

        self.model = self.load_model(session)
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the Gemma 3 model.

        Uses the `max_length` from the :obj:`max.pipelines.config.PipelineConfig`
        if provided, otherwise falls back to the `max_position_embeddings` from
        the HuggingFace configuration's text config.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The calculated maximum sequence length.
        """
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
        """Gets the parameters required to configure the KV cache for Gemma 3.

        Delegates to the :obj:`Gemma3Config.construct_kv_params` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            pipeline_config: The MAX Engine pipeline configuration.
            devices: The list of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache
                (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        return Gemma3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

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

    def load_model(self, session: InferenceSession) -> Model:
        """Loads the compiled Gemma 3 model into the MAX Engine session.

        Args:
            session: The MAX Engine inference session.

        Returns:
            The loaded MAX Engine model object.
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        timer = CompilationTimer("model")
        graph = self._build_graph()
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model

    def load_logprobs_model(self, session: InferenceSession) -> Model:
        # TODO: Perhaps 'levels' ought to be configurable.
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = self.kv_params
        fetch_types = kv_params.get_symbolic_inputs()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev: list[PagedCacheValues] = []
        for i in range(len(self.devices)):
            start_idx = i * len_of_kv_tuple_per_dev
            kv_caches_per_dev.append(
                PagedCacheValues(
                    kv_blocks=kv_inputs_flat[start_idx].buffer,
                    cache_lengths=kv_inputs_flat[start_idx + 1].tensor,
                    lookup_table=kv_inputs_flat[start_idx + 2].tensor,
                    max_lengths=kv_inputs_flat[start_idx + 3].tensor,
                )
            )
        return kv_caches_per_dev

    # For text-only models, we should be using all the weights.  This is
    # overridden for Gemma3 multi-modal.
    _strict_state_dict_loading = True

    def _build_graph(self) -> Graph:
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        # Create input_row_offsets_type for each device
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef(device.label, device.id),
            )
            for device in self.devices
        ]
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
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

        state_dict_prefix = "language_model." if self._is_multimodal else ""
        float8_config = parse_float8_config(
            huggingface_config,
            state_dict,
            self.dtype,
            state_dict_name_prefix=state_dict_prefix,
            ignored_modules_prefix=state_dict_prefix or "model.",
        )

        model_config = Gemma3Config.initialize_from_config(
            self.pipeline_config, huggingface_config
        )
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            float8_config=float8_config,
        )
        nn_model = Gemma3(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        self.state_dict = nn_model.state_dict(auto_initialize=False)

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=[
                tokens_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *signals.input_types(),
                *flattened_kv_types,
            ],
        ) as graph:
            # Unpack inputs following InternVL pattern
            tokens, return_n_logits, *variadic_args = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract KV cache inputs
            kv_cache = self._unflatten_kv_inputs(variadic_args)

            outputs = nn_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_cache_inputs_per_dev=kv_cache,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)
        return graph

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

        assert isinstance(model_inputs, Gemma3Inputs)
        gemma3_inputs: Gemma3Inputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = gemma3_inputs.tokens.to_numpy()
        assert gemma3_inputs.input_row_offsets[0].device == self.logprobs_device
        input_row_offsets = gemma3_inputs.input_row_offsets[0].to_numpy()

        # Determine if we have full logits for all tokens or only last-token logits.
        # Full logits are only available when return_logits is ALL or VARIABLE.
        has_full_logits = self.return_logits in (
            ReturnLogits.ALL,
            ReturnLogits.VARIABLE,
        )

        # If echo is requested but we don't have full logits, raise an error.
        if any(batch_echo) and not has_full_logits:
            raise ValueError(
                "Log probabilities with echo=true requires enable_echo=true "
                "in the pipeline configuration to return logits for all tokens."
            )

        # Pass logits=None when we only have last-token logits.
        # compute_log_probabilities_ragged will use next_token_logits instead.
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

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma 3 model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        assert isinstance(model_inputs, Gemma3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the Gemma 3 model.

        Args:
            replica_batches: A sequence of sequences of :obj:`TextContext` objects representing
                the input prompts for each replica.
            kv_cache_inputs: Optional inputs required by the KV cache manager.

        Returns:
            The prepared :obj:`ModelInputs` object for the initial execution step.
        """
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None
        assert isinstance(kv_cache_inputs, KVCacheInputsSequence)

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        # Create input_row_offsets for each device
        input_row_offsets_tensors = [
            Buffer.from_numpy(input_row_offsets).to(device)
            for device in self.devices
        ]

        return Gemma3Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation.

        Args:
            next_tokens: The tensor containing the token IDs generated in the previous step.
            prev_model_inputs: The :obj:`ModelInputs` used in the previous execution step.

        Returns:
            The prepared :obj:`ModelInputs` object for the next execution step.
        """
        assert isinstance(prev_model_inputs, Gemma3Inputs)

        row_offsets_size = prev_model_inputs.input_row_offsets[0].shape[0]

        next_row_offsets = [
            self._input_row_offsets_prealloc[:row_offsets_size].to(device)
            for device in self.devices
        ]

        return Gemma3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
