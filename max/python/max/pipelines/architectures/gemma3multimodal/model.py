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
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.kv_cache import (
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedCacheValues,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from transformers import AutoConfig

from .gemma3multimodal import Gemma3Multimodal, SigLipVisionModel
from .model_config import Gemma3Config, Gemma3MultimodalConfig, VisionConfig

logger = logging.getLogger("max.pipelines")


class Gemma3Inputs(ModelInputs):
    """A class representing inputs for the Gemma3 model.

    This class encapsulates the input tensors required for the Gemma3 model
    execution, including optional vision inputs for multimodal models.
    """

    tokens: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Tensor]
    """List of tensors containing the offsets for each row in the ragged input
    sequence, one per device."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    # Vision inputs for multimodal support
    pixel_values: list[Tensor] | None = None
    """Pixel values for vision inputs."""

    image_token_indices: list[Tensor] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: list[Tensor],
        return_n_logits: Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None = None,
        pixel_values: list[Tensor] | None = None,
        image_token_indices: list[Tensor] | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets: Input row offsets (ragged tensors).
            return_n_logits: Number of logits to return.
            signal_buffers: Device buffers for distributed communication.
            kv_cache_inputs: Inputs for the KV cache.
            pixel_values: Optional pixel values for vision inputs.
            image_token_indices: Optional indices of image tokens in the input sequence.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs
        self.pixel_values = pixel_values
        self.image_token_indices = image_token_indices

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None and len(self.pixel_values) > 0
        self.return_n_logits = return_n_logits


class Gemma3_MultiModalModel(
    AlwaysSignalBuffersMixin,
    PipelineModel[TextAndVisionContext],
    KVCacheMixin,
):
    """Gemma 3 multimodal pipeline model for text generation.

    This class integrates the Gemma 3 multimodal architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    # Type annotations for the models - vision_model may be None if no vision weights
    vision_model: Model | None
    language_model: Model

    # The vision and text towers are in the same weights file, but are in
    # separate models, so load_state_dict will naturally be loading subsets in
    # each case.
    _strict_state_dict_loading = False

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
            return_logits: The number of top logits to return from the model
                execution.
        """
        # Save the full config (with vision_config) before passing text_config to parent
        self.full_huggingface_config = huggingface_config

        hf_quant_config = getattr(
            huggingface_config, "quantization_config", None
        )
        # To the language model section of the config (`text_config`), add a
        # reference to the top level `quantization_config` for compatibility
        # with the base Gemma3Model, if text_config doesn't already have one
        if hf_quant_config and not hasattr(
            huggingface_config.text_config, "quantization_config"
        ):
            huggingface_config.text_config.quantization_config = hf_quant_config

        super().__init__(
            pipeline_config,
            session,
            huggingface_config.text_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.signal_buffers = Signals(
            devices=(DeviceRef(d.label, d.id) for d in devices)
        ).buffers()
        self.vision_model, self.language_model = self.load_model(session)
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Gemma3MultimodalConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Gemma3MultimodalConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3MultimodalConfig.get_num_layers(huggingface_config)

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
        """Estimates the size of the KV cache required for the Gemma 3 model in bytes.

        Args:
            pipeline_config: The configuration for the pipeline.
            available_cache_memory: The total memory available for the KV cache
                in bytes.
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) the
                model will run on.
            kv_cache_config: Configuration settings for the KV cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The data type for the KV cache (:obj:`max.dtype.DType`).

        Returns:
            The estimated size of the KV cache in bytes.
        """
        return estimate_kv_cache_size(
            params=Gemma3MultimodalConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=Gemma3_MultiModalModel.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Gemma3MultimodalConfig.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    # @classmethod
    # def estimate_activation_memory(
    #     cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    # ) -> int:
    #     """Estimates the activation memory required for Gemma3 model execution.

    #     This accounts for the temporary memory buffers used during model execution,
    #     particularly for the vision encoder and language model activations.

    #     Based on empirical analysis:
    #     - Vision encoder uses ~128MiB per image
    #     - Language model uses ~100KB per token for intermediate activations

    #     These values are based on buffer plan analysis and runtime validation.
    #     The vision encoder memory scales with the number of images that can be
    #     processed concurrently, which is limited by prefill_chunk_size / num_image_tokens
    #     where num_image_tokens=256 for Gemma3.

    #     Args:
    #         pipeline_config: Pipeline configuration
    #         huggingface_config: HuggingFace model configuration

    #     Returns:
    #         Estimated activation memory in bytes
    #     """
    #     # Vision encoder memory estimation
    #     vision_memory_per_image = 128 * 1024 * 1024  # 128 MiB per image

    #     # Get mm_tokens_per_image from config (default 256 for Gemma3)
    #     mm_tokens_per_image = getattr(huggingface_config, "mm_tokens_per_image", 256)

    #     max_images = pipeline_config.prefill_chunk_size // mm_tokens_per_image
    #     # Ensure at least 1 image worth of memory
    #     max_images = max(1, max_images)

    #     if not pipeline_config.enable_chunked_prefill:
    #         max_images += 1

    #     vision_activation_memory = max_images * vision_memory_per_image

    #     # ~100KB per token for intermediate activations
    #     llm_memory_per_token = 100 * 1024  # 100 KiB
    #     llm_activation_memory = (
    #         pipeline_config.prefill_chunk_size * llm_memory_per_token
    #     )

    #     total_activation_memory = (
    #         vision_activation_memory + llm_activation_memory
    #     )

    #     return (
    #         len(pipeline_config.model_config.device_specs)
    #         * total_activation_memory
    #     )

    def _separate_state_dicts(
        self, state_dict: dict
    ) -> tuple[dict, dict, dict]:
        """Separate the full state dict into vision, projector, and language components.

        After the weight adapter has run, the keys will be:
        - Vision: embeddings.*, encoder.*, post_layernorm.*
          (vision_tower.vision_model. prefix stripped by adapter)
        - Projector: multi_modal_projector.*
        - Language: language_model.* (language_model.model. -> language_model.)

        Args:
            state_dict: The full state dictionary with all model weights.

        Returns:
            Tuple of (vision_state_dict, projector_state_dict, llm_state_dict)
        """
        vision_state_dict = {}
        projector_state_dict = {}
        llm_state_dict = {}

        # Log summary for debugging
        all_keys = list(state_dict.keys())
        logger.info(f"Separating {len(all_keys)} total weights")
        logger.debug(f"First 10 weight keys: {all_keys[:10]}")

        for key, value in state_dict.items():
            if (
                key.startswith("embeddings.")
                or key.startswith("encoder.")
                or key.startswith("post_layernorm.")
            ):
                vision_state_dict[key] = value
            elif key.startswith("multi_modal_projector."):
                projector_state_dict[key] = value
            elif key.startswith("language_model."):
                llm_state_dict[key] = value
            else:
                llm_state_dict[key] = value

        logger.info(
            f"Separated weights: {len(vision_state_dict)} vision, "
            f"{len(projector_state_dict)} projector, "
            f"{len(llm_state_dict)} language"
        )

        return vision_state_dict, projector_state_dict, llm_state_dict

    def load_model(
        self, session: InferenceSession
    ) -> tuple[Model | None, Model]:
        """Loads the compiled Gemma 3 multimodal models into the MAX Engine session.

        Args:
            session: The MAX Engine inference session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        vision_state_dict, projector_state_dict, llm_state_dict = (
            self._separate_state_dicts(state_dict)
        )

        self.state_dict = state_dict
        self._vision_state_dict_for_graph = vision_state_dict
        self._projector_state_dict_for_graph = projector_state_dict
        self._llm_state_dict_for_graph = llm_state_dict

        if vision_state_dict or projector_state_dict:
            logger.info("Building and compiling vision model with projector...")
            before = time.perf_counter()
            vision_graph, vision_and_projector_state_dict = (
                self._build_vision_graph()
            )
            vision_model = session.load(
                vision_graph, weights_registry=vision_and_projector_state_dict
            )
            after = time.perf_counter()
            logger.info(
                f"Building and compiling vision model took {after - before:.6f} seconds"
            )
        else:
            logger.warning(
                "No vision weights found in model checkpoint. "
                "Vision model will not be available. This is expected for text-only models."
            )
            vision_model = None

        if len(llm_state_dict) > 0:
            logger.info("Building and compiling language model...")
            before = time.perf_counter()
            language_graph = self._build_language_graph()

            language_weights_for_graph = self.state_dict

            language_model = session.load(
                language_graph, weights_registry=language_weights_for_graph
            )
            after = time.perf_counter()
            logger.info(
                f"Building and compiling language model took {after - before:.6f} seconds"
            )
        else:
            raise ValueError(
                "No language weights found in model checkpoint. "
                "This appears to be a vision-only checkpoint, which is not supported for multimodal inference."
            )

        return vision_model, language_model

    def load_logprobs_model(self, session: InferenceSession) -> Model:
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = Gemma3Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev: list[PagedCacheValues] = []
        for i in range(n_devices):
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

    def _build_vision_graph(self) -> tuple[Graph, dict]:
        """Build the vision encoder graph.

        This graph processes image inputs through the SigLip vision encoder
        to produce vision embeddings.

        Returns:
            Tuple of (graph, state_dict) where state_dict contains the vision weights.
        """
        device_ref = DeviceRef.from_device(self.devices[0])

        vision_state_dict = self._vision_state_dict_for_graph

        huggingface_config = self.full_huggingface_config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)

        if hf_vision_config is None:
            raise ValueError(
                "No vision_config found in HuggingFace config. "
                "This model may not be a multimodal Gemma3 model."
            )

        vision_config = VisionConfig.generate(hf_vision_config)

        pixel_values_type = TensorType(
            DType.float32,
            shape=[
                "batch_size",
                vision_config.num_channels,
                vision_config.image_size,
                vision_config.image_size,
            ],
            device=device_ref,
        )

        vision_model = SigLipVisionModel(vision_config, devices=[device_ref])

        if vision_state_dict:
            vision_model.load_state_dict(
                state_dict=vision_state_dict,
                strict=True,
            )

        from .gemma3multimodal import MultimodalProjector

        text_hidden_size = self.full_huggingface_config.text_config.hidden_size

        projector = MultimodalProjector(
            vision_hidden_size=vision_config.hidden_size,
            text_hidden_size=text_hidden_size,
            dtype=self.dtype,
            device=device_ref,
        )

        if self._projector_state_dict_for_graph:
            projector_state_dict_stripped = {
                k.replace("multi_modal_projector.", ""): v
                for k, v in self._projector_state_dict_for_graph.items()
            }
            projector.load_state_dict(
                state_dict=projector_state_dict_stripped,
                weight_alignment=1,
                strict=True,
            )

        with Graph(
            "gemma3_vision",
            input_types=[pixel_values_type],
        ) as graph:
            pixel_values = graph.inputs[0].tensor
            vision_embeddings = vision_model(
                pixel_values, output_hidden_states=False
            )

            batch_size = vision_embeddings.shape[0]
            num_patches = vision_embeddings.shape[1]
            vision_hidden_size = vision_embeddings.shape[2]

            vision_embeddings_flat = vision_embeddings.reshape(
                [batch_size * num_patches, vision_hidden_size]
            )

            projected_embeddings = projector(vision_embeddings_flat)

            graph.output(projected_embeddings)

            projector_state_dict_for_registry = {
                k.replace("multi_modal_projector.", ""): v
                for k, v in self._projector_state_dict_for_graph.items()
            }
            combined_state_dict = {
                **vision_state_dict,
                **projector_state_dict_for_registry,
            }
            return graph, combined_state_dict

    def _build_language_graph(self) -> Graph:
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
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

        llm_state_dict = self._llm_state_dict_for_graph

        full_state_dict = {
            **self._vision_state_dict_for_graph,
            **{f"language_model.{k}": v for k, v in llm_state_dict.items()},
        }

        model_config = Gemma3MultimodalConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.full_huggingface_config,
            state_dict=full_state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            attention_bias=getattr(
                self.full_huggingface_config, "attention_bias", False
            ),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        nn_model = Gemma3Multimodal(model_config)
        llm_state_dict_clean = {}
        for key, value in llm_state_dict.items():
            if key.startswith("language_model."):
                clean_key = key.replace("language_model.", "", 1)
                llm_state_dict_clean[clean_key] = value

        logger.info(
            f"Loading {len(llm_state_dict_clean)} language model weights"
        )
        logger.info(
            f"Sample language weight keys: {list(llm_state_dict_clean.keys())[:5]}"
        )

        nn_model.language_model.load_state_dict(
            llm_state_dict_clean,
            strict=True,
        )
        self.state_dict = nn_model.language_model.state_dict(
            auto_initialize=False
        )

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        if hasattr(self.huggingface_config, "text_config"):
            hidden_size = self.huggingface_config.text_config.hidden_size
        else:
            hidden_size = self.huggingface_config.hidden_size
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=["num_image_tokens", hidden_size],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        kv_inputs = self.kv_manager.input_symbols()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=[
                tokens_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *image_embeddings_types,
                *image_token_indices_types,
                *signals.input_types(),
                *flattened_kv_types,
            ],
        ) as graph:
            tokens, return_n_logits, *variadic_args = graph.inputs

            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            kv_cache = self._unflatten_kv_inputs(variadic_args)

            outputs = nn_model.language_model(
                tokens.tensor,
                signal_buffers,
                kv_cache,
                return_n_logits.tensor,
                input_row_offsets,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )
            graph.output(*outputs)
        return graph

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, Gemma3Inputs)
        gemma3_inputs: Gemma3Inputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = gemma3_inputs.tokens.to_numpy()
        assert gemma3_inputs.input_row_offsets[0].device == self.logprobs_device
        input_row_offsets = gemma3_inputs.input_row_offsets[0].to_numpy()

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
        """Executes the Gemma 3 multimodal model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, including
                token IDs, attention masks/offsets, KV cache inputs, and optionally
                vision inputs (pixel_values, image_token_indices).

        Returns:
            An object containing the output logits from the model execution.
        """
        assert isinstance(model_inputs, Gemma3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        image_embeddings: list[Tensor]
        image_token_indices: list[Tensor]

        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None

            if self.vision_model is None:
                raise ValueError(
                    "Vision inputs provided but no vision model is available. "
                    "Please use a multimodal checkpoint with vision weights."
                )

            logger.info(
                f"Processing {len(model_inputs.pixel_values)} vision inputs"
            )

            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values
            )

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Tensor)
            ]

            if model_inputs.image_token_indices is not None:
                image_token_indices = model_inputs.image_token_indices
            else:
                image_token_indices = [
                    Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
                    for dev in self.devices
                ]

            logger.info(
                f"Generated {len(image_embeddings)} image embedding tensors"
            )

        else:
            if hasattr(self.huggingface_config, "text_config"):
                hidden_size = self.huggingface_config.text_config.hidden_size
            else:
                hidden_size = self.huggingface_config.hidden_size

            image_embeddings = [
                Tensor.zeros(
                    shape=[0, hidden_size],
                    dtype=self.dtype,
                ).to(dev)
                for dev in self.devices
            ]
            image_token_indices = [
                Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
                for dev in self.devices
            ]

        if self.language_model is None:
            raise ValueError(
                "Language model is not available. "
                "This appears to be a vision-only checkpoint. "
                "Please use a full multimodal Gemma3 checkpoint that includes both vision and language weights."
            )

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Tensor] | None:
        """Batches up pixel_values for vision processing.

        Args:
            context_batch: Sequence of contexts that may contain vision data.

        Returns:
            List of tensors (one per device) containing pixel values, or None if no images.
        """
        images = []
        for context in context_batch:
            if context.needs_vision_encoding:
                next_images = context.next_images
                if not next_images:
                    continue

                for image_meta in next_images:
                    pixel_values = image_meta.pixel_values
                    # pixel_values should be shape (C, H, W) for each image
                    images.append(pixel_values)

        if not images:
            return None

        stacked_images = np.stack(images, axis=0)
        tensor = Tensor.from_numpy(stacked_images)
        return [tensor.to(dev) for dev in self.devices]

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Tensor] | None:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.

        For Gemma3, we find where <image> tokens (self.image_token_id) appear
        in the token sequence. These positions mark where vision embeddings
        should be inserted.

        Args:
            context_batch: Sequence of contexts that may contain image tokens.

        Returns:
            List of tensors containing all batched indices (one per device), or None if no indices found.
        """
        indices_and_offsets = []
        batch_offset = 0

        image_token_id = self.full_huggingface_config.image_token_index

        for ctx in context_batch:
            if "image_token_indices" in ctx.extra_model_args:
                indices = ctx.extra_model_args["image_token_indices"]
                indices_and_offsets.append(indices + batch_offset)
            else:
                input_ids = ctx.next_tokens
                special_image_token_mask = input_ids == image_token_id
                indices = np.where(special_image_token_mask)[0]

                if len(indices) > 0:
                    indices_and_offsets.append(indices + batch_offset)

            batch_offset += ctx.active_length

        if not indices_and_offsets:
            return [
                Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
                for dev in self.devices
            ]

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )
        return [Tensor.from_numpy(np_indices).to(dev) for dev in self.devices]

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the Gemma 3 model.

        Args:
            context_batch: A sequence of :obj:`TextAndVisionContext` objects representing
                the input prompts.
            kv_cache_inputs: Optional inputs required by the KV cache manager.
            return_n_logits: Number of logits to return.

        Returns:
            The prepared :obj:`Gemma3Inputs` object for the initial execution step.
        """
        assert kv_cache_inputs is not None
        assert isinstance(kv_cache_inputs, KVCacheInputsSequence)

        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )

        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        input_row_offsets_tensors = [
            Tensor.from_numpy(input_row_offsets).to(device)
            for device in self.devices
        ]

        pixel_values = self._prepare_vision_inputs(context_batch)
        image_token_indices = self._batch_image_token_indices(context_batch)

        return Gemma3Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            pixel_values=pixel_values,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
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
            pixel_values=None,
            image_token_indices=None,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager:
        """Loads and initializes the KVCacheManager for the Gemma 3 model.

        Configures the KV cache manager based on model parameters, pipeline settings,
        and available memory.

        Args:
            session: The MAX Engine inference session.
            available_cache_memory: The amount of memory available for the KV cache in bytes.

        Returns:
            An initialized :obj:`KVCacheManager` instance.
        """
        return load_kv_manager(
            params=Gemma3MultimodalConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=Gemma3MultimodalConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )
