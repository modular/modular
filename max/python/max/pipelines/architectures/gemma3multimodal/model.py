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
from typing import Any, cast
from collections.abc import Sequence

from max.driver import Device, Tensor, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    Value,
    BufferValue,
    DeviceRef,
    TensorValue,
)
from max.graph.weights import Weights, WeightsAdapter, WeightData
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedCacheValues,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    KVCacheConfig,
    KVCacheMixin,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding
)
from transformers import AutoConfig

from max.pipelines.architectures.gemma3.gemma3 import Gemma3
from .model_config import Gemma3ForConditionalGenerationConfig
from .gemma3multimodal import Gemma3TextModel, Gemma3VisionModel
from .weight_adapters import (
    convert_safetensor_state_dict,
    #convert_vision_model_state_dict,
)

import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("max.pipelines")

class Gemma3MultiModalModelInputs(ModelInputs):
    """A class representing inputs for the Gemma3 multi modal model.

    This class encapsulates the input tensors required for the Gemma3 multi
    modal model, for text and vision processing
    """

    tokens: npt.NDArray[np.integer[Any]] | Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | Tensor | list[Tensor]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    # Vision inputs.
    pixel_values: list[Tensor] | None = None
    """Pixel values for vision inputs."""

    image_token_indices: list[Tensor] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""

    return_n_logits: Tensor
    """Number of logits to return, used by speculative decoding for example."""

    def __init__(
        self,
        tokens: npt.NDArray[np.integer[Any]] | Tensor,
        input_row_offsets: npt.NDArray[np.integer[Any]] | Tensor | list[Tensor],
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
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs
        self.return_n_logits = return_n_logits
        self.pixel_values = pixel_values
        self.image_token_indices = image_token_indices

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Gemma3_MultiModalModel(PipelineModel[TextAndVisionContext], KVCacheMixin):
    """Gemma 3 multimodal pipeline model for text generation.

    This class integrates the Gemma 3 multimodal architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    language_model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    vision_model: Model
    """The compiled and initialized MAX Engine vision model ready for inference."""

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
        # TODO where on earth did this come from
        # hf_quant_config = getattr(
        #     huggingface_config, "quantization_config", None
        # )
        # # To the language model section of the config (`text_config`), add a
        # # reference to the top level `quantization_config` for compatibility
        # # with the base Gemma3Model, if text_config doesn't already have one
        # if hf_quant_config and not hasattr(
        #     huggingface_config.text_config, "quantization_config"
        # ):
        #     huggingface_config.text_config.quantization_config = hf_quant_config

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

        # Initialize signal buffers for distributed execution
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
            )
            for dev in self.devices
        ]

        # logger.info(f"\nPIPELINE CONFIG:\n{self.pipeline_config}\n--------------")
        # logger.info(f"\nHUGGINGFACE CONFIG:\n{self.huggingface_config}\n")
        # logger.info(f"\nHF info: {self.huggingface_config.model_type}, {self.huggingface_config.__class__.__name__}\n")

        # TODO will it work, using the gemma3.load_model() approach? consider session and init args
        self.vision_model, self.language_model = self.load_model(session)
        logger.info(f"Language model: {self.language_model}")

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        # huggingface_config = getattr(
        #     huggingface_config, "text_config", huggingface_config
        # )
        return Gemma3ForConditionalGenerationConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Gemma 3.

        Delegates to the :obj:`Gemma3ForConditionalGenerationConfig.get_kv_params` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            n_devices: The number of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache
                (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        # huggingface_config = getattr(
        #     huggingface_config, "text_config", huggingface_config
        # )
        return Gemma3ForConditionalGenerationConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration.

        Delegates to the :obj:`Gemma3ForConditionalGenerationConfig.get_num_layers` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers.
        """
        return Gemma3ForConditionalGenerationConfig.get_num_layers(huggingface_config)

    @staticmethod
    def estimate_kv_cache_size(
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
            params=Gemma3ForConditionalGenerationConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=Gemma3_MultiModalModel.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Gemma3ForConditionalGenerationConfig.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def prepare_initial_token_inputs(
            self,
            context_batch: Sequence[TextAndVisionContext],
            kv_cache_inputs: KVCacheInputs | None = None,
            return_n_logits: int = 1,
        ) -> ModelInputs:
        logger.info("*** prepare_initial_token_inputs ***")
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_row_offsets_tensors = [
            Tensor.from_numpy(input_row_offsets).to(device)
            for device in self.devices
        ]

        # TODO
        # image_token_indices = self._batch_image_token_indices(context_batch)
        # pixel_values = self._prepare_vision_inputs(context_batch)

        return Gemma3MultiModalModelInputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            #pixel_values=None,
            #image_token_indices=None,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        logger.info("*** prepare_next_token_inputs ***")

        prev_model_inputs = cast(Gemma3MultiModalModelInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets[0].shape[0]

        next_row_offsets = [
            self._input_row_offsets_prealloc[:row_offsets_size].to(device)
            for device in self.devices
        ]

        return Gemma3MultiModalModelInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            pixel_values=None,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        logger.info("*** execute ***")
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        # Check if input_row_offsets is a list or a single tensor
        if isinstance(model_inputs.input_row_offsets, list):
            input_row_offsets_list = model_inputs.input_row_offsets
        else:
            # For backward compatibility, distribute the single tensor to all devices
            if isinstance(model_inputs.input_row_offsets, np.ndarray):
                # Convert numpy array to tensor first
                tensor = Tensor.from_numpy(model_inputs.input_row_offsets)
                input_row_offsets_list = [
                    tensor.to(device) for device in self.devices
                ]
            else:
                # Already a tensor
                input_row_offsets_list = [
                    model_inputs.input_row_offsets.to(device)
                    for device in self.devices
                ]

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets_list,
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

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Gemma3 MultiModal models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        logger.info("*** build language model (code from gemma3/model.py) ***")
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])
        graph = self._build_language_graph()
        language_model = session.load(graph, weights_registry=self.state_dict)



        logger.info("*** build vision model (code from InternVL/model.py) ***")

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        llm_weights_dict = convert_safetensor_state_dict(
            weights_dict
        )
        # vision_model_weights_dict = convert_vision_model_state_dict(
        #     weights_dict
        # )

        # Generate Gemma3 config from HuggingFace config
        model_config = Gemma3ForConditionalGenerationConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=llm_weights_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Build and compile vision model
        vision_graph, vision_model_state_dict = self._build_vision_graph(
            model_config, llm_weights_dict # vision_model_weights_dict
        )
        
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )

        return vision_model, language_model

    def _build_language_graph(self):
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
        model_config = Gemma3ForConditionalGenerationConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        nn_model = Gemma3TextModel(model_config)
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
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                kv_collections=kv_cache,
            )
            graph.output(*outputs)
        return graph
    
    def _build_vision_graph(
        self, config: Gemma3ForConditionalGenerationConfig, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""
        # Define input types for the vision model
        # Use static dimensions from the vision config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        # Calculate number of patches in each dimension
        height_patches = image_size // patch_size
        width_patches = image_size // patch_size

        # Expect pre-extracted patches from the tokenizer.
        # Use bfloat16 to match the tokenizer's output.
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "batch_size",
                    height_patches,
                    width_patches,
                    3,
                    patch_size,
                    patch_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Initialize graph with input types
        with Graph(
            "gemma3_vision",
            input_types=[*pixel_values_types, *signals.input_types()],
        ) as graph:
            # Build vision model architecture.
            vision_model = Gemma3VisionModel(config)
            vision_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs (one per device).
            pixel_values = [
                inp.tensor for inp in graph.inputs[: len(self.devices)]
            ]

            # Extract signal buffers (one per device).
            signal_buffers = [
                inp.buffer for inp in graph.inputs[len(self.devices) :]
            ]

            # Execute vision model: pixel_values -> image_embeddings.
            image_embeddings = vision_model(pixel_values, signal_buffers)

            # Set graph outputs.
            graph.output(*image_embeddings)

            return graph, vision_model.state_dict()

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
            params=Gemma3ForConditionalGenerationConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=Gemma3ForConditionalGenerationConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )


    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = Gemma3ForConditionalGenerationConfig.get_kv_params(
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
