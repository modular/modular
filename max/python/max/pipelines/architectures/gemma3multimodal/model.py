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
import numpy as np
import numpy.typing as npt

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Type, Value
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.core import TextAndVisionContext
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")

# from ..gemma3.model import Gemma3Model
from max.pipelines.architectures.gemma3.model import Gemma3Model



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

        # TODO will it work, using the gemma3.load_model() approach? consider session and init argsZZ
        self.vision_model, self.language_model = self.load_model(session)
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        huggingface_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return super().calculate_max_seq_len(
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
        """Gets the parameters required to configure the KV cache for Gemma 3.

        Delegates to the :obj:`Gemma3Config.get_kv_params` static method.

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
        return super().get_kv_params(
            huggingface_config.text_config,
            n_devices,
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
        return super().get_num_layers(huggingface_config.text_config)

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
        return super().estimate_kv_cache_size(
            pipeline_config,
            available_cache_memory,
            devices,
            huggingface_config.text_config,
            kv_cache_config,
            cache_dtype,
        )

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """
        Args: etc
        """
        print("CALL OVERRIDDEN!")
    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Gemma3 MultiModal models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """

        # *** language model (code from gemma3/model.py) ***
        # self._input_row_offsets_prealloc = Tensor.from_numpy(
        #     np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        # ).to(self.devices[0])

        # graph = self._build_graph()
        # language_model = session.load(graph, weights_registry=self.state_dict)
        language_model = gemma3.load_model(session)


        # *** vision model (code from InternVL/model.py) ***
        # Pre-allocation for multi-step execution
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        input_row_offsets_prealloc_host = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        )
        self._input_row_offsets_prealloc = [
            input_row_offsets_prealloc_host.to(dev) for dev in self.devices
        ]

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        llm_weights_dict = convert_internvl_language_model_state_dict(
            weights_dict
        )
        vision_model_weights_dict = convert_internvl_vision_model_state_dict(
            weights_dict
        )

        # Generate Gemma3 config from HuggingFace config
        gemma3_vl_config = Gemma3VLConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_weights_dict,
            vision_state_dict=vision_model_weights_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Build and compile vision model
        vision_graph, vision_model_state_dict = self._build_vision_graph(
            gemma3_vl_config, vision_model_weights_dict
        )
        
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )

        return vision_model, language_model

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len


    def _build_vision_graph(
        self, config: Gemma3VLConfig, state_dict: dict[str, WeightData]
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
            vision_model = Gemma3VLVisionModel(config)
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