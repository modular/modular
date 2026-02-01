# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Mistral3 text encoder for Flux2 pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from max.driver import Device, DeviceSpec
from max.engine import InferenceSession
from max.graph import TensorValue
from max.graph.weights import Weights, WeightsFormat
from max.interfaces import (
    RequestID,
    SamplingParams,
    SamplingParamsInput,
    TextGenerationRequest,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    PipelineConfig,
    SupportedEncoding,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.pipelines.lib.memory_estimation import MemoryEstimator
from transformers import AutoConfig

from .arch import mistral3_arch
from .model import Mistral3Model
from .tokenizer import Mistral3Tokenizer

logger = logging.getLogger(__name__)


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder wrapper implementing ComponentModel interface.

    This class wraps Mistral3Model to function as a text encoder for Flux2 pipeline.
    It uses the full Mistral3 text generation infrastructure internally but exposes
    a simpler interface that returns hidden states from all layers.

    Note: Although text encoding is a single forward pass operation and doesn't
    actually use KV cache for multi-step generation, the compiled model graph
    requires KV cache inputs as part of its interface. We allocate minimal KV
    cache to satisfy the graph requirements.
    """

    config_name = "config.json"

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Mistral3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)

        # Extract model path from config
        self._text_encoder_path = config.get("text_encoder_path") or config.get(
            "model_path"
        )
        if not self._text_encoder_path:
            raise ValueError(
                "model_path or text_encoder_path must be provided in config"
            )

        self._root_model_path = config.get("root_model_path")

        # Text encoder uses single forward pass, so minimal KV cache is sufficient
        self.device_memory_utilization = config.get(
            "device_memory_utilization", 0.3
        )

        # Lazy initialization attributes (set in load_model)
        self._mistral_model: Mistral3Model | None = None
        self._session: InferenceSession | None = None
        self._tokenizer: Mistral3Tokenizer | None = None
        self._pipeline_config: PipelineConfig | None = None

        # Load model during initialization
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load pretrained model weights and compile the model graph.

        Returns:
            Compiled model callable (Model instance).
        """
        device_specs = [
            DeviceSpec.cpu(id=d.id) if d.label == "cpu"
            else DeviceSpec.accelerator(id=d.id) if d.label == "gpu"
            else DeviceSpec(id=d.id, device_type=d.label)
            for d in self.devices
        ]
        self._pipeline_config = PipelineConfig(
            model_path=self._text_encoder_path,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
            device_specs=device_specs,
        )
        model_config = self._pipeline_config.model
        model_config.kv_cache.device_memory_utilization = (
            self.device_memory_utilization
        )

        self._session = InferenceSession(devices=self.devices)
        huggingface_config = AutoConfig.from_pretrained(self._text_encoder_path)

        arch_config = mistral3_arch.config.initialize(self._pipeline_config)
        model_weights_size = Mistral3Model.estimate_weights_size(
            self._pipeline_config
        )
        activation_memory_size = Mistral3Model.estimate_activation_memory(
            self._pipeline_config, huggingface_config
        )
        MemoryEstimator.estimate_memory_footprint(
            self._pipeline_config,
            model_config,
            arch_config,
            self.devices,
            model_weights_size,
            activation_memory_size,
        )

        adapter = mistral3_arch.weight_adapters.get(
            WeightsFormat.safetensors, None
        )
        self._mistral_model = Mistral3Model(
            pipeline_config=self._pipeline_config,
            session=self._session,
            huggingface_config=huggingface_config,
            encoding=self.encoding,
            devices=self.devices,
            kv_cache_config=model_config.kv_cache,
            weights=self.weights,
            adapter=adapter,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
        )

        self._tokenizer = Mistral3Tokenizer(
            model_path=self._text_encoder_path,
            pipeline_config=self._pipeline_config,
            root_model_path=self._root_model_path,
        )
        return self._mistral_model.model

    def __call__(
        self,
        input_ids: TensorValue | np.ndarray,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Apply Mistral3 text encoder forward pass.

        Args:
            input_ids: Input token IDs as numpy array (preferred) or MAX TensorValue.
                       Passing numpy directly avoids unnecessary GPU->CPU transfer.
            attention_mask: Attention mask (not used, kept for compatibility).
            position_ids: Position IDs (not used, kept for compatibility).

        Returns:
            Tuple of hidden states from all layers as MAX TensorValues.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._mistral_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(input_ids, np.ndarray):
            arr = input_ids
        elif hasattr(input_ids, "to_numpy"):
            arr = input_ids.to_numpy()
        else:
            arr = np.asarray(input_ids)
        input_ids_list = arr.ravel().tolist()

        sampling_params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(max_new_tokens=1),
            sampling_params_defaults=self._pipeline_config.model.sampling_params_defaults,
        )
        request = TextGenerationRequest(
            request_id=RequestID(),
            model_name=self._text_encoder_path or "",
            prompt=input_ids_list,
            sampling_params=sampling_params,
        )

        try:
            asyncio.get_running_loop()
            loop = asyncio.new_event_loop()
            with ThreadPoolExecutor() as pool:
                fut = pool.submit(
                    loop.run_until_complete,
                    self._tokenizer.new_context(request),
                )
                context = fut.result()
            loop.close()
        except RuntimeError:
            context = asyncio.run(self._tokenizer.new_context(request))

        request_id = context.request_id
        replica_idx = 0
        num_steps = 1
        try:
            self._mistral_model.kv_manager.claim(
                request_id, replica_idx=replica_idx
            )
            self._mistral_model.kv_manager.alloc(
                context, replica_idx=replica_idx, num_steps=num_steps
            )
            kv_cache_inputs_list = (
                self._mistral_model.kv_manager.get_runtime_inputs(
                    [[context]], num_steps=num_steps
                )
            )
            kv_cache_inputs = kv_cache_inputs_list[0]
            model_inputs = self._mistral_model.prepare_initial_token_inputs(
                replica_batches=[[context]],
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=1,
            )
            model_outputs = self._mistral_model.execute(model_inputs=model_inputs)

            if model_outputs.hidden_states is None:
                logger.warning(
                    "Model did not return hidden states; "
                    "graph may not be compiled with return_hidden_states=ALL_LAYERS"
                )
                raise RuntimeError(
                    "Model did not return hidden states. "
                    "Ensure the model graph was compiled with "
                    "return_hidden_states=ALL_LAYERS."
                )
            return model_outputs.hidden_states
        finally:
            self._mistral_model.kv_manager.release(
                request_id, replica_idx=replica_idx
            )

    @property
    def session(self) -> InferenceSession:
        """Return the InferenceSession instance.

        Returns:
            InferenceSession: The compiled inference session.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._session
