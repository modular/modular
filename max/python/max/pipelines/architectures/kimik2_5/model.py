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
"""Implements the Kimi-K2.5 nn.model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs
from max.nn.layer import Module
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)

from ..deepseekV3.model import DeepseekV3Inputs
from .context import KimiK2_5TextAndVisionContext
from .kimik2_5 import KimiK2_5
from .model_config import KimiK2_5Config, KimiK2_5TextConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class KimiK2_5ModelInputs(DeepseekV3Inputs):
    """A class representing inputs for the KimiK2_5M model.

    This class encapsulates the input tensors required for the KimiK2_5M model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing.
    """

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed multimodal merge indices for the image embeddings.

    These are the locations of the image_token_id in the inputs fed to the model.

    Some indices may be negative, which means that they are ignored by the multimodal merge."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    grid_thw: list[Buffer] | None = None
    """Grid dimensions (temporal, height, width) for each image/video, shape (n_images, 3) per device."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention per device."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs per device."""

    vision_position_ids: list[Buffer] | None = None
    """Vision rotary position IDs per device."""

    max_grid_size: list[Buffer] | None = None
    """Maximum grid size for vision inputs per device."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class KimiK2_5Model(
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[KimiK2_5TextAndVisionContext],
):
    """A Kimi-K2.5 pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    embedding_model: Model
    """The compiled model for text embedding."""

    model_config: KimiK2_5Config | None
    """The Kimi-K2.5 model configuration."""

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

        self.model_config = None
        self._session = session  # reuse for on-device casts

        self.vision_model, self.language_model, self.embedding_model = (
            self.load_model(session)
        )

    def _create_language_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> KimiK2_5TextConfig:
        """Create model configuration from huggingface config."""
        raise NotImplementedError

    def load_model(
        self, session: InferenceSession
    ) -> tuple[Model, Model, Model]:
        """Load the compiled models into the MAX Engine session."""

        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        # `_host_input_row_offsets_prealloc` tensor needs to reserve space for
        # `max_batch_size` of requests on each DP rank.
        dp_size = self.pipeline_config.model.data_parallel_degree
        max_batch_size *= dp_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(self.devices[0])
        )

        # create batch context lengths tensor for each device
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

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

        # Split state dict into vision and language model components
        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in state_dict.items():
            if key.startswith("vision_tower.") or key.startswith(
                "mm_projector."
            ):
                vision_state_dict[key] = value
            elif key.startswith("language_"):
                llm_state_dict[key] = value
            else:
                raise ValueError(
                    f"Key: {key} is not part of the vision or language model"
                )

        # Create the LM model first
        config = self._create_language_model_config(state_dict)

        n_devices = len(self.devices)
        # PipelineConfig.ep_size exists in config.config; type checker may not resolve it here.
        if n_devices > 1 and self.pipeline_config.ep_size != n_devices:  # type: ignore[attr-defined]
            raise ValueError("Only the EP strategy is supported.")

        self.ep_comm_initializer: EPCommInitializer | None = None
        # Skip EP initialization in virtual device mode (compilation-only)
        # since NVSHMEM functions cannot be linked without real GPU devices.
        # We still keep ep_config to generate the correct graph structure.
        if config.ep_config is not None and not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP initialization is successful."
                )

        # Generate the full KimiK2_5Config from HuggingFace config and LM config
        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=config,
        )
        self.model_config = kimik2_5_config

        # Use the local non-optional variable to satisfy typing.
        self.model: Module = KimiK2_5(kimik2_5_config)
        self.model.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Build and compile vision model
        timer = CompilationTimer("vision model")
        vision_graph = self._build_vision_graph(
            kimik2_5_config, vision_state_dict
        )
        timer.mark_build_complete()
        vision_model = session.load(
            vision_graph, weights_registry=vision_state_dict
        )
        timer.done()

        # Build and compile language model
        timer = CompilationTimer("language model")
        language_graph = self._build_language_graph(
            kimik2_5_config, llm_state_dict
        )
        timer.mark_build_complete()
        language_model = session.load(
            language_graph, weights_registry=llm_state_dict
        )
        timer.done()

        # Build and compile language model for embeddings
        timer = CompilationTimer("language embedding model")
        embedding_graph = self._build_language_embedding_graph(
            kimik2_5_config, llm_state_dict
        )
        timer.mark_build_complete()
        embedding_model = session.load(
            embedding_graph, weights_registry=llm_state_dict
        )
        timer.done()

        return vision_model, language_model, embedding_model

    def _build_vision_graph(
        self, config: KimiK2_5Config, state_dict: dict[str, WeightData]
    ) -> Graph:
        """Build the vision model graph for processing images."""
        assert isinstance(self.model, KimiK2_5)
        vision_encoder = self.model.vision_encoder
        multimodal_projector = self.model.multimodal_projector

        # Define vision graph input types - one per device
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", config.vision_config.vt_hidden_size],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        grid_thw_types = [
            TensorType(
                DType.int64,
                shape=["n_images", 3],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        max_seqlen_types = [
            TensorType(
                DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        vision_rot_pos_ids_types = [
            TensorType(
                DType.int32,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        signal_buffer_types: list[BufferType] = signals.input_types()

        max_grid_size_types = [
            TensorType(
                DType.int32,
                shape=[],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        # Build the vision graph
        with Graph(
            "kimik2_5_vision",
            input_types=tuple(
                [
                    *pixel_values_types,
                    *grid_thw_types,
                    *cu_seqlens_types,
                    *max_seqlen_types,
                    *vision_rot_pos_ids_types,
                    *signal_buffer_types,
                    *max_grid_size_types,
                ]
            ),
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            grid_thw_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_seqlen_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            rot_pos_ids_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            signal_buffers = [inp.buffer for inp in all_inputs]

            max_grid_size_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            # Execute vision transformer
            image_embeddings = vision_encoder(
                pixel_values=pixel_values_list,
                grid_thw=grid_thw_list,
                cu_seqlens=cu_seqlens_list,
                max_seqlen=max_seqlen_list,
                rot_pos_ids=rot_pos_ids_list,
                signal_buffers=signal_buffers,
                max_grid_size=max_grid_size_list,
            )
            # Ensure we have a valid output
            assert image_embeddings is not None, (
                "Vision encoder must return a valid output"
            )
            image_embeddings = multimodal_projector(image_embeddings)

            graph.output(image_embeddings)

            return graph

    def _build_language_embedding_graph(
        self, config: KimiK2_5Config, state_dict: dict[str, WeightData]
    ) -> Graph:
        """Build the language model graph for text generation with image embeddings."""
        assert isinstance(self.model, KimiK2_5)
        language_model = self.model.language_model
        assert language_model is not None, "Language model must be initialized"

        # Create the graph
        with Graph(
            "deepseekV3_embedding_graph",
            input_types=language_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                _,
                _,
                _,
                _,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)
            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            outputs = language_model.embed_tokens(
                tokens.tensor,
                signal_buffers,
            )

            graph.output(*outputs)

        return graph

    def _build_language_graph(
        self, config: KimiK2_5Config, state_dict: dict[str, WeightData]
    ) -> Graph:
        """Build the language model graph for text generation with image embeddings."""
        assert isinstance(self.model, KimiK2_5)
        language_model = self.model.language_model
        assert language_model is not None, "Language model must be initialized"

        # Create the graph
        with Graph(
            "deepseekV3_graph",
            input_types=language_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)
            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            # Unmarshal the KV cache arguments.
            fetch_types = self.kv_params.get_symbolic_inputs()[0]
            len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
            kv_caches_per_dev = self._unflatten_kv_inputs(
                [next(variadic_args_iter) for _ in range(len_of_kv_inputs)]
            )

            # Unmarshal the batch context lengths
            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            # all remaining arguments are for EP inputs
            ep_model_inputs = list(variadic_args_iter)

            outputs = language_model(
                tokens.tensor,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                devices_input_row_offsets.tensor,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)

        return graph

    def _merge_input_ids_with_image_features(
        self, image_embeddings: list[Buffer], text_embeddings: list[Buffer]
    ) -> list[Buffer]:
        raise NotImplementedError

    def _process_model_outputs(self, outputs: list[Buffer]) -> ModelOutputs:
        raise NotImplementedError

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the KimiK2_5 model with the prepared inputs."""
        assert isinstance(model_inputs, KimiK2_5ModelInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "KimiK2_5 requires KV cache inputs"
        )

        if model_inputs.has_vision_inputs:
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.max_grid_size is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.grid_thw is not None

            assert self.model_config is not None, (
                "Model config must be initialized"
            )

            image_embeddings = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.grid_thw,
                *model_inputs.cu_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.vision_position_ids,
                *model_inputs.signal_buffers,
                *model_inputs.max_grid_size,
            )  # output should be projected via multimodal_projector as part of vision_graph

            assert len(image_embeddings) == len(self.devices)
            # assert image embeddings have the same hidden size as language model
            for output in image_embeddings:
                assert isinstance(output, Buffer)
                assert (
                    output.shape[1]
                    == self.huggingface_config.text_config.hidden_size
                )

        text_embeddings = self.embedding_model.execute(
            model_inputs.tokens, *model_inputs.signal_buffers
        )

        self._merge_input_ids_with_image_features(
            image_embeddings, text_embeddings
        )

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else self.ep_comm_initializer.model_inputs()
        )

        # Execute language model with text and image embeddings
        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.host_input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.data_parallel_splits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            *model_inputs.batch_context_lengths,
            *ep_inputs,
        )

        return self._process_model_outputs(model_outputs)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        raise NotImplementedError

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV3Inputs:
        raise NotImplementedError
