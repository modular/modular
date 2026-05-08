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

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    DeviceSpec,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.buffer_utils import cast_tensor_to
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces.request import RequestID
from max.nn.comm import Signals
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    upper_bounded_default,
)
from max.pipelines.lib.config.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from max.pipelines.lib.quant import parse_quant_config
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.support.algorithm import flatten2d
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

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

    precomputed_image_embeddings: list[Buffer] | None = None
    """Pre-computed image embeddings from VisionEncoderCache."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    grid_thws: list[Buffer] | None = None
    """Grid dimensions (temporal, height, width) for each image/video, shape (n_images, 3) per device."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention per device."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs per device."""

    vision_position_ids: list[Buffer] | None = None
    """Vision rotary position IDs per device."""

    language_image_embeddings: list[Buffer] = field(default_factory=list)
    """Per-device image embeddings for the language model graph.
    Shape [0, hidden_size] during decode, [num_patches, hidden_size] during prefill."""

    language_image_token_indices: list[Buffer] = field(default_factory=list)
    """Per-device scatter indices for the language model graph.
    Shape [0] during decode, [num_image_tokens] during prefill."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns the language model input ABI tuple."""
        return (
            self.tokens,
            *self.language_image_embeddings,
            *self.language_image_token_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class KimiK2_5Model(
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[KimiK2_5TextAndVisionContext],
):
    """A Kimi-K2.5 pipeline model for multimodal text generation."""

    _GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE = 8 * 1024**3

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")
        self.session = session
        self._ve_cache: VisionEncoderCache[KimiK2_5TextAndVisionContext] = (
            VisionEncoderCache(
                max_entries=pipeline_config.runtime.max_vision_cache_entries
            )
        )
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

        self.vision_model, self.language_model = self.load_model(session)

    @property
    def model(self) -> Model:
        """Expose language model for graph capture/replay.

        Only the language model is captured since vision runs
        during prefill
        """
        return self.language_model

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        encoding = pipeline_config.model.quantization_encoding
        if (
            encoding is not None
            and is_float4_encoding(encoding)
            and kv_cache_config.kv_cache_format is None
        ):
            cache_dtype = DType.float8_e4m3fn
        return KimiK2_5TextConfig.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.text_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for DeepseekV2, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.text_config.max_position_embeddings})."
            ) from e

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> KimiK2_5TextConfig:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config.text_config

        # quantization_config lives at the top level of the HF config, not
        # under text_config. Propagate it so parse_quant_config() finds it.
        if hasattr(self.huggingface_config, "quantization_config"):
            config.quantization_config = (
                self.huggingface_config.quantization_config
            )

        # data_parallel_degree controls the attention strategy:
        #   == num_devices  ->  DP attention  (each device owns a batch shard)
        #   == 1            ->  TP attention  (heads sharded, tokens replicated)
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree
        max_batch_total_tokens = (
            self.pipeline_config.runtime.max_batch_total_tokens
        )
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif self.pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = self.dtype
        quant_config = parse_quant_config(config, state_dict, dtype)

        # Check if EP should be configured
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % len(self.devices) != 0:
                raise ValueError(
                    "If you are running with expert parallelism, ep_size must"
                    " be set to the total number of GPUs across nodes."
                )
            n_nodes = ep_size // len(self.devices)

            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=data_parallel_degree,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            )

            is_mxfp4 = quant_config is not None and quant_config.is_mxfp4
            ep_dispatch_dtype = DType.uint8 if is_mxfp4 else dtype

            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_gpus_per_node=len(self.devices),
                n_nodes=n_nodes,
                dispatch_quant_config=None,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            )

            if config.n_shared_experts == 1 and not is_mxfp4:
                # Only enable shared expert fusion if the shared expert is of
                # the same shape and dtype as routed experts.
                ep_kwargs["fused_shared_expert"] = True

            if quant_config is not None:
                ep_kwargs["dispatch_quant_config"] = quant_config

            ep_config = EPConfig(**ep_kwargs)

        norm_dtype = state_dict[
            "language_model.layers.0.self_attn.kv_a_layernorm.weight"
        ].dtype

        if config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            correction_bias_dtype = state_dict[correction_bias_key].dtype
        else:
            correction_bias_dtype = None

        # Initialize config with parameters from pipeline_config
        model_config = KimiK2_5TextConfig.initialize(self.pipeline_config)

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states

        num_devices = len(self.devices)
        if num_devices > 1:
            if ep_size > 1:
                attn_strategy = "TP" if data_parallel_degree == 1 else "DP"
                moe_strategy = "EP"
            else:
                attn_strategy = "TP"
                moe_strategy = "TP"
            logger.info(
                f"KimiK2_5: data_parallel_degree={data_parallel_degree},"
                f" ep_size={ep_size}. Use {attn_strategy}-attention +"
                f" {moe_strategy}-MoE strategy."
            )

        return model_config

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of our model."""
        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        n_gpus_per_node = len(model_config.device_specs)

        encoding = pipeline_config.model.quantization_encoding
        assert encoding is not None

        def _n_elems_to_bytes(n_elems: int) -> int:
            dtype = supported_encoding_dtype(encoding).size_in_bytes
            if is_float4_encoding(encoding):
                # Account for the scales. For NVFP4 format, every 16 FP4 elements
                # share one FP8 scale factor. The size of the scales is one
                # eighth of the size of the FP4 quants (8 bits / (16 * 4 bits)).
                return int(n_elems // 2 * dtype * 1.125)
            else:
                return n_elems * dtype

        assert model_config.huggingface_config is not None
        config = model_config.huggingface_config.text_config
        assert config is not None
        n_sparse_layers = (
            config.num_hidden_layers - config.first_k_dense_replace
        )
        n_mtp_layers = config.num_nextn_predict_layers

        # Note: All the following calculations are not exact, but they are
        # better than directly using the raw weights size.

        # First, Calculate the lm_head/embed_tokens size.
        # There are always in Bf16.
        lm_head_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        embed_tokens_size = lm_head_size

        # Subtract the lm_head/embed_tokens size from the weights size
        weights_size -= lm_head_size + embed_tokens_size
        weights_size -= (lm_head_size + embed_tokens_size) * n_mtp_layers

        # We don't use the MTP module for now, so subtract the MTP attn/moe size.
        # Estimate the MTP module size by assuming the MTP layer is of the same
        # size as a sparse model layer.
        weights_size = int(
            weights_size * n_sparse_layers / (n_sparse_layers + n_mtp_layers)
        )

        # Calculate the routing experts and the shared experts size.
        expert_elems = (
            config.moe_intermediate_size * config.hidden_size * 3
        )  # A factor of 3 accounts for the gate/up/down proj weights.
        expert_size = _n_elems_to_bytes(expert_elems)
        routing_experts_size = (
            n_sparse_layers * config.n_routed_experts * expert_size
        )
        shared_experts_size = (
            n_sparse_layers * config.n_shared_experts * expert_size
        )

        # Estimate the size of the attention weights.
        attn_weights_size = (
            weights_size - routing_experts_size - shared_experts_size
        )

        # If we use DP attention, attention weights are duplicated on each DP rank.
        total_size = attn_weights_size * model_config.data_parallel_degree

        # The shared experts are duplicated on each device.
        total_size += shared_experts_size * n_gpus_per_node

        ep_size = max(pipeline_config.runtime.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # we don't support mixing EP and TP strategies yet.
            # ep_size must be equal to n_gpus_per_node * n_nodes
            assert ep_size % n_gpus_per_node == 0
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Add back the lm_head/embed_tokens size, they will never be duplicated.
        total_size += lm_head_size + embed_tokens_size

        return total_size

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for model execution.

        This accounts for temporary memory buffers used during model execution,
        such as intermediate activations and working buffers.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration

        Returns:
            Estimated activation memory in bytes
        """

        encoding = pipeline_config.model.quantization_encoding
        assert encoding is not None
        mla_activation_memory: int = 0
        moe_activation_memory: int = 0
        ep_buffer_memory = 0

        # During the prefill, we need to up-project all the KV cache for
        # current requests. The total context length of requests in a batch
        # should be limited by max_batch_total_tokens.
        if pipeline_config.runtime.pipeline_role != "decode_only":
            max_kv_length: int = 0

            if pipeline_config.runtime.max_batch_total_tokens is None:
                # If max_batch_total_tokens is not set, we use max_length.
                max_kv_length = pipeline_config.model.max_length or 0
            else:
                max_kv_length = pipeline_config.runtime.max_batch_total_tokens

            mla_activation_memory += (
                pipeline_config.model.data_parallel_degree
                * 2  # 2 for K and V
                * max_kv_length
                * huggingface_config.text_config.num_attention_heads
                * huggingface_config.text_config.qk_nope_head_dim
                * pipeline_config.model.kv_cache.cache_dtype.size_in_bytes
            )

        # Estimate buffer and activation memory during Expert Parallel MoE.
        if pipeline_config.runtime.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)

            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=pipeline_config.runtime.ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )

            # Calculate the maximum number of tokens a rank may receive during
            # all-to-all routing. Each token selects top_k experts, and in the
            # worst case all selections land on one rank.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                huggingface_config.text_config.n_routed_experts,
                pipeline_config.runtime.ep_size
                * huggingface_config.text_config.num_experts_per_tok,
            )

            if pipeline_config.runtime.ep_use_allreduce:
                max_recv_tokens_per_rank = (
                    pipeline_config.runtime.max_batch_input_tokens
                    * min(
                        huggingface_config.text_config.n_routed_experts
                        // n_gpus_per_node,
                        huggingface_config.text_config.num_experts_per_tok,
                    )
                )

            # The maximal activation memory usage happens at the second
            # grouped_matmul in the MoE layer. The input for that matmul would
            # of shape [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.text_config.moe_intermediate_size
                * supported_encoding_dtype(encoding).size_in_bytes
            )

            # The output would be of shape [max_recv_tokens_per_rank, hidden_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.text_config.hidden_size
                * DType.bfloat16.size_in_bytes  # output is always bfloat16.
            )

            # Adding 256MB per GPU to account for misc items (e.g. FP8 scalars).
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            # EP SHMEM communication buffers are persistent (allocated once at
            # model init, not freed between layers).
            n_nodes = pipeline_config.runtime.ep_size // n_gpus_per_node

            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=huggingface_config.text_config.hidden_size,
                dispatch_dtype=supported_encoding_dtype(encoding),
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=huggingface_config.text_config.n_routed_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=huggingface_config.text_config.num_experts_per_tok,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node

            logger.info(
                "Estimated EP SHMEM buffer memory: "
                f"{to_human_readable_bytes(ep_buffer_memory)}"
            )

        # We only need to consider the maximum of the MLA and MoE activation
        # memories, because the MLA and MoE layers are executed sequentially.
        activation_memory = max(mla_activation_memory, moe_activation_memory)
        activation_memory += ep_buffer_memory

        if pipeline_config.runtime.device_graph_capture:
            graph_capture_headroom = (
                cls._GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE
                * len(pipeline_config.model.device_specs)
            )
            activation_memory += graph_capture_headroom
            logger.info(
                "Added graph capture headroom to activation memory: %s",
                to_human_readable_bytes(graph_capture_headroom),
            )

        if activation_memory != 0:
            logger.info(
                f"Estimated activation memory: {to_human_readable_bytes(activation_memory)}"
            )

        return activation_memory

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Load the model with the given weights."""

        max_batch_size = self.pipeline_config.runtime.max_batch_size
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
                huggingface_config=self.huggingface_config.text_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        # Create the LM model first
        config = self._create_model_config(state_dict)

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
        self.nn_model = KimiK2_5(self.model_config)
        self.nn_model.load_state_dict(
            state_dict, weight_alignment=1, strict=True
        )
        self.state_dict = self.nn_model.state_dict()
        logger.info("Loaded Weights")

        # Load the vision + language model.
        with CompilationTimer("vision + language model") as timer:
            # Create a new module to hold both models
            module = Graph.empty_module()

            # Build the vision graph in the module
            self._build_vision_graph(kimik2_5_config, state_dict, module=module)

            # Build the language graph in the module
            language_graph = self._build_language_graph(config, module=module)
            timer.mark_build_complete()
            vision_model, language_model = session.load_all(
                language_graph, weights_registry=self.state_dict
            )

        return vision_model, language_model

    def _build_vision_graph(
        self,
        config: KimiK2_5Config,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> Graph:
        """Build the vision model graph for processing images."""
        assert isinstance(self.nn_model, KimiK2_5)
        vision_encoder = self.nn_model.vision_encoder

        # Define vision graph input types - one per device
        # pixel_values are raw NCHW patches fed into PatchEmbedding's Conv2d.
        pixel_values_types = [
            TensorType(
                config.vision_config.dtype,
                shape=[
                    "n_patches",
                    config.vision_config.in_channels,
                    config.vision_config.patch_size,
                    config.vision_config.patch_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        grid_thws_types = [
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
                DType.int64,
                shape=["n_patches"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        signal_buffer_types: list[BufferType] = signals.input_types()

        # Build the vision graph
        with Graph(
            "kimik2_5_vision_graph",
            input_types=tuple(
                [
                    *pixel_values_types,
                    *grid_thws_types,
                    *cu_seqlens_types,
                    *max_seqlen_types,
                    *vision_rot_pos_ids_types,
                    *signal_buffer_types,
                ]
            ),
            module=module,
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            grid_thws_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_seqlen_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            rot_pos_ids_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            n_signal_buffers = len(signal_buffer_types)
            signal_buffers = [
                inp.buffer for inp in all_inputs[:n_signal_buffers]
            ]

            # Execute vision transformer (includes patch merger projection).
            # max_h and max_w are computed at runtime inside Transformer.__call__
            # from the grid_thws input via ops.max.
            image_embeddings = vision_encoder(
                pixel_values=pixel_values_list,
                grid_thws=grid_thws_list,
                input_row_offsets=cu_seqlens_list,
                max_seq_len=max_seqlen_list,
                position_ids=rot_pos_ids_list,
                signal_buffers=signal_buffers,
            )
            assert image_embeddings is not None, (
                "Vision encoder must return a valid output"
            )

            graph.output(*image_embeddings)

            return graph

    def _build_language_graph(
        self,
        config: KimiK2_5TextConfig,
        module: Module | None = None,
    ) -> Graph:
        """Build the language model graph for text generation with image embeddings."""
        assert isinstance(self.nn_model, KimiK2_5)
        language_model = self.nn_model.language_model
        assert language_model is not None, "Language model must be initialized"

        # Create the graph
        with Graph(
            "kimik2_5_language_graph",
            input_types=language_model.input_types(self.kv_params),
            module=module,
        ) as graph:
            n = len(self.devices)
            tokens, all_inputs = graph.inputs[0], graph.inputs[1:]
            image_embeddings, all_inputs = all_inputs[:n], all_inputs[n:]
            image_token_indices, all_inputs = all_inputs[:n], all_inputs[n:]
            (
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = all_inputs

            variadic_args_iter = iter(variadic_args)
            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            # Unmarshal the KV cache arguments.
            fetch_types = (
                self.kv_params.get_symbolic_inputs().inputs[0].flatten()
            )
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
                tokens=tokens.tensor,
                image_embeddings=[v.tensor for v in image_embeddings],
                image_token_indices=[v.tensor for v in image_token_indices],
                signal_buffers=signal_buffers,
                kv_collections=kv_caches_per_dev,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=devices_input_row_offsets.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                data_parallel_splits=data_parallel_splits.tensor,
                batch_context_lengths=batch_context_lengths,
                ep_inputs=ep_model_inputs,
            )

            graph.output(*outputs)

        return graph

    @cached_property
    def _empty_image_embeddings(
        self,
    ) -> list[Buffer]:
        """Empty ``[0, D]`` image embeddings shared across all non-vision calls.

        The language model ABI always includes image embeddings and scatter
        indices in its input tuple, even during text-only prefill and decode.
        These zero-length buffers act as no-op placeholders so the scatter
        sees zero indices and does nothing.
        """
        image_embeddings = Buffer.zeros(
            shape=[
                0,
                self.huggingface_config.text_config.hidden_size,
            ],
            dtype=DType.bfloat16,
        ).to(self.devices)
        return image_embeddings

    @cached_property
    def _empty_image_image_token_indices(self) -> list[Buffer]:
        """Empty ``[0]`` scatter indices for text-only and decode calls."""
        return Buffer.zeros(
            shape=[0],
            dtype=DType.int32,
        ).to(self.devices)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, KimiK2_5ModelInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "KimiK2_5 requires KV cache inputs"
        )
        if model_inputs.has_vision_inputs:
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.grid_thws is not None
            assert self.model_config is not None

            image_embeddings = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.grid_thws,
                *model_inputs.cu_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.vision_position_ids,
                *model_inputs.signal_buffers,
            )

            assert len(image_embeddings) == len(self.devices)
            for output in image_embeddings:
                assert isinstance(output, Buffer)
                assert (
                    output.shape[1]
                    == self.huggingface_config.text_config.hidden_size
                )
            assert (
                model_inputs.image_token_indices[0].shape[0]
                == image_embeddings[0].shape[0]
            ), (
                f"The size of scatter indices must match the number of image embeddings. "
                f"Got: {model_inputs.image_token_indices[0].shape[0]} != {image_embeddings[0].shape[0]}"
            )

            # Update language model placeholders with actual vision outputs.
            model_inputs.language_image_embeddings = image_embeddings
            model_inputs.language_image_token_indices = (
                model_inputs.image_token_indices
            )

        model_outputs = self.language_model.execute(*model_inputs.buffers)
        return self._process_model_outputs(model_outputs)

    def release(self, request_id: RequestID) -> None:
        """Release vision encoder cache entries for a completed request."""
        self._ve_cache.release_request(request_id)

    def _process_model_outputs(
        self, model_outputs: list[Buffer]
    ) -> ModelOutputs:
        num_outputs = len(model_outputs)

        # Possible output configurations:
        # - 4 outputs: next_token_logits, logits, logit_offsets + hidden_states
        # - 3 outputs: next_token_logits, logits, logit_offsets (variable logits)
        # - 2 outputs: next_token_logits + hidden_states
        # - 1 output: next_token_logits only

        if num_outputs == 4:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            assert isinstance(model_outputs[3], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif num_outputs == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        elif num_outputs == 2:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
                hidden_states=model_outputs[1],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def _collect_vision_encoder_request_metadata(
        self,
        context_batch: Sequence[KimiK2_5TextAndVisionContext],
        uncached_contexts: Sequence[KimiK2_5TextAndVisionContext],
        vision_inputs: dict[str, list[Buffer]],
    ) -> dict[str, Any]:
        """Collect debug metadata for a vision-encoder invocation."""
        patch_size = int(self.model_config.vision_config.patch_size)
        merge_cfg = self.model_config.vision_config.merge_kernel_size
        if isinstance(merge_cfg, int):
            merge_h, merge_w = merge_cfg, merge_cfg
        else:
            merge_h = int(merge_cfg[0])
            merge_w = int(merge_cfg[1])
        merge_sq = merge_h * merge_w

        request_ids = [str(ctx.request_id) for ctx in context_batch]
        total_images = sum(len(ctx.images) for ctx in context_batch)
        next_images_total = sum(len(ctx.next_images) for ctx in context_batch)
        context_sequence_lengths = [
            {
                "request_id": str(ctx.request_id),
                "active_sequence_length": int(ctx.tokens.active_length),
                "processed_sequence_length": int(ctx.tokens.processed_length),
                "total_sequence_length": len(ctx.tokens),
                "context_needs_vision_encoding": bool(
                    ctx.needs_vision_encoding
                ),
                "image_count": len(ctx.images),
                "next_image_count": len(ctx.next_images),
            }
            for ctx in context_batch
        ]

        per_image_metadata: list[dict[str, Any]] = []
        image_counter = 0
        for ctx in uncached_contexts:
            for i, (img, thw) in enumerate(
                zip(ctx.images, ctx.grid_thws, strict=True)
            ):
                if (
                    img.image_hash is None
                    or self._ve_cache.lookup(img.image_hash) is not None
                ):
                    continue

                t = int(thw[0])
                h = int(thw[1])
                w = int(thw[2])
                pixel_values_shape = [int(x) for x in img.pixel_values.shape]
                per_image_metadata.append(
                    {
                        "request_id": str(ctx.request_id),
                        "image_index_in_request": i,
                        "image_index_in_uncached_batch": image_counter,
                        "image_hash_present": img.image_hash is not None,
                        "token_span": {
                            "start_idx": int(img.start_idx),
                            "end_idx": int(img.end_idx),
                            "length": int(img.end_idx - img.start_idx),
                        },
                        "size_before_patching": {
                            "temporal_frames": t,
                            "height_px": h * patch_size,
                            "width_px": w * patch_size,
                        },
                        "size_after_patching": {
                            "patch_grid_thw": [t, h, w],
                            "patch_tensor_shape": pixel_values_shape,
                        },
                        "size_after_patch_merger": {
                            "merge_kernel_hw": [merge_h, merge_w],
                            "merged_token_count": int((t * h * w) // merge_sq),
                        },
                    }
                )
                image_counter += 1

        images_needing_encoding = len(per_image_metadata)
        if vision_inputs["grid_thws"]:
            expected_from_inputs = int(vision_inputs["grid_thws"][0].shape[0])
        else:
            expected_from_inputs = 0

        return {
            "request_ids": request_ids,
            "batch_size": len(context_batch),
            "uncached_context_count": len(uncached_contexts),
            "vision_cache_enabled": bool(self._ve_cache.enabled),
            "vision_input_dtype": str(self.model_config.vision_config.dtype),
            "vision_input_shapes": {
                key: [int(x) for x in values[0].shape] if values else []
                for key, values in vision_inputs.items()
            },
            "images": {
                "total_images_in_batch": total_images,
                "next_images_total": next_images_total,
                "images_needing_encoding": images_needing_encoding,
                "images_not_needing_encoding": (
                    total_images - images_needing_encoding
                ),
                "images_already_encoded_in_prior_steps": (
                    total_images - next_images_total
                ),
                "images_skipped_due_to_cache": max(
                    0, next_images_total - images_needing_encoding
                ),
                "images_needing_encoding_from_vision_input_shapes": (
                    expected_from_inputs
                ),
            },
            "context_sequence_lengths": context_sequence_lengths,
            "per_image_metadata": per_image_metadata,
        }

    def _prepare_vision_inputs(
        self,
        context_batch: Sequence[KimiK2_5TextAndVisionContext],
    ) -> dict[str, list[Buffer]] | None:
        """Assemble per-device vision encoder ``Buffer``s for uncached images.

        Skips images already in the vision encoder cache. Prepares pixel
        values, grid THWs, cumulative sequence lengths, max sequence
        length, and RoPE position IDs.

        Args:
            context_batch: Contexts with at least one uncached image
                (from ``get_uncached_contexts``).

        Returns:
            Dictionary of named per-device ``Buffer`` lists, or ``None``
            when all images are already cached.
        """
        all_pixel_values_list: list[npt.NDArray[Any]] = []
        all_grid_thws_list: list[npt.NDArray[np.int64]] = []
        all_position_ids_list: list[npt.NDArray[np.int64]] = []

        for ctx in context_batch:
            pos_offset = 0
            for i, img in enumerate(ctx.images):
                thw = ctx.grid_thws[i]
                n_pos = int(thw[0] * thw[1] * thw[2])

                if (
                    img.image_hash is not None
                    and self._ve_cache.lookup(img.image_hash) is None
                ):
                    all_pixel_values_list.append(img.pixel_values)
                    all_grid_thws_list.append(thw)
                    all_position_ids_list.append(
                        ctx.position_ids[pos_offset : pos_offset + n_pos]
                    )

                pos_offset += n_pos

        if not all_pixel_values_list:
            return None

        all_pixel_values = np.concatenate(all_pixel_values_list, axis=0)
        all_grid_thws_np = np.vstack(all_grid_thws_list).astype(np.int64)

        # Cumulative patch-sequence lengths for packed full-attention.
        seq_lens = [int(np.prod(g)) for g in all_grid_thws_np]
        cu_seqlens_np = np.zeros(len(seq_lens) + 1, dtype=np.uint32)
        np.cumsum(seq_lens, out=cu_seqlens_np[1:])

        max_seqlen_np = np.array([max(seq_lens)], dtype=np.uint32)

        position_ids_np = np.concatenate(all_position_ids_list).astype(np.int64)

        device0 = self.devices[0]
        vision_dtype = self.model_config.vision_config.dtype
        # GPU-backed vision inputs: create on device0, cast to vision dtype, then replicate
        pixel_values_f32 = Buffer.from_numpy(all_pixel_values).to(device0)
        pixel_values_buf = (
            pixel_values_f32
            if pixel_values_f32.dtype == vision_dtype
            else cast_tensor_to(
                pixel_values_f32, vision_dtype, session=self.session
            )
        )
        grid_thws_buf = Buffer.from_numpy(all_grid_thws_np).to(device0)
        cu_seqlens_buf = Buffer.from_numpy(cu_seqlens_np).to(device0)
        vision_position_ids_buf = Buffer.from_numpy(position_ids_np).to(device0)
        max_seqlen_buf = Buffer.from_numpy(max_seqlen_np)
        return {
            "pixel_values": [pixel_values_buf.to(d) for d in self.devices],
            "grid_thws": [grid_thws_buf.to(d) for d in self.devices],
            "cu_seqlens": [cu_seqlens_buf.to(d) for d in self.devices],
            "max_seqlen": [max_seqlen_buf for _ in self.devices],
            "vision_position_ids": [
                vision_position_ids_buf.to(d) for d in self.devices
            ],
        }

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> KimiK2_5ModelInputs:
        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        # Allocate the model inputs on pinned memory for faster h2d
        # transfer speeds. If model is on host, then fall back to normal
        # pageable memory. We initialize these empty max tensors by exporting
        # to numpy over dlpack and using numpy methods.
        # TODO: move rest of inputs to pinned memory
        device0 = self.devices[0]
        pinned = not device0.is_host

        # If we are not in decode only mode, we need to create a list of
        # tensors containing the context length of each batch. Need by MLA
        # prefill.
        if self.pipeline_config.runtime.pipeline_role != "decode_only":

            def align_length(length: int) -> int:
                page_size = self.kv_cache_config.kv_cache_page_size
                return (length + page_size - 1) // page_size * page_size

            for i, batch in enumerate(replica_batches):
                curr_length = sum(
                    [align_length(ctx.tokens.current_position) for ctx in batch]
                )
                self._batch_context_lengths_prealloc_cpu[i][0] = curr_length

            if dp != len(self.devices):
                assert dp == 1
                # Duplicate the batch context lengths for each device.
                for dev_idx in range(1, len(self.devices)):
                    self._batch_context_lengths_prealloc_cpu[dev_idx][0] = (
                        self._batch_context_lengths_prealloc_cpu[0][0].item()
                    )

        context_batch = flatten2d(replica_batches)
        # Create tokens
        tokens: Buffer
        pinned_input_row_offsets: Buffer
        if len(context_batch) == 0:
            if pinned:
                tokens = DevicePinnedBuffer(
                    shape=[0], dtype=DType.int64, device=device0
                )
            else:
                tokens = Buffer(shape=[0], dtype=DType.int64, device=device0)
            host_input_row_offsets = Buffer.zeros(shape=[1], dtype=DType.uint32)

            if pinned:
                pinned_input_row_offsets = DevicePinnedBuffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            else:
                pinned_input_row_offsets = Buffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            device_input_row_offsets = pinned_input_row_offsets.to(device0)
        else:
            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            num_tokens = sum(ctx.tokens.active_length for ctx in context_batch)
            tokens_host: Buffer
            if pinned:
                tokens_host = DevicePinnedBuffer(
                    shape=(num_tokens,),
                    dtype=DType.int64,
                    device=device0,
                )
            else:
                tokens_host = Buffer(
                    shape=(num_tokens,),
                    dtype=DType.int64,
                    device=device0,
                )
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_host.to_numpy(),
            )
            tokens = tokens_host.to(device0)

            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            # Get input_row_offsets: start and end position of each batch in the
            # combined total_seq_len dimension.
            input_row_offsets = np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )

            # FIXME GEX-3121: There is a bug when using pinned buffer as graph cpu input:
            # `Expected Device(type=cpu,id=0), but was on device Device(type=gpu,id=0)`
            # Thus we set up both a non-pinned and a pinned cpu buffer as workaround.
            host_input_row_offsets = Buffer(
                shape=(len(context_batch) + 1,),
                dtype=DType.uint32,
            )
            host_input_row_offsets.to_numpy()[:] = input_row_offsets[:]

            if pinned:
                pinned_input_row_offsets = DevicePinnedBuffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            else:
                pinned_input_row_offsets = Buffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            pinned_input_row_offsets.to_numpy()[:] = input_row_offsets[:]
            device_input_row_offsets = pinned_input_row_offsets.to(device0)

        data_parallel_splits = Buffer.from_numpy(
            compute_data_parallel_splits(replica_batches)
        )

        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else tuple(self.ep_comm_initializer.model_inputs())
        )

        uncached_contexts = self._ve_cache.get_uncached_contexts(context_batch)

        if uncached_contexts:
            vision_inputs = self._prepare_vision_inputs(uncached_contexts)
            assert vision_inputs is not None

            vision_metadata = self._collect_vision_encoder_request_metadata(
                context_batch=context_batch,
                uncached_contexts=uncached_contexts,
                vision_inputs=vision_inputs,
            )
            try:
                vision_embeds = self.vision_model.execute(
                    *vision_inputs["pixel_values"],
                    *vision_inputs["grid_thws"],
                    *vision_inputs["cu_seqlens"],
                    *vision_inputs["max_seqlen"],
                    *vision_inputs["vision_position_ids"],
                    *self.signal_buffers,
                )
            except Exception as err:
                failure_payload = {
                    "stage": "prepare_initial_token_inputs",
                    "error": {
                        "type": type(err).__name__,
                        "message": str(err),
                        "repr": repr(err),
                    },
                    "metadata": vision_metadata,
                }
                logger.exception(
                    "Vision encoder failed. Request metadata: %s",
                    json.dumps(failure_payload, sort_keys=True),
                )
                raise
            assert len(vision_embeds) == len(self.devices)

            merge_size = self.model_config.vision_config.merge_kernel_size
            merge_sq = merge_size[0] * merge_size[1]
            token_counts = [
                int(thw[0] * thw[1] * thw[2]) // merge_sq
                for ctx in uncached_contexts
                for img, thw in zip(ctx.images, ctx.grid_thws, strict=True)
                if img.image_hash is not None
                and self._ve_cache.lookup(img.image_hash) is None
            ]
        else:
            vision_embeds = self._empty_image_embeddings
            token_counts = []

        precomputed_image_embeddings, image_token_indices_np = (
            self._ve_cache.prepare_vision_outputs(
                context_batch,
                uncached_contexts,
                vision_embeds,
                token_counts,
                n_devices=len(self.devices),
                empty_embeddings=self._empty_image_embeddings,
            )
        )
        image_token_indices_buf = Buffer.from_numpy(image_token_indices_np).to(
            self.devices[0]
        )
        image_token_indices = [
            image_token_indices_buf.to(d) for d in self.devices
        ]

        return KimiK2_5ModelInputs(
            tokens=tokens,
            input_row_offsets=device_input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            precomputed_image_embeddings=precomputed_image_embeddings,
            image_token_indices=image_token_indices,
            language_image_embeddings=precomputed_image_embeddings,
            language_image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> KimiK2_5ModelInputs:
        assert isinstance(prev_model_inputs, KimiK2_5ModelInputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._device_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        next_host_input_row_offsets = self._host_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        return KimiK2_5ModelInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            host_input_row_offsets=next_host_input_row_offsets,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
            ep_inputs=prev_model_inputs.ep_inputs,
            language_image_embeddings=self._empty_image_embeddings,
            language_image_token_indices=self._empty_image_image_token_indices,
        )
