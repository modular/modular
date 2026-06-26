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
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, cast

from max.driver import (
    Buffer,
    Device,
    DeviceSpec,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.pipelines.modeling.config_enums import is_float4_encoding
from max.pipelines.request import RequestID
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from ..deepseekV3.model import DeepseekV3Inputs
from .batch_processor import KimiK2_5BatchProcessor
from .context import KimiK2_5TextAndVisionContext
from .kimi_nvfp4_policy import infer_kimi_nvfp4_weight_flags
from .kimik2_5 import KimiK2_5
from .model_config import KimiK2_5Config, KimiK2_5TextConfig
from .weight_adapters import (
    preshuffle_mxfp4_b_experts,
    preshuffle_mxfp4_b_scales,
)

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

    model_config_cls: ClassVar[type[Any]] = KimiK2_5Config
    batch_processor_cls: ClassVar[type[KimiK2_5BatchProcessor]] = (
        KimiK2_5BatchProcessor
    )

    """Conservative coefficient for vision encoder peak transient memory.

    Per-patch peak transient working memory in the encoder, in units of
    ``vt_hidden_size`` bytes. Captures the in-layer attention working set
    (packed QKV bf16 + fp32 Q/K upcast in RoPE + attention output + residual)
    which dominates the MLP working set for the Kimi-VL config. See
    ``layers/vision/attention.py::_apply_rope`` for the fp32 upcast that drives
    this; rounded up from ~16x to leave headroom.
    """

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

        if self._batch_processor is not None:
            assert isinstance(self._batch_processor, KimiK2_5BatchProcessor)
            self._batch_processor.bind_vision_cache(self._ve_cache)
            assert self.model_config is not None
            self._batch_processor.bind_model_config(self.model_config)
            self._batch_processor.bind_vision_encoder(
                vision_model=self.vision_model,
                session=self.session,
            )
            self._batch_processor.bind_ep_comm_initializer(
                self.ep_comm_initializer
            )

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

        # Kimi K2.5 expects expert B weights in the 5D layout that the AMD
        # `mxfp4_grouped_matmul_amd_preb` kernel reads, and the per-expert
        # B-scales in the 4D-cell layout the same kernel addresses via
        # `Shuffler.scale_4d_byte_off`. The OG weight adapter only renames
        # keys, so do both CPU preshuffles here and flip the QuantConfig
        # flag so `MoEQuantized` dispatches to the preb path. Must stay in
        # lockstep with the weight adapter.
        if quant_config is not None and quant_config.is_mxfp4:
            preshuffle_mxfp4_b_experts(state_dict)
            preshuffle_mxfp4_b_scales(state_dict)
            quant_config = replace(quant_config, mxfp4_preshuffled_b=True)
        shared_experts_weight_dtype, dense_mlp_layers_without_quant = (
            infer_kimi_nvfp4_weight_flags(
                state_dict,
                first_k_dense_replace=config.first_k_dense_replace,
                quant_config=quant_config,
            )
        )
        if quant_config is not None and shared_experts_weight_dtype is not None:
            quant_config = replace(
                quant_config,
                shared_experts_weight_dtype=shared_experts_weight_dtype,
            )

        # Check if EP should be configured
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % len(self.devices) != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({len(self.devices)}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={len(self.devices)}."
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

            if (
                config.n_shared_experts == 1
                and not is_mxfp4
                and quant_config is not None
                and quant_config.shared_experts_use_quant(dtype)
            ):
                # Only enable shared expert fusion when shared tensors match
                # routed NVFP4 experts (false for nvidia/Kimi-K2.6-NVFP4).
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
        model_config.dense_mlp_layers_without_quant = (
            dense_mlp_layers_without_quant
        )
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

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Load the model with the given weights."""

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
            module = Module()

            # Build the vision graph in the module
            vision_graph = self._build_vision_graph(
                kimik2_5_config, state_dict, module=module
            )

            # Build the language graph in the module
            language_graph = self._build_language_graph(config, module=module)
            timer.mark_build_complete()
            models = session.load_all(module, weights_registry=self.state_dict)
            vision_model = models[vision_graph.name]
            language_model = models[language_graph.name]

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
            kv_inputs = self.kv_params.unflatten_kv_inputs(variadic_args_iter)
            assert isinstance(kv_inputs, KVCacheInputs)
            kv_caches_per_dev = list(kv_inputs.inputs)

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
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    def release(self, request_id: RequestID) -> None:
        """Release vision encoder cache entries for a completed request."""
        self._ve_cache.release_request(request_id)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> KimiK2_5ModelInputs:
        """Delegates to the batch processor; typed for Eagle subclasses."""
        if self._batch_processor is not None:
            return cast(
                KimiK2_5ModelInputs,
                self._batch_processor.prepare_initial_token_inputs(
                    replica_batches,
                    kv_cache_inputs=kv_cache_inputs,
                    return_n_logits=return_n_logits,
                ),
            )
        raise RuntimeError("No batch processor configured for KimiK2_5Model")
