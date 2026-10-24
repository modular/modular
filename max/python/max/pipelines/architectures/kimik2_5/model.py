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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, ClassVar, cast

from max import tree
from max.driver import (
    Buffer,
    Device,
    DeviceSpec,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph import Module as GraphModule
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface, KVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.eplb_runtime import EplbRuntime
from max.pipelines.lib.eplb_stats import (
    EplbStatsAccumulator,
    EplbStatsMetadata,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.modeling.config_enums import is_float4_encoding
from max.pipelines.weights.quant import parse_quant_config
from max.profiler import traced
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model import DeepseekV3Inputs
from .batch_processor import (
    KimiK2_5BatchProcessor,
    KimiK2_5BatchProcessorBase,
)
from .context import KimiK2_5TextAndVisionContext
from .kimi_nvfp4_policy import infer_kimi_nvfp4_weight_flags
from .kimik2_5 import KimiK2_5
from .layers.vision.transformer import Transformer
from .model_config import KimiK2_5Config, KimiK2_5TextConfig
from .weight_adapters import _VISION_PREFIXES

logger = logging.getLogger("max.pipelines")


def _fp8_attention_quant_config(
    state_dict: Mapping[str, WeightData],
    quant_config: QuantConfig | None,
    num_layers: int,
) -> QuantConfig | None:
    """Returns the quantization of FP8 attention beside FP4 experts, if any.

    AMD's MXFP4 Kimi checkpoints store the MLA projections in FP8 with one
    scale per output row, and their Quark config quantizes activations per
    token at runtime.
    """
    if quant_config is None or not quant_config.is_fp4:
        return None
    scale = state_dict.get(
        "language_model.layers.0.self_attn.q_a_proj.weight_scale"
    )
    if scale is None:
        return None
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
            dtype=scale.dtype,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE, dtype=scale.dtype
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(range(num_layers)),
        format=QuantFormat.FBGEMM_FP8,
    )


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

    eplb_counter_buffers: list[Buffer] = field(default_factory=list)
    """Per-device EP counter buffers for the language model graph."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns the language model input ABI tuple."""
        return (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
            *self.eplb_counter_buffers,
        )


class KimiK2_5Model(
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[
        KimiK2_5TextAndVisionContext, KimiK2_5TextConfig, Model
    ],
):
    """A Kimi-K2.5 pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = KimiK2_5Config
    batch_processor_cls: ClassVar[type[KimiK2_5BatchProcessorBase[Any]]] = (
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

    _compile_group: ClassVar[str] = "kimik2_5"
    """The group the vision and language graphs compile into one artifact
    under."""

    _eplb_stats_accumulator: EplbStatsAccumulator | None = None
    """The EPLB routing histogram accumulator."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")
        self.session = session
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        self._models = self.load_graphs(session)

        if self._batch_processor is not None:
            assert isinstance(self._batch_processor, KimiK2_5BatchProcessorBase)
            assert self.model_config is not None
            self._batch_processor.bind_model_config(self.model_config)
            assert self.vision_model is not None
            self._batch_processor.bind_vision_encoder(
                vision_model=self.vision_model,
                session=self.session,
            )
            self._batch_processor.bind_ep_comm_initializer(
                self.ep_comm_initializer
            )

    @property
    def vision_model(self) -> Model:
        """The compiled vision model for processing images."""
        return self._models[GraphRole.VISION]

    @property
    def language_model(self) -> Model:
        """The compiled language model for text generation."""
        return self._models[GraphRole.DECODER]

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
        encoding = _select_quantization_encoding(
            pipeline_config.model, KimiK2_5Config.DEFAULT_ENCODING
        )
        if (
            is_float4_encoding(encoding)
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
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV3, KimiK2_5TextConfig]:
        """Builds the language model over the config's devices.

        It embeds tokens only; a caller that runs it beside the vision
        encoder :meth:`build_vision_graph` builds merges the images in. A
        caller that sized the cache passes its KV params, and one that reads
        selected layers names them. It starts expert parallelism's runtime
        through ``context`` before it shards the experts.
        """
        state_dict = _state_dict_under(context, ("language_model.",))
        config = cls.finalized_config(
            context.pipeline_config,
            state_dict,
            max_seq_len=context.max_seq_len,
            max_batch_total_tokens=context.max_batch_total_tokens,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParams)
            config.kv_params = context.kv_params
        if context.target_layer_ids:
            config.eagle_aux_hidden_state_layer_ids = list(
                context.target_layer_ids
            )
        context.start_ep(config.ep_config)
        module = DeepseekV3(config)
        module.load_state_dict(
            {
                name.removeprefix("language_model."): value
                for name, value in state_dict.items()
            },
            weight_alignment=1,
            strict=True,
        )
        return module, config

    @classmethod
    def build_vision_graph(
        cls, context: ModelLoadContext
    ) -> tuple[Graph, dict[str, Any]]:
        """Builds the vision encoder's graph and the weights it loads, for a
        caller that runs the language model itself."""
        state_dict = _state_dict_under(context, _VISION_PREFIXES)
        config = KimiK2_5Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        vision_encoder = Transformer(config.vision_config)
        vision_encoder.load_state_dict(
            {
                name.removeprefix("vision_encoder."): value
                for name, value in state_dict.items()
            },
            weight_alignment=1,
            strict=True,
        )
        spec = GraphSpec(
            vision_encoder,
            "kimik2_5_vision_graph",
            vision_encoder.input_types(),
            weights=vision_encoder.state_dict(),
        )
        return spec.trace(), dict(spec.weights_registry())

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> KimiK2_5TextConfig:
        """Create model configuration from huggingface config."""
        return self.finalized_config(
            self.pipeline_config,
            state_dict,
            max_seq_len=self.max_seq_len,
            max_batch_total_tokens=self.planned_max_batch_total_tokens,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

    @classmethod
    def finalized_config(
        cls,
        pipeline_config: PipelineConfig,
        state_dict: dict[str, WeightData],
        *,
        max_seq_len: int,
        max_batch_total_tokens: int | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates,
    ) -> KimiK2_5TextConfig:
        """Builds the language model's config, finalized against
        ``state_dict``, which it may preshuffle in place."""
        huggingface_config = pipeline_config.model.huggingface_config
        config = huggingface_config.text_config
        num_devices = len(pipeline_config.model.device_specs)

        # quantization_config lives at the top level of the HF config, not
        # under text_config. Propagate it so parse_quant_config() finds it.
        if hasattr(huggingface_config, "quantization_config"):
            config.quantization_config = huggingface_config.quantization_config

        # data_parallel_degree controls the attention strategy:
        #   == num_devices  ->  DP attention  (each device owns a batch shard)
        #   == 1            ->  TP attention  (heads sharded, tokens replicated)
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        # Initialize config with parameters from pipeline_config
        model_config = KimiK2_5TextConfig.initialize(
            pipeline_config, max_seq_len=max_seq_len
        )

        dtype = model_config.dtype
        quant_config = parse_quant_config(config, state_dict, dtype)

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
        ep_size = pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % num_devices != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({num_devices}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={num_devices}."
                )
            n_nodes = ep_size // num_devices

            runtime = pipeline_config.runtime
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=data_parallel_degree,
                use_allreduce=runtime.ep_use_allreduce,
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
                n_gpus_per_node=num_devices,
                n_nodes=n_nodes,
                dispatch_quant_config=None,
                use_allreduce=runtime.ep_use_allreduce,
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

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.attn_quant_config = _fp8_attention_quant_config(
            state_dict, quant_config, config.num_hidden_layers
        )
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.dense_mlp_layers_without_quant = (
            dense_mlp_layers_without_quant
        )
        model_config.return_logits = return_logits
        model_config.return_hidden_states = return_hidden_states

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
        model_config.eplb_profile_enabled = pipeline_config.runtime.eplb_profile
        return model_config

    def _load_state_dict(self) -> dict[str, Any]:
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
        return state_dict

    @override
    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns.

        Sets up expert parallelism and builds the modules with
        :meth:`_init_distributed_runtime` first, since the graphs trace them.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        # TODO(kathywu): Drop once the speculative-decoding subclasses build
        # their graphs through `graph_specs`.
        if type(self).load_model is not KimiK2_5Model.load_model:
            vision_model, language_model = self.load_model(session)
            return {
                GraphRole.VISION: vision_model,
                GraphRole.DECODER: language_model,
            }

        state_dict = self._load_state_dict()
        config = self._create_model_config(state_dict)
        self._init_distributed_runtime(session, config, state_dict)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(config)
            timer.mark_build_complete()
            self.compiled_modules = compile_and_init_graphs(session, specs)
        return {
            role: compiled.engine_model
            for role, compiled in self.compiled_modules.items()
        }

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the vision and language graphs; implemented only by the
        speculative-decoding subclasses that don't build through
        :meth:`graph_specs` yet."""
        raise NotImplementedError

    @override
    def graph_specs(
        self, model_config: KimiK2_5TextConfig
    ) -> dict[str, GraphSpec]:
        """Returns the vision encoder and the decoder, in one compile group."""
        return {
            GraphRole.VISION: self._vision_spec(),
            GraphRole.DECODER: self._language_spec(),
        }

    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        config: KimiK2_5TextConfig,
        state_dict: dict[str, Any],
    ) -> None:
        self._eplb = EplbRuntime(
            self.pipeline_config.runtime,
            self._eplb_stats_metadata(),
            self.devices,
        )
        if config.ep_config is not None:
            self._eplb.install_placement(
                config.ep_config,
                n_groups=getattr(
                    self.huggingface_config.text_config, "n_group", 1
                ),
            )

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
            self._eplb.bind(self.ep_comm_initializer)
        # ------------------------------------------------------------------
        # Generate the full KimiK2_5Config from HuggingFace config and LM config
        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=config,
            max_seq_len=self.max_seq_len,
        )
        self.model_config = kimik2_5_config
        self.nn_model = KimiK2_5(self.model_config)
        self.nn_model.load_state_dict(
            self._eplb.remap_weights(self.nn_model, state_dict),
            weight_alignment=1,
            strict=True,
        )
        self.state_dict = self.nn_model.state_dict()
        logger.info("Loaded Weights")

        self._eplb_stats_accumulator = self._eplb.accumulator

    def _vision_spec(self) -> GraphSpec:
        """Returns the vision encoder of the modules
        :meth:`_init_distributed_runtime` built."""
        assert isinstance(self.nn_model, KimiK2_5)
        vision_encoder = self.nn_model.vision_encoder
        return GraphSpec(
            vision_encoder,
            "kimik2_5_vision_graph",
            vision_encoder.input_types(),
            weights=self.state_dict,
            group=self._compile_group,
        )

    def _build_vision_graph(
        self,
        config: KimiK2_5Config,
        state_dict: dict[str, WeightData],
        module: GraphModule,
    ) -> tuple[Graph, dict[str, Any]]:
        """Traces the vision encoder into ``module``, for the subclasses that
        still load through ``load_model``."""
        del config, state_dict
        spec = self._vision_spec()
        return spec.trace(module), dict(spec.weights_registry())

    def _language_spec(self) -> GraphSpec:
        assert isinstance(self.nn_model, KimiK2_5)
        language_model = self.nn_model.language_model
        language_model.graph_kv_params = self.kv_params
        return GraphSpec(
            language_model,
            "kimik2_5_language_graph",
            language_model.input_types(self.kv_params),
            weights=self.state_dict,
            group=self._compile_group,
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        """Kimi packs inline in :meth:`vision_execute` (chunked encode)."""
        return

    def vision_execute(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Run the chunked vision encoder over the cache-selected images.

        The chunked encode (packing + per-chunk graph runs + re-concatenation to
        per-image order) stays encapsulated in the batch processor; the cache
        only ever sees the per-image-ordered output.
        """
        assert isinstance(self._batch_processor, KimiK2_5BatchProcessorBase)
        return self._batch_processor.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` image embeddings for non-vision steps.

        Cached: this is hit on every text-only / decode step, so it must not
        allocate per call.
        """
        if not hasattr(self, "_cached_empty_vision_embeddings"):
            hidden_size = self.huggingface_config.text_config.hidden_size
            host = Buffer.zeros(shape=[0, hidden_size], dtype=DType.bfloat16)
            self._cached_empty_vision_embeddings = [host.to(d) for d in devices]
        return self._cached_empty_vision_embeddings

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, KimiK2_5ModelInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "KimiK2_5 requires KV cache inputs"
        )
        model_outputs = self.language_model.execute(*model_inputs.buffers)
        if self._eplb_stats_accumulator is not None:
            self._eplb_stats_accumulator.record_batch_total_tokens(
                int(model_inputs.tokens.shape[0])
            )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    @cached_property
    def _frozen_ep_inputs(self) -> tuple[Buffer, ...]:
        """Persistent EP inputs, including the EPLB tables when enabled."""
        if self.ep_comm_initializer is None:
            return ()
        return tuple(self.ep_comm_initializer.model_inputs())

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> KimiK2_5ModelInputs:
        """Delegates to the batch processor; typed for Eagle subclasses."""
        if self._batch_processor is not None:
            model_inputs = cast(
                KimiK2_5ModelInputs,
                self._batch_processor.prepare_initial_token_inputs(
                    replica_batches,
                    kv_cache_inputs=kv_cache_inputs,
                    return_n_logits=return_n_logits,
                ),
            )
            # Graph-capture warmup packs ``.buffers`` straight from prepared
            # inputs, so the EP / EPLB inputs must be set here, not in
            # ``execute()``.
            if self._eplb_stats_accumulator is not None:
                model_inputs.eplb_counter_buffers = (
                    self._eplb_stats_accumulator.device_buffers
                )
            model_inputs.ep_inputs = self._frozen_ep_inputs
            return model_inputs
        raise RuntimeError("No batch processor configured for KimiK2_5Model")

    def _eplb_stats_metadata(self) -> EplbStatsMetadata:
        """Returns shape descriptor for the language layers (row i == layer i)."""
        text = self.huggingface_config.text_config
        num_layers = text.num_hidden_layers
        first_k_dense = int(getattr(text, "first_k_dense_replace", 0) or 0)
        moe_freq = int(getattr(text, "moe_layer_freq", 1) or 1)
        moe_idx = tuple(
            i
            for i in range(num_layers)
            if i >= first_k_dense and i % moe_freq == 0
        )
        return EplbStatsMetadata(
            num_layers=num_layers,
            num_moe_layers=len(moe_idx),
            moe_layer_indices=moe_idx,
            num_logical_experts=text.n_routed_experts,
            num_experts_per_token=text.num_experts_per_tok,
        )


def _state_dict_under(
    context: ModelLoadContext, prefixes: tuple[str, ...]
) -> dict[str, WeightData]:
    """Adapts only the checkpoint's weights whose names start with
    ``prefixes``."""
    assert context.adapter is not None
    return context.adapter(
        {
            name: weight
            for name, weight in context.weights.items()
            if name.startswith(prefixes)
        },
        huggingface_config=context.huggingface_config,
        pipeline_config=context.pipeline_config,
    )
