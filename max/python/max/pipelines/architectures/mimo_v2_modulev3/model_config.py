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
"""Config for MiMo-V2.6-Flash on ModuleV3 (``MiMoV2ForCausalLM``).

The model runs FP8 W8A8 dense linears and W4A8 MXFP4 experts. Both KV cache
groups hold K (192) and V (128) unpadded, packed into one row per head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.ep_config import EPConfig, calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import (
    CacheLeafParamInterface,
    KVCacheParamInterface,
    MHAKVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ModelOutputConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_runtime import (
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override

from .hybrid_kv_params import hybrid_swa_full_kv_params
from .layers.attention import QkvChunkLayout
from .quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
    MiMoV2QuantScheme,
    moe_layers,
    parse_quant_scheme,
)

SLIDING = "sliding_attention"
FULL = "full_attention"


def layer_types(huggingface_config: AutoConfig) -> list[str]:
    """Returns each decoder layer's KV group, from ``hybrid_layer_pattern``."""
    pattern = huggingface_config.hybrid_layer_pattern
    return [
        SLIDING if pattern[i] == 1 else FULL
        for i in range(huggingface_config.num_hidden_layers)
    ]


def qkv_chunk_layout(config: AutoConfig, sliding: bool) -> QkvChunkLayout:
    """Returns the chunk layout of ``qkv_proj`` for one attention type.

    The chunk count is the tensor-parallel degree the checkpoint was saved
    at, which is the full-attention KV head count, ``num_key_value_heads``
    (4). A different count misreads the full-attention scale grids, which
    then fail the load on their shape.

    Args:
        config: The checkpoint's top-level Hugging Face config.
        sliding: Whether the layer is sliding-window.

    Returns:
        The layout.
    """

    def attn(key: str) -> int:
        # As in the reference: sliding layers read ``swa_<key>`` if present.
        value = getattr(config, f"swa_{key}", None) if sliding else None
        return int(getattr(config, key) if value is None else value)

    heads, kv_heads = attn("num_attention_heads"), attn("num_key_value_heads")
    head_dim, v_head_dim = attn("head_dim"), attn("v_head_dim")
    chunks = config.num_key_value_heads
    if heads % chunks or kv_heads % chunks:
        raise ValueError(
            f"MiMo-V2: {heads} query and {kv_heads} KV heads do not split "
            f"into {chunks} qkv_proj chunks."
        )
    return QkvChunkLayout(
        chunks=chunks,
        q_rows=heads * head_dim // chunks,
        k_rows=kv_heads * head_dim // chunks,
        v_rows=kv_heads * v_head_dim // chunks,
    )


def validate_devices(device_specs: Sequence[DeviceSpec]) -> None:
    """Raises unless every device is an NVIDIA SM100 (B200-class) GPU.

    The W4A8 experts' grouped MXFP8 quantize and MXFP4 matmul are SM100 kernels;
    elsewhere the graph fails late, with an error that does not name the
    cause.

    Args:
        device_specs: The devices to run on.

    Raises:
        ValueError: If a device is not an SM100 GPU.
    """
    for device in load_devices(device_specs):
        arch = device.architecture_name if device.api == "cuda" else device.api
        if not arch.startswith("sm_10"):
            raise ValueError(
                "MiMo-V2 runs only on NVIDIA SM100 (B200-class) GPUs, whose "
                f"kernels its W4A8 experts use; {device.label}:{device.id} "
                f"is {arch!r}."
            )


def _validate(config: AutoConfig) -> None:
    """Raises on any config value this implementation does not handle."""
    expected = {
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "attention_projection_layout": "fused_qkv",
    }
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(
                f"MiMo-V2: {key} is {getattr(config, key, None)!r}; only "
                f"{value!r} is implemented."
            )
    if (config.n_group or 1, config.topk_group or 1) != (1, 1):
        raise ValueError(
            f"MiMo-V2: n_group {config.n_group} and topk_group "
            f"{config.topk_group} are not the single-group router."
        )
    if getattr(config, "n_shared_experts", None):
        raise ValueError("MiMo-V2: shared experts are not implemented.")
    if config.routed_scaling_factor not in (None, 1.0):
        raise ValueError(
            f"MiMo-V2: routed_scaling_factor {config.routed_scaling_factor}"
            " is not implemented."
        )
    # The reference reads the RoPE type from either key and would apply
    # YaRN or any other scaling named there.
    for key in ("rope_parameters", "rope_scaling"):
        rope = getattr(config, key, None) or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise ValueError(
                f"MiMo-V2: {key} has rope_type {rope_type!r}; only default "
                "RoPE is implemented."
            )
    # Sliding and full layers share every width except their KV head count.
    for key in ("num_attention_heads", "head_dim", "v_head_dim"):
        if getattr(config, f"swa_{key}") != getattr(config, key):
            raise ValueError(
                f"MiMo-V2: swa_{key} differs from {key}; only the KV head "
                "count may differ between sliding and full layers."
            )


def ep_dispatch_row_budget(pipeline_config: PipelineConfig) -> int:
    """Returns the rows one forward step carries; sizes the EP buffers.

    The scheduler's ``max_batch_input_tokens`` bounds a step's active rows
    -- chunked prefill makes it hard -- so a forward's dispatch buffers fit
    it. A speculative verify merges ``draft_width`` draft rows beside each
    request's active token through every MoE layer, rows the budget never
    charges, so it counts them too, as DeepSeek-V4.1's DSpark EP config
    does: the dispatch buffers are fixed at build and the dispatch kernel
    rejects a wider shard.

    Args:
        pipeline_config: The pipeline whose scheduler budget and verify
            width bound the step's rows.

    Raises:
        ValueError: Speculating without ``max_batch_size``, which bounds the
            requests a verify step carries and so the draft rows too.
    """
    runtime = pipeline_config.runtime
    speculative = pipeline_config.speculative
    if speculative is None:
        return runtime.max_batch_input_tokens
    if runtime.max_batch_size is None:
        raise ValueError(
            "MiMo-V2 speculative decoding under expert parallelism needs"
            " max_batch_size, which bounds the draft rows a verify step"
            " adds to max_batch_input_tokens"
        )
    return (
        runtime.max_batch_input_tokens
        + runtime.max_batch_size * speculative.draft_width
    )


SPLIT_KV_MAX_QUERY_ROWS = 128
"""Query rows (a request's query tokens times the Q heads per KV head) up to
which attention over a split cache honors the metadata's partition count:
``SPLIT_KV_1Q_MAX_Q_ROWS`` in the SM100 dispatch. Past it a step runs the
unsplit prefill tile, which is far slower on a long cache."""


def check_split_verify_rows(
    num_attention_heads: int, kv_params: CacheLeafParamInterface
) -> None:
    """Refuses a speculative width that overflows a group's verify step.

    A block draft (DFlash) verifies its block per request, and an
    autoregressive draft its drafts and the bonus token. ``num_draft_tokens``
    is the drafter's block only on the tree a speculative pairing stamps
    with it; the base tree carries ``num_speculative_tokens`` there, so the
    target's build checks the stamped tree too.

    Raises:
        ValueError: If a split group's verify step has more than
            :data:`SPLIT_KV_MAX_QUERY_ROWS` query rows.
    """
    if isinstance(kv_params, MultiKVCacheParams):
        for child in kv_params.children.values():
            check_split_verify_rows(num_attention_heads, child)
        return
    if (
        not isinstance(kv_params, MHAKVCacheParams)
        or not kv_params.has_split_head_dims
        or kv_params.speculative_method is None
    ):
        return
    tokens = kv_params.num_draft_tokens
    if kv_params.speculative_method not in ("dflash", "dflash2", "dspark"):
        tokens += 1
    rows = tokens * (num_attention_heads // kv_params.n_kv_heads)
    if rows > SPLIT_KV_MAX_QUERY_ROWS:
        raise ValueError(
            f"MiMo-V2: a {kv_params.speculative_method} verify step attends "
            f"{tokens} tokens per request, {rows} query rows, over the "
            f"split KV cache; at most {SPLIT_KV_MAX_QUERY_ROWS} keep its "
            "split-K decode route."
        )


@dataclass(kw_only=True)
class MiMoV2Config(
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
    ExpertParallelDeploymentConfig,
):
    """Model configuration for MiMo-V2.6-Flash.

    ``attention_chunk_size`` in the checkpoint config is a leftover key and is
    deliberately not read: the model has no chunked local attention.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float4_e2m1fnx2"}

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    layer_types: list[str]
    """Each decoder layer's KV group, ``sliding_attention`` or
    ``full_attention``."""

    num_attention_heads: int
    head_dim: int
    """Q and K head dim."""
    v_head_dim: int
    qkv_layouts: dict[str, QkvChunkLayout]
    """The fused ``qkv_proj`` chunk layout of each KV group."""
    rotary_dim: int
    """Leading Q/K dims that RoPE rotates, NeoX style."""
    rope_thetas: dict[str, float]
    sliding_window: int
    """Keys a sliding layer attends to, the query's own included."""
    attention_value_scale: float
    sinks: dict[str, bool]
    """Whether each KV group has a learned per-head attention sink."""

    intermediate_size: int
    """Dense MLP width."""
    moe_layers: set[int]

    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool

    quant: MiMoV2QuantScheme
    dtype: DType
    devices: list[DeviceRef]
    kv_params: KVCacheParamInterface
    max_seq_len: int

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    target_layer_ids: list[int] | None = None
    """For ``ReturnHiddenStates.SELECTED_LAYERS``, the layers whose outputs
    (post-residual, before the final norm) are returned."""

    ep_config: EPConfig | None = None
    """Expert-parallel communication config, set by the pipeline model when
    ``runtime.ep_size > 1``. ``None`` leaves the routed experts channel-
    sharded (tensor parallel)."""

    fused_sliding_qkv: bool = False
    """Whether sliding-window layers prepare Q, K and V with the fused
    chunked QKV op instead of separate cache stores and RoPE. Off by
    default; unsupported with data-parallel attention."""
    fused_full_qkv: bool = False
    """Whether full-attention layers use fused chunked QKV preparation.
    Off by default, independently of sliding fusion; unsupported with
    data-parallel attention."""

    data_parallel_degree: int = 1
    """Attention data-parallel replicas. Reserved for the DP coupling; a
    degree above 1 is refused until the DP-attn mode is built."""

    ep_dispatch_token_layout: ClassVar[str] = "dp_replicated"
    ep_row_budget: int = 0
    """The rows one EP forward step carries, from the scheduler's budget;
    0 reads ``runtime.max_batch_input_tokens``."""

    chunked_prefill: bool = True
    """Whether chunked prefill is on, which the EP dispatch buffers need."""

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Completes the output policy without changing checkpoint weights."""
        del huggingface_config, state_dict
        return replace(
            self,
            return_logits=outputs.return_logits,
            return_hidden_states=outputs.return_hidden_states,
        )

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the exchange while preserving MiMo's supported EP modes."""
        n_devices = len(runtime.mesh.devices)
        dp_degree = runtime.data_parallel_degree
        use_allreduce = dp_degree <= 1
        if dp_degree > 1 and dp_degree != n_devices:
            raise ValueError(
                "MiMo-V2 data-parallel attention requires one replica per"
                " device"
            )
        if runtime.ep_size != n_devices:
            raise ValueError(
                "MiMo-V2 serves expert parallelism on one node only: "
                "ep_size must equal the device count"
            )
        if runtime.ep_use_allreduce != use_allreduce:
            raise NotImplementedError(
                "MiMo-V2 requires allreduce EP with TP attention and "
                "P2P EP with DP attention"
            )
        if not self.chunked_prefill:
            raise ValueError(
                "MiMo-V2 expert parallelism requires chunked prefill"
            )
        if self.n_routed_experts % runtime.ep_size:
            raise ValueError("MiMo-V2 devices must divide the routed experts")
        return EPConfig(
            dispatch_dtype=DType.float8_e4m3fn,
            dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
            combine_dtype=DType.bfloat16,
            hidden_size=self.hidden_size,
            top_k=self.num_experts_per_tok,
            n_experts=self.n_routed_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=self.ep_row_budget
                or runtime.max_batch_input_tokens,
                ep_size=runtime.ep_size,
                data_parallel_degree=dp_degree,
                use_allreduce=use_allreduce,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=1,
            use_allreduce=use_allreduce,
        )

    def apply_model_load_context(self, context: ModelLoadContext) -> Self:
        """Applies the speculative target's KV limits and selected-layer taps."""
        assert context.kv_params is not None
        assert isinstance(context.kv_params, KVCacheParamInterface)
        check_split_verify_rows(
            context.huggingface_config.num_attention_heads, context.kv_params
        )
        if (
            context.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
            and context.pipeline_config.runtime.ep_size > 1
            and context.pipeline_config.model.data_parallel_degree <= 1
        ):
            raise NotImplementedError(
                "MiMo-V2 speculative expert parallelism requires DP attention"
            )
        return replace(
            self,
            max_seq_len=self.max_seq_len
            + spec_decode_cache_slack(context.kv_params),
            target_layer_ids=list(context.target_layer_ids) or None,
        )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Completes placement and rejects unsupported attention replicas."""
        n_devices = len(runtime.mesh.devices)
        if runtime.data_parallel_degree > 1 and (
            runtime.ep_size != n_devices
            or runtime.data_parallel_degree != n_devices
        ):
            raise ValueError(
                "MiMo-V2 data-parallel attention requires one replica per"
                " device and expert parallelism over every device"
            )
        return replace(
            self,
            ep_config=runtime.ep,
            data_parallel_degree=runtime.data_parallel_degree,
        )

    @property
    def attention_scale(self) -> float:
        """The softmax scale, from the unpadded Q/K head dim."""
        return self.head_dim**-0.5

    @override
    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Builds the ``{sliding_attention, full_attention}`` KV tree.

        Both groups cache each head's 192-wide key and 128-wide value
        unpadded, packed into one row
        (:attr:`~max.nn.kv_cache.KVCacheParams.v_head_dim`). The sliding
        group's window also bounds its split-K partition count.
        """
        return hybrid_swa_full_kv_params(
            layer_types=layer_types(huggingface_config),
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            allow_kv_head_replication=allow_kv_head_replication,
            sliding_n_kv_heads=huggingface_config.swa_num_key_value_heads,
            full_n_kv_heads=huggingface_config.num_key_value_heads,
            sliding_v_head_dim=huggingface_config.v_head_dim,
            full_v_head_dim=huggingface_config.v_head_dim,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes the config from the pipeline config.

        Args:
            pipeline_config: The pipeline configuration.
            model_config: The model configuration to read. Defaults to
                ``pipeline_config.model``.
            max_seq_len: The maximum sequence length to build for.

        Returns:
            The initialized config.

        Raises:
            ValueError: If a device is not an SM100 GPU, or the checkpoint is
                not one this implementation handles.
        """
        model_config = model_config or pipeline_config.model
        validate_devices(model_config.device_specs)
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but it could not be loaded."
            )
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            encoding, model_config.kv_cache.kv_cache_format
        )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            model_config.kv_cache,
            cache_dtype,
        )
        config = cls.from_huggingface_config(
            huggingface_config,
            devices=devices,
            kv_params=kv_params,
            max_seq_len=max_seq_len,
            data_parallel_degree=model_config.data_parallel_degree,
        )
        config.ep_row_budget = (
            ep_dispatch_row_budget(pipeline_config)
            if pipeline_config.runtime.ep_size > 1
            else 0
        )
        config.chunked_prefill = pipeline_config.runtime.enable_chunked_prefill
        return config

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: AutoConfig,
        *,
        devices: list[DeviceRef],
        kv_params: KVCacheParamInterface,
        max_seq_len: int,
        data_parallel_degree: int = 1,
    ) -> Self:
        """Builds the config from the checkpoint's HuggingFace config.

        Args:
            huggingface_config: The checkpoint's top-level config.
            devices: The devices to build for.
            kv_params: The KV cache tree, from :meth:`construct_kv_params`.
            max_seq_len: The maximum sequence length to build for.
            data_parallel_degree: Attention replicas over ``devices``. The
                divisibility checks below are the tensor-parallel sharding
                constraints, so they hold over one replica's devices --
                where the weights shard -- not over the whole mesh, where
                every one-device replica holds them unsharded.

        Returns:
            The config.

        Raises:
            ValueError: If the config describes a variant this implementation
                does not handle, or a quantization other than Xiaomi's.
        """
        hf = huggingface_config
        _validate(hf)
        num_devices = max(len(devices) // max(data_parallel_degree, 1), 1)
        layouts = {
            SLIDING: qkv_chunk_layout(hf, sliding=True),
            FULL: qkv_chunk_layout(hf, sliding=False),
        }
        for group, layout in layouts.items():
            # Every rank holds whole qkv_proj chunks, whose FP8 block grid
            # it pads to.
            if layout.chunks % num_devices:
                raise ValueError(
                    f"MiMo-V2: {num_devices} devices do not divide the "
                    f"{layout.chunks} {group} qkv_proj chunks."
                )
        if hf.moe_intermediate_size % (128 * num_devices):
            raise ValueError(
                f"MiMo-V2: expert width {hf.moe_intermediate_size} does not "
                f"split into whole 128-row scale granules over {num_devices} "
                "devices."
            )
        if hf.intermediate_size % (128 * num_devices):
            raise ValueError(
                f"MiMo-V2: dense width {hf.intermediate_size} does not split "
                f"into whole 128-row scale granules over {num_devices} "
                "devices."
            )
        check_split_verify_rows(hf.num_attention_heads, kv_params)
        rotary_dim = int(hf.head_dim * hf.partial_rotary_factor)
        if rotary_dim % 2:
            raise ValueError(f"MiMo-V2: rotary dim {rotary_dim} is odd.")
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            rms_norm_eps=hf.layernorm_epsilon,
            layer_types=layer_types(hf),
            num_attention_heads=hf.num_attention_heads,
            head_dim=hf.head_dim,
            v_head_dim=hf.v_head_dim,
            qkv_layouts=layouts,
            rotary_dim=rotary_dim,
            rope_thetas={SLIDING: hf.swa_rope_theta, FULL: hf.rope_theta},
            sliding_window=hf.sliding_window,
            attention_value_scale=hf.attention_value_scale,
            sinks={
                SLIDING: bool(hf.add_swa_attention_sink_bias),
                FULL: bool(hf.add_full_attention_sink_bias),
            },
            intermediate_size=hf.intermediate_size,
            moe_layers=set(moe_layers(hf)),
            n_routed_experts=hf.n_routed_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            moe_intermediate_size=hf.moe_intermediate_size,
            norm_topk_prob=hf.norm_topk_prob,
            quant=parse_quant_scheme(hf),
            dtype=DType.bfloat16,
            devices=devices,
            kv_params=kv_params,
            max_seq_len=max_seq_len,
        )
