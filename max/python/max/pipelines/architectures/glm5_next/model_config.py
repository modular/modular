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
"""Config for GLM-5.3-Flash (``glm5_next``) on the ModuleV3 API.

Everything that does not depend on the module API -- the hybrid schedule, the
KDA dimensions, the three-child cache tree and the state and batch sizing -- is
read through :mod:`.hybrid_layout`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.ep_config import EPConfig, calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    GraphRole,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
    ModelOutputConfig,
)
from max.pipelines.lib.registration_runtime import (
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights import resolve_hf_quant_config
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import Self, override

from . import hybrid_layout
from .checkpoint import (
    _stored_hc_fn_dtype,
    check_raw_projections,
    checkpoint_quant_scheme,
)
from .hybrid_layout import (
    LINEAR_ATTENTION,
    MLA_KERNEL_LATENT_WIDTH,
    SPARSE_ATTENTION,
    layer_indices,
    recurrent_state_bytes,
)
from .quantization import Glm5NextQuantScheme
from .vision.config import Glm5NextVisionConfig
from .vision.image_processor import Glm5NextImageProcessor

__all__ = ["Glm5NextConfig"]

_DECLARED_DTYPES = {
    "bfloat16": DType.bfloat16,
    "float16": DType.float16,
    "float32": DType.float32,
}


def _declared_dtype(text_config: AutoConfig) -> DType | None:
    """The dtype the checkpoint declares for its unquantized tensors."""
    for attr in ("dtype", "torch_dtype"):
        declared = getattr(text_config, attr, None)
        if isinstance(declared, str) and declared in _DECLARED_DTYPES:
            return _DECLARED_DTYPES[declared]
    return None


@dataclass(kw_only=True)
class Glm5NextConfig(
    ArchConfigWithKVCache,
    ExpertParallelDeploymentConfig,
    ArchConfigWithVisionCache,
):
    """Configuration for GLM-5.3-Flash on ModuleV3."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float8_e4m3fn"}

    takes_batch_context_lengths: ClassVar[bool] = True

    # The decoder's weights carry their own `language_model.` prefix, so it
    # takes the whole checkpoint and binds only its own names.
    weight_prefixes: ClassVar[Mapping[str, str]] = MappingProxyType(
        {GraphRole.VISION: "vision_encoder."}
    )

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParamInterface
    """A ``MultiKVCacheParams`` tree with an ``mla`` and an ``indexer`` leaf."""
    devices: list[DeviceRef]
    quantization_encoding: SupportedEncoding | None = None

    mesh: DeviceMesh | None = None
    """Device mesh for sharding across multiple devices."""

    ep_config: EPConfig | None = None
    """Expert-parallel configuration, or ``None`` for replicated/TP experts.

    Set by :meth:`apply_runtime` when ``--ep-size > 1``.
    """

    data_parallel_degree: int = 1
    """Number of data-parallel attention replicas. Selects the decoder layer's
    parallelism mode together with :attr:`ep_config`."""

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    norm_topk_prob: bool = True
    hidden_act: str = "silu"

    max_position_embeddings: int = 4096
    """Maximum positional embeddings as defined by the original model."""
    max_seq_len: int
    """Maximum sequence length as defined by the MAX Engine pipeline
    configuration."""

    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_interleave: bool = True
    scoring_func: str = "sigmoid"
    attention_bias: bool = False
    attention_dropout: float = 0.0

    correction_bias_dtype: DType | None = None
    max_batch_context_length: int = 131072
    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"

    quant_config: QuantConfig | None = None
    """Block-scaled quantization config when the checkpoint is FP8/FP4.

    Set from the checkpoint by :meth:`finalize`; ``None`` for bf16
    checkpoints.
    """

    mla_o_proj_quantized: bool = True
    """Whether the MLA output projection is quantized."""

    dense_mlp_layers_without_quant: frozenset[int] = frozenset()
    """Dense prefix layers (indices ``< first_k_dense_replace``) that skip MLP quant."""

    unpadded_vocab_size: int | None = None
    """Logical vocab size when the checkpoint pads ``lm_head``.

    The padded tail is masked to ``-inf`` so it cannot be sampled.
    """

    # Added parameters for the Indexer used in DeepSeek Sparse Attention.
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    indexer_types: list[str] = field(default_factory=list)
    # GLM-5.x sets indexer_rope_interleave=true.
    indexer_rope_interleave: bool = False

    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention family: KDA or sparse MLA."""

    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    state_pool_dtype: DType = DType.float32
    """Storage dtype of the KDA conv and recurrent pools."""

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hc_fn_dtype: DType = DType.float32
    """Storage dtype of the mHC stream projections."""

    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True

    mla_latent_pad_to: int = 576
    """Latent width the MLA kernels compile against; the 512-wide NoPE latent
    is zero-padded to it on the way in."""

    swiglu_limit: float = 10.0
    """Clamp applied before the activation in every MLP and expert."""

    index_share_for_mtp_iteration: bool = True
    """Whether the MTP draft's later steps reuse its first step's top-k
    selection."""

    hf_quantization_config: dict[str, Any] | None = None
    quant_scheme: Glm5NextQuantScheme | None = None
    declared_dtype: DType | None = None
    """The dtype the checkpoint declares for its unquantized tensors."""

    vision_config: Glm5NextVisionConfig | None = None
    """Vision tower configuration; ``None`` for a text-only checkpoint."""

    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Resolves checkpoint precision.

        The decoder returns only the last token's logits, whatever
        ``outputs`` asks for; the fused MTP graph sets its own outputs.

        Raises:
            ValueError: If the checkpoint's ``modules_to_not_convert``
                disagrees with which MLA projections carry a scale.
        """
        del outputs
        check_raw_projections(state_dict)
        self.quant_config = self.resolve_quant_scheme(state_dict)
        for name, weight in state_dict.items():
            if name.endswith("e_score_correction_bias"):
                self.correction_bias_dtype = weight.dtype
                break
        # Held as stored: the fused pre-site op reads a bfloat16 projection,
        # and the unfused path widens it to float32, which is exact.
        self.hc_fn_dtype = _stored_hc_fn_dtype(state_dict)
        return self

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(
                "'silu' is the only hidden_act currently supported"
            )

        rope_type = self.rope_scaling and self.rope_scaling.get(
            "rope_type", self.rope_scaling.get("type")
        )
        if rope_type and rope_type != "yarn":
            raise ValueError(
                "'yarn' is the only rope_scaling type currently supported"
            )

        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")

        # A ``shared`` layer reuses the top-k from the most recent ``full``
        # layer, so the first layer must be ``full``. An empty schedule means
        # every layer is full.
        if self.indexer_types and self.indexer_types[0] != "full":
            raise ValueError(
                "indexer_types[0] must be 'full': the first layer cannot be "
                "'shared' because no preceding full indexer layer exists to "
                f"reuse a top-k selection from (got {self.indexer_types[0]!r})."
            )

        if self.data_parallel_degree != 1:
            raise ValueError(
                "GLM-5.3-Flash is tensor-parallel only: data_parallel_degree "
                f"must be 1, got {self.data_parallel_degree}."
            )
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "GLM-5.3-Flash is NoPE: qk_rope_head_dim must be 0, got "
                f"{self.qk_rope_head_dim}."
            )
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}."
            )

    @override
    def get_kv_params(self) -> KVCacheParamInterface:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_seq_len

    @override
    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """Bounds against the *text* config's ``max_position_embeddings``."""
        return upper_bounded_default(
            upper_bound=hybrid_layout.get_text_config(
                huggingface_config
            ).max_position_embeddings,
            default=model_config.max_length,
        )

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the expert-parallel buffers for ``runtime``.

        Tokens dispatch in the checkpoint's FP8 or NVFP4 format when it is
        quantized, else in bf16. The shared expert fuses with the routed ones
        only when it has their shape and weight layout.
        """
        quant_config = self.quant_config
        quantized_dispatch = quant_config is not None and (
            self.dtype.is_float8() or quant_config.is_nvfp4
        )
        n_devices = len(runtime.mesh.devices)
        return EPConfig(
            dispatch_dtype=self.dtype if quantized_dispatch else DType.bfloat16,
            dispatch_quant_config=quant_config if quantized_dispatch else None,
            combine_dtype=DType.bfloat16,
            hidden_size=self.hidden_size,
            top_k=self.num_experts_per_tok,
            n_experts=self.n_routed_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=runtime.max_batch_input_tokens,
                ep_size=runtime.ep_size,
                data_parallel_degree=runtime.data_parallel_degree,
                use_allreduce=runtime.ep_use_allreduce,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=runtime.ep_size // n_devices,
            fused_shared_expert=self.n_shared_experts == 1
            and (
                quant_config is None
                or quant_config.shared_experts_use_quant(self.dtype)
            ),
            use_allreduce=runtime.ep_use_allreduce,
        )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Returns the config placed on the deployment's mesh."""
        return replace(
            self,
            mesh=runtime.mesh,
            ep_config=runtime.ep,
            max_batch_context_length=(
                runtime.max_batch_total_tokens or self.max_batch_context_length
            ),
        )

    @property
    def mla_kv_params(self) -> KVCacheParams:
        """The MLA leaf of the cache tree."""
        assert isinstance(self.kv_params, MultiKVCacheParams)
        mla = self.kv_params.children["mla"]
        assert isinstance(mla, KVCacheParams)
        return mla

    @property
    def compute_dtype(self) -> DType:
        """Dtype of activations and every unquantized weight."""
        if self.quant_scheme is not None:
            return self.quant_scheme.compute_dtype
        return self.declared_dtype or DType.bfloat16

    @property
    def kda_layers(self) -> tuple[int, ...]:
        """Indices of the KDA linear-attention layers."""
        return layer_indices(self.layer_types, LINEAR_ATTENTION)

    @property
    def sparse_attention_layers(self) -> tuple[int, ...]:
        """Indices of the sparse-MLA layers."""
        return layer_indices(self.layer_types, SPARSE_ATTENTION)

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Bytes one largest-budget image occupies in the vision cache."""
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        if spec is None:
            return 0
        hidden_size, dtype = spec
        max_tokens = Glm5NextImageProcessor().max_image_tokens
        return max_tokens * hidden_size * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """One decoder-width bfloat16 row per merged vision token.

        ``None`` for a checkpoint without a vision tower. The config shim
        always sets a ``vision_config``, so its presence alone does not say.
        """
        vision_config = getattr(huggingface_config, "vision_config", None)
        if not hasattr(vision_config, "patch_size"):
            return None
        text_config = hybrid_layout.get_text_config(huggingface_config)
        return text_config.hidden_size, DType.bfloat16

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """How many layers the text tower has."""
        return hybrid_layout.get_num_layers(huggingface_config)

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Builds the MLA, indexer and recurrent-state children of the cache."""
        return hybrid_layout.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def resolve_quant_scheme(
        self, state_dict: Mapping[str, Any]
    ) -> QuantConfig | None:
        """Builds the mixed-precision map from the checkpoint's weight scales.

        The precision map is read off which weights carry a scale, by
        :mod:`.quantization`'s derivation; the checkpoint's names are mapped to
        the ones that derivation reads.
        """
        self.quant_scheme = checkpoint_quant_scheme(
            self.hf_quantization_config, self.declared_dtype, state_dict
        )
        return self.quant_scheme.config if self.quant_scheme else None

    def per_request_state_bytes(self) -> int:
        """Returns the GPU bytes one request's recurrent state occupies.

        The count is summed over every device.
        """
        return recurrent_state_bytes(self.kv_params, len(self.devices))

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the checkpoint's nested ``text_config``."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be loaded."
            )
        text_config = hybrid_layout.get_text_config(huggingface_config)
        # Every MTP pairing encodes images, so a checkpoint without a tower
        # would otherwise fail only after the fused graph compiled and the
        # cache was allocated.
        if (
            pipeline_config.speculative
            and cls.get_vision_cache_row_spec(huggingface_config) is None
        ):
            raise ValueError(
                "--speculative-method mtp needs GLM-5.3-Flash's vision tower, "
                "and this checkpoint has none. Serve it without speculative "
                "decoding."
            )

        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, kv_cache_config.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        kda_dims = hybrid_layout.declared_kda_dims(text_config)
        read_linear = hybrid_layout.linear_attn_field

        declared_dtype = _declared_dtype(text_config)
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        vision_config: Glm5NextVisionConfig | None = None
        if hf_vision_config is not None and hasattr(
            hf_vision_config, "patch_size"
        ):
            vision_config = Glm5NextVisionConfig.initialize_from_config(
                hf_vision_config, dtype=declared_dtype or DType.bfloat16
            )
        # With the cache off, a step's encoder rows cover only the images in
        # its window while the scatter indices cover every image, so a prompt
        # that spans steps gets rows on the wrong placeholders.
        runtime = pipeline_config.runtime
        if (
            vision_config is not None
            and runtime.vision_cache_utilization == 0
            and (
                runtime.enable_chunked_prefill
                or kv_cache_config.enable_prefix_caching
            )
        ):
            raise ValueError(
                "GLM-5.3-Flash needs the vision encoder cache with chunked "
                "prefill or prefix caching on: set --vision-cache-utilization "
                "above 0."
            )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            data_parallel_degree=model_config.data_parallel_degree,
            quantization_encoding=quantization_encoding,
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            n_shared_experts=text_config.n_shared_experts,
            n_routed_experts=text_config.n_routed_experts,
            routed_scaling_factor=text_config.routed_scaling_factor,
            kv_lora_rank=text_config.kv_lora_rank,
            q_lora_rank=text_config.q_lora_rank,
            qk_rope_head_dim=text_config.qk_rope_head_dim,
            qk_nope_head_dim=text_config.qk_nope_head_dim,
            v_head_dim=text_config.v_head_dim,
            topk_method=text_config.topk_method,
            n_group=text_config.n_group,
            topk_group=text_config.topk_group,
            num_experts_per_tok=text_config.num_experts_per_tok,
            first_k_dense_replace=text_config.first_k_dense_replace,
            norm_topk_prob=text_config.norm_topk_prob,
            hidden_act=text_config.hidden_act,
            max_position_embeddings=text_config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=text_config.rms_norm_eps,
            scoring_func=text_config.scoring_func,
            attention_bias=text_config.attention_bias,
            attention_dropout=text_config.attention_dropout,
            # Prefill runs the sparse decode kernel: the prefill route needs
            # absorbed FP8 scales, and this checkpoint ships ``kv_b_proj`` in
            # BF16.
            graph_mode="decode",
            index_head_dim=text_config.index_head_dim,
            index_n_heads=text_config.index_n_heads,
            index_topk=text_config.index_topk,
            indexer_types=list(
                getattr(text_config, "indexer_types", None)
                or ["full"] * text_config.num_hidden_layers
            ),
            layer_types=hybrid_layout.resolve_layer_types(text_config),
            linear_num_heads=kda_dims.num_heads,
            linear_head_dim=kda_dims.head_dim,
            linear_conv_kernel_dim=kda_dims.conv_kernel_dim,
            linear_lower_bound=read_linear(
                text_config, "gate_lower_bound", -5.0
            ),
            state_pool_dtype=hybrid_layout.declared_state_dtype(
                kv_cache_config
            ),
            hc_mult=getattr(text_config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(text_config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(text_config, "hc_eps", 1e-6),
            index_kpool=getattr(text_config, "index_kpool", 4),
            index_kpool_compress=getattr(
                text_config, "index_kpool_compress", True
            ),
            index_kpool_always_select_tail=getattr(
                text_config, "index_kpool_always_select_tail", True
            ),
            mla_latent_pad_to=max(
                text_config.kv_lora_rank + text_config.qk_rope_head_dim,
                MLA_KERNEL_LATENT_WIDTH,
            ),
            swiglu_limit=getattr(text_config, "swiglu_limit", 10.0),
            index_share_for_mtp_iteration=getattr(
                text_config, "index_share_for_mtp_iteration", True
            ),
            hf_quantization_config=resolve_hf_quant_config(
                huggingface_config, {}
            ),
            declared_dtype=declared_dtype,
            vision_config=vision_config,
        )
