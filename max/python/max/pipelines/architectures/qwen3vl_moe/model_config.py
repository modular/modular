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
"""Config for Qwen3-VL and Qwen3-VL-MoE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import select_quantization_encoding
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ArchConfigWithVisionCache,
    ModelOutputConfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_runtime import DeploymentConfig
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights import parse_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override

from .data_processing import QWEN3VL_MAX_PIXELS

__all__ = ["Qwen3VLConfig", "VisionConfig"]


@dataclass
class VisionConfig:
    """The vision tower's geometry."""

    patch_size: int
    """Vision transformer patch size."""

    temporal_patch_size: int
    """Vision transformer temporal patch size."""

    in_channels: int
    """Vision transformer number of input channels."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    depth: int
    """Number of vision transformer layers."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    out_hidden_size: int
    """Output hidden size of the vision encoder, the language model's."""

    deepstack_visual_indexes: list[int]
    """The blocks whose outputs the decoder also adds to its early layers."""

    spatial_merge_size: int
    """Spatial merge size for the vision encoder."""

    num_position_embeddings: int
    """Number of position embeddings for the vision encoder."""

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig
    ) -> VisionConfig:
        """Initializes the tower's geometry from its Hugging Face config."""
        return cls(
            patch_size=hf_vision_config.patch_size,
            temporal_patch_size=hf_vision_config.temporal_patch_size,
            in_channels=hf_vision_config.in_channels,
            hidden_size=hf_vision_config.hidden_size,
            num_attention_heads=hf_vision_config.num_heads,
            depth=hf_vision_config.depth,
            intermediate_size=hf_vision_config.intermediate_size,
            out_hidden_size=hf_vision_config.out_hidden_size,
            deepstack_visual_indexes=list(
                hf_vision_config.deepstack_visual_indexes
            ),
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            num_position_embeddings=hf_vision_config.num_position_embeddings,
        )

    @property
    def embedding_width(self) -> int:
        """The width of a row the tower hands the decoder.

        A row holds the merged patch's embedding, then each deepstack
        layer's, side by side.
        """
        return self.out_hidden_size * (1 + len(self.deepstack_visual_indexes))


def _declared_dtype(text_config: AutoConfig) -> DType:
    """The dtype the checkpoint declares for its unquantized tensors.

    ``PretrainedConfig`` turns the JSON string into a ``torch.dtype``, so
    this reads the name off either form.
    """
    for attr in ("dtype", "torch_dtype"):
        declared = getattr(text_config, attr, None)
        if declared is None:
            continue
        name = (
            declared
            if isinstance(declared, str)
            else str(declared).removeprefix("torch.")
        )
        if name in ("bfloat16", "float16", "float32"):
            return DType[name]
    return DType.bfloat16


@dataclass(kw_only=True)
class Qwen3VLConfig(
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
    DeploymentConfig,
):
    """Configuration for the Qwen3-VL decoder and its vision tower."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
        "float8_e4m3fn",
    }

    # The decoder binds its own names out of the whole checkpoint; the tower
    # takes the weights under its prefix.
    weight_prefixes: ClassVar[Mapping[str, str]] = MappingProxyType(
        {GraphRole.VISION: "vision_encoder."}
    )

    devices: list[DeviceRef]
    """Devices the decoder is parallelized over."""

    dtype: DType
    """The dtype the encoding stores the decoder's weights in."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    max_seq_len: int
    kv_params: KVCacheParamInterface

    mrope_section: list[int]
    """Rotary frequencies per temporal, height and width axis."""

    num_experts: int
    """Number of experts in each MoE layer, 0 for a dense model."""

    num_experts_per_tok: int
    """Number of experts per token in the MoE layer."""

    moe_intermediate_size: int
    """Intermediate size in the MoE layer."""

    mlp_only_layers: list[int]
    """Layers that keep a dense MLP in an MoE model."""

    decoder_sparse_step: int
    """Every this many layers is an MoE layer."""

    vision_config: VisionConfig
    """Vision encoder configuration."""

    declared_dtype: DType
    """The dtype of the checkpoint's unquantized tensors and the vision rows."""

    quantization_encoding: SupportedEncoding | None = None

    # Set by `finalize`.
    rms_norm_eps: float = 1e-6
    norm_dtype: DType | None = None
    tie_word_embeddings: bool = False
    quant_config: QuantConfig | None = None
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    @property
    def compute_dtype(self) -> DType:
        """The dtype of activations and every unquantized weight.

        An FP8 encoding stores only the projections in FP8.
        """
        if self.dtype == DType.float8_e4m3fn:
            return DType.bfloat16
        return self.dtype

    @property
    def head_dim(self) -> int:
        """The attention head width."""
        params = self.kv_params
        assert hasattr(params, "head_dim")
        return int(params.head_dim)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Returns whether layer ``layer_idx`` routes to experts."""
        return (
            layer_idx not in self.mlp_only_layers
            and self.num_experts > 0
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @staticmethod
    def _text_config(huggingface_config: AutoConfig) -> AutoConfig:
        return getattr(huggingface_config, "text_config", huggingface_config)

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """The language model's hidden layer count."""
        return int(
            Qwen3VLConfig._text_config(huggingface_config).num_hidden_layers
        )

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
    ) -> KVCacheParamInterface:
        """Grouped-attention KV params from the text config."""
        return super().construct_kv_params(
            cls._text_config(huggingface_config),
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )

    @override
    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """Bounds against the text config's ``max_position_embeddings``."""
        return super().calculate_max_seq_len(
            cls._text_config(huggingface_config), model_config
        )

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Bounds one cached image by the image processor's pixel ceiling.

        Every image is resized under :data:`QWEN3VL_MAX_PIXELS`, so its rows
        are at most that over the patch area and the spatial merge.
        """
        vision_config = huggingface_config.vision_config
        merge = vision_config.spatial_merge_size
        max_tokens = QWEN3VL_MAX_PIXELS // (
            vision_config.patch_size**2 * merge**2
        )
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        assert spec is not None
        width, dtype = spec
        return max_tokens * width * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """One row per merged vision token, with its deepstack embeddings."""
        vision = VisionConfig.initialize_from_config(
            huggingface_config.vision_config
        )
        return (
            vision.embedding_width,
            _declared_dtype(cls._text_config(huggingface_config)),
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
        """Initializes the config from the pipeline's Hugging Face config."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be loaded."
            )
        text_config = cls._text_config(huggingface_config)
        quantization_encoding = select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
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
            cache_dtype_for_encoding(
                quantization_encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        rope_scaling = getattr(text_config, "rope_scaling", None) or {}
        return cls(
            devices=devices,
            dtype=supported_encoding_dtype(quantization_encoding),
            hidden_size=text_config.hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            num_hidden_layers=text_config.num_hidden_layers,
            intermediate_size=text_config.intermediate_size,
            vocab_size=text_config.vocab_size,
            rope_theta=get_rope_theta(text_config),
            max_seq_len=max_seq_len,
            kv_params=kv_params,
            mrope_section=list(rope_scaling["mrope_section"]),
            # A dense checkpoint has no experts, so every layer is an MLP.
            num_experts=getattr(text_config, "num_experts", 0),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 1),
            moe_intermediate_size=getattr(
                text_config,
                "moe_intermediate_size",
                text_config.intermediate_size,
            ),
            mlp_only_layers=list(getattr(text_config, "mlp_only_layers", [])),
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),
            vision_config=VisionConfig.initialize_from_config(
                huggingface_config.vision_config
            ),
            declared_dtype=_declared_dtype(text_config),
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Sets the fields that need the checkpoint's weights."""
        text_config = self._text_config(huggingface_config)
        # The checkpoint keeps its quantization config at the top level.
        if hasattr(huggingface_config, "quantization_config"):
            text_config.quantization_config = (
                huggingface_config.quantization_config
            )
        decoder_weights = {
            name: weight
            for name, weight in state_dict.items()
            if not name.startswith("vision_encoder.")
        }
        self.quant_config = parse_quant_config(
            text_config, decoder_weights, self.dtype
        )
        norm = decoder_weights.get("layers.0.input_layernorm.weight")
        self.norm_dtype = norm.dtype if norm is not None else None
        self.rms_norm_eps = text_config.rms_norm_eps
        # The top-level value says whether the checkpoint has an lm_head;
        # the text config's can disagree.
        self.tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )
        self.return_logits = outputs.return_logits
        self.return_hidden_states = outputs.return_hidden_states
        return self
