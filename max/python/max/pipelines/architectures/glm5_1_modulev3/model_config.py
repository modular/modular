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
"""Config for GLM-5.x (GlmMoeDsa) models, in the ModuleV3 API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.ep_config import EPConfig, calculate_ep_max_tokens_per_rank
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_runtime import (
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override

from ..deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
    check_replica_layout,
    resolve_dcp_q_replicate,
    resolve_indexer_types,
)
from ..glm5_1.model_config import glm_rope_scaling, glm_unpadded_vocab_size


@dataclass(kw_only=True)
class Glm5_1Config(DeepseekV3_2Config, ExpertParallelDeploymentConfig):
    """Configuration for GLM-5.x models (ModuleV3).

    GLM shares DeepSeek-V3.2's sparse-MLA decoder; it differs in declaring
    standard RoPE rather than YaRN, and in padding ``lm_head`` past the
    tokenizer's vocabulary.
    """

    takes_batch_context_lengths: ClassVar[bool] = True
    ep_dispatch_token_layout: ClassVar[str | None] = "tp_sharded"
    """How tokens reach the expert dispatch. Without allreduce, attention
    reduce-scatters them across the TP ranks first, which is the layout
    GLM-5.3's recipe serves; exporters pick the matching EP backend from it.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float4_e2m1fnx2",
        "float8_e4m3fn",
        "bfloat16",
    }

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initialize config, mapping GLM default RoPE to ``rope_scaling=None``."""
        model_config = model_config or pipeline_config.model
        config = model_config.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=get_rope_theta(config),
            rope_scaling=glm_rope_scaling(config),
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            graph_mode=graph_mode,
            data_parallel_degree=model_config.data_parallel_degree,
            decode_context_parallel_degree=(
                model_config.decode_context_parallel_degree
            ),
            dcp_q_replicate=resolve_dcp_q_replicate(
                model_config.dcp_q_replicate
            ),
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
            indexer_types=resolve_indexer_types(
                config, config.num_hidden_layers
            ),
            indexer_rope_interleave=getattr(
                config, "indexer_rope_interleave", False
            ),
            quantization_encoding=quantization_encoding,
            unpadded_vocab_size=glm_unpadded_vocab_size(pipeline_config),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Returns the config completed with the checkpoint's quantization.

        The decoder returns only the last token's logits, whatever
        ``outputs`` asks for; the fused MTP graph sets its own outputs.

        Raises:
            KeyError: If a ``noaux_tc`` router has no correction bias.
        """
        del outputs
        quant_config = None
        if self.dtype in (
            DType.float8_e4m3fn,
            DType.uint8,
            DType.float4_e2m1fn,
        ):
            quant_config = parse_quant_config(
                huggingface_config, state_dict, self.dtype
            )
        correction_bias_dtype = self.correction_bias_dtype
        if self.topk_method == "noaux_tc":
            bias = next(
                (
                    value
                    for key, value in state_dict.items()
                    if key.endswith("e_score_correction_bias")
                ),
                None,
            )
            if bias is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            correction_bias_dtype = bias.dtype
        return replace(
            self,
            quant_config=quant_config,
            correction_bias_dtype=correction_bias_dtype,
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
        """Returns the config placed on the deployment's mesh.

        Raises:
            NotImplementedError: If the replicas or the decode context
                parallel group don't fit the mesh (see
                :func:`check_replica_layout`).
        """
        check_replica_layout(self, runtime)
        return replace(
            self,
            mesh=runtime.mesh,
            ep_config=runtime.ep,
            max_batch_context_length=(
                runtime.max_batch_total_tokens or self.max_batch_context_length
            ),
        )
