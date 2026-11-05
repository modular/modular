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
"""Config for the Beam architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.ep_config import EPConfig, calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.outputs import ReturnLogits
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
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
from max.pipelines.lib.registration_runtime import (
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights.quant import parse_quant_config
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import Self, override


def _check_parallel_layout(
    pipeline_config: PipelineConfig, n_devices: int
) -> None:
    """Rejects a layout Beam's decoder does not build.

    Raises:
        NotImplementedError: Data parallelism, or the allreduce EP backend.
        ValueError: An EP size other than the device count, or EP with
            chunked prefill off.
    """
    dp = pipeline_config.model.data_parallel_degree
    if dp != 1:
        raise NotImplementedError(
            "Beam runs attention tensor parallel over every device; got "
            f"data_parallel_degree={dp}."
        )
    runtime = pipeline_config.runtime
    if runtime.ep_size != n_devices:
        # The routed experts do not fit replicated on every device.
        raise ValueError(
            f"Beam needs --ep-size {n_devices}, the device count: on more "
            "than one device its routed experts run expert parallel over "
            f"all of them. Got ep_size={runtime.ep_size}."
        )
    if n_devices == 1:
        return
    if runtime.ep_use_allreduce:
        raise NotImplementedError(
            "Beam runs expert parallelism with peer-to-peer dispatch; drop "
            "--ep-use-allreduce."
        )
    if not runtime.enable_chunked_prefill:
        # The dispatch/combine buffers are sized to max_tokens_per_rank,
        # which the batch constructor treats as a soft limit: with chunking
        # off it admits an oversized prompt whole, and the dispatch overflows
        # the fixed buffers.
        raise ValueError(
            "Beam expert parallelism requires chunked prefill: the EP "
            "dispatch buffers are sized to max_batch_input_tokens, which "
            "without chunking is a soft limit a longer prompt exceeds."
        )


@dataclass(kw_only=True)
class BeamConfig(
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
    ExpertParallelDeploymentConfig,
):
    """Configuration for Beam decoder-only MoE models.

    Beam alternates sliding-window RoPE layers with full-attention NoPE
    layers, puts a dense SwiGLU MLP on the first layers and a group-limited
    sigmoid MoE on the rest, and wraps every sublayer in sandwich RMSNorms
    joined by a depth-scaled residual.
    """

    ep_dispatch_token_layout: ClassVar[str | None] = "tp_sharded"
    """Attention reduce-scatters the tokens over the TP ranks before the
    expert dispatch."""
    export_ep_init: ClassVar[bool] = True

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    """Intermediate size of the dense MLP layers."""
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    rms_norm_eps: float
    sliding_window: int
    rope_theta: float
    """RoPE base of the sliding-attention layers; full-attention layers have
    no positional encoding."""
    layer_types: list[str]
    """``"sliding_attention"`` or ``"full_attention"``, one per layer."""
    mlp_layer_types: list[str]
    """``"dense"`` or ``"sparse"``, one per layer."""
    final_logit_softcapping: float | None
    max_seq_len: int
    dtype: DType
    """The weight dtype the encoding selects: packed FP4 on an NVFP4
    checkpoint, whose unquantized layers are still bfloat16."""
    devices: list[DeviceRef]
    kv_params: KVCacheParamInterface
    quant_config: QuantConfig | None = None
    """The NVFP4 config of the routed experts, set from the checkpoint."""
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    mesh: DeviceMesh | None = None
    """The decoder's tensor-parallel mesh, set by :meth:`apply_runtime`."""
    ep_config: EPConfig | None = None
    """The expert-parallel config on more than one device, set by
    :meth:`apply_runtime`."""
    use_subgraphs: bool = True

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
        """Builds the sliding/full two-leaf KV cache tree."""
        return hybrid_swa_full_kv_params(
            layer_types=huggingface_config.layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            allow_kv_head_replication=allow_kv_head_replication,
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
        """Initializes a BeamConfig from the pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: The MAX model configuration to read the HuggingFace
                config from. Defaults to ``pipeline_config.model``.
            max_seq_len: The maximum sequence length the config allows.

        Returns:
            The config, without ``quant_config``, which needs the weights.
        """
        model_config = model_config or pipeline_config.model
        hf = model_config.huggingface_config
        if hf is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}'."
            )
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        _check_parallel_layout(pipeline_config, len(device_refs))
        rope_parameters = hf.rope_parameters
        if rope_parameters.get("full_attention") is not None:
            raise ValueError(
                "Beam full-attention layers run without RoPE, but the config "
                f"gives them {rope_parameters['full_attention']!r}."
            )
        sliding_rope = rope_parameters["sliding_attention"]
        if sliding_rope.get("rope_type", "default") != "default":
            raise ValueError(
                "Beam supports only default RoPE on sliding layers, got "
                f"{sliding_rope['rope_type']!r}."
            )
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=cls.get_head_dim(hf),
            intermediate_size=hf.intermediate_size,
            moe_intermediate_size=hf.moe_intermediate_size,
            num_experts=hf.num_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            n_group=hf.n_group,
            topk_group=hf.topk_group,
            norm_topk_prob=hf.norm_topk_prob,
            routed_scaling_factor=float(hf.routed_scaling_factor),
            rms_norm_eps=hf.rms_norm_eps,
            sliding_window=hf.sliding_window,
            rope_theta=float(sliding_rope["rope_theta"]),
            layer_types=list(hf.layer_types),
            mlp_layer_types=list(hf.mlp_layer_types),
            final_logit_softcapping=hf.final_logit_softcapping,
            max_seq_len=max_seq_len,
            dtype=supported_encoding_dtype(quantization_encoding),
            devices=device_refs,
            kv_params=cls.construct_kv_params(
                huggingface_config=hf,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=model_config.kv_cache,
                cache_dtype=cache_dtype_for_encoding(
                    quantization_encoding,
                    model_config.kv_cache.kv_cache_format,
                ),
            ),
            use_subgraphs=pipeline_config.model.use_subgraphs,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Returns the config completed with the checkpoint and the outputs.

        Args:
            huggingface_config: The checkpoint's model configuration.
            state_dict: The adapted checkpoint weights.
            outputs: The outputs the pipeline requires.

        Returns:
            A copy with the routed experts' NVFP4 config, on a quantized
            checkpoint, and the logits ``outputs`` asks for.

        Raises:
            NotImplementedError: If, on more than one device, the checkpoint
                leaves some sparse layer's routed experts unquantized.
        """
        quant_config = None
        if self.dtype == DType.uint8:
            quant_config = parse_quant_config(
                huggingface_config, state_dict, self.dtype
            )
        if quant_config is not None and len(self.devices) > 1:
            # One EP dispatch format serves every sparse layer.
            unquantized = [
                i
                for i, mlp_type in enumerate(self.mlp_layer_types)
                if mlp_type == "sparse"
                and i not in quant_config.mlp_quantized_layers
            ]
            if unquantized:
                raise NotImplementedError(
                    "Beam expert parallelism dispatches NVFP4 to every sparse "
                    "layer, but the checkpoint leaves the routed experts of "
                    f"layers {unquantized} unquantized."
                )
        return replace(
            self,
            quant_config=quant_config,
            return_logits=outputs.return_logits,
        )

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the expert-parallel buffers for ``runtime``.

        Tokens dispatch in the experts' format: NVFP4 on a quantized
        checkpoint, else bfloat16.
        """
        n_devices = len(runtime.mesh.devices)
        return EPConfig(
            dispatch_dtype=self.dtype,
            dispatch_quant_config=self.quant_config,
            combine_dtype=DType.bfloat16,
            hidden_size=self.hidden_size,
            top_k=self.num_experts_per_tok,
            n_experts=self.num_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=runtime.max_batch_input_tokens,
                ep_size=runtime.ep_size,
                data_parallel_degree=runtime.data_parallel_degree,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=1,
        )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Returns the config placed on the deployment's mesh."""
        return replace(self, mesh=runtime.mesh, ep_config=runtime.ep)
