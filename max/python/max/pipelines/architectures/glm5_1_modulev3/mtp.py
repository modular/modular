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
"""The NextN (MTP) layer a GLM-5.x checkpoint ships, registered as a draft.

The layer sits past the target's last one, at ``num_hidden_layers``, and
runs only beside its target in a spec-decode pairing. It embeds with the
target's ``embed_tokens`` and projects with the target's ``lm_head``, which
its proposer reads off the target.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, Weights
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
    ModelOutputConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.lib.registration_draft import DraftRegistration
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.tensor import default_dtype
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from ..deepseekV3_2_modulev3 import weight_adapters as target_adapters
from ..deepseekV3_2_modulev3.deepseekV3_2 import rotary_embedding
from ..deepseekV3_2_modulev3.nextn import DeepseekV3_2NextN
from .model_config import Glm5_1Config

__all__ = [
    "Glm5_1MTP",
    "Glm5_1MTPArchConfig",
    "convert_mtp_state_dict",
    "mtp_draft_registration",
]

_MTP_HEAD_MAP = {
    "shared_head.norm.": "shared_head_norm.",
    "enorm.": "enorm.",
    "hnorm.": "hnorm.",
    "eh_proj.": "eh_proj.",
}

_MTP_SHARED = ("embed_tokens.", "shared_head.head.")
"""The layer's copies of the target's embedding and head, which it reads off
the target instead."""

_TARGET_FINAL_NORM = "language_model.norm.weight"


class Glm5_1MTP(DeepseekV3_2NextN):
    """The NextN layer, constructed from its config alone."""

    def __init__(self, config: Glm5_1Config) -> None:
        qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            super().__init__(
                config,
                attention_scale=rotary_embedding(config).compute_scale(
                    math.sqrt(1.0 / qk_head_dim)
                ),
                ep_batch_manager=None
                if config.ep_config is None
                else EPBatchManager(config.ep_config),
            )


@dataclass
class Glm5_1MTPArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV tree."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Glm5_1MTPArchConfig:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Returns a one-layer view of each of the target's caches."""
        target = Glm5_1Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        assert isinstance(target, MultiKVCacheParams)
        children: dict[str, KVCacheParams] = {}
        for name, leaf in target.children.items():
            assert isinstance(leaf, KVCacheParams)
            children[name] = replace(leaf, num_layers=1)
        return MultiKVCacheParams.from_params(children)


def convert_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **kwargs: object,
) -> dict[str, WeightData]:
    """Converts the NextN layer's weights out of a GLM-5.x checkpoint.

    The layer converts as one more target layer, so it takes the target's
    quantization layout and ModuleV3 names. Its copies of the target's
    embedding and head are left out.
    """
    num_layers = huggingface_config.num_hidden_layers
    # The target conversion drops the layer past the last one; one more layer
    # keeps it.
    with_mtp = copy.copy(huggingface_config)
    with_mtp.num_hidden_layers = num_layers + 1
    mtp_prefix = f"language_model.layers.{num_layers}."

    converted: dict[str, WeightData] = {}
    final_norm: WeightData | None = None
    for name, value in target_adapters.convert_safetensor_state_dict(
        state_dict, with_mtp, **kwargs
    ).items():
        if name == _TARGET_FINAL_NORM:
            final_norm = value
        if not name.startswith(mtp_prefix):
            continue
        suffix = name[len(mtp_prefix) :]
        if suffix.startswith(_MTP_SHARED):
            continue
        for before, after in _MTP_HEAD_MAP.items():
            if suffix.startswith(before):
                converted[f"{after}{suffix[len(before) :]}"] = value
                break
        else:
            converted[f"decoder_layer.{suffix}"] = value
    # Some checkpoints share the head norm with the target's final norm and
    # do not emit it under the layer.
    if "shared_head_norm.weight" not in converted and final_norm is not None:
        converted["shared_head_norm.weight"] = final_norm
    return converted


def _quantized(state_dict: Mapping[str, WeightData], subtree: str) -> bool:
    """Whether the layer's ``subtree`` ships a quantized ``weight_scale``.

    NVFP4 checkpoints leave the whole layer in bf16, with no scales.
    """
    return any(subtree in key and ".weight_scale" in key for key in state_dict)


def configure_mtp_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> Glm5_1Config:
    """Narrows the target's config to the NextN layer.

    The layer owns a one-layer cache, computes its own top-k, and may be
    quantized differently from the target's layers. It also starts expert
    parallelism, sized for the layer's dispatch: its bf16 experts dispatch
    wider than an NVFP4 target's, and the target shares the buffers.

    Raises:
        KeyError: If the checkpoint has no NextN layer.
    """
    if "eh_proj.weight" not in state_dict:
        raise KeyError(
            "The checkpoint ships no NextN layer: expected eh_proj.weight"
            f" past layer {context.huggingface_config.num_hidden_layers - 1}"
        )
    kv_params = context.kv_params
    assert isinstance(kv_params, MultiKVCacheParams)
    pipeline_config = context.pipeline_config
    # The target's own config, as its builder finalizes it.
    target = Glm5_1Config.initialize(
        pipeline_config, max_seq_len=context.max_seq_len
    ).finalize(
        context.huggingface_config,
        parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            target_adapters.convert_safetensor_state_dict,
        ),
        ModelOutputConfig(),
    )
    data_parallel_degree = pipeline_config.model.data_parallel_degree
    settings = pipeline_config.runtime
    runtime = ModuleRuntime(
        mesh=device_mesh(
            [ref.to_device() for ref in kv_params.devices],
            data_parallel_degree,
        ),
        data_parallel_degree=data_parallel_degree,
        max_batch_input_tokens=settings.max_batch_input_tokens,
        max_batch_total_tokens=context.max_batch_total_tokens,
        ep_size=settings.ep_size,
        ep_use_allreduce=settings.ep_use_allreduce,
    )
    experts_quantized = _quantized(state_dict, ".mlp.experts.")
    if runtime.ep_size > 1:
        ep = target.expert_parallel_config(runtime)
        if not experts_quantized:
            ep = replace(
                ep,
                dispatch_dtype=DType.bfloat16,
                dispatch_quant_config=None,
                fused_shared_expert=target.n_shared_experts == 1,
            )
        context.start_ep(ep)
        runtime = replace(runtime, ep=ep)
    config = target.apply_runtime(runtime)

    quant_config = config.quant_config
    if quant_config is not None:
        layer_idx = max(config.num_hidden_layers, config.first_k_dense_replace)
        mlp_layers = set(quant_config.mlp_quantized_layers)
        attn_layers = set(quant_config.attn_quantized_layers)
        if experts_quantized:
            mlp_layers.add(layer_idx)
        if _quantized(state_dict, ".self_attn."):
            attn_layers.add(layer_idx)
        quant_config = replace(
            quant_config,
            mlp_quantized_layers=mlp_layers,
            attn_quantized_layers=attn_layers,
        )
    return replace(
        config,
        kv_params=kv_params,
        # The layer computes its own top-k, so it indexes every step.
        indexer_types=[],
        quant_config=quant_config,
    )


mtp_draft_registration = DraftRegistration[Glm5_1MTP, Glm5_1Config](
    method="mtp", configure=configure_mtp_draft
)
