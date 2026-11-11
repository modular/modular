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
"""The config and draft registration of GLM-5.3-Flash's MTP layer on ModuleV3.

The layer runs only beside its ModuleV3 target, inside a fused spec-decode
module. The framework constructs it from the config :func:`configure_mtp_draft`
returns and loads the weights the draft's adapter names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    WeightTransform,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
    ModelOutputConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.lib.registration_draft import DraftRegistration
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..glm5_next.model_config import Glm5NextConfig
from ..glm5_next.mtp import Glm5NextMTP
from ..glm5_next.weight_adapters import glm5_next_with_mtp_weight_rules

__all__ = [
    "MTPGlm5NextDraftArchConfig",
    "configure_mtp_draft",
    "mtp_draft_registration",
]


@dataclass
class MTPGlm5NextDraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound, and no KV leaf.

    The layer writes the last slot of the target's latent, indexer and tail
    ring caches, which the target's config declares when it speculates.
    """

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> MTPGlm5NextDraftArchConfig:
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
    ) -> None:
        """Returns ``None``, because the layer writes into the target's caches.

        Raises:
            ValueError: If the accelerator is neither an NVIDIA GPU nor a
                gfx950 AMD GPU, the only AMD target the KDA ops build for.
        """
        del huggingface_config, pipeline_config, devices
        del kv_cache_config, cache_dtype
        api = accelerator_api()
        if api == "cuda":
            return
        if api == "hip" and accelerator_architecture_name().startswith(
            "gfx950"
        ):
            return
        raise ValueError(
            "GLM-5.3-Flash MTP runs only on NVIDIA and gfx950 GPUs, and this"
            f" host's accelerator API is {api!r}. Serve it without"
            " --speculative-method."
        )


def configure_mtp_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> Glm5NextConfig:
    """Returns the target's config, finalized for the MTP layer.

    The precision map is read off the whole checkpoint, MTP layer included,
    so the layer's FP8 projections build as FP8. The layer runs over the
    target's KV params, which ``context`` names, and starts expert
    parallelism over the runtime the target shares.

    Raises:
        ValueError: If the checkpoint ships no MTP layer.
    """
    if "eh_proj.weight" not in state_dict:
        raise ValueError(
            "The checkpoint ships no MTP layer: expected its eh_proj.weight"
            " past the target's last layer. Serve it without"
            " --speculative-method."
        )
    kv_params = context.kv_params
    assert isinstance(kv_params, MultiKVCacheParams)
    pipeline_config = context.pipeline_config
    config = Glm5NextConfig.initialize(
        pipeline_config, max_seq_len=context.max_seq_len
    ).finalize(
        context.huggingface_config,
        parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            WeightTransform(
                glm5_next_with_mtp_weight_rules(context.huggingface_config)
            ),
        ),
        ModelOutputConfig(),
    )
    config.kv_params = kv_params
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
    if runtime.ep_size > 1:
        ep = config.expert_parallel_config(runtime)
        context.start_ep(ep)
        runtime = replace(runtime, ep=ep)
    return config.apply_runtime(runtime)


mtp_draft_registration = DraftRegistration[Glm5NextMTP, Glm5NextConfig](
    method="mtp", configure=configure_mtp_draft
)
