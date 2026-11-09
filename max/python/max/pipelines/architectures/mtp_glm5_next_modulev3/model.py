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
"""Builds GLM-5.3-Flash's MTP layer as a ModuleV3 module.

The layer runs only beside its ModuleV3 target, inside a fused spec-decode
module, and compiles with the weights :meth:`checkpoint_weights` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager
from max.pipelines.lib import (
    CheckpointRole,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    WeightEntry,
    WeightTransform,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
    ModelOutputConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.tensor import default_device, default_dtype
from transformers import AutoConfig

from ..glm5_next_modulev3.model_config import Glm5NextConfig
from ..glm5_next_modulev3.mtp import Glm5NextMTP
from ..glm5_next_modulev3.weight_adapters import (
    glm5_next_with_mtp_weight_rules,
)
from ..mtp_glm5_next.model import MTPGlm5NextDraftArchConfig

__all__ = ["MTPGlm5NextModuleV3DraftArchConfig"]


@dataclass
class MTPGlm5NextModuleV3DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The ModuleV3 draft's config, which builds its module.

    Its sequence bound and its lack of a KV leaf are the graph-API draft's:
    the layer writes the last slot of the target's latent, indexer and tail
    ring caches.
    """

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> MTPGlm5NextModuleV3DraftArchConfig:
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
        """Returns ``None``, as the graph-API draft does.

        It does so on the accelerators the draft builds for.

        Raises:
            ValueError: If the accelerator is neither an NVIDIA GPU nor a
                gfx950 AMD GPU.
        """
        MTPGlm5NextDraftArchConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Glm5NextMTP, Glm5NextConfig]:
        """Builds the MTP layer over the KV params ``context`` names.

        Those are the target's.

        Its config is the target's, with the precision map read off the
        whole checkpoint, MTP layer included, so its FP8 projections build as
        FP8. It embeds and projects through the target's embedding and head.
        It dispatches its experts through an expert-parallel manager of its
        own, over the runtime the target shares.
        """
        pipeline_config = context.pipeline_config
        state_dict = parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            WeightTransform(
                glm5_next_with_mtp_weight_rules(context.huggingface_config)
            ),
        )
        config = Glm5NextConfig.initialize(
            pipeline_config, max_seq_len=context.max_seq_len
        ).finalize(context.huggingface_config, state_dict, ModelOutputConfig())
        assert isinstance(context.kv_params, MultiKVCacheParams)
        config.kv_params = context.kv_params
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        settings = pipeline_config.runtime
        runtime = ModuleRuntime(
            mesh=device_mesh(
                [ref.to_device() for ref in context.kv_params.devices],
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
        config = config.apply_runtime(runtime)
        ep_batch_manager = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        assert config.mesh is not None
        with F.lazy(), default_dtype(dtype), default_device(config.mesh):
            draft = Glm5NextMTP(config, ep_batch_manager)
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: Glm5NextConfig
    ) -> dict[str, Any]:
        """Returns the MTP layer's weights, under the draft module's names.

        Raises:
            ValueError: If the checkpoint ships no MTP layer.
        """
        del config
        weights = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        if not weights:
            raise ValueError(
                "no MTP layer in the checkpoint; GLM-5.3-Flash MTP needs the"
                " layer the checkpoint ships past its last decoder layer"
            )
        return weights

    @classmethod
    def weight_entries(
        cls, context: ModelLoadContext, config: Glm5NextConfig
    ) -> list[WeightEntry]:
        """Returns the entry that names the MTP layer's weights.

        The layer ships in the target's checkpoint, which the draft loads as
        its own; its rules leave out the target's weights.
        """
        del config
        assert isinstance(context.adapter, WeightTransform)
        return [WeightEntry(CheckpointRole.DRAFT, context.adapter)]
