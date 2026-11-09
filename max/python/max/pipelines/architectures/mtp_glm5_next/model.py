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
"""The MTP layer a GLM-5.3-Flash checkpoint ships: its config and builder.

The layer runs only beside its target, inside a unified spec-decode graph.
The config builds the layer's module for that graph from the target's own
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
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
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..glm5_next.model import Glm5NextModel
from ..glm5_next.model_config import Glm5NextConfig
from ..glm5_next.mtp import Glm5NextMTP
from ..glm5_next.weight_adapters import glm5_next_with_mtp_weight_rules

__all__ = ["MTPGlm5NextDraftArchConfig"]


@dataclass
class MTPGlm5NextDraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound, and no KV leaf."""

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

        It writes the last slot of the target's latent, indexer and tail ring
        caches, which the target's config declares when it speculates.

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

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Glm5NextMTP, Glm5NextConfig]:
        """Builds the MTP layer from the checkpoint that ``context`` names.

        Its config is the target's, with the precision map read off the
        whole checkpoint, MTP layer included, so its FP8 projections load as
        FP8. It embeds and projects through the target's embedding and head,
        so spec decoding shares those onto it.

        Raises:
            ValueError: If the checkpoint ships no MTP layer.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config,
            context.weights,
            WeightTransform(
                glm5_next_with_mtp_weight_rules(context.huggingface_config)
            ),
        )
        config = Glm5NextModel.finalized_config(
            context.pipeline_config,
            state_dict,
            max_seq_len=context.max_seq_len,
            max_batch_total_tokens=context.max_batch_total_tokens,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, MultiKVCacheParams)
            config.kv_params = context.kv_params
        mtp_state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        if not mtp_state_dict:
            raise ValueError(
                "no MTP layer in the checkpoint; GLM-5.3-Flash MTP needs the"
                " layer the checkpoint ships past its last decoder layer"
            )
        context.start_ep(config.ep_config)
        draft = Glm5NextMTP(config)
        draft.load_state_dict(mtp_state_dict, weight_alignment=1, strict=True)
        return draft, config

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
