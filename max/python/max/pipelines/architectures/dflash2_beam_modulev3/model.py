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
"""The Beam DFlash2 drafter's arch config, which builds its module.

The drafter runs only beside a Beam target, so it has no pipeline model: its
arch config sizes its KV leaf and builds the module a fused spec-decode
graph holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.tensor import default_device, default_dtype
from transformers import AutoConfig
from typing_extensions import Self

from ..dflash_mimo_v2 import DrafterExport
from ..dflash_mimo_v2.drafter import sampleable_vocab_size
from .dflash2_beam import DFlash2Beam
from .model_config import (
    DFlash2BeamConfig,
    DFlash2BeamDraftConfig,
    drafter_kv_params,
)

__all__ = ["DFlash2BeamDraftArchConfig"]


def _checkpoint(context: ModelLoadContext) -> dict[str, WeightData]:
    draft_model = context.pipeline_config.draft_model
    assert draft_model is not None
    return parse_state_dict_from_weights(
        context.pipeline_config,
        context.weights,
        context.adapter,
        hf_config=draft_model.huggingface_config,
        model_config=draft_model,
    )


@dataclass
class DFlash2BeamDraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
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
    ) -> KVCacheParams:
        """Returns the drafter's context cache, from its own geometry."""
        del cache_dtype
        return drafter_kv_params(
            huggingface_config, pipeline_config, devices, kv_cache_config
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlash2Beam, DFlash2BeamDraftConfig]:
        """Builds the drafter from the draft checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns, which must name every parameter.
        It embeds and projects through its target's embedding and head,
        which its proposer reads.

        Raises:
            ValueError: If the method isn't ``dflash2``, or the checkpoint
                and the module disagree on a weight's name.
            NotImplementedError: Under data parallelism, which Beam's target
                does not serve.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dflash2():
            raise ValueError(
                "The Beam DFlash2 drafter requires --speculative-method"
                f" dflash2, got {speculative.speculative_method!r}."
            )
        if pipeline_config.model.data_parallel_degree != 1:
            raise NotImplementedError(
                "The Beam DFlash2 drafter shards its heads over the target's"
                " tensor-parallel devices; data parallelism is not supported."
            )
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        block = speculative.draft_width + 1
        config = DFlash2BeamConfig.from_huggingface_config(
            draft_model.huggingface_config,
            block_size=block,
            devices=list(kv_params.devices),
            kv_params=kv_params,
            # The block's rows sit past the longest context.
            max_seq_len=context.max_seq_len + 2 * block,
        )
        mesh = device_mesh([ref.to_device() for ref in config.devices], 1)
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = DFlash2Beam(config)
        # A compiled module drops a weight no parameter names and leaves the
        # parameter at its initial value, so the names must match exactly.
        names = {name for name, _ in draft.parameters}
        stored = set(_checkpoint(context))
        if names != stored:
            raise ValueError(
                "The Beam DFlash2 checkpoint does not name the drafter's"
                f" parameters: missing {sorted(names - stored)[:8]},"
                f" unexpected {sorted(stored - names)[:8]}."
            )
        paths = draft_model.resolved_weight_paths()
        return draft, DFlash2BeamDraftConfig(
            module=config,
            sampleable_vocab_size=sampleable_vocab_size(pipeline_config),
            drafter_export=DrafterExport(
                directory=Path(paths[0]).parent,
                target_layer_ids=config.target_layer_ids,
                num_speculative_tokens=speculative.draft_width,
                spec_block_size=block,
            ),
        )

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DFlash2BeamDraftConfig
    ) -> dict[str, Any]:
        """Returns the weights the built drafter compiles with, by name."""
        del config
        return dict(_checkpoint(context))
