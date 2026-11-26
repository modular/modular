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
"""The DSpark drafter Inkling 2 ships in ``mtp.safetensors``, as a draft.

The drafter has caches of its own, beside the target's in the shared pool,
and reads the target's embedding and LM head, which its proposer passes in;
so the checkpoint's ``mtp.safetensors`` tensors are all it loads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from max.driver import CPU, Device
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat
from max.nn import Module
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.lib import (
    GraphRole,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import DraftRegistration
from max.pipelines.lib.weight_transform import Ignore, WeightRule
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

from .dspark import Inkling2DSpark, construct_dspark_kv_params
from .dspark_config import Inkling2DSparkConfig

__all__ = [
    "Inkling2DSparkDraft",
    "Inkling2DSparkDraftConfig",
    "inkling2_dspark_draft_registration",
    "inkling2_dspark_draft_weight_rules",
]

MASK_TOKEN_ID = 0
"""The id that fills the block's mask slots; no embedding reads it."""


@dataclass(kw_only=True)
class Inkling2DSparkDraftConfig(ArchConfigWithBoundedMaxSeqLen):
    """The drafter's config, its caches and the sequence bound it rotates to."""

    max_seq_len: int
    dspark: Inkling2DSparkConfig | None = None
    kv_params: MultiKVCacheParams | None = None
    device: Device | None = None

    @property
    def hidden_size(self) -> int:
        """The hidden size the drafter reads and drafts in, the target's.

        The spec-decode pairing compares it with the target's, and
        :meth:`Inkling2DSparkConfig.from_hf` has already refused a checkpoint
        whose ``mtp_config`` disagrees, so this is that one comparison's
        reader rather than a second source.
        """
        assert self.dspark is not None
        return self.dspark.hidden_size

    @property
    def mask_token_id(self) -> int:
        """The id a block's mask slots carry, checked against the vocabulary.

        The drafter replaces every mask row's embedding with its own
        ``mask_embed``, so the id only gives the block its shape.
        """
        return MASK_TOKEN_ID

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        """The target layers whose inputs the drafter reads.

        The drafter also reads the final-norm output, id
        ``num_hidden_layers``, which names no layer; the verify taps all
        eight from ``dspark.target_layer_ids`` itself.
        """
        assert self.dspark is not None
        return tuple(self.dspark.tapped_layer_inputs)

    @property
    def vocab_size(self) -> int:
        """The vocabulary the drafter drafts over, the target's."""
        assert self.dspark is not None
        return self.dspark.vocab_size

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
    ) -> MultiKVCacheParams:
        """The drafter's own caches, on the target's devices and pool."""
        return construct_dspark_kv_params(
            Inkling2DSparkConfig.from_hf(huggingface_config),
            kv_cache_config,
            devices,
            cache_dtype,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=pipeline_config.speculative.speculative_method
            if pipeline_config.speculative is not None
            else None,
        )


class Inkling2DSparkDraft(Module[..., Any]):
    """Holds the drafter under the checkpoint's own tensor names."""

    def __init__(self, config: Inkling2DSparkDraftConfig) -> None:
        super().__init__()
        assert config.dspark is not None
        assert config.kv_params is not None
        assert config.device is not None
        self.dspark = Inkling2DSpark(
            config.dspark,
            config.kv_params,
            max_seq_len=config.max_seq_len,
            device=config.device,
        )
        self.dspark.name_transparent = True


def configure_inkling2_dspark(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> Inkling2DSparkDraftConfig:
    """Finalizes the drafter's config from the target's checkpoint."""
    del state_dict
    pipeline_config = context.pipeline_config
    assert isinstance(context.kv_params, MultiKVCacheParams)
    return Inkling2DSparkDraftConfig(
        max_seq_len=context.max_seq_len,
        dspark=Inkling2DSparkConfig.from_hf(
            pipeline_config.model.huggingface_config
        ),
        kv_params=context.kv_params,
        # The rotary tables build on the host and move with their first use.
        device=CPU(),
    )


def inkling2_dspark_draft_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Keeps ``mtp.safetensors``' tensors: everything outside ``model.``."""
    del huggingface_config
    return (Ignore(r"^model\.", reason="the target's own weights"),)


inkling2_dspark_draft_registration = DraftRegistration[
    Inkling2DSparkDraft, Inkling2DSparkDraftConfig
](method="dspark", configure=configure_inkling2_dspark)


def inkling2_dspark_draft_arch(target_arch: Any) -> Any:
    """The draft architecture, the target's with the drafter's roles."""
    return replace(
        target_arch,
        name="Inkling2DSparkDraftModel_ModuleV3",
        pipeline_model=None,
        batching=None,
        use_weight_rules=True,
        modules={GraphRole.DRAFT: Inkling2DSparkDraft},
        config=Inkling2DSparkDraftConfig,
        draft=inkling2_dspark_draft_registration,
        weight_rules=inkling2_dspark_draft_weight_rules,
        default_weights_format=WeightsFormat.safetensors,
        checkpoint_drafts={},
        spec_decode_graph_names={},
        supports_device_graph_capture=True,
    )
