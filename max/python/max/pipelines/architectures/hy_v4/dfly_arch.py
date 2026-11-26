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
"""Registers the Hy4 VL DFly2 drafter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    KVCacheConfig,
    PipelineConfig,
    SupportedArchitecture,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import (
    DraftRegistration,
    TargetHiddenStates,
)
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from .dfly import HYV4DFlyDrafter, draft_config
from .model_config import HYV4VLConfig
from .tokenizer import HYV4Tokenizer
from .weight_adapters import hy_v4_dfly_weight_rules

__all__ = ["hy_v4_dfly_draft_arch"]


class HYV4DFlyDraftArchConfig(HYV4VLConfig):
    """The drafter's config: its target's, with no cache of its own."""

    @staticmethod
    def construct_kv_params(  # type: ignore[override]
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> None:
        """Returns ``None``: the drafter reads and writes the target's MLA."""
        del huggingface_config, pipeline_config, devices
        del kv_cache_config, cache_dtype

    @property
    def mask_token_id(self) -> int:
        """Fills the block past its bonus token."""
        return self.dfly.mask_token_id


def _configure(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> HYV4VLConfig:
    """Narrows the target's config to the drafter at the served width."""
    pipeline = context.pipeline_config
    assert pipeline.speculative is not None
    assert isinstance(context.kv_params, MultiKVCacheParams)
    config = HYV4DFlyDraftArchConfig.initialize(
        pipeline, max_seq_len=context.max_seq_len
    )
    config = replace(
        config,
        kv_params=context.kv_params,
        quant_config=parse_quant_config(
            pipeline.model.huggingface_config, dict(state_dict), config.dtype
        ),
    )
    return draft_config(config, pipeline.speculative.draft_width)


def _target_hidden_states(
    speculative: SpeculativeConfig, config: HYV4VLConfig
) -> TargetHiddenStates:
    """Names no target layers: the verify taps what the drafter reads.

    The default reads ``target_layer_ids``, which this config has none of.
    """
    del speculative, config
    return TargetHiddenStates(mode=ReturnHiddenStates.ALL_NORMALIZED)


# The drafter ships in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's.
hy_v4_dfly_draft_arch = SupportedArchitecture(
    name="DFlyHYV4VLDraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["tencent/Hy4-FP8"],
    default_encoding=HYV4VLConfig.DEFAULT_ENCODING,
    supported_encodings=HYV4VLConfig.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    modules={GraphRole.DRAFT: HYV4DFlyDrafter},
    draft=DraftRegistration[HYV4DFlyDrafter, HYV4VLConfig](
        method="dfly",
        configure=_configure,
        target_hidden_states=_target_hidden_states,
    ),
    tokenizer=HYV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_rules=hy_v4_dfly_weight_rules,
    use_weight_rules=True,
    config=HYV4DFlyDraftArchConfig,
)
