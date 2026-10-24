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
"""The MTP depths an Inkling checkpoint ships: their config and module builder.

The depths run only beside their target, inside a unified spec-decode graph.
The config builds the depths' module for that graph from the target's own
checkpoint.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

from ..inkling.model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    InklingConfig,
    InklingMTPConfig,
    parse_inkling_mtp_config,
)
from ..inkling.mtp import InklingMultiTokenPredictor

__all__ = [
    "MTPInklingDraftArchConfig",
    "inkling_mtp_draft_width",
]


def _mtp_config(huggingface_config: AutoConfig) -> InklingMTPConfig:
    mtp = parse_inkling_mtp_config(huggingface_config)
    if mtp is None:
        raise ValueError(
            "Inkling MTP requires checkpoint"
            " mtp_config.num_nextn_predict_layers > 0"
        )
    return mtp


def inkling_mtp_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: AutoConfig,
    draft_huggingface_config: AutoConfig | None,
) -> int:
    """Returns how many depths the draft builds: the requested width, capped
    at the depths the checkpoint ships."""
    del draft_huggingface_config
    return _mtp_config(target_huggingface_config).num_depths_for(speculative)


class MTPInklingDraftArchConfig(InklingConfig):
    """The depths' config: their target's, with a KV leaf per attention
    flavor the depths use."""

    @property
    def hidden_size(self) -> int:
        """Returns the width of the hidden states the depths chain."""
        return self.text_config.hidden_size

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Returns the target's attention leaves, each sized for the depths
        of its flavor; one depth writes one layer."""
        target = InklingConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        speculative = pipeline_config.speculative
        assert speculative is not None
        mtp = _mtp_config(huggingface_config)
        local = mtp.local_flags(mtp.num_depths_for(speculative))
        counts = {
            GLOBAL_ATTENTION: local.count(False),
            LOCAL_ATTENTION: local.count(True),
        }
        return MultiKVCacheParams.from_params(
            {
                key: replace(
                    target.child(key, MHAKVCacheParams), num_layers=count
                )
                for key, count in counts.items()
                if count
            }
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[InklingMultiTokenPredictor, MTPInklingDraftArchConfig]:
        """Builds the depths the speculative width needs from the target's
        checkpoint, which ``context`` names.

        They embed and project through the target's embedding and head,
        which their proposer shares onto them.

        Raises:
            ValueError: If the checkpoint ships no MTP depths.
        """
        assert isinstance(context.kv_params, MultiKVCacheParams)
        speculative = context.pipeline_config.speculative
        assert speculative is not None
        config = MTPInklingDraftArchConfig.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        config.mtp = _mtp_config(context.huggingface_config)
        config.kv_params = context.kv_params
        draft = InklingMultiTokenPredictor(
            config, config.mtp.num_depths_for(speculative), context.kv_params
        )
        draft.load_state_dict(
            parse_state_dict_from_weights(
                context.pipeline_config, context.weights, context.adapter
            ),
            weight_alignment=1,
            strict=True,
        )
        return draft, config
