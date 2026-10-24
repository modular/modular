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
"""KV trees and drafter settings for the ModuleV3 MiMo-V2.6-Flash + DFlash
pipeline.

The drafter's context cache is always the last KV group, after the target's
``sliding_attention`` and ``full_attention``, which the fused speculative
graph nests under ``target`` -- the same tree the graph-API build takes, so
Mach's block-spec adapter binds either export unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
)
from transformers import AutoConfig
from typing_extensions import override

from ..dflash_mimo_v2.drafter import (
    drafter_config,
    drafter_kv_params,
    mimo_dflash_draft_width,
    read_dflash_config,
    repo_file,
    sampleable_vocab_size,
)
from ..mimo_v2_modulev3.model_config import MiMoV2Config

__all__ = [
    "DRAFT",
    "UnifiedDflashMiMoV2Config",
    "drafter_config",
    "drafter_kv_params",
    "mimo_dflash_draft_width",
    "read_dflash_config",
    "repo_file",
    "sampleable_vocab_size",
    "verify_widened_row_budget",
]


def verify_widened_row_budget(pipeline_config: PipelineConfig) -> int:
    """Returns the EP dispatch row budget for the fused spec graph's DP
    build.

    The scheduler's token budget charges a decoding request its one active
    token, but the fused verify step merges ``draft_width`` draft rows
    beside it through every MoE layer -- rows the budget never counts --
    so a full decode batch of ``max_batch_size`` requests carries rows
    past ``max_batch_input_tokens``. The dispatch buffers are fixed at
    build and the dispatch kernel rejects a wider shard, so the budget
    counts the drafts, as DeepSeek-V4.1's DSpark EP config does.

    Args:
        pipeline_config: The pipeline whose scheduler budget and verify
            width bound the rows.

    Raises:
        ValueError: Without ``max_batch_size``, which bounds the requests
            a verify step carries and so the draft rows too.
    """
    runtime = pipeline_config.runtime
    speculative = pipeline_config.speculative
    assert speculative is not None
    if runtime.max_batch_size is None:
        raise ValueError(
            "the fused MiMo-V2 DFlash spec graph under expert "
            "parallelism needs max_batch_size, which bounds the draft "
            "rows a verify step adds to max_batch_input_tokens"
        )
    return (
        runtime.max_batch_input_tokens
        + runtime.max_batch_size * speculative.draft_width
    )


DRAFT = "draft"
"""The drafter's KV group, the tail of every MiMo tree that has one."""


def with_num_draft_tokens(
    params: MultiKVCacheParams, num_draft_tokens: int
) -> MultiKVCacheParams:
    """Returns ``params`` with ``num_draft_tokens`` on every leaf.

    The graph-API unified package owned this; the generic composition on
    main stamps the count itself, and this package builds its own fused
    tree, so it lives here now.

    Args:
        params: The KV tree to stamp.
        num_draft_tokens: The draft count every leaf carries.
    """
    children = {}
    for name, leaf in params.children.items():
        assert isinstance(leaf, KVCacheParams)
        children[name] = replace(leaf, num_draft_tokens=num_draft_tokens)
    return MultiKVCacheParams.from_params(children)


class UnifiedDflashMiMoV2Config(MiMoV2Config):
    """The target's config, with the drafter's group in its KV tree.

    ``kv_params`` is ``{"target": {sliding, full}, "draft": ...}``, so memory
    planning prices the drafter's cache.
    """

    @override
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
        """Builds ``{"target": {sliding, full}, "draft": ...}``."""
        draft_model = pipeline_config.draft_model
        assert draft_model is not None
        dflash = read_dflash_config(
            repo_file(draft_model.huggingface_weight_repo, "config.json")
        )
        draft = drafter_kv_params(
            dflash, pipeline_config, devices, kv_cache_config
        )
        target = with_num_draft_tokens(
            super().construct_kv_params(
                huggingface_config,
                pipeline_config,
                devices,
                kv_cache_config,
                cache_dtype,
                allow_kv_head_replication=allow_kv_head_replication,
            ),
            draft.num_draft_tokens,
        )
        return MultiKVCacheParams.from_params({"target": target, DRAFT: draft})
