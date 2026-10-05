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
"""KV trees and drafter settings for MiMo-V2.6-Flash with its DFlash drafter.

The drafter's context cache is always the last KV group, after the target's
``sliding_attention`` and ``full_attention``. The fused speculative graph nests
the target's two under ``target``; the base graph that writes the drafter's
context lists all three flat. Both flatten to the same group order.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import huggingface_hub
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, KVConnectorType, MultiKVCacheParams
from max.pipelines.lib import (
    HuggingFaceRepo,
    KVCacheConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config
from transformers import AutoConfig
from typing_extensions import override

from ..dflash_mimo_v2 import DFlashMiMoV2Config
from ..mimo_v2.model_config import FULL, SLIDING, MiMoV2Config

DRAFT = "draft"
"""The drafter's KV group, the tail of every MiMo tree that has one."""

DFLASH_DIR = "dflash"
"""Where both MiMo-V2.6-Flash checkpoints keep the drafter."""


def mimo_dflash_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns how many of the block's proposals a step verifies.

    The block keeps its trained width whatever this is. Unset verifies all of
    them, ``block_size - 1``.

    Raises:
        ValueError: If the requested width does not fit in the block.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash requires a draft model.")
    block = parse_dflash_draft_hf_config(draft_huggingface_config).block_size
    if block is None:
        raise ValueError("The MiMo-V2 DFlash drafter declares no block_size.")
    # The shared DFlash policy (``dflash_draft_width``) replaces any requested
    # K with ``block_size - 1``, since a drafter is defined only at its
    # trained block. MiMo keeps drafting that block whatever K is, and its
    # ``head`` slices the first K proposals, so a smaller K is well defined.
    width = speculative.num_speculative_tokens
    if width is None:
        return block - 1
    if not 1 <= width <= block - 1:
        raise ValueError(
            f"MiMo-V2 DFlash verifies 1 to {block - 1} of its {block}-wide"
            f" block; got --num-speculative-tokens {width}."
        )
    return width


def repo_file(repo: HuggingFaceRepo, name: str) -> Path:
    """Returns a file of ``repo``'s folder, local or downloaded."""
    if repo.subfolder:
        name = f"{repo.subfolder}/{name}"
    if repo.repo_type == "local":
        return Path(repo.local_path) / name
    return Path(
        huggingface_hub.hf_hub_download(
            repo.repo_id, name, revision=repo.revision
        )
    )


def sampleable_vocab_size(pipeline_config: PipelineConfig) -> int:
    """Returns the tokenizer's size; ``lm_head`` rows past it are padding."""
    return len(
        PIPELINE_REGISTRY.get_active_tokenizer(
            pipeline_config.model.huggingface_model_repo
        )
    )


def read_dflash_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Reads a drafter's ``config.json``, raw, as the drafter module parses it."""
    return json.loads(Path(path).read_text())


def drafter_kv_params(
    dflash_config: Mapping[str, Any],
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
) -> KVCacheParams:
    """The drafter's context cache: its own geometry, always BF16.

    It is windowed like the drafter's attention. Under speculation it carries
    the block's width as its draft count, which every group of a tree shares.
    """
    speculative = pipeline_config.speculative
    return kv_cache_config.to_params(
        dtype=DType.bfloat16,
        n_kv_heads=int(dflash_config["num_key_value_heads"]),
        head_dim=int(dflash_config["head_dim"]),
        num_layers=int(dflash_config["num_hidden_layers"]),
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=(
            speculative.speculative_method if speculative else None
        ),
        num_draft_tokens=(
            int(dflash_config["block_size"]) if speculative else 0
        ),
        window_size=int(dflash_config["sliding_window"]),
    )


def with_num_draft_tokens(
    params: MultiKVCacheParams, num_draft_tokens: int
) -> MultiKVCacheParams:
    """``params`` with ``num_draft_tokens`` on every leaf."""
    children = {}
    for name, leaf in params.children.items():
        assert isinstance(leaf, KVCacheParams)
        children[name] = replace(leaf, num_draft_tokens=num_draft_tokens)
    return MultiKVCacheParams.from_params(children)


def drafter_config(
    dflash_config: Mapping[str, Any],
    kv_params: KVCacheParams,
    devices: list[DeviceRef],
    max_seq_len: int,
) -> DFlashMiMoV2Config:
    """The drafter's module config, with RoPE reaching a block past the end."""
    return DFlashMiMoV2Config.from_dflash_config(
        dflash_config,
        devices=devices,
        kv_params=kv_params,
        max_seq_len=max_seq_len + 2 * int(dflash_config["block_size"]),
    )


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
        """Builds ``{"target": {sliding, full}, "draft": ...}``.

        Raises:
            ValueError: If the cache is configured for dKV.
        """
        # The drafter's 1,024-key window gives the tree a second sliding
        # width beside the target's, which DKVConnector refuses: its touch
        # windows every sliding group by one width. The window saves real
        # memory, since the drafter's context is 20 KB a token, so DFlash on
        # MAX runs MiMo without dKV.
        if kv_cache_config.kv_connector_config.type == KVConnectorType.dkv:
            raise ValueError(
                "MiMo-V2 DFlash cannot use the dKV connector: the drafter's"
                " context window and the target's sliding window are two"
                " sliding widths, and dKV serves one."
            )
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


class MiMoV2DFlashContextConfig(MiMoV2Config):
    """The base graph's config with the drafter's group as a third, tail group.

    The drafter is the one the target checkpoint ships in ``dflash/``.
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
        """Builds ``{sliding, full, draft}``."""
        if pipeline_config.speculative is not None:
            raise ValueError(
                "The MiMo-V2 base graph that writes the drafter's context"
                " verifies nothing; build it without a speculative config."
            )
        dflash = read_dflash_config(
            repo_file(
                pipeline_config.model.huggingface_weight_repo,
                f"{DFLASH_DIR}/config.json",
            )
        )
        target = super().construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )
        return MultiKVCacheParams.from_params(
            {
                SLIDING: target.children[SLIDING],
                FULL: target.children[FULL],
                DRAFT: drafter_kv_params(
                    dflash, pipeline_config, devices, kv_cache_config
                ),
            }
        )
