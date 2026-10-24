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
"""Reading a MiMo-V2 DFlash drafter from its checkpoint folder.

The drafter's ``config.json`` is read raw, as the drafter module parses it,
and its weights are a safetensors file beside a trained mask embedding.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import huggingface_hub
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, load_weights
from max.nn.kv_cache import KVCacheParams, KVConnectorType
from max.pipelines.lib import (
    HuggingFaceRepo,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config
from transformers import AutoConfig

from .dflash_mimo_v2 import DFlashMiMoV2
from .model_config import DFlashMiMoV2Config
from .weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING,
    MASK_EMBEDDING_FILE,
    convert_safetensor_state_dict,
    load_mask_embedding,
)


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


@dataclass(frozen=True)
class DrafterExport:
    """What an export built into the graph from the drafter.

    ``gen_mef`` records it in the sidecar with the hashes of the files in
    ``directory``, which Mach re-hashes at boot.
    """

    directory: Path
    """The drafter directory whose files the graph was built from."""
    target_layer_ids: list[int]
    num_speculative_tokens: int
    """The verify width of the speculative graph."""
    spec_block_size: int
    """The drafter's block width."""


def drafter_export(
    config: DFlashMiMoV2Config,
    directory: Path,
    *,
    num_speculative_tokens: int,
) -> DrafterExport:
    """Describes a graph built from ``config`` for its export sidecar."""
    return DrafterExport(
        directory=directory,
        target_layer_ids=list(config.target_layer_ids),
        num_speculative_tokens=num_speculative_tokens,
        spec_block_size=config.block_size,
    )


def load_drafter(
    repo: HuggingFaceRepo, config: DFlashMiMoV2Config, subfolder: str = ""
) -> tuple[dict[str, WeightData], WeightData, Path]:
    """Loads and checks every drafter tensor and the mask embedding.

    Returns:
        The drafter's tensors without the mask embedding, the mask embedding,
        and the directory they were read from.
    """
    prefix = f"{subfolder}/" if subfolder else ""
    weights_path = repo_file(repo, prefix + DRAFT_WEIGHTS_FILE)
    mask = load_mask_embedding(
        repo_file(repo, prefix + MASK_EMBEDDING_FILE), config
    )
    state = convert_safetensor_state_dict(
        dict(load_weights([weights_path]).items()), config, mask
    )
    del state[MASK_EMBEDDING]
    return state, mask, weights_path.parent


@dataclass(frozen=True)
class DFlashMiMoV2DraftConfig:
    """The drafter module's config, and what the pairing reads beside it."""

    module: DFlashMiMoV2Config
    sampleable_vocab_size: int
    """The tokenizer's size. The target's head rows from here up are padding
    with live weights, so neither verify nor the draft samples them."""
    drafter_export: DrafterExport
    """What the graph bakes in of the drafter, for an export's sidecar."""

    @property
    def hidden_size(self) -> int:
        """Returns the drafter's hidden size."""
        return self.module.hidden_size

    @property
    def block_size(self) -> int:
        """Returns the block width the drafter was trained at."""
        return self.module.block_size

    @property
    def mask_token_id(self) -> int:
        """Returns the token the block's drafted slots stand for."""
        return self.module.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the drafter reads."""
        return self.module.target_layer_ids


@dataclass
class DFlashMiMoV2DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf.

    The drafter is served only beside a MiMo-V2 target, so nothing else of
    it is read from here.
    """

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> DFlashMiMoV2DraftArchConfig:
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
        """Returns the drafter's context cache, from its raw config.

        Raises:
            ValueError: If the cache is configured for dKV.
        """
        del huggingface_config, cache_dtype
        # The connector itself serves this tree per leaf; the graph-API
        # build keeps the refusal by policy (all model work is
        # ModuleV3-only).
        if kv_cache_config.kv_connector_config.type == KVConnectorType.dkv:
            raise ValueError(
                "MiMo-V2 DFlash supports the dKV connector only on its "
                "ModuleV3 build. Pass --prefer-module-v3 to use it."
            )
        draft_model = pipeline_config.draft_model
        assert draft_model is not None
        return drafter_kv_params(
            read_dflash_config(
                repo_file(draft_model.huggingface_weight_repo, "config.json")
            ),
            pipeline_config,
            devices,
            kv_cache_config,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlashMiMoV2, DFlashMiMoV2DraftConfig]:
        """Builds the drafter from the draft checkpoint ``context`` names.

        It reads the folder's ``config.json`` raw, as the drafter module
        parses it, and loads its weights and trained mask embedding from the
        same folder. It embeds and projects through its target's embedding
        and head, so spec decoding shares those onto it.

        Raises:
            ValueError: If the method isn't ``dflash``.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if speculative.speculative_method != "dflash":
            raise ValueError(
                "The MiMo-V2 DFlash drafter requires --speculative-method"
                f" dflash, got {speculative.speculative_method!r}."
            )
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        repo = draft_model.huggingface_weight_repo
        config = drafter_config(
            read_dflash_config(repo_file(repo, "config.json")),
            kv_params,
            list(kv_params.devices),
            context.max_seq_len,
        )
        state, mask_embedding, directory = load_drafter(repo, config)
        draft = DFlashMiMoV2(config, mask_embedding=mask_embedding)
        draft.load_state_dict(state, weight_alignment=1, strict=True)
        return draft, DFlashMiMoV2DraftConfig(
            module=config,
            sampleable_vocab_size=sampleable_vocab_size(pipeline_config),
            drafter_export=drafter_export(
                config,
                directory,
                num_speculative_tokens=speculative.draft_width,
            ),
        )
