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
"""Config for DFlash Llama3 unified pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max.nn import ReturnHiddenStates
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self

from ..llama3.model_config import ArchConfigWithKVCache, Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class DflashDraftHFConfig:
    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int | None = None
    num_target_layers: int | None = None


def parse_dflash_draft_hf_config(
    huggingface_config: Any,
) -> DflashDraftHFConfig:
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    dflash_cfg = _get(huggingface_config, "dflash_config", None)
    mask_token_id = _get(dflash_cfg, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "DFlash draft HF config is missing ``dflash_config.mask_token_id``."
        )
    target_layer_ids_raw = _get(dflash_cfg, "target_layer_ids", None)
    if target_layer_ids_raw is None:
        raise ValueError(
            "DFlash draft HF config is missing"
            " ``dflash_config.target_layer_ids``."
        )
    if not isinstance(target_layer_ids_raw, (list, tuple)):
        raise ValueError(
            "DFlash dflash_config.target_layer_ids must be a list of ints,"
            f" got {type(target_layer_ids_raw).__name__}."
        )
    target_layer_ids = [int(x) for x in target_layer_ids_raw]
    if not target_layer_ids:
        raise ValueError(
            "DFlash dflash_config.target_layer_ids must be non-empty."
        )

    raw_block_size = _get(
        dflash_cfg, "block_size", _get(huggingface_config, "block_size", None)
    )
    block_size = int(raw_block_size) if raw_block_size is not None else None
    raw_num_target_layers = _get(
        dflash_cfg,
        "num_target_layers",
        _get(huggingface_config, "num_target_layers", None),
    )
    num_target_layers = (
        int(raw_num_target_layers)
        if raw_num_target_layers is not None
        else None
    )

    return DflashDraftHFConfig(
        mask_token_id=int(mask_token_id),
        target_layer_ids=target_layer_ids,
        block_size=block_size,
        num_target_layers=num_target_layers,
    )


def resolve_dflash_num_speculative_tokens(
    pipeline_config: PipelineConfig,
    *,
    warn: bool = True,
) -> int:
    """Returns the resolved DFlash draft width.

    The DFlash drafter's behavior is only defined at its trained
    ``block_size``, so the width is ``block_size - 1``; a mismatching
    explicit ``num_speculative_tokens`` is overridden with a warning. When
    the draft checkpoint declares no ``block_size``, an explicit
    ``--num-speculative-tokens`` is required and returned as-is. Pure: the
    caller's config is never mutated or copied.
    """
    speculative = pipeline_config.speculative
    assert speculative is not None
    assert pipeline_config.draft_model is not None
    dflash_hf = parse_dflash_draft_hf_config(
        pipeline_config.draft_model.huggingface_config
    )
    if dflash_hf.block_size is None:
        if speculative.num_speculative_tokens is None:
            raise ValueError(
                "The DFlash draft checkpoint declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return speculative.num_speculative_tokens
    expected_spec = dflash_hf.block_size - 1
    actual_spec = speculative.num_speculative_tokens
    if warn and actual_spec is not None and actual_spec != expected_spec:
        logger.warning(
            "DFlash draft was trained at block_size=%d, so"
            " num_speculative_tokens is being overridden from %d to"
            " %d. The DFlash draft's behavior is only defined at"
            " its trained block_size.",
            dflash_hf.block_size,
            actual_spec,
            expected_spec,
        )
    return expected_spec


@dataclass(kw_only=True)
class UnifiedDflashLlama3Config(ArchConfigWithKVCache):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
    }

    target: Llama3Config
    draft: Llama3Config
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0
    quantization_encoding: SupportedEncoding | None = None

    def __post_init__(self) -> None:
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target.target_layer_ids = list(self.target_layer_ids)
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != len(self.draft.devices):
            raise ValueError(
                "Target and draft must have the same number of devices."
                f" Got target={len(self.target.devices)}"
                f" draft={len(self.draft.devices)}."
            )
        if len(self.target.devices) != 1:
            raise ValueError(
                "DFlash currently supports a single device only. Got"
                f" {len(self.target.devices)} devices."
            )

    def validate_dflash_fields(self) -> None:
        """Strict validation run from ``UnifiedDflashLlama3Model.load_model``
        once the DFlash-specific fields have been populated from the draft
        HF config — ``__post_init__`` accepts the empty-placeholder config
        produced by :meth:`initialize` so we can't enforce these there.
        """
        if not self.target_layer_ids:
            raise ValueError(
                "DFlash requires non-empty target_layer_ids (one per draft"
                " hidden layer)."
            )
        if len(self.target_layer_ids) != self.draft.num_hidden_layers:
            raise ValueError(
                "DFlash invariant: len(target_layer_ids) must equal the"
                " draft's num_hidden_layers."
                f" Got len(target_layer_ids)={len(self.target_layer_ids)}"
                f" draft.num_hidden_layers={self.draft.num_hidden_layers}."
            )
        if not 0 <= self.mask_token_id < self.target.vocab_size:
            raise ValueError(
                "DFlash mask_token_id must be in [0, target.vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" target.vocab_size={self.target.vocab_size}."
            )
        if self.block_size > 0:
            expected_spec = self.block_size - 1
            actual_spec = self.speculative_config.num_speculative_tokens
            if actual_spec is not None and actual_spec != expected_spec:
                # Check only, never written back: the trained width is
                # resolved as a plain int by
                # :func:`resolve_dflash_num_speculative_tokens` and
                # threaded by the model.
                logger.warning(
                    "DFlash draft was trained at block_size=%d, so"
                    " num_speculative_tokens is being overridden from %d to"
                    " %d. The DFlash draft's behavior is only defined at"
                    " its trained block_size.",
                    self.block_size,
                    actual_spec,
                    expected_spec,
                )

    def resolve_block_size(self, *, default: int | None = None) -> int:
        if self.block_size > 0:
            return self.block_size
        if default is not None:
            return default
        num_spec = self.speculative_config.num_speculative_tokens
        if num_spec is None:
            raise ValueError(
                "The DFlash draft checkpoint declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return num_spec + 1

    def get_kv_params(self) -> KVCacheParamInterface:
        target_kv_params = self.target.get_kv_params()
        draft_kv_params = self.draft.get_kv_params()
        return MultiKVCacheParams.from_params(
            {"target": target_kv_params, "draft": draft_kv_params}
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        assert pipeline_config.speculative is not None

        speculative_config = pipeline_config.speculative
        resolved: int | None = None
        if speculative_config.num_speculative_tokens is None:
            # Unset resolves to the drafter's trained width: unlike the
            # DSpark pipelines, this model never re-derives the baked
            # ``num_draft_tokens`` after load, so the value used here is
            # the one serving runs with. The resolved value lives on this
            # arch config's own speculative section; the caller's
            # pipeline_config is never mutated or copied.
            resolved = resolve_dflash_num_speculative_tokens(pipeline_config)
            speculative_config = speculative_config.model_copy(
                update={"num_speculative_tokens": resolved}
            )

        assert pipeline_config.draft_model is not None
        assert pipeline_config.draft_model.huggingface_config is not None

        target_config = Llama3Config.initialize_from_config(
            pipeline_config,
            model_config.huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )
        # Resolved at construction by PipelineConfig._resolve_max_length.
        draft_max_length = pipeline_config.draft_model.max_length
        assert draft_max_length is not None
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config,
            pipeline_config.draft_model.huggingface_config,
            pipeline_config.draft_model,
            max_seq_len=draft_max_length,
        )
        if resolved is not None:
            # ``initialize_from_config`` derives KV from the raw pipeline
            # config, where the unset width would bake num_draft_tokens=0.
            target_config.kv_params = replace(
                target_config.kv_params, num_draft_tokens=resolved
            )
            draft_config.kv_params = replace(
                draft_config.kv_params, num_draft_tokens=resolved
            )

        # Empty placeholder values for the DFlash-specific fields;
        # ``UnifiedDflashLlama3Model.load_model`` parses the draft HF config
        # and constructs the real values, then re-instantiates the config.
        return cls(
            target=target_config,
            draft=draft_config,
            speculative_config=speculative_config,
            target_layer_ids=[],
            mask_token_id=0,
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config, model_config
        )
