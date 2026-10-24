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
"""Picks the proposer a unified spec-decode graph drafts through.

The speculative method picks the proposer, as vLLM's model runner picks its
drafter and SGLang's ``SpeculativeAlgorithm.create_worker`` its worker. A
proposer calls its draft module by the method's convention, so one proposer
serves every draft checkpoint its method runs, whichever architecture built
the draft.
"""

from __future__ import annotations

from typing import Any

from max.nn.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

from .block_driver import BlockProposer
from .config import SpeculativeConfig, SpeculativeMethod
from .cross_attention import CrossAttentionDraft, CrossAttentionMTPProposer
from .dflash import DFlashDraft, DFlashProposer
from .dflash2 import DFlash2Draft, DFlash2Proposer
from .driver import SequentialProposer
from .dspark import (
    DSparkDraft,
    DSparkProposer,
    DSparkTargetHeadDraft,
    DSparkTargetHeadProposer,
    TensorParallelDSparkDraft,
    TensorParallelDSparkProposer,
)
from .eagle import EagleProposer
from .mla import Eagle3MHAProposer, Eagle3MLAProposer, MLADraft, MTPProposer
from .shared_head import SharedHeadMTPDraft, SharedHeadMTPProposer

__all__ = [
    "BLOCK_METHODS",
    "SEQUENTIAL_METHODS",
    "block_proposer",
    "capture_layers",
    "sequential_proposer",
    "target_hidden_states",
]

SEQUENTIAL_METHODS: frozenset[SpeculativeMethod] = frozenset({"eagle", "mtp"})
"""The one-token-per-step methods :func:`sequential_proposer` runs."""

BLOCK_METHODS: frozenset[SpeculativeMethod] = frozenset(
    {"dflash", "dflash2", "dspark"}
)
"""The block-drafting methods :func:`block_proposer` runs."""


def target_hidden_states(
    speculative: SpeculativeConfig, draft_config: Any
) -> ReturnHiddenStates:
    """Returns the target hidden states the configured draft reads.

    A draft that names target layers reads their outputs. Any other
    one-token-per-step draft reads the target's last normalized layer.
    """
    if capture_layers(speculative, draft_config):
        return ReturnHiddenStates.SELECTED_LAYERS
    return ReturnHiddenStates.ALL_NORMALIZED


def capture_layers(
    speculative: SpeculativeConfig, draft_config: Any
) -> tuple[int, ...]:
    """Returns the target layers a draft reads, as MAX indices.

    Every block draft's config names them in ``target_layer_ids``, and an
    EAGLE3 draft's names its auxiliary layers there, as vLLM's
    ``eagle_aux_hidden_state_layer_ids`` does.
    """
    if speculative.is_block_draft():
        return tuple(draft_config.target_layer_ids)
    return tuple(getattr(draft_config, "target_layer_ids", None) or ())


def sequential_proposer(
    speculative: SpeculativeConfig,
    draft: Module,
    draft_config: Any,
    target_huggingface_config: Any,
    *,
    distributed: bool = False,
) -> SequentialProposer[Any]:
    """Returns the proposer for a one-token-per-step method.

    A draft beside a target that takes the distributed signature takes it
    too, as every DeepSeek-family draft does. An EAGLE3 head there caches
    MLA latents or its own heads.

    Raises:
        ValueError: If no proposer runs the configured method.
    """
    del target_huggingface_config
    if speculative.is_eagle():
        if distributed:
            assert isinstance(draft, MLADraft)
            if isinstance(draft_config.kv_params, MLAKVCacheParams):
                return Eagle3MLAProposer(draft)
            return Eagle3MHAProposer(draft)
        return EagleProposer(draft, draft_config.hidden_size)
    if speculative.is_mtp():
        # A head without a cache of its own attends into its target's.
        if getattr(draft_config, "kv_params", None) is None:
            assert isinstance(draft, CrossAttentionDraft)
            return CrossAttentionMTPProposer(draft, draft_config.hidden_size)
        # A head over an attention cache projects through its target's head.
        if isinstance(draft_config.kv_params, MHAKVCacheParams):
            assert isinstance(draft, SharedHeadMTPDraft)
            return SharedHeadMTPProposer(draft, draft_config.hidden_size)
        assert isinstance(draft, MLADraft)
        return MTPProposer(draft)
    raise ValueError(
        f"No proposer runs {speculative.speculative_method!r} one token at a"
        " time."
    )


def block_proposer(
    speculative: SpeculativeConfig,
    draft: Module,
    draft_config: Any,
    target_huggingface_config: Any,
    *,
    distributed: bool = False,
) -> BlockProposer[Any, Any]:
    """Returns the proposer for a block-drafting method.

    The draft's config names the mask token and the draft's width. A draft
    that projects through its target's head applies the target's scales.

    Raises:
        ValueError: If no proposer runs the configured method.
    """
    text_config = target_huggingface_config.get_text_config()
    if speculative.is_dflash2():
        assert isinstance(draft, DFlash2Draft)
        return DFlash2Proposer(
            draft,
            block_size=draft_config.block_size,
            # The selector walks a path through every slot after the anchor.
            num_drafts=draft_config.block_size - 1,
            mask_token_id=draft_config.mask_token_id,
            hidden_size=draft_config.hidden_size,
        )
    if speculative.speculative_method == "dflash":
        assert isinstance(draft, DFlashDraft)
        return DFlashProposer(
            draft,
            # A draft trained at a fixed width drafts at it; any other drafts
            # the anchor slot plus the drafted tokens.
            block_size=(
                getattr(draft_config, "block_size", None)
                or speculative.draft_width + 1
            ),
            num_drafts=speculative.draft_width,
            mask_token_id=draft_config.mask_token_id,
            hidden_size=draft_config.hidden_size,
            embedding_multiplier=getattr(
                text_config, "embedding_multiplier", 1.0
            ),
            logits_scaling=getattr(text_config, "logits_scaling", 1.0),
            final_logit_softcapping=getattr(
                text_config, "final_logit_softcapping", None
            ),
            sampleable_vocab_size=getattr(
                draft_config, "sampleable_vocab_size", None
            ),
        )
    if speculative.is_dspark():
        # A DSpark draft sharded like its distributed target drafts at every
        # slot when its checkpoint samples from the anchor.
        if distributed:
            assert isinstance(draft, TensorParallelDSparkDraft)
            anchor = draft_config.sample_from_anchor
            return TensorParallelDSparkProposer(
                draft,
                block_size=speculative.draft_width + (0 if anchor else 1),
                mask_token_id=draft_config.mask_token_id,
                hidden_size=draft_config.hidden_size,
                samples_from_anchor=anchor,
            )
        # A DSpark draft with its own head drops the anchor slot; one that
        # holds its target's drafts at every slot.
        if getattr(draft, "has_own_lm_head", False):
            assert isinstance(draft, DSparkDraft)
            return DSparkProposer(
                draft,
                block_size=speculative.draft_width + 1,
                mask_token_id=draft_config.mask_token_id,
                hidden_size=draft_config.hidden_size,
            )
        assert isinstance(draft, DSparkTargetHeadDraft)
        return DSparkTargetHeadProposer(
            draft,
            block_size=speculative.draft_width,
            mask_token_id=draft_config.mask_token_id,
            hidden_size=draft_config.hidden_size,
            final_logit_softcapping=draft_config.final_logit_softcapping,
        )
    raise ValueError(
        f"No proposer runs {speculative.speculative_method!r} a block at a"
        " time."
    )
