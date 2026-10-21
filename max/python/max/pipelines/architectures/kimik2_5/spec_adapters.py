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
"""Speculative-decoding adapters for a Kimi K2.5 target.

The target is a :class:`DeepseekV3`, but it differs from the DeepSeek
speculators in two ways that keep it off :class:`DeepseekV3Target`: it may
scatter vision embeddings into the merged sequence before running its stack,
and its Eagle3 aux capture layers arrive as separate tensors per device rather
than concatenated into one. Both drafts take the DeepSeek MLA draft call
signature, so the proposers below only declare their names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from max.dtype import DType
from max.graph import (
    BufferType,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import captures_by_device
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    Proposed,
    SequentialBatch,
)
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.spec_adapters import DeepseekV3MLAProposer

__all__ = [
    "Eagle3KimiK25Proposer",
    "Eagle3MHAKimiK25Proposer",
    "KimiK25Target",
]

_TargetHidden = list[list[TensorValue]]
"""Kimi's Eagle3 captures: one list of aux layers per device."""


class KimiK25Target:
    """The Kimi K2.5 target entry point, optionally scattering vision."""

    def __init__(self, target: DeepseekV3, *, enable_vision: bool) -> None:
        self.target = target
        self.enable_vision = enable_vision

    def verify(self, batch: SequentialBatch) -> Verified[_TargetHidden]:
        if self.enable_vision:
            # Embed, scatter the image embeddings at their merge positions,
            # then run the rest of the stack. Prefill merges zero draft tokens
            # per row, so positions stay valid without remapping; decode
            # introduces no new images.
            h_per_dev = self.target.embed_tokens(
                batch.merged_tokens, batch.signal_buffers
            )
            h_per_dev = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_d,
                    multimodal_embeddings=img_emb_d,
                    image_token_indices=img_idx_d,
                )
                for h_d, img_emb_d, img_idx_d in zip(
                    h_per_dev,
                    batch.vision_embeddings,
                    batch.vision_scatter_indices,
                    strict=True,
                )
            ]
            outputs = self.target._process_hidden_states(
                h_per_dev,
                batch.signal_buffers,
                batch.kv_collections,
                batch.return_n_logits,
                list(batch.merged_offsets_per_dev),
                batch.dist.host_merged_offsets,
                batch.dist.data_parallel_splits,
                batch.batch_context_lengths,
                batch.ep_inputs,
            )
        else:
            outputs = self.target(
                batch.merged_tokens,
                batch.signal_buffers,
                batch.kv_collections,
                batch.return_n_logits,
                batch.merged_offsets_per_dev,
                batch.dist.host_merged_offsets,
                batch.dist.data_parallel_splits,
                batch.batch_context_lengths,
                batch.ep_inputs,
            )
        return Verified(
            logits=outputs[1],
            hidden=captures_by_device(outputs[3:], batch.n_devs),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.target.ep_manager is None:
            return ()
        return self.target.ep_manager.input_types()


class Eagle3KimiK25Proposer(DeepseekV3MLAProposer[_TargetHidden]):
    """The MLA-draft Eagle3 proposer."""

    split_prefix = "eagle3"
    carry_dim_names = CarryDimNames(prefix="draft_step", per_device=True)
    uses_thinking_phase = True
    # In decode mode ALL-hs == LAST-hs, and ALL returns per-device hidden
    # states directly, avoiding the LAST path's allgather.
    step_hidden_mode = ReturnHiddenStates.ALL


class Eagle3MHAKimiK25Proposer(DeepseekV3MLAProposer[_TargetHidden]):
    """The MHA-draft Eagle3 proposer.

    An MHA draft has no MLA partition count to retarget, and its step-0 pass
    needs a wider dispatch metadata than the cache manager resolved.
    """

    split_prefix = "eagle3_mha"
    carry_dim_names = CarryDimNames(prefix="draft_mha_step", per_device=True)
    uses_thinking_phase = True
    step_hidden_mode = ReturnHiddenStates.ALL
    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
    )

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: _TargetHidden,
    ) -> Proposed:
        return super().prefill(
            replace(
                batch,
                draft_kv_collections=[
                    patch_draft0_kv_cache(kv)
                    for kv in batch.draft_kv_collections
                ],
            ),
            tokens,
            target_hidden,
        )


def patch_draft0_kv_cache(kv: PagedCacheValues) -> PagedCacheValues:
    """Returns ``kv`` with ``attention_dispatch_metadata`` re-sized for the
    unified-eagle draft's step-0 prefill."""

    decode_md = kv.draft_attention_dispatch_metadata
    assert decode_md is not None
    step0_max_prompt = kv.max_prompt_length.cast(DType.int64).reshape([1]) + 1
    step0_md = ops.concat(
        [decode_md[0:1], step0_max_prompt, decode_md[2:4]],
        axis=0,
    )
    return replace(kv, attention_dispatch_metadata=step0_md)
