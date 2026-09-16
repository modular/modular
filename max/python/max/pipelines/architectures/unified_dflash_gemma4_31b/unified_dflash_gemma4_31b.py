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
"""Unified DFlash Gemma4 nn.Module: target + KV materialize + draft block.

The unified DFlash graph (merge -> target -> reject -> materialize -> block
forward) over a Gemma4 target. Differences from the DSpark sibling, all from
the z-lab DFlash runtime reference (vLLM ``qwen3_dflash.py``):

- The drafter owns no embedding and no head. Block slots embed through the
  target's ``ScaledWordEmbedding`` (gemma's ``sqrt(hidden)`` scale, which the
  reference reproduces explicitly for gemma4 targets) and draft logits come
  from the target's tied ``lm_head``.
- No markov chain and no ``d2t``: draft ids are a plain argmax over the
  target vocabulary.
- The tap count is set by ``fc``'s in-features (six taps into five layers),
  not by the draft's layer count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeGraphSignature,
    SpecDecodeInputTypeSpec,
)
from max.pipelines.speculative.unified_graph_ops import apply_overlap_bitmask
from typing_extensions import override

from ..dflash_llama3 import DFlashLlama3
from ..gemma4.gemma4 import Gemma4TextModel
from .model_config import UnifiedDflashGemma4_31BConfig


def _block_dispatch_metadata(meta: TensorValue | None, k: int) -> TensorValue:
    """Rebuilds the MHA dispatch metadata at the draft block's query width.

    The 4-int CPU buffer is ``[batch_size, q_max_seq_len, num_partitions,
    max_cache_valid_length]``. ``q_max_seq_len`` becomes the block width
    ``k`` and ``num_partitions`` is zeroed so the decode kernel recomputes
    the split-K count for the draft's own head geometry instead of reusing
    the target's.

    Args:
        meta: The leaf's verify-width dispatch metadata buffer.
        k: The draft block width (anchor slot plus drafted tokens).

    Returns:
        The rebuilt dispatch metadata buffer.
    """
    assert meta is not None
    cpu = DeviceRef.CPU()
    return ops.concat(
        [
            meta[0:1],
            ops.constant(k, DType.int64, device=cpu).reshape((1,)),
            ops.constant(0, DType.int64, device=cpu).reshape((1,)),
            meta[3:4],
        ],
        axis=0,
    )


@dataclass
class UnifiedDflashGemma4_31BValues:
    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    return_n_logits: TensorValue
    signal_buffers: list[BufferValue]
    sliding_kv_collection: PagedCacheValues
    global_kv_collection: PagedCacheValues
    draft_kv_collection: PagedCacheValues
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue
    in_thinking_phase: TensorValue
    pinned_bitmask: TensorValue | None = None
    wait_payload: BufferValue | None = None
    device_bitmask_scratch: BufferValue | None = None


class UnifiedDflashGemma4_31B(SpecDecodeGraphSignature, Module):
    """Fused module: merge → target → reject → materialize → draft block."""

    def __init__(
        self,
        config: UnifiedDflashGemma4_31BConfig,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        # AcceptanceSampler dispatches to the synthetic path first, which
        # ignores token bitmasks — grammar constraints would silently stop
        # being enforced while the serve layer believes they are.
        if (
            enable_structured_output
            and config.speculative_config.synthetic_acceptance_rate is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rate is incompatible with structured"
                " output: the synthetic acceptance path ignores token"
                " bitmasks. This arch enables the bitmask path by default"
                " for tool-call grammars; for synthetic-acceptance"
                " benchmarking pass --tool-parser none and leave"
                " --enable-structured-output off."
            )
        self.enable_structured_output = enable_structured_output
        self.block_size = config.effective_block_size
        # The anchor slot carries the committed/bonus token and never
        # predicts, so the block drafts block_size - 1 tokens per step.
        self.num_speculative_tokens = self.block_size - 1
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if config.speculative_config.use_relaxed_acceptance_for_thinking:
            relaxed_topk = config.speculative_config.relaxed_topk
            relaxed_delta = config.speculative_config.relaxed_delta
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                config.speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )

        self.target = Gemma4TextModel(config.target)
        self.draft = DFlashLlama3(
            config.draft,
            num_context_features=len(self.target_layer_ids),
            layer_types=config.layer_types or None,
        )
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def _unified_kv_params(self) -> MultiKVCacheParams:
        return MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft_kv_params,
            }
        )

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedDflashGemma4_31BValues:
        graph_inputs = self.decode_inputs(inputs)
        return UnifiedDflashGemma4_31BValues(
            tokens=graph_inputs.tokens,
            input_row_offsets=graph_inputs.input_row_offsets,
            draft_tokens=graph_inputs.draft_tokens,
            return_n_logits=graph_inputs.return_n_logits,
            signal_buffers=graph_inputs.signal_buffers,
            sliding_kv_collection=graph_inputs.kv(
                "target", "sliding_attention"
            )[0],
            global_kv_collection=graph_inputs.kv("target", "full_attention")[0],
            draft_kv_collection=graph_inputs.kv("draft")[0],
            seed=graph_inputs.seed,
            temperature=graph_inputs.temperature,
            top_k=graph_inputs.top_k,
            max_k=graph_inputs.max_k,
            top_p=graph_inputs.top_p,
            min_top_p=graph_inputs.min_top_p,
            in_thinking_phase=graph_inputs.thinking_phase,
            pinned_bitmask=graph_inputs.pinned_bitmask,
            wait_payload=graph_inputs.wait_payload,
            device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
        )

    @override
    @property
    def input_spec(self) -> SpecDecodeInputTypeSpec:
        """Single-device DFlash graph. Signal buffers are declared even
        though the graph is single-device: Gemma4's embedding/lm_head layers
        use collectives unconditionally.
        """
        return SpecDecodeInputTypeSpec(
            devices=self.config.target.devices,
            distributed=False,
            include_signal_buffers=True,
            include_in_thinking_phase=True,
            enable_structured_output=self.enable_structured_output,
        )

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface:
        return self._unified_kv_params()

    def _empty_vision_inputs(self) -> tuple[TensorValue, TensorValue]:
        """Zero-row image embeddings + scatter indices for the text-only
        target forward (the vision merge scatter is a no-op)."""
        device = self.config.target.devices[0]
        hidden = self.config.target.text_config.hidden_size
        empty_embeds = ops.constant(
            np.zeros((0, hidden), dtype=np.float32),
            DType.float32,
            device=device,
        ).cast(self.config.target.unquantized_dtype)
        empty_indices = ops.range(
            0, 0, 1, out_dim=0, dtype=DType.int32, device=device
        )
        return empty_embeds, empty_indices

    def __call__(
        self,
        inputs: UnifiedDflashGemma4_31BValues,
    ) -> tuple[TensorValue, ...]:
        device = inputs.tokens.device
        K = self.block_size
        signal_buffers = inputs.signal_buffers
        # Pre-step committed length; the global (full-attention) leaf tracks
        # the logical sequence length.
        pre_cache_lengths = ops.rebind(
            inputs.global_kv_collection.cache_lengths, ["batch_size"]
        )

        merged_tokens, merged_offsets = self.merger(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.draft_tokens,
        )
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        empty_embeds, empty_indices = self._empty_vision_inputs()
        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            [inputs.sliding_kv_collection],
            [inputs.global_kv_collection],
            inputs.return_n_logits,
            [merged_offsets],
            [empty_embeds],
            [empty_indices],
        )
        target_logits = target_outputs[1]
        target_hs_concat = fuse_captured_hidden_states(
            captures_by_device(target_outputs[3:], 1)
        )[0]

        seed_scalar = inputs.seed[0]
        # Grammar constraining is target/verify-side only: every K+1 verify
        # row is masked before acceptance, so a grammar-violating draft is
        # deterministically rejected and its replacement sampled from the
        # masked residual. The draft chain itself stays unconstrained.
        effective_bitmasks = apply_overlap_bitmask(
            inputs.pinned_bitmask,
            inputs.wait_payload,
            inputs.device_bitmask_scratch,
            num_steps=inputs.draft_tokens.shape[1],
            device=device,
        )
        num_accepted, recovered, bonus = self.acceptance_sampler(
            inputs.draft_tokens,
            target_logits,
            seed=seed_scalar,
            temperature=inputs.temperature,
            top_k=inputs.top_k,
            max_k=inputs.max_k,
            top_p=inputs.top_p,
            min_top_p=inputs.min_top_p,
            in_thinking_phase=inputs.in_thinking_phase,
            token_bitmasks=effective_bitmasks,
        )

        num_steps_u32 = _shape_to_scalar(
            inputs.draft_tokens.shape[1], device, dtype=DType.uint32
        )
        zero = ops.constant(0, DType.uint32, device=device)
        is_prefill = (num_steps_u32 == zero).broadcast_to(["batch_size"])
        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (inputs.draft_tokens == magic_token)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        num_steps = _shape_to_scalar(
            inputs.draft_tokens.shape[1], device, dtype=DType.int32
        )
        is_dummy_draft = num_magic_tokens == num_steps.broadcast_to(
            ["batch_size"]
        )
        num_accepted = ops.where(
            is_prefill | is_dummy_draft,
            ops.constant(0, num_accepted.dtype, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )
        prompt_lens = (
            inputs.input_row_offsets[1:] - inputs.input_row_offsets[:-1]
        ).rebind(["batch_size"])
        decode_commit = (num_accepted + 1).cast(DType.uint32)
        # A dummy-draft row inside a K>0 batch is either a decode row with no
        # real drafts (prompt_lens == 1 == decode_commit, both branches equal)
        # or a prefill row riding a mixed batch, which commits its whole chunk
        # into the draft-KV bump. The nested ``ops.where`` calls keep
        # ``is_dummy_draft`` -- an empty-axis reduction when num_steps == 0 --
        # off the prefill path.
        mixed_commit = ops.where(is_dummy_draft, prompt_lens, decode_commit)
        commit_lengths = ops.where(is_prefill, prompt_lens, mixed_commit)

        target_tokens = ops.concat([recovered, bonus], axis=1)
        gather_idx = ops.where(
            is_prefill,
            ops.constant(0, DType.int64, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted.cast(DType.int64),
        )
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(gather_idx, axis=-1),
            batch_dims=1,
        )

        ctx_hidden = self.draft.project_target_hidden(target_hs_concat)

        # The draft leaf already carries draft blocks + lookup_table /
        # max_prompt_length / max_cache_length / dispatch metadata (mirroring
        # the target leaves); only cache_lengths is overridden with the
        # pre-step value.
        draft_kv_collection = replace(
            inputs.draft_kv_collection,
            cache_lengths=pre_cache_lengths,
        )

        self.draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=merged_offsets,
            kv_collection=draft_kv_collection,
        )

        # DFlash runs the draft as a full block (the accepted token plus the
        # mask-token tail): K query rows per sequence on every batch type.
        # Neither manager-provided metadata fits that geometry: the leaf's
        # attention_dispatch_metadata is the TARGET/verify key, whose
        # q_max_seq_len is the batch's max prompt length — equal to the block
        # width only on decode batches, and far larger on a prefill batch,
        # where the oversized query bound drives the block's layer-0 flash
        # attention to produce NaN. The manager's
        # draft_attention_dispatch_metadata is resolved one row narrow
        # (num_draft_tokens_per_step = K - 1) with the target's partition
        # count. Rebuild the dispatch buffer at the block's true width.
        bumped_cache_lengths = pre_cache_lengths + commit_lengths
        block_kv_collection = replace(
            draft_kv_collection,
            cache_lengths=bumped_cache_lengths,
            attention_dispatch_metadata=_block_dispatch_metadata(
                draft_kv_collection.attention_dispatch_metadata, K
            ),
            max_prompt_length=ops.constant(
                K, DType.uint32, device=DeviceRef.CPU()
            ).broadcast_to([1]),
        )

        next_tokens_2d = ops.unsqueeze(next_tokens, axis=1)
        mask_const = ops.constant(
            self.mask_token_id, DType.int64, device=device
        )
        mask_tail = mask_const.broadcast_to(["batch_size", K - 1])
        block_ids = ops.concat([next_tokens_2d, mask_tail], axis=1)
        block_ids_flat = block_ids.reshape((-1,))

        # Slot 0 = the anchor/bonus token, slots 1..K-1 = the mask token. The
        # target's ScaledWordEmbedding applies gemma's sqrt(hidden) scale,
        # which is what the drafter was trained against (the reference
        # multiplies the shared embedding rows by the same factor).
        block_embeds = self.target.embed_tokens(block_ids_flat, signal_buffers)[
            0
        ]

        block_indices = ops.range(
            start=0,
            stop=inputs.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device,
            dtype=DType.uint32,
        )
        draft_block_offsets = block_indices * ops.constant(
            K, DType.uint32, device=device
        )

        block_hs = self.draft.forward_block(
            input_embeds=block_embeds,
            kv_collection=block_kv_collection,
            input_row_offsets=draft_block_offsets,
        )

        # Anchor-slot drop: slot 0's output is untrained; only the K-1 mask
        # slots produce drafts. Neither the target's final-logit softcapping
        # nor its logits_scaling is applied here — both are monotone in the
        # logit, so the greedy argmax below is unchanged by them.
        block_hs_2d = block_hs.reshape(
            ("batch_size", K, self.config.draft.hidden_size)
        )
        draft_logits = self.target.lm_head(
            [block_hs_2d[:, 1:, :]], signal_buffers
        )[0]
        next_draft_tokens = ops.argmax(draft_logits, axis=-1).reshape(
            ("batch_size", K - 1)
        )

        # Force num_accepted=0 in the prefill output even if the sampler
        # happened to "accept" garbage drafts, so downstream metrics don't
        # report bogus acceptances.
        num_accepted_out = ops.where(
            is_prefill,
            ops.constant(0, num_accepted.dtype, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        return (num_accepted_out, next_tokens, next_draft_tokens)
