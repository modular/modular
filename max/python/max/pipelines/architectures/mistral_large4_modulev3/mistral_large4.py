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
"""Mistral Large 4, ModuleV3, forking the DeepSeek-V3.2 DSA spine.

The stack is DeepSeek-V3.2's: sparse MLA with a lightning indexer, residual
adds around a gated FFN. What differs is which layer does what, and that is
read from the config's explicit per-layer lists rather than re-derived:

* a **rope** or **NoPE** latent row (576 or 512 bytes),
* a **full** indexer or a carry-forward of the previous full layer's top-k,
* an MLA latent **owner** or a **reader** of another layer's rows.

Every block is emitted as a subgraph whose name encodes all three, so call
sites that differ in any of them cannot share a body.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.kernels import row_invariant_matmul
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.pipelines.speculative.driver import gather_variable_rows
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Replicated, Unknown

from ..deepseekV3_2_modulev3.deepseekV3_2 import project_to_logits
from ..deepseekV3_modulev3.deepseekV3 import gather_last_tokens
from ..deepseekV4.layers.ragged import segment_ids
from .config import (
    INDEXER_GROUP,
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    MistralLarge4Config,
    indexer_slot_map,
    kv_slot_map,
    mla_group,
)
from .layers.moe import MistralLarge4MoE
from .layers.rope import LeRope, MistralLarge4RotaryEmbedding
from .layers.ssmax import log_positions
from .layers.transformer_block import (
    MistralLarge4TransformerBlock,
    RowInvariantDecodeBlocker,
    row_invariant_decode_blocker,
)
from .model_config import pins_verify_split

_ML4_MOE_TYPES = (QuantizedMoE, MistralLarge4MoE)
"""Every FFN here that a shared subgraph body can serve: the TP and EP MoEs
are :class:`QuantizedMoE` subclasses, and the replicated one is this
architecture's own, so both families have to be named."""

logger = logging.getLogger("max.pipelines")


@functools.cache
def _log_row_invariance_off(blocker: RowInvariantDecodeBlocker) -> None:
    """Logs, once per blocker, that decode steps keep the default kernels."""
    logger.log(
        logging.WARNING if blocker.warn else logging.INFO,
        "Mistral Large 4: %s, so decode steps keep the default kernels and a"
        " DSpark verify's rows need not match them bit for bit",
        blocker.reason,
    )


def _pin_verify_split(
    key: KVCacheParamInterface, rope_layers: Sequence[bool]
) -> None:
    """Pins the latent leaves whose sparse dispatch this model keys.

    The capture key is resolved from the cache tree the graph's inputs are
    flattened from, not from the config's copy of it, so the decision has to
    land on that tree's leaves: a key that did not know would size the verify's
    split from the live cache and capture the launch once per cache length.

    Args:
        key: The cache tree the graph's inputs are flattened from.
        rope_layers: One rope flag per layer, which names the latent groups.
    """
    assert isinstance(key, MultiKVCacheParams)
    for rope in sorted(set(rope_layers)):
        leaf = key.children[mla_group(rope)]
        assert isinstance(leaf, MLAKVCacheParams)
        leaf.sparse_row_invariant_verify = True


def _row_positions(
    tokens: TensorValue,
    input_row_offsets: TensorValue,
    cache_lengths: TensorValue,
) -> TensorValue:
    """Returns each token's absolute 0-based position, ``[tokens]``.

    A token's position is its request's cache length plus its index within the
    chunk, which is what SSMax's ``log(pos + 1)`` reads.
    """
    off = ops.cast(input_row_offsets, DType.int32)
    bid, index = segment_ids(off, tokens.shape[0], input_row_offsets.device)
    local = index - ops.gather(off, bid, axis=0)
    starts = ops.cast(cache_lengths, DType.int32)
    return ops.cast(ops.gather(starts, bid, axis=0) + local, DType.uint32)


_row_positions_op = F.functional(_row_positions)


def concat_taps(taps: Mapping[int, Tensor], order: Sequence[int]) -> Tensor:
    """Concatenates the captured taps in the drafter's declared column order.

    The order is the checkpoint's tap list, not the order the layer loop
    happened to produce them in: the drafter's projection columns are
    order-dependent, and its checkpoint treats the list as authoritative.

    Args:
        taps: Each captured layer's post-layer residual stream.
        order: The layer ids, in declaration order.

    Returns:
        ``[tokens, len(order) * hidden]``, one tap after another.
    """
    return F.concat([taps[layer] for layer in order], axis=-1)


def row_positions(
    tokens: Tensor, input_row_offsets: Tensor, cache_lengths: Tensor
) -> Tensor:
    """Returns each token's absolute 0-based position.

    Args:
        tokens: The step's tokens, read for their count.
        input_row_offsets: ``[batch + 1]`` ragged row offsets.
        cache_lengths: ``[batch]`` tokens already cached per request.
    """
    return _row_positions_op(tokens, input_row_offsets, cache_lengths)


class MistralLarge4TextModel(
    Module[
        [
            Tensor,
            PagedCacheValues,
            PagedCacheValues | None,
            PagedCacheValues,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            EPCommBuffers | None,
            Tensor | None,
            Tensor | None,
        ],
        tuple[Tensor, ...],
    ]
):
    """The Mistral Large 4 language model."""

    def __init__(
        self,
        config: MistralLarge4Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        """Builds the embedding, the decoder stack, the norms and the head.

        Args:
            config: The resolved config, whose per-layer lists drive the stack.
            kv_params: The cache tree the graph's inputs are flattened from,
                which decides whether decode takes the verify's kernels.
            ep_batch_manager: The expert-parallel runtime, when EP is on.

        Raises:
            ValueError: If a layer has no slot in its MLA cache leaf.
        """
        self.ep_batch_manager = ep_batch_manager
        self.config = config
        yarn = config.yarn
        assert yarn is not None
        # The table's magnitude is left unscaled (attention_factor) and its
        # frequencies are LeRoPE-scaled; either assumption breaking would move
        # every logit, so they are checked rather than assumed.
        assert yarn.attention_factor == 1.0, (
            "Mistral Large 4's rope takes no YaRN magnitude scale"
        )
        assert yarn.lerope, "Mistral Large 4's rope carries the LeRoPE term"
        rope_width = config.qk_rope_head_dim
        self.lerope = LeRope(rope_width)
        self.rope = MistralLarge4RotaryEmbedding(
            dim=rope_width,
            max_seq_len=config.max_seq_len,
            theta=yarn.rope_theta,
            factor=yarn.factor,
            beta_fast=yarn.beta_fast,
            beta_slow=yarn.beta_slow,
            original_max_position_embeddings=(
                yarn.original_max_position_embeddings
            ),
        )
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size
        )
        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size, out_dim=config.vocab_size, bias=False
        )

        fp8_config = config.fp8_config
        nvfp4_config = config.nvfp4_config
        self.kv_slots = kv_slot_map(config.rope_layers, config.kv_source_layers)
        self.indexer_slots = indexer_slot_map(config.indexer_types)
        self.kv_params = config.kv_params

        blocker = row_invariant_decode_blocker(config, kv_params)
        if blocker is not None:
            _log_row_invariance_off(blocker)
        self.row_invariant_decode = blocker is None
        if pins_verify_split(self.row_invariant_decode, config.graph_mode):
            _pin_verify_split(kv_params, config.rope_layers)
        layers: list[MistralLarge4TransformerBlock] = []
        for layer_idx in range(config.num_hidden_layers):
            rope = config.rope_layers[layer_idx]
            if layer_idx not in self.kv_slots[mla_group(rope)]:
                raise ValueError(
                    f"layer {layer_idx} has no slot in its MLA cache leaf"
                )
            layers.append(
                MistralLarge4TransformerBlock(
                    config=config,
                    layer_idx=layer_idx,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    # A NoPE layer folds its rope dims into the content dims:
                    # 128 content dims against a 512-wide latent.
                    qk_nope_head_dim=(
                        config.qk_nope_head_dim
                        if rope
                        else config.qk_nope_head_dim + config.qk_rope_head_dim
                    ),
                    qk_rope_head_dim=config.qk_rope_head_dim if rope else 0,
                    kv_lora_rank=config.kv_lora_rank,
                    v_head_dim=config.v_head_dim,
                    attention_scale=config.attention_scale,
                    rms_norm_eps=config.rms_norm_eps,
                    kv_params=config.mla_leaf(rope),
                    index_n_heads=config.index_n_heads,
                    index_head_dim=config.index_head_dim,
                    index_topk=config.index_topk,
                    owns_indexer=config.indexer_types[layer_idx] == "full",
                    owns_kv=config.kv_source_layers[layer_idx] == layer_idx,
                    is_moe=config.moe_layers[layer_idx],
                    moe_intermediate_size=config.moe_intermediate_size,
                    dense_intermediate_size=config.intermediate_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_token=config.num_experts_per_tok,
                    attention_quant_config=fp8_config,
                    expert_quant_config=nvfp4_config,
                    graph_mode=config.graph_mode,
                    buffer_size=config.max_batch_context_length,
                    rope_interleaved=config.rope_interleave,
                    index_norm_eps=config.index_norm_eps,
                    ep_batch_manager=ep_batch_manager,
                    row_invariant_decode=self.row_invariant_decode,
                )
            )

        self.layers = ModuleList[MistralLarge4TransformerBlock](layers)
        self.mesh = config.mesh
        self.emit_last_token_logits = True
        self.return_logits = ReturnLogits.LAST_TOKEN
        self.return_hidden_states = ReturnHiddenStates.NONE
        self.target_layer_ids = tuple(config.target_layer_ids)

    def _block_name(self, layer: MistralLarge4TransformerBlock) -> str:
        """Names a block's subgraph by its three structural choices."""
        rope = "rope" if layer.self_attn.qk_rope_head_dim else "nope"
        indexer = (
            "full_indexer" if layer.self_attn.indexer else "shared_indexer"
        )
        kv = "owner" if layer.self_attn.owns_kv else "reader"
        return f"decoder_{rope}_{indexer}_{kv}"

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        nope_kv_collection: PagedCacheValues | None,
        indexer_kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: EPCommBuffers | None = None,
        image_embeddings: Tensor | None = None,
        image_token_indices: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Runs the embedding and the decoder stack over the tokens."""
        del data_parallel_splits, input_row_offsets_i64
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        h, taps = self._stack(
            tokens,
            kv_collection,
            nope_kv_collection,
            indexer_kv_collection,
            input_row_offsets,
            batch_context_length,
            comm_buffers,
            image_embeddings,
            image_token_indices,
        )
        # Read the output modes at build time, not construction: a caller may
        # ask this graph for a mode it does not build.
        self._check_output_modes()
        outputs: list[Tensor] = []
        if self.emit_last_token_logits:
            last_token_h = gather_last_tokens(h, input_row_offsets)
            outputs.append(self._project(last_token_h))
        if self.return_logits is ReturnLogits.VARIABLE:
            outputs.extend(
                self._variable_logits(h, input_row_offsets, return_n_logits)
            )
        if self.return_hidden_states is ReturnHiddenStates.SELECTED_LAYERS:
            outputs.append(self._taps(taps))
        return tuple(outputs)

    def verify(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        nope_kv_collection: PagedCacheValues | None,
        indexer_kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        comm_buffers: EPCommBuffers | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the target over the merged verify rows and returns its taps.

        A speculative verify scores every row the block acceptance compares,
        so it takes no last-token head and no batch context length: the fused
        graph's signature has a slot for neither, and the prefill plan's
        staging budget is then a host constant
        (:meth:`~.layers.sparse_mla.MistralLarge4SparseMLA.create_mla_prefill_metadata`).

        Args:
            tokens: ``[merged_seq_len]`` the step's merged tokens.
            kv_collection: The rope layers' latent cache.
            nope_kv_collection: The NoPE layers' latent cache.
            indexer_kv_collection: The indexer's key cache.
            return_n_logits: How many trailing rows of each request score.
            input_row_offsets: ``[batch + 1]`` merged ragged offsets.
            comm_buffers: The EP collective buffers, when EP is on.

        Returns:
            The ``[rows, vocab]`` float32 variable logits, their row offsets,
            and the captured taps concatenated in the drafter's column order.

        Raises:
            ValueError: If the graph was not built to capture the drafter's
                taps, or declares no tap layer at all.
        """
        if self.return_hidden_states is not ReturnHiddenStates.SELECTED_LAYERS:
            raise ValueError(
                "a Mistral Large 4 verify returns the drafter's taps, so it"
                " needs return_hidden_states=SELECTED_LAYERS; got"
                f" {self.return_hidden_states.value}"
            )
        if not self.target_layer_ids:
            raise ValueError(
                "a Mistral Large 4 verify returns the drafter's taps, and this"
                " graph declares none"
            )
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        h, taps = self._stack(
            tokens,
            kv_collection,
            nope_kv_collection,
            indexer_kv_collection,
            input_row_offsets,
            None,
            comm_buffers,
        )
        logits, offsets = self._variable_logits(
            h, input_row_offsets, return_n_logits
        )
        return logits, offsets, self._taps(taps)

    def _place(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Stages the batch onto every rank of the mesh.

        Every rank holds every row -- attention is tensor-parallel over the
        heads -- so the tokens and their offsets are broadcast, which is a
        collectives' kernel rather than a device copy and so is legal inside a
        device graph capture.

        Args:
            tokens: The step's tokens, ``[seq_len]``.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.

        Returns:
            Both, on the mesh.
        """
        if self.mesh is None:
            return tokens, input_row_offsets
        if self.mesh.num_devices > 1:
            return (
                F.distributed_broadcast(tokens, self.mesh),
                F.distributed_broadcast(input_row_offsets, self.mesh),
            )
        return tokens.to(self.mesh), input_row_offsets.to(self.mesh)

    def _stack(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        nope_kv_collection: PagedCacheValues | None,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        batch_context_length: Tensor | None,
        comm_buffers: EPCommBuffers | None,
        image_embeddings: Tensor | None = None,
        image_token_indices: Tensor | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Runs the embedding and the decoder stack.

        Args:
            tokens: The step's tokens, ``[seq_len]``, already staged on every
                rank of the mesh.
            kv_collection: The rope layers' latent cache.
            nope_kv_collection: The NoPE layers' latent cache.
            indexer_kv_collection: The indexer's key cache.
            input_row_offsets: ``[batch + 1]`` ragged row offsets, staged like
                ``tokens``.
            batch_context_length: The host page-aligned context length the
                dense prefill plan stages, or ``None`` for a graph that has no
                slot for it.
            comm_buffers: The EP collective buffers, when EP is on.
            image_embeddings: The vision rows this step scatters into the
                prompt, ``[rows, hidden]``, or ``None`` for a graph whose
                ABI carries no vision inputs.
            image_token_indices: The prompt rows the embeddings land in,
                ``[rows]``; out-of-bounds entries are skipped, so a batch
                with no image this step (empty embeddings and indices) is a
                no-op.

        Returns:
            The final hidden stream, and each captured tap by layer id.
        """
        h = self.tok_embeddings(tokens)
        if image_embeddings is not None:
            assert image_token_indices is not None
            if self.mesh is not None and self.mesh.num_devices > 1:
                # Attention runs TP over the heads with every rank holding
                # every row, so each rank scatters the same replicated
                # embeddings at the same replicated indices.
                image_embeddings = F.distributed_broadcast(
                    image_embeddings, self.mesh
                )
                image_token_indices = F.distributed_broadcast(
                    image_token_indices, self.mesh
                )
            # No embedding norm sits between the table and the first
            # block, so the splice lands directly on the embedded prompt,
            # exactly where the reference's masked_scatter does: the image,
            # break and end tokens are ordinary prompt rows everywhere but
            # the image tokens' own embeddings.
            h = F_merge_multimodal_embeddings(
                h,
                image_embeddings.cast(h.dtype),
                image_token_indices,
            )
        freqs_cis = F.cast(self.rope.freqs_cis_from(self.lerope.alpha), h.dtype)
        positions = row_positions(
            tokens, input_row_offsets, kv_collection.cache_lengths
        )
        if self.mesh is not None and self.mesh.num_devices > 1:
            # Attention runs TP over the heads with every rank holding every
            # row, so each token's position is the same on every rank. Claim
            # that placement: the position vector's own ops leave it Unknown,
            # and SSMax multiplies it against a head-sharded scale.
            positions = positions.rebind_mapping(
                DeviceMapping.replicated(self.mesh)
            )

        fp8_config = self.config.fp8_config
        # The prefill plan depends only on the batch's layout, so build it once
        # instead of once per layer, and keep its buffer length host-resident.
        mla_prefill_metadata = None
        if self.layers[0].self_attn.graph_mode != "decode":
            mla_prefill_metadata = self.layers[
                0
            ].self_attn.create_mla_prefill_metadata(
                input_row_offsets, batch_context_length
            )
        # The taps an in-checkpoint drafter reads: the post-layer residual
        # stream of each named layer, in the checkpoint's declared order.
        # Capturing reads the stream and changes nothing about it, so the
        # logits below are the ones this graph would otherwise produce.
        capture = (
            set(self.target_layer_ids)
            if self.return_hidden_states is ReturnHiddenStates.SELECTED_LAYERS
            else set()
        )
        taps: dict[int, Tensor] = {}
        topk_indices: Tensor | None = None
        carried: dict[int, tuple[Tensor, Tensor]] = {}
        # Each layer's SSMax takes the same per-row log, so compute it once.
        ssmax_log_positions = log_positions(positions, h.dtype)
        for idx, layer in enumerate(self.layers):
            rope = self.config.rope_layers[idx]
            group = mla_group(rope)
            mla_collection = kv_collection if rope else nope_kv_collection
            assert mla_collection is not None
            owns_indexer = self.config.indexer_types[idx] == "full"
            layer_idx_tensor = F.constant(
                self.kv_slots[group][idx], DType.uint32, device=CPU()
            )
            indexer_layer_idx_tensor = F.constant(
                self.indexer_slots.get(idx, 0), DType.uint32, device=CPU()
            )
            shared_kv = (
                None
                if layer.self_attn.owns_kv
                else carried[self.config.kv_source_layers[idx]]
            )
            call: Callable[
                ..., tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]
            ] = layer
            if isinstance(layer.mlp, _ML4_MOE_TYPES):
                # A subgraph body is shared only between call sites that agree
                # on all three structural choices, so the name encodes them.
                # A reader also takes its source's projection, whose width
                # depends on whether the source projects an indexer key too.
                name = self._block_name(layer)
                if shared_kv is not None:
                    name += f"_{int(shared_kv[0].shape[-1])}"
                call = as_subgraph(layer, name=name)
            h, topk_indices, *rest = call(
                h,
                mla_collection,
                indexer_kv_collection if owns_indexer else None,
                freqs_cis,
                positions,
                input_row_offsets,
                layer_idx_tensor,
                indexer_layer_idx_tensor,
                fp8_config,
                topk_indices,
                not owns_indexer,
                shared_kv,
                comm_buffers,
                mla_prefill_metadata=mla_prefill_metadata,
                ssmax_log_positions=ssmax_log_positions,
            )
            if layer.self_attn.owns_kv:
                carried_kv = rest[0]
                assert carried_kv is not None
                # Keyed by the owner's own layer id, so a reader always takes
                # the latent of the layer ``kv_source_layers`` names, whatever
                # else sits between them in the schedule.
                carried[idx] = carried_kv
            if idx in capture:
                taps[idx] = h

        # A tap list that names no layer of this graph would otherwise return
        # a shorter output tuple than the caller reads, shifting every later
        # output by one.
        missing = capture - set(taps)
        if missing:
            raise ValueError(
                f"the taps {sorted(missing)} name no layer of this graph's"
                f" {len(self.layers)} layers"
            )
        return h, taps

    def _taps(self, taps: dict[int, Tensor]) -> Tensor:
        """Returns the captured layers' residuals, concatenated in tap order.

        Args:
            taps: Each captured layer's post-layer residual stream.

        Returns:
            ``[tokens, n_taps * hidden]``, one tap after another, in the
            checkpoint's declared order (see :func:`concat_taps`).
        """
        return concat_taps(taps, self.target_layer_ids)

    def _check_output_modes(self) -> None:
        """Rejects output modes this graph does not build.

        The graph emits every position's logits or the last token's, and no
        hidden states. Asking for anything else is a caller error, not a mode
        the graph may silently ignore.

        Raises:
            NotImplementedError: If the requested logits or hidden-state mode
                is not built, or a per-replica mode is asked of data
                parallelism.
            ValueError: If the last-token logits are skipped without asking
                for variable logits.
        """
        supported_logits = (ReturnLogits.LAST_TOKEN, ReturnLogits.VARIABLE)
        supported_hidden: tuple[ReturnHiddenStates, ...] = (
            ReturnHiddenStates.NONE,
        )
        if self.target_layer_ids:
            # The only hidden states this graph captures are the drafter's
            # taps, so asking for them without a tap list stays an error.
            supported_hidden = (
                ReturnHiddenStates.NONE,
                ReturnHiddenStates.SELECTED_LAYERS,
            )
        if (
            self.return_logits not in supported_logits
            or self.return_hidden_states not in supported_hidden
        ):
            raise NotImplementedError(
                f"return_logits={self.return_logits} with"
                f" return_hidden_states={self.return_hidden_states} is not"
                " implemented on ModuleV3."
            )
        if (
            self.return_logits is ReturnLogits.VARIABLE
            and self.config.data_parallel_degree > 1
        ):
            # Each replica holds only its own rows; the graph-API model
            # allgathers them, which is not built here yet.
            raise NotImplementedError(
                "VARIABLE logits require data_parallel_degree=1 on ModuleV3."
            )
        if not self.emit_last_token_logits and (
            self.return_logits is not ReturnLogits.VARIABLE
        ):
            raise ValueError(
                "Skipping the last-token logits requires VARIABLE logits."
            )

    def _project(self, h: Tensor) -> Tensor:
        """Returns the soft-capped float32 logits of ``h``."""
        # A row-invariant LM head runs the same kernel at every row count,
        # so a decode step's logits match the DSpark verify's bit for bit.
        logits = project_to_logits(
            h,
            self.norm,
            (
                (lambda rows: row_invariant_matmul(rows, self.lm_head.weight))
                if self.row_invariant_decode
                else self.lm_head
            ),
            self.mesh,
            self.config.vocab_size,
            self.config.unpadded_vocab_size,
        )
        # `final_logit_softcapping` is 30, applied once, on the LM head only.
        cap = self.config.final_logit_softcapping
        if cap:
            logits = F.tanh(logits * (1.0 / cap)) * cap
        return logits

    def _variable_logits(
        self, h: Tensor, input_row_offsets: Tensor, return_n_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns the last ``return_n_logits`` rows' logits and their offsets."""
        assert self.mesh is not None
        variable_h, offsets = gather_variable_rows(
            h, input_row_offsets, return_n_logits, self.mesh
        )
        return self._project(variable_h), offsets

    def module_outputs(self, outputs: tuple[Tensor, ...]) -> ModuleV3Outputs:
        """Names the outputs of :meth:`forward` for the pipeline model."""
        rest = iter(outputs)
        next_token_logits = next(rest)
        logits = logit_offsets = hidden_states = None
        if self.return_logits is ReturnLogits.VARIABLE:
            logits, logit_offsets = next(rest), next(rest)
        if self.return_hidden_states is ReturnHiddenStates.SELECTED_LAYERS:
            hidden_states = next(rest)
        return ModuleV3Outputs(
            next_token_logits=next_token_logits,
            logits=logits,
            logit_offsets=logit_offsets,
            hidden_states=hidden_states,
        )


class MistralLarge4(Module[..., ModuleV3Outputs]):
    """Top-level wrapper that unflattens the variadic KV cache arguments."""

    def __init__(
        self,
        config: MistralLarge4Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        """Wraps the language model.

        Args:
            config: The resolved config.
            kv_params: The cache tree the graph's inputs are flattened from.
            ep_batch_manager: The expert-parallel runtime, when EP is on.
        """
        super().__init__()
        self.language_model = MistralLarge4TextModel(
            config, kv_params, ep_batch_manager
        )
        self.config = config
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the model over its graph inputs."""
        mesh = self.config.mesh
        assert mesh is not None

        dp_degree = self.config.data_parallel_degree
        batch_context_lengths = variadic_args[:dp_degree]
        variadic_args = variadic_args[dp_degree:]
        image_embeddings: Tensor | None = None
        image_token_indices: Tensor | None = None
        if self.config.encodes_images:
            # The vision-merge inputs end the ABI, after the KV cache's
            # leaves and the expert-parallel buffers, which is where the
            # pipeline's input types declare them.
            *rest, image_embeddings, image_token_indices = variadic_args
            variadic_args = tuple(rest)
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        if dp_degree > 1:
            raise NotImplementedError(
                "Mistral Large 4 ModuleV3 does not implement data-parallel"
                " batching yet."
            )
        batch_context_lengths_tensor = batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in variadic_args)
        # A ``MultiKVCacheParams`` tree unflattens to a dict keyed by child
        # name, so the cache leaves are addressed by name, not by position.
        kv_tree = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_tree, dict)

        # Replicas serve different requests, so their caches vary; tensor-
        # parallel devices all store the same latent cache.
        kv_mapping = DeviceMapping(
            mesh,
            tuple(
                Unknown() if name == "dp" else Replicated()
                for name in mesh.axis_names
            ),
        )

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, kv_mapping)

        nope_kv_collection = (
            collection(NOPE_MLA_GROUP) if NOPE_MLA_GROUP in kv_tree else None
        )
        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            # Any variadic graph values left after the KV cache are the EP
            # communication buffers. Wrap them as one forward argument so they
            # thread through each MoE block's subgraph boundary.
            comm_buffers = self.ep_batch_manager.comm_buffers(list(kv_inputs))
        outputs = self.language_model(
            tokens,
            collection(ROPE_MLA_GROUP),
            nope_kv_collection,
            collection(INDEXER_GROUP),
            return_n_logits,
            input_row_offsets,
            batch_context_lengths_tensor,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
            image_embeddings,
            image_token_indices,
        )
        return self.language_model.module_outputs(outputs)

    def verify(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        kv_by_group: Mapping[
            str, Sequence[KVCacheInputsPerDevice[Tensor, Tensor]]
        ],
        comm_buffers: EPCommBuffers | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the speculative verify over the merged rows.

        The fused graph's caches arrive named rather than as a flattened tree:
        the rope layers' latent is the driver's primary leaf and the NoPE
        layers' and the indexer's ride in its passthrough. Nothing here reads a
        batch context length, because the canonical speculative signature
        declares one only under data parallelism (see
        :meth:`MistralLarge4TextModel.verify`).

        Args:
            tokens: ``[merged_seq_len]`` the step's merged tokens.
            return_n_logits: How many trailing rows of each request score.
            input_row_offsets: ``[batch + 1]`` merged ragged offsets.
            kv_by_group: Each cache leaf the target reads, keyed by its name in
                the tree, one entry per device.
            comm_buffers: The EP collective buffers, when EP is on.

        Returns:
            The ``[rows, vocab]`` float32 variable logits, their row offsets,
            and the drafter's taps in the checkpoint's column order.

        Raises:
            ValueError: If a leaf the target reads is missing.
        """
        mesh = self.config.mesh
        assert mesh is not None
        missing = [
            name
            for name in (ROPE_MLA_GROUP, INDEXER_GROUP)
            if name not in kv_by_group
        ]
        if missing:
            raise ValueError(
                f"the Mistral Large 4 verify is missing KV leaves: {missing}"
            )
        # Replicas serve different requests, so their caches vary; tensor-
        # parallel devices all store the same latent cache.
        mapping = DeviceMapping(
            mesh,
            tuple(
                Unknown() if name == "dp" else Replicated()
                for name in mesh.axis_names
            ),
        )

        def collection(name: str) -> PagedCacheValues:
            return PagedCacheValues.from_upstream(kv_by_group[name], mapping)

        nope_kv_collection = (
            collection(NOPE_MLA_GROUP)
            if NOPE_MLA_GROUP in kv_by_group
            else None
        )
        return self.language_model.verify(
            tokens,
            collection(ROPE_MLA_GROUP),
            nope_kv_collection,
            collection(INDEXER_GROUP),
            return_n_logits,
            input_row_offsets,
            comm_buffers,
        )
