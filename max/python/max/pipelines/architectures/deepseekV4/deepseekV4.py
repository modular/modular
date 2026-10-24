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

"""Implements the DeepSeek-V4-Flash model.

Module and weight names mirror the checkpoint's own layout (which follows the
reference ``inference/model.py``) so the weight adapter stays a single rename.
See ``weight_adapters.py``.

One forward serves prefill, chunked prefill and decode: it takes a ragged
batch (``T`` tokens, ``input_row_offsets``) plus the paged cache leaves, reads
the cache at each request's ``cache_lengths`` and appends its chunk
(``layers/attention.py``, ``layers/ragged.py``). :meth:`DeepseekV4.serve` is
that entry with the pipeline's logits contract; :meth:`DeepseekV4.__call__`
is the padded ``[batch, seq_len]`` form the gates drive, expressed as the
uniform-offsets case of the same path. Without a cache it is a plain prefill
from position 0.

DSpark contributes nothing to the trunk's logits, so its stages are a module
of their own, :class:`DeepseekV4DSpark`. After a trunk forward, its
:meth:`~DeepseekV4DSpark.fill_cache` writes the stages' windows from the
trunk states :meth:`DeepseekV4.trunk` collects, and
:meth:`~DeepseekV4DSpark.block` / :meth:`~DeepseekV4DSpark.head` draft a
block per request (``layers/dspark.py``).

With ``config.use_subgraphs``, trunk blocks that share a compress ratio and a
routing kind share one subgraph body (:func:`subgraph_layer_groups`), called
once per layer with that layer's weights and its place in the cache leaves
(:class:`~.layers.LayerIndex`). Only the cached forward does; the cache-less
gates' path stays inline.

The block's two mHC sites are :class:`~max.nn.HyperConnection` submodules
(``hc_attn.hc_fn``), which the checkpoint stores flat on the block
(``hc_attn_fn``); the weight adapter renames them. The head's contraction has
no Sinkhorn step and keeps the checkpoint's flat ``hc_head_*`` names.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorValue,
    Type,
    Value,
    Weight,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.embedding import Embedding
from max.nn.hyper_connection import HyperConnection
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    forward_sequential_layers,
    logits_postprocess,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_kv_symbolic_input_tree,
)
from max.tree import Tree

from .layers import (
    DeepseekV4Attention,
    DeepseekV4Cache,
    DeepseekV4Indexer,
    DeepseekV4MoE,
    DSparkAttention,
    DSparkMarkovHead,
    LayerIndex,
    RaggedRows,
    expand_copies,
    hc_head,
    hc_post,
)
from .layers.quantization import linear_for
from .model_config import DeepseekV4Config


def _adopt_shards(
    source: Module, replicas: Sequence[Module], devices: list[DeviceRef]
) -> None:
    """Point each replica's weights at the matching shard of ``source``'s.

    ``replicas`` are structurally identical to ``source`` (built from the same
    config), so they are walked in step. A weight with no strategy yet is
    replicated; a ``Linear``'s goes through its setter, which is what shards a
    block-scaled fp8 ``weight_scale`` along with the weight.
    """
    n = len(devices)
    if isinstance(source, Linear) and source.sharding_strategy is None:
        source.sharding_strategy = ShardingStrategy.replicate(n)
    for name, value in list(vars(source).items()):
        if not isinstance(value, Weight):
            continue
        if value.sharding_strategy is None:
            value.sharding_strategy = ShardingStrategy.replicate(n)
        for replica, shard in zip(replicas, value.shard(devices), strict=True):
            setattr(replica, name, shard)
    for name, child in source.sublayers.items():
        _adopt_shards(
            child, [replica.sublayers[name] for replica in replicas], devices
        )


def head_logits(head: Linear, x: TensorValue) -> TensorValue:
    """float32 logits from a GEMM in the head's stored dtype.

    The reference upcasts the bf16 head and runs the GEMM in float32; on the
    real checkpoint that is a 2 GiB weight read per step on each device. A
    bf16 GEMM halves it; the logits are rounded to bf16 before the float32
    cast, which is also what vLLM's bf16 ``ParallelLMHead`` returns. A float32
    checkpoint keeps the float32 GEMM (both casts are no-ops).
    """
    weight = head.weight
    logits = ops.matmul(ops.cast(x, weight.dtype), ops.transpose(weight, 0, 1))
    return ops.cast(logits, DType.float32)


def _hyper_connection(
    config: DeepseekV4Config, device: DeviceRef
) -> HyperConnection:
    """One per-block mHC site, ``attn`` or ``ffn``.

    Its weights are float32 in the checkpoint and stay float32: the mixer runs
    in fp32 in the reference, and the Sinkhorn division chain is not safe in
    bf16. The head's triple is narrower and is declared on the model.
    """
    return HyperConnection(
        config.hidden_size,
        config.hc_mult,
        device,
        DType.float32,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
    )


def _hc_pre(
    x: TensorValue, hc: HyperConnection
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Contract ``[b, s, hc, d]`` to ``[b, s, d]``, and hand back post and comb.

    ``post`` and ``comb`` are consumed by ``hc_post`` after the sublayer runs:
    they are read off the state *before* the sublayer, not after it.
    """
    b, s = x.shape[0], x.shape[1]
    post, comb, y, _ = hc(ops.reshape(x, [-1, hc.hc_mult, hc.hidden_size]))
    return (
        ops.reshape(y, [b, s, hc.hidden_size]),
        ops.reshape(post, [b, s, hc.hc_mult]),
        ops.reshape(comb, [b, s, hc.hc_mult, hc.hc_mult]),
    )


def subgraph_layer_groups(config: DeepseekV4Config) -> list[list[int]]:
    """The trunk layers that can share one subgraph body, in layer order.

    A body is fixed by its operand types and its weights. Both follow from
    the layer's compress ratio -- which leaves it touches, whether it has a
    compressor and an indexer, its rotary schedule -- and from whether its
    gate is hash-routed (``tid2eid`` in place of ``bias``).
    """
    groups: dict[tuple[int, bool], list[int]] = {}
    for layer_idx in range(config.num_hidden_layers):
        key = (
            config.layer_compress_ratio(layer_idx),
            config.layer_is_hash_routed(layer_idx),
        )
        groups.setdefault(key, []).append(layer_idx)
    return list(groups.values())


class DeepseekV4Block(Module):
    """One decoder block: mHC-wrapped attention then mHC-wrapped MoE."""

    attention_cls: type[DeepseekV4Attention] = DeepseekV4Attention

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        device: DeviceRef,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.attn = self.attention_cls(config, layer_idx, device, max_seq_len)
        self.ffn = DeepseekV4MoE(config, layer_idx, device)
        # Every norm weight is float32, as the reference declares it (the
        # checkpoint's bf16 is upcast on load); the norm still computes in
        # the activation dtype.
        self.attn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_attn = _hyper_connection(config, device)
        self.hc_ffn = _hyper_connection(config, device)

    @staticmethod
    def tensor_parallel(
        blocks: Sequence[DeepseekV4Block],
        xs: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        token_ids: Sequence[TensorValue],
        caches: Sequence[DeepseekV4Cache],
        signal_buffers: Sequence[BufferValue],
        indices: Sequence[LayerIndex | None] | None = None,
    ) -> list[TensorValue]:
        """:meth:`__call__` on every device; ``blocks`` holds a replica each.

        Each device's attention covers its share of the heads and returns a
        partial sum of the output projection; the all-reduce completes it
        before the mHC merge, which every device runs whole. The indexer's
        heads are split the same way, so its scores are all-reduced before the
        top-k. The MoE splits its routed experts across the devices and
        all-reduces their sum (:meth:`DeepseekV4MoE.tensor_parallel`).
        ``indices`` is the layer's place in each device's leaves, ``None``
        for constants.
        """
        if indices is None:
            indices = [None] * len(blocks)
        contracted = [
            _hc_pre(x, block.hc_attn)
            for block, x in zip(blocks, xs, strict=True)
        ]
        states = [
            block.attn.begin(block.attn_norm(h), row, cache, index)
            for block, (h, _, _), row, cache, index in zip(
                blocks, contracted, rows, caches, indices, strict=True
            )
        ]
        if blocks[0].attn.indexer is not None:
            indexers, scores, valid = [], [], []
            for block, state in zip(blocks, states, strict=True):
                assert block.attn.indexer is not None
                assert state.index_score is not None
                assert state.valid is not None
                indexers.append(block.attn.indexer)
                scores.append(state.index_score)
                valid.append(state.valid)
            selected = DeepseekV4Indexer.select_tensor_parallel(
                indexers, scores, valid, signal_buffers
            )
            for state, candidates in zip(states, selected, strict=True):
                state.candidates = candidates
        partials = [
            block.attn.end(state, row, cache)
            for block, state, row, cache in zip(
                blocks, states, rows, caches, strict=True
            )
        ]
        attended = ops.allreduce.sum(partials, signal_buffers)
        xs = [
            hc_post(h, residual, post, comb)
            for h, residual, (_, post, comb) in zip(
                attended, xs, contracted, strict=True
            )
        ]

        contracted = [
            _hc_pre(x, block.hc_ffn)
            for block, x in zip(blocks, xs, strict=True)
        ]
        moved = DeepseekV4MoE.tensor_parallel(
            [block.ffn for block in blocks],
            [
                block.ffn_norm(h)
                for block, (h, _, _) in zip(blocks, contracted, strict=True)
            ],
            token_ids,
            signal_buffers,
        )
        return [
            hc_post(h, x, post, comb)
            for h, x, (_, post, comb) in zip(moved, xs, contracted, strict=True)
        ]

    def __call__(
        self,
        x: TensorValue,
        rows: RaggedRows,
        token_ids: TensorValue,
        cache: DeepseekV4Cache | None = None,
        index: LayerIndex | None = None,
    ) -> TensorValue:
        """``[1, T, hc, d]`` in, ``[1, T, hc, d]`` out.

        Two identical wrappings. Each reads ``post`` and ``comb`` off the state
        *before* its sublayer runs, so the mixing weights describe the incoming
        stream, not the outgoing one. ``index`` is the block's place in the
        leaves, ``None`` for constants.
        """
        residual = x
        h, post, comb = _hc_pre(x, self.hc_attn)
        h = self.attn(self.attn_norm(h), rows, cache, index)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = _hc_pre(x, self.hc_ffn)
        h = self.ffn(self.ffn_norm(h), token_ids)
        return hc_post(h, residual, post, comb)


class DSparkBlock(DeepseekV4Block):
    """One speculative stage. Reference: ``inference/model.py::DSparkBlock``.

    Structurally a decoder block with a different attention, plus whatever the
    stage's position gives it. Only the ends are special: ``mtp.0`` owns the
    projection that brings the trunk's hidden states in, ``mtp.2`` owns the
    output side, and ``mtp.1`` carries no DSpark-specific parameter at all.

    ``embed`` and ``head`` are *not* here. The reference assigns the trunk's
    own modules onto every stage, and the checkpoint has no ``mtp.*`` copy of
    either; :class:`DeepseekV4DSpark`'s caller passes them in.
    """

    attention_cls = DSparkAttention

    def __init__(
        self,
        config: DeepseekV4Config,
        stage_id: int,
        device: DeviceRef,
        max_seq_len: int,
    ) -> None:
        # compress_ratios carries one entry per MTP stage after the trunk's,
        # and they are zero: DSpark stages have no compressor and the
        # reference asserts as much.
        layer_idx = config.num_hidden_layers + stage_id
        super().__init__(config, layer_idx, device, max_seq_len)
        if config.layer_compress_ratio(layer_idx) != 0:
            raise ValueError(
                f"DSpark stage {stage_id} must have compress_ratio 0, got "
                f"{config.layer_compress_ratio(layer_idx)}"
            )
        self.stage_id = stage_id
        self.block_size = config.dspark_block_size
        self.hc_mult = config.hc_mult

        self.is_first = stage_id == 0
        self.is_last = stage_id == len(config.dspark_target_layer_ids) - 1

        if self.is_first:
            # fp8 in both the real and the minimized checkpoint, with a
            # 128x128 block scale, so it takes the same native path as the
            # attention projections.
            self.main_proj = linear_for(
                config,
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                device,
            )
            # float32 like every V4 norm weight; the adapter upcasts the
            # checkpoint's bf16.
            self.main_norm = RMSNorm(
                config.hidden_size, DType.float32, config.rms_norm_eps
            )
        if self.is_last:
            self.norm = RMSNorm(
                config.hidden_size, DType.float32, config.rms_norm_eps
            )
            self.markov_head = DSparkMarkovHead(config, device)
            self.hc_head_fn = Weight(
                name="hc_head_fn",
                dtype=DType.float32,
                shape=(config.hc_mult, config.hc_mult * config.hidden_size),
                device=device,
            )
            self.hc_head_base = Weight(
                name="hc_head_base",
                dtype=DType.float32,
                shape=(config.hc_mult,),
                device=device,
            )
            self.hc_head_scale = Weight(
                name="hc_head_scale",
                dtype=DType.float32,
                shape=(1,),
                device=device,
            )

    def project_main(self, main_hidden: TensorValue) -> TensorValue:
        """The trunk state the stages attend to, ``[1, T, d]``. ``mtp.0`` only."""
        return self.main_norm(self.main_proj(main_hidden))

    def decode(
        self,
        x: TensorValue,
        rows: RaggedRows,
        cache: DeepseekV4Cache,
        block_ids: TensorValue,
    ) -> TensorValue:
        """One stage over a ragged batch of draft blocks.

        Not ``__call__``: a stage reads its window below each block rather
        than appending the block to it, so it is a different entry point
        rather than an override.

        Args:
            x: ``[1, b * K, hc, d]`` the draft stream.
            rows: The blocks' bookkeeping (:meth:`DSparkAttention.decode`).
            cache: The paged leaves, windows already materialized.
            block_ids: ``[1, b * K]`` the block's token ids.
        """
        residual = x
        h, post, comb = _hc_pre(x, self.hc_attn)
        assert isinstance(self.attn, DSparkAttention)
        h = self.attn.decode(self.attn_norm(h), rows, cache, self.block_size)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = _hc_pre(x, self.hc_ffn)
        # The gate on an MTP stage is score-routed, so block_ids goes unread.
        h = self.ffn(self.ffn_norm(h), block_ids)
        return hc_post(h, residual, post, comb)

    @staticmethod
    def decode_tensor_parallel(
        stages: Sequence[DSparkBlock],
        xs: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[DeepseekV4Cache],
        block_ids: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """:meth:`decode` on every device; ``stages`` holds a replica each.

        Split as :meth:`DeepseekV4Block.tensor_parallel`: the attention's
        output projection is a partial sum over this device's heads, the MoE
        sums this device's experts, and each is all-reduced before its mHC
        merge.
        """
        contracted = [
            _hc_pre(x, stage.hc_attn)
            for stage, x in zip(stages, xs, strict=True)
        ]
        partials = []
        for stage, (h, _, _), row, cache in zip(
            stages, contracted, rows, caches, strict=True
        ):
            assert isinstance(stage.attn, DSparkAttention)
            partials.append(
                stage.attn.decode(
                    stage.attn_norm(h), row, cache, stage.block_size
                )
            )
        attended = ops.allreduce.sum(partials, signal_buffers)
        xs = [
            hc_post(h, residual, post, comb)
            for h, residual, (_, post, comb) in zip(
                attended, xs, contracted, strict=True
            )
        ]

        contracted = [
            _hc_pre(x, stage.hc_ffn)
            for stage, x in zip(stages, xs, strict=True)
        ]
        moved = DeepseekV4MoE.tensor_parallel(
            [stage.ffn for stage in stages],
            [
                stage.ffn_norm(h)
                for stage, (h, _, _) in zip(stages, contracted, strict=True)
            ],
            block_ids,
            signal_buffers,
        )
        return [
            hc_post(h, x, post, comb)
            for h, x, (_, post, comb) in zip(moved, xs, contracted, strict=True)
        ]

    def forward_head(
        self,
        x: TensorValue,
        anchor_ids: TensorValue,
        head: Linear,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Draft tokens and their logits out of the last stage.

        Every block position predicts one token, the anchor's included, so a
        block of ``K`` drafts ``K``. The Markov loop is serial and cannot be
        batched: step ``i``'s logit bias is keyed on the token sampled at step
        ``i - 1`` (the anchor at step 0), which does not exist until that step
        has run. ``K`` is static, so it unrolls.

        Sampling is greedy -- ``sample()`` short-circuits to ``argmax`` at
        temperature 0.

        Args:
            x: ``[1, b * K, hc, d]`` the last stage's output stream.
            anchor_ids: ``[b]`` the token each block starts with.
            head: The trunk's LM head, shared not copied.
            sampler: Draws step ``i``'s ``[b]`` token from its ``[b, vocab]``
                biased logits in place of the argmax.

        Returns:
            ``[b, K]`` draft token ids and ``[b, K, vocab]`` float32 biased
            logits.
        """
        k = self.block_size
        contracted = hc_head(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.hc_mult,
            self.norm_eps,
            self.hc_eps,
        )
        vocab = int(head.weight.shape[0])
        logits = head_logits(head, self.norm(contracted))
        batch = anchor_ids.shape[0]
        logits = ops.reshape(
            ops.rebind(ops.reshape(logits, [-1, vocab]), [batch * k, vocab]),
            [batch, k, vocab],
        )

        projection = self.markov_head.projection()
        current = anchor_ids
        out_ids = []
        biased = []
        for i in range(k):
            step = logits[:, i] + self.markov_head(current, projection)
            biased.append(step)
            if sampler is None:
                current = ops.squeeze(ops.argmax(step, axis=-1), axis=-1)
            else:
                current = sampler(step, i)
            current = ops.cast(current, anchor_ids.dtype)
            out_ids.append(current)
        return ops.stack(out_ids, axis=1), ops.stack(biased, axis=1)


class _TensorParallelBlock(Module):
    """One trunk layer on every device, as one call.

    ``source`` is the block the replicas' weights are shards of. A shard is
    sliced in the graph out of its source weight, so the source's weights are
    the ones a shared subgraph body must take as placeholders.
    """

    def __init__(
        self, source: DeepseekV4Block, blocks: Sequence[DeepseekV4Block]
    ) -> None:
        super().__init__()
        self.source = source
        self.blocks = LayerList(list(blocks))
        self.replicas = tuple(blocks)

    def layer_indices(self) -> list[LayerIndex]:
        """Each replica's place in its device's leaves, as constants."""
        return [block.attn.layer_index() for block in self.replicas]

    def __call__(
        self,
        xs: list[TensorValue],
        rows: list[RaggedRows],
        token_ids: list[TensorValue],
        caches: list[DeepseekV4Cache],
        signal_buffers: list[BufferValue],
        indices: list[LayerIndex],
    ) -> list[TensorValue]:
        return DeepseekV4Block.tensor_parallel(
            self.replicas, xs, rows, token_ids, caches, signal_buffers, indices
        )


class DeepseekV4(Module):
    """The DeepSeek-V4-Flash language model."""

    def __init__(
        self, config: DeepseekV4Config, device: DeviceRef | None = None
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = config.devices[0]

        self.embed = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            device=device,
        )
        self.layers = LayerList(
            [
                DeepseekV4Block(config, layer_idx, device, config.max_seq_len)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        # The final mHC contraction before the LM head. Unlike the per-block
        # sites this one has no Sinkhorn step -- the reference's ``hc_head``
        # applies a plain sigmoid to the mixes -- so it emits ``hc_mult``
        # numbers rather than ``mix_hc``, and its scale is a single scalar.
        self.hc_head_fn = Weight(
            name="hc_head_fn",
            dtype=DType.float32,
            shape=(config.hc_mult, config.hc_mult * config.hidden_size),
            device=device,
        )
        self.hc_head_base = Weight(
            name="hc_head_base",
            dtype=DType.float32,
            shape=(config.hc_mult,),
            device=device,
        )
        self.hc_head_scale = Weight(
            name="hc_head_scale",
            dtype=DType.float32,
            shape=(1,),
            device=device,
        )
        # Stored dtype, not the reference's float32: see ``head_logits``.
        self.head = Linear(
            config.hidden_size, config.vocab_size, config.dtype, device
        )
        # On a tensor-parallel replica, each trunk layer across every replica
        # (:meth:`tensor_parallel_replicas`). A tuple, so they do not register
        # as this model's sublayers.
        self.tp_blocks: tuple[_TensorParallelBlock, ...] = ()
        self.subgraph_groups = (
            subgraph_layer_groups(config) if config.use_subgraphs else None
        )

    @staticmethod
    def collect_main_hidden(hiddens: list[TensorValue]) -> TensorValue:
        """The trunk states DSpark reads, ``[b, s, d * n_targets]``.

        Each one is the block output at a layer in ``dspark_target_layer_ids``,
        **averaged over the hc axis** and only then concatenated. Taking copy 0
        instead would give the right shape and the wrong numbers.
        """
        return ops.concat(
            [ops.squeeze(ops.mean(h, axis=2), axis=2) for h in hiddens],
            axis=-1,
        )

    def contract_head(self, x: TensorValue) -> TensorValue:
        """``[b, s, hc, d]`` -> ``[b, s, d]``, the last thing before ``norm``."""
        return hc_head(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.hc_mult,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )

    def _lm_head(self, x: TensorValue) -> TensorValue:
        """float32 logits off the head (:func:`head_logits`)."""
        return head_logits(self.head, x)

    def trunk(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: DeepseekV4Cache | None,
    ) -> tuple[TensorValue, TensorValue]:
        """Run a ragged batch through the blocks and the head contraction.

        Args:
            tokens: ``[1, T]`` token ids.
            rows: The batch's token bookkeeping (``layers/ragged.py``).
            cache: The paged leaves to read and append to, or ``None``.

        Returns:
            ``[1, T, hidden]`` the contracted stream before ``norm``, and
            ``[1, T, hidden * n_targets]`` the trunk state DSpark would read.
        """
        h = expand_copies(self.embed(tokens), self.config.hc_mult)
        targets = set(self.config.dspark_target_layer_ids)
        collected: list[TensorValue] = []

        def collect(layer_idx: int, out: list[TensorValue]) -> None:
            if layer_idx in targets:
                collected.append(out[0])

        def inputs_for_layer(
            layer_idx: int, hs: list[TensorValue]
        ) -> list[Tree[Any]]:
            block = self.layers[layer_idx]
            assert isinstance(block, DeepseekV4Block)
            index = block.attn.layer_index()
            return [hs[0], rows, tokens, cache, index]

        (h,) = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=[h],
            on_layer_output=collect,
            subgraph_layer_groups=(
                self.subgraph_groups if cache is not None else None
            ),
            name_for_subgraph=lambda g: f"dsv4_block_{g}",
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
        )
        return self.contract_head(h), self.collect_main_hidden(collected)

    def __call__(
        self,
        tokens: TensorValue,
        seq_len: int,
        cache: DeepseekV4Cache | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Run a padded chunk of token ids through the trunk.

        The gates' entry: a ``[batch, seq_len]`` chunk is the ragged batch
        with uniform row offsets, so this reshapes and defers to
        :meth:`trunk`.

        Args:
            tokens: ``[batch, seq_len]`` token ids at positions
                ``cache.cache_lengths ..``; a prefill from position 0 when
                there is no cache.
            seq_len: Static chunk length; ``1`` for decode.
            cache: The paged leaves to read and append to, or ``None``.

        Returns:
            ``[batch, seq_len, vocab_size]`` float32 logits, and the
            ``[batch, seq_len, hidden * n_targets]`` trunk state DSpark would
            read. The logits are float32, as the reference's are
            (:func:`head_logits`).
        """
        batch = int(tokens.shape[0])
        rows = RaggedRows.uniform(
            batch,
            seq_len,
            tokens.device,
            cache.cache_lengths if cache is not None else None,
            ratios=self.config.window_ratios,
        )
        x, main_hidden = self.trunk(
            ops.reshape(tokens, [1, batch * seq_len]), rows, cache
        )
        logits = self._lm_head(self.norm(x))
        return (
            ops.reshape(logits, [batch, seq_len, self.config.vocab_size]),
            ops.reshape(main_hidden, [batch, seq_len, main_hidden.shape[2]]),
        )

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the serving graph's inputs; its leaves are its inputs.

        A single device takes no signal buffers.
        """
        devices = list(self.config.devices)
        return ragged_kv_symbolic_input_tree(
            kv_params=self.config.kv_params,
            device_refs=devices,
            include_signal_buffers=len(devices) > 1,
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the serving graph, whose inputs are :meth:`input_tree`'s
        leaves.

        The graph runs :meth:`serve` on a single device and
        :meth:`serve_tensor_parallel` over :meth:`tensor_parallel_replicas`
        on several, so a multi-device trace shards this model's weights.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        config = self.config
        devices = list(config.devices)
        n_dev = len(devices)

        def forward(
            tokens: Value[Any],
            input_row_offsets: Value[Any],
            return_n_logits: Value[Any],
            *variadic_args: Value[Any],
        ) -> DecoderOutputs:
            if n_dev == 1:
                cache = DeepseekV4Cache.from_groups(
                    config,
                    config.kv_params.unflatten_basic_kv_tree(
                        iter(variadic_args)
                    ),
                )
                return self.serve(
                    tokens.tensor,
                    input_row_offsets.tensor,
                    return_n_logits.tensor,
                    cache,
                )
            signal_buffers = [v.buffer for v in variadic_args[:n_dev]]
            groups = config.kv_params.unflatten_basic_kv_tree(
                iter(variadic_args[n_dev:])
            )
            caches = [
                DeepseekV4Cache.from_groups(config, groups, device_idx=i)
                for i in range(n_dev)
            ]
            tokens_per_dev = ops.distributed_broadcast(
                tokens.tensor, signal_buffers
            )
            offsets_per_dev = ops.distributed_broadcast(
                input_row_offsets.tensor, signal_buffers
            )
            return DeepseekV4.serve_tensor_parallel(
                self.tensor_parallel_replicas(devices),
                tokens_per_dev,
                offsets_per_dev,
                return_n_logits.tensor,
                caches,
                signal_buffers,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    def serve(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        cache: DeepseekV4Cache,
    ) -> DecoderOutputs:
        """The serving graph body: ragged tokens in, the pipeline's logits out.

        Args:
            tokens: ``[T]`` token ids of the whole batch.
            input_row_offsets: ``[batch + 1]`` uint32 row offsets.
            return_n_logits: ``[1]`` int64, trailing logits per request.
            cache: The paged leaves.

        Returns:
            What :func:`~max.nn.transformer.logits_postprocess` returns for
            ``config.return_logits`` / ``config.return_hidden_states``.
            ``SELECTED_LAYERS`` returns the trunk state DSpark reads, the one
            capture of the layers in ``dspark_target_layer_ids``.
        """
        t = tokens.shape[0]
        rows = RaggedRows.from_offsets(
            input_row_offsets,
            t,
            cache.cache_lengths,
            dict.fromkeys(self.config.window_ratios),
        )
        x, main_hidden = self.trunk(ops.reshape(tokens, [1, t]), rows, cache)
        return logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            input_row_offsets,
            return_n_logits,
            norm=self.norm,
            lm_head=self._lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
            capture_hidden_states=[[main_hidden]],
        )

    def tensor_parallel_replicas(
        self, devices: Sequence[DeviceRef]
    ) -> list[DeepseekV4]:
        """One copy of the model per device, for the multi-device graph.

        Every weight of a copy is a shard of this model's, so this model stays
        the one the weights registry is built from while the copies are what
        the graph runs. Attention, its indexer included, splits its heads
        across the devices (:meth:`DeepseekV4Attention.shard_heads`) and the
        MoE its routed experts (:meth:`DeepseekV4MoE.shard_experts`), and
        every other weight is replicated, so each device computes the rest of
        the forward whole. Call it once, after :meth:`load_state_dict`.
        """
        replicas = [DeepseekV4(self.config, device) for device in devices]
        _shard_blocks(self, replicas, "layers", devices)
        tp_blocks = []
        for i, source in enumerate(self.layers):
            assert isinstance(source, DeepseekV4Block)
            blocks = []
            for replica in replicas:
                block = replica.layers[i]
                assert isinstance(block, DeepseekV4Block)
                blocks.append(block)
            tp_blocks.append(_TensorParallelBlock(source, blocks))
        for replica in replicas:
            replica.tp_blocks = tuple(tp_blocks)
        return replicas

    @staticmethod
    def serve_tensor_parallel(
        replicas: Sequence[DeepseekV4],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        caches: Sequence[DeepseekV4Cache],
        signal_buffers: Sequence[BufferValue],
    ) -> DecoderOutputs:
        """:meth:`serve` over :meth:`tensor_parallel_replicas`, one per device.

        ``tokens``, ``input_row_offsets`` and ``caches`` are per device; the
        logits come from device 0, whose stream every device holds a copy of.
        """
        lead = replicas[0]
        t = tokens[0].shape[0]
        windows = dict.fromkeys(lead.config.window_ratios)
        rows = [
            RaggedRows.from_offsets(offsets, t, cache.cache_lengths, windows)
            for offsets, cache in zip(input_row_offsets, caches, strict=True)
        ]
        selected = (
            lead.config.return_hidden_states
            == ReturnHiddenStates.SELECTED_LAYERS
        )
        x, main_hidden = DeepseekV4.trunk_tensor_parallel(
            replicas,
            [ops.reshape(tok, [1, t]) for tok in tokens],
            rows,
            caches,
            signal_buffers,
            collect_main_hidden=selected,
        )
        return logits_postprocess(
            ops.reshape(x, [t, lead.config.hidden_size]),
            input_row_offsets[0],
            return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=lead.config.return_logits,
            return_hidden_states=lead.config.return_hidden_states,
            capture_hidden_states=[main_hidden] if selected else None,
        )

    @staticmethod
    def serve_paged(
        replicas: Sequence[DeepseekV4],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        leaves: Mapping[str, Sequence[PagedCacheValues]],
        signal_buffers: Sequence[BufferValue],
    ) -> DecoderOutputs:
        """:meth:`serve` on one device, or across the replicas' devices.

        Args:
            replicas: The model, or :meth:`tensor_parallel_replicas`.
            tokens: ``[T]`` token ids, on each device.
            input_row_offsets: ``[batch + 1]`` row offsets, on each device.
            return_n_logits: ``[1]`` int64, trailing logits per request.
            leaves: Each cache leaf's values, per device, by its key in
                ``config.kv_leaf_specs()``.
            signal_buffers: One per device, empty on one device.
        """
        lead = replicas[0]
        caches = [
            DeepseekV4Cache.from_leaves(lead.config, leaves, device_idx=i)
            for i in range(len(replicas))
        ]
        if len(replicas) == 1:
            return lead.serve(
                tokens[0],
                input_row_offsets[0],
                return_n_logits,
                caches[0],
            )
        return DeepseekV4.serve_tensor_parallel(
            replicas,
            tokens,
            input_row_offsets,
            return_n_logits,
            caches,
            signal_buffers,
        )

    @staticmethod
    def trunk_tensor_parallel(
        replicas: Sequence[DeepseekV4],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[DeepseekV4Cache],
        signal_buffers: Sequence[BufferValue],
        *,
        collect_main_hidden: bool = True,
    ) -> tuple[TensorValue, list[TensorValue]]:
        """:meth:`trunk` over :meth:`tensor_parallel_replicas`.

        Args:
            replicas: One copy of the model per device.
            tokens: ``[1, T]`` token ids, on each device.
            rows: The batch's token bookkeeping, per device.
            caches: The paged leaves, per device.
            signal_buffers: One per device.
            collect_main_hidden: Whether to collect the DSpark states.

        Returns:
            Device 0's ``[1, T, hidden]`` contracted stream -- every device
            holds the same one -- and, per device, the ``[1, T, hidden *
            n_targets]`` trunk state DSpark reads (empty when not collected).
        """
        lead = replicas[0]
        targets = set(lead.config.dspark_target_layer_ids)
        hs = [
            expand_copies(replica.embed(tok), lead.config.hc_mult)
            for replica, tok in zip(replicas, tokens, strict=True)
        ]
        collected: list[list[TensorValue]] = [[] for _ in replicas]

        def collect(layer_idx: int, out: list[TensorValue]) -> None:
            if collect_main_hidden and layer_idx in targets:
                for per_dev, h in zip(collected, out, strict=True):
                    per_dev.append(h)

        if len(lead.tp_blocks) != len(lead.layers):
            raise ValueError(
                "replicas must come from DeepseekV4.tensor_parallel_replicas"
            )

        def inputs_for_layer(
            layer_idx: int, xs: list[TensorValue]
        ) -> list[Tree[Any]]:
            return [
                xs,
                list(rows),
                list(tokens),
                list(caches),
                list(signal_buffers),
                lead.tp_blocks[layer_idx].layer_indices(),
            ]

        hs = forward_sequential_layers(
            list(lead.tp_blocks),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=hs,
            on_layer_output=collect,
            subgraph_layer_groups=lead.subgraph_groups,
            name_for_subgraph=lambda g: f"dsv4_block_tp_{g}",
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
        )
        main_hidden = (
            [lead.collect_main_hidden(per_dev) for per_dev in collected]
            if collect_main_hidden
            else []
        )
        return lead.contract_head(hs[0]), main_hidden


def _shard_blocks(
    source: Module,
    replicas: Sequence[Module],
    blocks: str,
    devices: Sequence[DeviceRef],
) -> None:
    """Splits the heads and experts of the blocks named ``blocks``.

    ``replicas`` are fresh copies of ``source``, one on each of
    ``devices``. Each keeps its device's heads and experts, and every other
    weight whole.
    """
    n = len(devices)
    for layer in getattr(source, blocks):
        assert isinstance(layer, DeepseekV4Block)
        layer.attn.shard_heads(n)
        layer.ffn.shard_experts(n)
    _adopt_shards(source, replicas, list(devices))
    for rank, replica in enumerate(replicas):
        for layer in getattr(replica, blocks):
            assert isinstance(layer, DeepseekV4Block)
            layer.attn.keep_local_heads(n)
            layer.ffn.keep_local_experts(n, rank)


class DeepseekV4DSpark(Module):
    """DeepSeek-V4's DSpark stages, which draft a block from the trunk's states.

    The stages ship in the trunk's checkpoint as ``mtp.*`` but run apart
    from it: they read the trunk's embedding and head, which the caller
    passes in, and their windows are layers ``num_hidden_layers ..`` of the
    trunk's own window leaf. Each method takes per-device lists, one entry
    on a single device, and :meth:`tensor_parallel_replicas` gives the
    per-device stages.
    """

    def __init__(
        self, config: DeepseekV4Config, device: DeviceRef | None = None
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = config.devices[0]
        self.mtp = LayerList(
            [
                DSparkBlock(config, stage_id, device, config.max_seq_len)
                for stage_id in range(len(config.dspark_target_layer_ids))
            ]
        )

    def tensor_parallel_replicas(
        self, devices: Sequence[DeviceRef]
    ) -> list[DeepseekV4DSpark]:
        """One copy of the stages per device, split as the trunk's blocks are.

        That covers the Markov head, which stays whole: sharding
        ``markov_w1`` by vocab as the reference's ``ParallelEmbedding`` does
        gives the same numbers (the all-reduce adds zeros), only with one
        collective per draft position. Call it once, after
        :meth:`load_state_dict`.
        """
        replicas = [DeepseekV4DSpark(self.config, device) for device in devices]
        _shard_blocks(self, replicas, "mtp", devices)
        return replicas

    @staticmethod
    def fill_cache(
        replicas: Sequence[DeepseekV4DSpark],
        main_hidden: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        leaves: Mapping[str, Sequence[PagedCacheValues]],
    ) -> None:
        """Materializes the stages' windows after a trunk forward.

        Each stage's attention writes the trunk's projected state for every
        token of the forward into its window; nothing else runs. Call it in
        the graph of the trunk forward that produced ``main_hidden``, with
        the leaves at that forward's starting lengths.

        Args:
            replicas: The stages, one copy per device.
            main_hidden: ``[1, T, d * n_targets]``, :meth:`DeepseekV4.trunk`'s
                second output, on each device.
            input_row_offsets: That forward's row offsets, on each device.
            leaves: The trunk's cache leaves by key, per device.
        """
        for i, replica in enumerate(replicas):
            cache = DeepseekV4Cache.from_leaves(
                replica.config, leaves, device_idx=i
            )
            rows = RaggedRows.from_offsets(
                input_row_offsets[i],
                main_hidden[i].shape[1],
                cache.cache_lengths,
            )
            first = replica.mtp[0]
            assert isinstance(first, DSparkBlock)
            main_x = first.project_main(main_hidden[i])
            for stage in replica.mtp:
                assert isinstance(stage, DSparkBlock)
                assert isinstance(stage.attn, DSparkAttention)
                stage.attn.prefill_cache(main_x, rows, cache)

    @staticmethod
    def block(
        replicas: Sequence[DeepseekV4DSpark],
        embeds: Sequence[Embedding],
        block_ids: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        leaves: Mapping[str, Sequence[PagedCacheValues]],
        signal_buffers: Sequence[BufferValue],
    ) -> TensorValue:
        """Runs ``b`` draft blocks through the three stages.

        Args:
            replicas: The stages, one copy per device.
            embeds: The trunk's embedding, on each device.
            block_ids: ``[1, b * K]`` each request's anchor followed by
                ``K - 1`` ``dspark_noise_token_id``, on each device; the
                noise token goes through the embedding like any other, it is
                not a mask.
            input_row_offsets: The blocks' row offsets, ``K`` rows per
                request, on each device.
            leaves: The trunk's cache leaves by key, per device, at one past
                each request's last committed token, windows already
                materialized.
            signal_buffers: One per device, empty on one device.

        Returns:
            Device 0's ``[1, b * K, hc, d]`` last-stage stream, which every
            device holds a copy of.
        """
        lead = replicas[0]
        caches = [
            DeepseekV4Cache.from_leaves(lead.config, leaves, device_idx=i)
            for i in range(len(replicas))
        ]
        rows = [
            RaggedRows.from_offsets(off, ids.shape[1], cache.cache_lengths)
            for off, ids, cache in zip(
                input_row_offsets, block_ids, caches, strict=True
            )
        ]
        xs = [
            expand_copies(embed(ids), lead.config.hc_mult)
            for embed, ids in zip(embeds, block_ids, strict=True)
        ]
        if len(replicas) == 1:
            x = xs[0]
            for stage in lead.mtp:
                assert isinstance(stage, DSparkBlock)
                x = stage.decode(x, rows[0], caches[0], block_ids[0])
            return x
        for stage_idx in range(len(lead.mtp)):
            stages = []
            for replica in replicas:
                stage = replica.mtp[stage_idx]
                assert isinstance(stage, DSparkBlock)
                stages.append(stage)
            xs = DSparkBlock.decode_tensor_parallel(
                stages, xs, rows, caches, block_ids, signal_buffers
            )
        return xs[0]

    def head(
        self,
        x: TensorValue,
        anchor_ids: TensorValue,
        head: Linear,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """``[b, K]`` draft ids and ``[b, K, vocab]`` logits off
        :meth:`block`'s output (:meth:`DSparkBlock.forward_head`), through
        the trunk's ``head``."""
        last = self.mtp[-1]
        assert isinstance(last, DSparkBlock)
        return last.forward_head(x, anchor_ids, head, sampler)

    def __call__(
        self,
        x: TensorValue,
        anchor_ids: TensorValue,
        head: Linear,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        # Alias for head to satisfy the Module ABC.
        return self.head(x, anchor_ids, head, sampler)
