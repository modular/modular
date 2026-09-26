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

DSpark is decode-only by construction and so contributes nothing to a
prefill; :meth:`DeepseekV4.fill_dspark_cache` writes the stages' windows after
a prefill and :meth:`DSparkBlock.decode` runs a draft block.

The mHC weights are declared flat on the block (``hc_attn_fn``, not
``hc_attn.fn``) because that is how the checkpoint names them. Grouping them
into a submodule would read better and cost a rename in the adapter, which is
the thing this file is arranged to avoid.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.embedding import Embedding
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.transformer import logits_postprocess

from .layers import (
    DeepseekV4Attention,
    DeepseekV4Cache,
    DeepseekV4Indexer,
    DeepseekV4MoE,
    DSparkAttention,
    DSparkConfidenceHead,
    DSparkMarkovHead,
    RaggedRows,
    expand_copies,
    hc_head,
    hc_mix_width,
    hc_post,
    hc_pre,
)
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


def _hc_parameters(
    config: DeepseekV4Config, device: DeviceRef, site: str
) -> tuple[Weight, Weight, Weight]:
    """The ``fn`` / ``base`` / ``scale`` triple for one mHC site.

    All three are float32 in the checkpoint and stay float32: the mixer runs in
    fp32 in the reference, and the Sinkhorn division chain is not safe in bf16.
    ``site`` is ``attn`` or ``ffn``; the head's triple is narrower and is
    declared on the model.
    """
    mix_hc = hc_mix_width(config.hc_mult)
    hc_dim = config.hc_mult * config.hidden_size
    return (
        Weight(
            name=f"hc_{site}_fn",
            dtype=DType.float32,
            shape=(mix_hc, hc_dim),
            device=device,
        ),
        Weight(
            name=f"hc_{site}_base",
            dtype=DType.float32,
            shape=(mix_hc,),
            device=device,
        ),
        Weight(
            name=f"hc_{site}_scale",
            dtype=DType.float32,
            shape=(3,),
            device=device,
        ),
    )


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
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.norm_eps = config.rms_norm_eps
        (
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
        ) = _hc_parameters(config, device, "attn")
        (
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
        ) = _hc_parameters(config, device, "ffn")

    @staticmethod
    def tensor_parallel(
        blocks: Sequence[DeepseekV4Block],
        xs: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        token_ids: Sequence[TensorValue],
        caches: Sequence[DeepseekV4Cache],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """:meth:`__call__` on every device; ``blocks`` holds a replica each.

        Each device's attention covers its share of the heads and returns a
        partial sum of the output projection; the all-reduce completes it
        before the mHC merge, which every device runs whole. The indexer's
        heads are split the same way, so its scores are all-reduced before the
        top-k. The MoE splits its routed experts across the devices and
        all-reduces their sum (:meth:`DeepseekV4MoE.tensor_parallel`).
        """
        contracted = [
            block._contract(
                x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            for block, x in zip(blocks, xs, strict=True)
        ]
        states = [
            block.attn.begin(block.attn_norm(h), row, cache)
            for block, (h, _, _), row, cache in zip(
                blocks, contracted, rows, caches, strict=True
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
            block._contract(
                x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
            )
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

    def _contract(
        self, x: TensorValue, fn: Weight, scale: Weight, base: Weight
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        return hc_pre(
            x,
            fn,
            scale,
            base,
            self.hc_mult,
            self.norm_eps,
            self.hc_eps,
            self.sinkhorn_iters,
        )

    def __call__(
        self,
        x: TensorValue,
        rows: RaggedRows,
        token_ids: TensorValue,
        cache: DeepseekV4Cache | None = None,
    ) -> TensorValue:
        """``[1, T, hc, d]`` in, ``[1, T, hc, d]`` out.

        Two identical wrappings. Each reads ``post`` and ``comb`` off the state
        *before* its sublayer runs, so the mixing weights describe the incoming
        stream, not the outgoing one.
        """
        residual = x
        h, post, comb = self._contract(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        h = self.attn(self.attn_norm(h), rows, cache)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._contract(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
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
    either; the model passes them in.
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
        self.noise_token_id = config.dspark_noise_token_id
        self.hc_mult = config.hc_mult

        self.is_first = stage_id == 0
        self.is_last = stage_id == len(config.dspark_target_layer_ids) - 1

        if self.is_first:
            self.main_proj = Linear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                config.dtype,
                device,
            )
            self.main_norm = RMSNorm(
                config.hidden_size, config.dtype, config.rms_norm_eps
            )
        if self.is_last:
            self.norm = RMSNorm(
                config.hidden_size, config.dtype, config.rms_norm_eps
            )
            self.markov_head = DSparkMarkovHead(config, device)
            self.confidence_head = DSparkConfidenceHead(config, device)
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
        """The trunk state the stages attend to, ``[b, s, d]``. ``mtp.0`` only."""
        return self.main_norm(self.main_proj(main_hidden))

    def forward_embed(
        self,
        main_hidden: TensorValue,
        token_ids: TensorValue,
        embed: Embedding,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Start a draft block. ``mtp.0`` only.

        Args:
            main_hidden: ``[b, s, d * len(target_layer_ids)]`` -- the trunk's
                hidden at each target layer, each one **mean-reduced over the
                hc axis first**, then concatenated. That mean is why the width
                is ``3 * 4096`` and not ``3 * 4 * 4096``.
            token_ids: ``[b]``, the token the trunk just produced.
            embed: The trunk's embedding, shared not copied.

        Returns:
            The draft stream ``[b, block_size, hc, d]``, the projected trunk
            state ``[b, s, d]``, and the draft token ids ``[b, block_size]``.
        """
        main_x = self.project_main(main_hidden)
        noise = ops.broadcast_to(
            ops.constant(
                self.noise_token_id, token_ids.dtype, token_ids.device
            ),
            [token_ids.shape[0], self.block_size - 1],
        )
        # Position 0 is the real token; the rest are a literal noise token that
        # goes through the embedding like any other, not a mask or a pad.
        draft_ids = ops.concat([ops.unsqueeze(token_ids, -1), noise], axis=1)
        return expand_copies(embed(draft_ids), self.hc_mult), main_x, draft_ids

    def decode(
        self,
        x: TensorValue,
        main_x: TensorValue,
        cache: DeepseekV4Cache,
        draft_ids: TensorValue,
    ) -> TensorValue:
        """One stage of a draft block, decode only.

        Not ``__call__``: a stage takes the trunk's state, which a trunk block
        has no equivalent of, so it is a different entry point rather than an
        override.

        At ``start_pos == 0`` the reference runs nothing but the attention's
        cache fill and returns its input, so there is no prefill path here;
        :meth:`DeepseekV4.fill_dspark_cache` is that fill.
        """
        residual = x
        h, post, comb = self._contract(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        assert isinstance(self.attn, DSparkAttention)
        h = self.attn.decode(self.attn_norm(h), main_x, cache)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._contract(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        # The gate on an MTP stage is score-routed, so draft_ids goes unread;
        # the reference passes the trunk's input_ids here, whose length does
        # not even match, for the same reason.
        h = self.ffn(self.ffn_norm(h), draft_ids)
        return hc_post(h, residual, post, comb)

    def forward_head(
        self,
        x: TensorValue,
        token_ids: TensorValue,
        head: Linear,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Draft tokens, logits and confidence out of the last stage.

        The Markov loop is serial and cannot be batched: step ``i``'s logit
        bias is keyed on the token sampled at step ``i``, which does not exist
        until step ``i`` has run. ``block_size`` is static, so it unrolls.

        Sampling is greedy -- ``sample()`` short-circuits to ``argmax`` at
        temperature 0, which is what the golden was generated at.
        """
        contracted = hc_head(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.hc_mult,
            self.norm_eps,
            self.hc_eps,
        )
        logits = ops.matmul(
            ops.cast(self.norm(contracted), DType.float32),
            ops.transpose(ops.cast(head.weight, DType.float32), 0, 1),
        )

        current = token_ids
        out_ids = [token_ids]
        biased = []
        embeds = []
        for i in range(self.block_size):
            bias, embed = self.markov_head(current)
            step = logits[:, i] + bias
            biased.append(step)
            embeds.append(embed)
            current = ops.cast(ops.argmax(step, axis=-1), token_ids.dtype)
            current = ops.squeeze(current, axis=-1)
            out_ids.append(current)

        confidence = self.confidence_head(contracted, ops.stack(embeds, axis=1))
        return (
            ops.stack(out_ids, axis=1),
            ops.stack(biased, axis=1),
            confidence,
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
        # float32, as the reference declares it; stored bf16, upcast on load.
        self.head = Linear(
            config.hidden_size, config.vocab_size, DType.float32, device
        )
        self.mtp = LayerList(
            [
                DSparkBlock(config, stage_id, device, config.max_seq_len)
                for stage_id in range(len(config.dspark_target_layer_ids))
            ]
            if config.dspark_stages
            else []
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
        """The reference's head: a float32 matmul."""
        return ops.matmul(
            ops.cast(x, DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )

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
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, rows, tokens, cache)
            if layer_idx in targets:
                collected.append(h)
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
            read. The logits are float32 because the reference's head is --
            it stores bf16 and upcasts before the matmul.
        """
        batch = int(tokens.shape[0])
        rows = RaggedRows.uniform(
            batch,
            seq_len,
            tokens.device,
            cache.cache_lengths if cache is not None else None,
        )
        x, main_hidden = self.trunk(
            ops.reshape(tokens, [1, batch * seq_len]), rows, cache
        )
        logits = self._lm_head(self.norm(x))
        return (
            ops.reshape(logits, [batch, seq_len, self.config.vocab_size]),
            ops.reshape(main_hidden, [batch, seq_len, main_hidden.shape[2]]),
        )

    def serve(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        cache: DeepseekV4Cache,
    ) -> tuple[TensorValue, ...]:
        """The serving graph body: ragged tokens in, the pipeline's logits out.

        Args:
            tokens: ``[T]`` token ids of the whole batch.
            input_row_offsets: ``[batch + 1]`` uint32 row offsets.
            return_n_logits: ``[1]`` int64, trailing logits per request.
            cache: The paged leaves.

        Returns:
            What :func:`~max.nn.transformer.logits_postprocess` returns for
            ``config.return_logits`` / ``config.return_hidden_states``.
        """
        t = tokens.shape[0]
        rows = RaggedRows.from_offsets(
            input_row_offsets, t, cache.cache_lengths
        )
        x, _ = self.trunk(ops.reshape(tokens, [1, t]), rows, cache)
        return logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            input_row_offsets,
            return_n_logits,
            norm=self.norm,
            lm_head=self._lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
        )

    def tensor_parallel_replicas(
        self, devices: Sequence[DeviceRef]
    ) -> list[DeepseekV4]:
        """One copy of the model per device, for the multi-device graph.

        Every weight of a copy is a shard of this model's, so this model stays
        the one the weights registry is built from while the copies are what
        the graph runs. Attention, its indexer included, splits its heads
        across the devices (:meth:`DeepseekV4Attention.shard_heads`) and the
        MoE its routed experts (:meth:`DeepseekV4MoE.shard_experts`); every
        other weight is replicated, and each device computes the rest of the
        forward whole.
        Call it once, after :meth:`load_state_dict`.
        """
        n = len(devices)
        for layer in self.layers:
            assert isinstance(layer, DeepseekV4Block)
            layer.attn.shard_heads(n)
            layer.ffn.shard_experts(n)
        replicas = [DeepseekV4(self.config, device) for device in devices]
        _adopt_shards(self, replicas, list(devices))
        for rank, replica in enumerate(replicas):
            for layer in replica.layers:
                assert isinstance(layer, DeepseekV4Block)
                layer.attn.keep_local_heads(n)
                layer.ffn.keep_local_experts(n, rank)
        return replicas

    @staticmethod
    def serve_tensor_parallel(
        replicas: Sequence[DeepseekV4],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        caches: Sequence[DeepseekV4Cache],
        signal_buffers: Sequence[BufferValue],
    ) -> tuple[TensorValue, ...]:
        """:meth:`serve` over :meth:`tensor_parallel_replicas`, one per device.

        ``tokens``, ``input_row_offsets`` and ``caches`` are per device; the
        logits come from device 0, whose stream every device holds a copy of.
        """
        lead = replicas[0]
        t = tokens[0].shape[0]
        rows = [
            RaggedRows.from_offsets(offsets, t, cache.cache_lengths)
            for offsets, cache in zip(input_row_offsets, caches, strict=True)
        ]
        token_rows = [ops.reshape(tok, [1, t]) for tok in tokens]
        hs = [
            expand_copies(replica.embed(tok), lead.config.hc_mult)
            for replica, tok in zip(replicas, token_rows, strict=True)
        ]
        for layer_idx in range(len(lead.layers)):
            blocks = []
            for replica in replicas:
                block = replica.layers[layer_idx]
                assert isinstance(block, DeepseekV4Block)
                blocks.append(block)
            hs = DeepseekV4Block.tensor_parallel(
                blocks,
                hs,
                rows,
                token_rows,
                caches,
                signal_buffers,
            )
        x = lead.contract_head(hs[0])
        return logits_postprocess(
            ops.reshape(x, [t, lead.config.hidden_size]),
            input_row_offsets[0],
            return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=lead.config.return_logits,
            return_hidden_states=lead.config.return_hidden_states,
        )

    def fill_dspark_cache(
        self,
        main_hidden: TensorValue,
        seq_len: int,
        cache: DeepseekV4Cache,
    ) -> None:
        """What ``forward_spec`` does at ``start_pos == 0``: each stage's
        attention writes the trunk's projected state into its window and
        nothing else runs. Call it after the trunk forward that produced
        ``main_hidden``, before ``cache_lengths`` advances.
        """
        first = self.mtp[0]
        assert isinstance(first, DSparkBlock)
        main_x = first.project_main(main_hidden)
        for stage in self.mtp:
            assert isinstance(stage, DSparkBlock)
            assert isinstance(stage.attn, DSparkAttention)
            stage.attn.prefill_cache(main_x, seq_len, cache)
