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

Status: prefill is complete end to end -- ``__call__`` takes token ids and
returns logits. Decode is not here, and is blocked rather than unwritten: V4's
compressed KV cache appends one entry per ``compress_ratio`` tokens, and MAX's
paged cache indexes slots by token position with no stride mode
(.agent/backlogs/192/ISSUES.md Issue 30). Prefill is unaffected because the
reference's ``start_pos == 0`` path reads freshly computed tensors and never
the cache -- it only writes it.

DSpark is likewise decode-only by construction and so contributes nothing to a
prefill; its stages are built and gated, but nothing calls them here.

The mHC weights are declared flat on the block (``hc_attn_fn``, not
``hc_attn.fn``) because that is how the checkpoint names them. Grouping them
into a submodule would read better and cost a rename in the adapter, which is
the thing this file is arranged to avoid.
"""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.embedding import Embedding
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from .layers import (
    DeepseekV4Attention,
    DeepseekV4MoE,
    DSparkAttention,
    DSparkConfidenceHead,
    DSparkMarkovHead,
    expand_copies,
    hc_head,
    hc_mix_width,
    hc_post,
    hc_pre,
)
from .model_config import DeepseekV4Config


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
        self.attn_norm = RMSNorm(
            config.hidden_size, config.dtype, config.rms_norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, config.dtype, config.rms_norm_eps
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
        seq_len: int,
        token_ids: TensorValue,
    ) -> TensorValue:
        """``[b, s, hc, d]`` in, ``[b, s, hc, d]`` out.

        Two identical wrappings. Each reads ``post`` and ``comb`` off the state
        *before* its sublayer runs, so the mixing weights describe the incoming
        stream, not the outgoing one.
        """
        residual = x
        h, post, comb = self._contract(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        h = self.attn(self.attn_norm(h), seq_len)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._contract(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        h = self.ffn(self.ffn_norm(h), token_ids)
        return hc_post(h, residual, post, comb)

    def decode_step(
        self,
        x: TensorValue,
        token_ids: TensorValue,
        pos: TensorValue,
        ring_pos: TensorValue,
        win_idxs: TensorValue,
        ratio_aux: dict[str, TensorValue] | None,
        state: dict[str, TensorValue],
    ) -> tuple[TensorValue, dict[str, TensorValue]]:
        """One decode token through the block. Same mHC wrapping as
        ``__call__``; only the attention entry point differs.

        ``ratio_aux`` carries the layer's ratio-specific host inputs
        (``comp_idxs`` / ``ape_idx`` / ``should`` / ``zone_pos`` /
        ``comp_pos``), shared across layers of the same ratio; ``None`` on
        ratio-0 layers. ``state`` holds this layer's buffers keyed ``ring`` /
        ``zone`` / ``kv_state`` / ``score_state``.
        """
        residual = x
        h, post, comb = self._contract(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        aux = ratio_aux or {}
        out, new_ring, new_zone, new_kvs, new_scs = self.attn.decode_token(
            self.attn_norm(h),
            pos,
            state["ring"],
            ring_pos,
            win_idxs,
            zone=state.get("zone"),
            comp_idxs=aux.get("comp_idxs"),
            kv_state=state.get("kv_state"),
            score_state=state.get("score_state"),
            ape_idx=aux.get("ape_idx"),
            should=aux.get("should"),
            zone_pos=aux.get("zone_pos"),
            comp_pos=aux.get("comp_pos"),
        )
        new_state = {"ring": new_ring}
        if new_zone is not None:
            assert new_kvs is not None and new_scs is not None
            new_state["zone"] = new_zone
            new_state["kv_state"] = new_kvs
            new_state["score_state"] = new_scs
        x = hc_post(out, residual, post, comb)

        residual = x
        h, post, comb = self._contract(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        h = self.ffn(self.ffn_norm(h), token_ids)
        return hc_post(h, residual, post, comb), new_state


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
        main_x = self.main_norm(self.main_proj(main_hidden))
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
        kv_cache: TensorValue,
        start_pos: int,
        seq_len: int,
        draft_ids: TensorValue,
    ) -> TensorValue:
        """One stage of a draft block, decode only.

        Not ``__call__``: a stage takes the trunk's state and a ring buffer,
        which a trunk block has no equivalent of, so it is a different entry
        point rather than an override.

        At ``start_pos == 0`` the reference runs nothing but the attention's
        cache fill and returns its input, so there is no prefill path here;
        ``DSparkAttention.prefill_cache`` is that fill, called by the model.
        """
        residual = x
        h, post, comb = self._contract(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        h = cast(DSparkAttention, self.attn).decode(
            self.attn_norm(h), main_x, kv_cache, start_pos, seq_len
        )
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

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
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
            config.hidden_size, config.dtype, config.rms_norm_eps
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
        self.head = Linear(
            config.hidden_size, config.vocab_size, config.dtype, device
        )
        self.mtp = LayerList(
            [
                DSparkBlock(config, stage_id, device, config.max_seq_len)
                for stage_id in range(len(config.dspark_target_layer_ids))
            ]
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

    def __call__(
        self, tokens: TensorValue, seq_len: int
    ) -> tuple[TensorValue, TensorValue]:
        """Prefill a padded batch of token ids.

        Args:
            tokens: ``[batch, seq_len]`` int32 token ids.
            seq_len: Static sequence length.

        Returns:
            ``[batch, seq_len, vocab_size]`` float32 logits, and the
            ``[batch, seq_len, hidden * n_targets]`` trunk state DSpark would
            read. The logits are float32 because the reference's head is --
            it stores bf16 and upcasts before the matmul.
        """
        h = expand_copies(self.embed(tokens), self.config.hc_mult)
        targets = set(self.config.dspark_target_layer_ids)
        collected: list[TensorValue] = []
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, seq_len, tokens)
            if layer_idx in targets:
                collected.append(h)

        x = self.norm(self.contract_head(h))
        logits = ops.matmul(
            ops.cast(x, DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )
        return logits, self.collect_main_hidden(collected)

    def prefill_with_state(
        self, tokens: TensorValue, seq_len: int
    ) -> tuple[TensorValue, list[dict[str, TensorValue]]]:
        """Prefill, also returning each trunk layer's cache-state tensors.

        Per layer: ``latent`` ``[b, s, head_dim]`` (the quantized per-token
        rows the reference writes into the sliding window); on compressed
        layers also ``comp_kv`` / ``comp_score`` ``[b, s, proj_dim]`` float32
        raw compressor projections, and ``zone``
        ``[b, s // ratio, head_dim]`` when at least one window closed. The
        host assembles the reference's ring layout and ``kv_state`` /
        ``score_state`` buffers from these; see ``Compressor.forward``'s
        ``start_pos == 0`` writes.
        """
        logits, _ = self(tokens, seq_len)
        states = []
        for layer in self.layers:
            states.append(dict(cast(DeepseekV4Block, layer).attn.exported))
        return logits, states

    def decode(
        self,
        token: TensorValue,
        pos: TensorValue,
        ring_pos: TensorValue,
        win_idxs: TensorValue,
        ratio_aux: dict[int, dict[str, TensorValue]],
        states: list[dict[str, TensorValue]],
    ) -> tuple[TensorValue, list[dict[str, TensorValue]]]:
        """One decode token through the trunk. DSpark stages are not run:
        the 10-step golden gate uses the DSpark-off baseline (the DSpark
        golden is not run-to-run reproducible, ISSUES Issue 28).

        Args:
            token: ``[b, 1]`` int32, the token at position ``pos``.
            pos: ``[1]`` int32 absolute position (``start_pos``).
            ring_pos: ``[1]`` int32, ``pos % window``.
            win_idxs: ``[b, 1, window]`` int32 decode window indices.
            ratio_aux: Ratio-specific host inputs, keyed by compress ratio.
            states: Per trunk layer, the buffers ``decode_step`` reads.

        Returns:
            ``[b, 1, vocab]`` float32 logits and the updated states.
        """
        h = expand_copies(self.embed(token), self.config.hc_mult)
        new_states: list[dict[str, TensorValue]] = []
        for layer_idx, layer in enumerate(self.layers):
            ratio = self.config.layer_compress_ratio(layer_idx)
            h, new_state = cast(DeepseekV4Block, layer).decode_step(
                h,
                token,
                pos,
                ring_pos,
                win_idxs,
                ratio_aux.get(ratio),
                states[layer_idx],
            )
            new_states.append(new_state)

        x = self.norm(self.contract_head(h))
        logits = ops.matmul(
            ops.cast(x, DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )
        return logits, new_states
