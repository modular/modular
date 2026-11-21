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
"""The Inkling 2 DSpark drafter on ModuleV3.

The drafter follows vLLM's ``Inkling2DSparkModel`` and its speculator
(``vllm/models/inkling2/nvidia/dspark.py`` and ``speculator.py``):

- Every layer, and the Markov head, mixes the target's tapped hidden states
  with its own softmaxed row of ``fuse_weight`` and normalizes the mixture
  with its ``dflare_norm``. A layer projects its mixture through
  ``k_proj_ctx`` and ``v_proj_ctx`` into the draft's cache at the context
  rows' positions; the mixture never runs through the layers.
- The block is the anchor (the last committed token, embedded and normalized
  through the target's table) followed by ``mask_embed`` rows. It runs
  through pre-norm layers whose attention sees the cached context and the
  whole block, bidirectionally; a local layer sees only the window before
  the block.
- The mask rows draft, the anchor does not. Each draft token's logits are the
  target's LM head over the row, plus the bias of a recurrent Markov head
  whose state advances one token at a time, then the target's softcap.

Parameter names mirror ``mtp.safetensors``, so the checkpoint loads as is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Embedding, Linear, Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import RowParallelLinear, col_parallel
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import (
    flash_attention_ragged,
    rope_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.kv_cache.cache_params import SpeculativeMethod
from max.nn.sequential import ModuleList
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor

from .dspark_config import Inkling2DSparkConfig
from .layers.norm import Inkling2RMSNorm
from .layers.placement import as_partial, as_sharded, reduce_partial

DSPARK_GLOBAL_ATTENTION = "dspark_global_attention"
"""The draft cache of the layers that see the whole context."""
DSPARK_LOCAL_ATTENTION = "dspark_local_attention"
"""The draft cache of the layers that see a window before the block."""


def dspark_cache_key(config: Inkling2DSparkConfig, layer_idx: int) -> str:
    """The draft cache a drafter layer reads and writes."""
    if config.is_local_attention(layer_idx):
        return DSPARK_LOCAL_ATTENTION
    return DSPARK_GLOBAL_ATTENTION


def construct_dspark_kv_params(
    config: Inkling2DSparkConfig,
    kv_cache_config: KVCacheConfig,
    devices: Sequence[DeviceRef],
    cache_dtype: DType,
    data_parallel_degree: int = 1,
    speculative_method: SpeculativeMethod | None = None,
) -> MultiKVCacheParams:
    """The drafter's own caches: one for its full layers, one for its local.

    A local layer reads only the window before the block and the block, so
    its cache keeps that many positions; a full layer keeps the context.
    Every block forward writes ``block_size`` rows past the committed tokens.
    """

    def params_for(num_layers: int, window_size: int | None) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            num_layers=num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            # The pool requires every leaf to name the target's method.
            speculative_method=speculative_method,
            num_draft_tokens=config.block_size,
            window_size=window_size,
        )

    local = [
        i
        for i in range(config.num_hidden_layers)
        if config.is_local_attention(i)
    ]
    children: dict[str, KVCacheParams] = {}
    num_global = config.num_hidden_layers - len(local)
    if num_global:
        children[DSPARK_GLOBAL_ATTENTION] = params_for(num_global, None)
    if local:
        children[DSPARK_LOCAL_ATTENTION] = params_for(
            len(local), config.attention_window(local[0])
        )
    return MultiKVCacheParams.from_params(children)


class Inkling2DSparkAttention(Module[..., Tensor]):
    """Block attention over the cached context and the block itself."""

    def __init__(
        self,
        config: Inkling2DSparkConfig,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx: int,
        layer_idx_in_cache: int,
    ) -> None:
        super().__init__()
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope = rope
        self.kv_params = kv_params
        self.layer_idx_in_cache = layer_idx_in_cache
        # MAX's windowed mask is per query (``k + window > q``); vLLM's is per
        # block (every row sees the window before the block start). Sizing it
        # from the block's last row makes that row exact; earlier rows reach up
        # to ``block_size - 1`` older keys.
        self.window = config.attention_window(layer_idx)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        # Heads shard over the target's tensor-parallel axis, as the target's
        # do: the draft's cache shares the target's pool, whose leaves hold
        # each device's share of the KV heads.
        self.q_proj = col_parallel(Linear(hidden, q_width, bias=False))
        self.k_proj = col_parallel(Linear(hidden, kv_width, bias=False))
        self.v_proj = col_parallel(Linear(hidden, kv_width, bias=False))
        self.k_proj_ctx = col_parallel(Linear(hidden, kv_width, bias=False))
        self.v_proj_ctx = col_parallel(Linear(hidden, kv_width, bias=False))
        self.o_proj = RowParallelLinear(q_width, hidden, bias=False)
        self.q_norm = Inkling2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Inkling2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_ctx = Inkling2RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )

    def _heads(self, x: Tensor, num_heads: int) -> Tensor:
        return x.reshape([-1, num_heads, self.head_dim])

    def _rope(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Rotates rotate-half pairs in float32 at the rows' cache positions."""
        return rope_ragged(
            x.cast(DType.float32),
            input_row_offsets,
            kv_collection.cache_lengths,
            self.rope.freqs_cis,
            interleaved=False,
            output_dtype=self.kv_params.dtype,
        )

    def _layer_idx(self) -> Tensor:
        return F.constant(self.layer_idx_in_cache, DType.uint32, device=CPU())

    def materialize(
        self,
        context: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes the K and V of this layer's mixed context rows."""
        k = self.k_norm_ctx(
            self._heads(self.k_proj_ctx(context), self.num_kv_heads)
        )
        k = self._rope(k, kv_collection, input_row_offsets)
        v = self._heads(self.v_proj_ctx(context), self.num_kv_heads)
        layer_idx = self._layer_idx()
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, v, input_row_offsets, layer_idx)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Writes the block's K and V, then attends over context and block."""
        # The head norms round to bfloat16 before RoPE, as the reference.
        q = self.q_norm(self._heads(self.q_proj(x), self.num_heads))
        k = self.k_norm(self._heads(self.k_proj(x), self.num_kv_heads))
        q = self._rope(q, kv_collection, input_row_offsets)
        k = self._rope(k, kv_collection, input_row_offsets)
        v = self._heads(self.v_proj(x), self.num_kv_heads)
        layer_idx = self._layer_idx()
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, v, input_row_offsets, layer_idx)
        out = flash_attention_ragged(
            self.kv_params,
            input=q,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.NULL_MASK
                if self.window is None
                else MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
            ),
            scale=self.head_dim**-0.5,
            local_window_size=-1 if self.window is None else self.window,
            output_dtype=x.dtype,
        )
        # Claim the heads axis before the reshape, as the target does, so the
        # flatten runs on each device's own heads.
        out = as_sharded(out, x.mesh, axis=1)
        out = out.reshape([-1, self.num_heads * self.head_dim])
        return reduce_partial(as_partial(self.o_proj(out), x.mesh))


class Inkling2DSparkLayer(Module[..., Tensor]):
    """A pre-norm drafter layer with a dense SwiGLU MLP."""

    def __init__(
        self,
        config: Inkling2DSparkConfig,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx: int,
        layer_idx_in_cache: int,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.dflare_norm = Inkling2RMSNorm(hidden, eps=eps)
        self.input_layernorm = Inkling2RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = Inkling2RMSNorm(hidden, eps=eps)
        self.self_attn = Inkling2DSparkAttention(
            config, rope, kv_params, layer_idx, layer_idx_in_cache
        )
        self.mlp = MLP(hidden, config.intermediate_size)

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer on the bfloat16 residual stream ``h``."""
        h = h + self.self_attn(
            self.input_layernorm(h), kv_collection, input_row_offsets
        )
        mlp = self.mlp(self.post_attention_layernorm(h))
        return h + reduce_partial(as_partial(mlp, h.mesh))


class Inkling2MarkovHead(Module[..., Tensor]):
    """The recurrent (``rnn_hidden``) bias on each draft token's logits.

    ``rnn_proj`` reads ``[state, previous token, hidden, f_rnn]`` and writes
    three gates; ``hidden_out`` lifts the rank-``r`` latent to the hidden
    width, where the target's LM head scores it.
    """

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = Embedding(config.vocab_size, dim=rank)
        self.rnn_proj = Linear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = Linear(rank, hidden, bias=True)
        self.dflare_norm = Inkling2RMSNorm(hidden, eps=config.rms_norm_eps)

    def context(self, hidden: Tensor, f_rnn: Tensor) -> Tensor:
        """The float32 bias plus ``[hidden, f_rnn]`` term of one draft step."""
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        bias = self.rnn_proj.bias
        assert isinstance(bias, Tensor)
        inputs = F.concat([hidden, f_rnn], axis=-1).cast(DType.float32)
        return inputs @ weight.cast(DType.float32).T + bias.cast(DType.float32)

    def step(
        self, state: Tensor, previous: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Advances the state by one token; returns it and the latent.

        Each op rounds to bfloat16, as the reference's torch ops do.
        """
        rank = self.markov_rank
        weight = self.rnn_proj.weight[:, : 2 * rank].cast(DType.float32)
        embedded = F.gather(self.markov_w1.weight, previous, axis=0)
        inputs = F.concat([state, embedded], axis=-1).cast(DType.float32)
        raw = (context + inputs @ weight.T).cast(DType.bfloat16)
        gate, candidate, out = raw.split([rank, rank, rank], axis=-1)
        gate = F.sigmoid(gate)
        keep = gate * state
        update = (1.0 - gate) * F.tanh(candidate)
        return keep + update, F.tanh(out)


class Inkling2DSpark(Module[..., Tensor]):
    """The drafter: mixed-context KV, the block's layers and the Markov head.

    The block's anchor embedding, the LM head and the softcap are the
    target's; the caller passes them in.
    """

    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(
        self,
        config: Inkling2DSparkConfig,
        kv_params: MultiKVCacheParams,
        *,
        max_seq_len: int,
        device: Device,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.rope = RotaryEmbedding(
            dim=hidden,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
        )
        layers = []
        self.layer_cache_keys: list[str] = []
        layers_in_cache: dict[str, int] = {}
        for i in range(config.num_hidden_layers):
            key = dspark_cache_key(config, i)
            leaf = kv_params.children[key]
            assert isinstance(leaf, KVCacheParams)
            layers.append(
                Inkling2DSparkLayer(
                    config,
                    self.rope,
                    leaf,
                    i,
                    layers_in_cache.get(key, 0),
                )
            )
            self.layer_cache_keys.append(key)
            layers_in_cache[key] = layers_in_cache.get(key, 0) + 1
        self.layers = ModuleList(layers)
        self.norm = Inkling2RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = Tensor.zeros(
            [config.num_fuse_rows, config.num_target_features]
        )
        self.mask_embed = Tensor.zeros([hidden])

    def mix(self, taps: Tensor, row: int, norm: Inkling2RMSNorm) -> Tensor:
        """The normalized ``fuse_weight[row]`` mixture of each row's taps.

        Args:
            taps: The target's hidden states, ``[rows, taps * hidden]``, in
                ``target_layer_ids`` order.
            row: The ``fuse_weight`` row: a layer index, or
                ``num_hidden_layers`` for the Markov head.
            norm: The ``dflare_norm`` that normalizes the mixture.
        """
        config = self.config
        alpha = F.softmax(self.fuse_weight.cast(DType.float32), axis=-1)[row]
        targets = taps.reshape(
            [-1, config.num_target_features, config.hidden_size]
        ).cast(DType.float32)
        weights = alpha.reshape([1, config.num_target_features, 1])
        mixed = F.sum(targets * weights, axis=1).reshape(
            [-1, config.hidden_size]
        )
        return norm(mixed)

    def materialize(
        self,
        taps: Tensor,
        kv_collections: Mapping[str, PagedCacheValues],
        input_row_offsets: Tensor,
    ) -> None:
        """Writes every layer's K and V of the context rows.

        Args:
            taps: The target's tapped hidden states, ``[rows, taps * hidden]``.
            kv_collections: The draft's caches at the context's position.
            input_row_offsets: The context rows' ragged offsets.
        """
        for i, (layer, key) in enumerate(
            zip(self.layers, self.layer_cache_keys, strict=True)
        ):
            layer.self_attn.materialize(
                self.mix(taps, i, layer.dflare_norm),
                kv_collections[key],
                input_row_offsets,
            )

    def f_rnn(self, taps: Tensor) -> Tensor:
        """The Markov head's context, from the last committed token's taps."""
        return self.mix(
            taps, self.config.num_hidden_layers, self.markov_head.dflare_norm
        )

    def embed_block(self, anchor_embeds: Tensor) -> Tensor:
        """Each sequence's block: its embedded anchor, then the mask rows.

        Args:
            anchor_embeds: The target's normalized embedding of each
                sequence's anchor, ``[seqs, hidden]``.

        Returns:
            ``[seqs * block_size, hidden]``, sequence by sequence.
        """
        config = self.config
        hidden, num_masks = config.hidden_size, config.block_size - 1
        anchors = anchor_embeds.cast(DType.bfloat16).reshape([-1, 1, hidden])
        masks = F.broadcast_to(
            self.mask_embed.cast(DType.bfloat16).reshape([1, 1, hidden]),
            [anchors.shape[0], num_masks, hidden],
        )
        return F.concat([anchors, masks], axis=1).reshape([-1, hidden])

    def forward(
        self,
        block_embeds: Tensor,
        kv_collections: Mapping[str, PagedCacheValues],
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the embedded blocks; returns their final-normed rows."""
        h = block_embeds
        for layer, key in zip(self.layers, self.layer_cache_keys, strict=True):
            h = layer(h, kv_collections[key], input_row_offsets)
        return self.norm(h)

    def draft(
        self,
        block_hidden: Tensor,
        anchors: Tensor,
        f_rnn: Tensor,
        lm_head: Tensor,
        *,
        final_logit_softcapping: float,
        unpadded_vocab_size: int,
        sampler: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        """Drafts ``block_size - 1`` tokens per sequence.

        The Markov head advances one token at a time from a zero state, its
        first step reading the anchor; each step's token is the argmax of the
        softcapped logits of its mask row plus the head's bias.

        Args:
            block_hidden: :meth:`forward`'s rows, ``[seqs * block, hidden]``.
            anchors: Each sequence's anchor token, ``[seqs]``.
            f_rnn: :meth:`f_rnn` of each sequence's last committed token.
            lm_head: The target's LM head weight, ``[vocab, hidden]``.
            final_logit_softcapping: The target's logit softcap.
            unpadded_vocab_size: The trained rows of the vocabulary; the
                padded tail is never drafted.
            sampler: Draws step ``i``'s token from its ``[seqs, vocab]``
                logits. The step takes an argmax when it is ``None``, and a
                sampled draft keeps every step's draw as it goes.

        Returns:
            The draft tokens, ``[seqs, block_size - 1]``.
        """
        config = self.config
        head = self.markov_head
        num_steps = config.block_size - 1
        rows = block_hidden.reshape([-1, config.block_size, config.hidden_size])
        state = F.broadcast_to(
            F.constant(0.0, DType.bfloat16, device=block_hidden.mapping),
            [rows.shape[0], config.markov_rank],
        )
        cap = final_logit_softcapping
        padded = config.vocab_size - unpadded_vocab_size
        masks = rows[:, 1:, :]
        flat = masks.reshape([-1, config.hidden_size])
        # Every step's base logits and ``[hidden, f_rnn]`` term at once; only
        # the state and the previous token are sequential.
        base_all = (flat @ lm_head.T).reshape([rows.shape[0], num_steps, -1])
        f_rnn_rows = F.broadcast_to(
            f_rnn.reshape([-1, 1, config.hidden_size]),
            [rows.shape[0], num_steps, config.hidden_size],
        ).reshape([-1, config.hidden_size])
        context_all = head.context(flat, f_rnn_rows).reshape(
            [rows.shape[0], num_steps, -1]
        )
        # ``hidden_out`` folded into the LM head, rounded to bfloat16 as the
        # reference folds it, so a step's bias reads ``rank`` columns per
        # token instead of the hidden width.
        out_weight = head.hidden_out.weight
        out_bias = head.hidden_out.bias
        assert isinstance(out_bias, Tensor)
        fold_weight = (out_weight.T @ lm_head.T).cast(DType.float32)
        fold_bias = (
            out_bias.reshape([1, config.hidden_size]) @ lm_head.T
        ).cast(DType.float32)
        tokens: list[Tensor] = []
        previous = anchors
        for step in range(num_steps):
            # Mask row ``step + 1`` predicts the token at its own position.
            state, latent = head.step(state, previous, context_all[:, step, :])
            bias = (latent.cast(DType.float32) @ fold_weight + fold_bias).cast(
                DType.bfloat16
            )
            # bf16(base + bias), softcapped in float32, rounded once more.
            # The softcap is elementwise, so each device caps its own
            # vocabulary shard and the gather moves bfloat16 instead of
            # float32.
            summed = (base_all[:, step, :] + bias).cast(DType.float32)
            logits = (cap * F.tanh(summed / cap)).cast(DType.bfloat16)
            if logits.is_distributed:
                logits = F.allgather(logits, tensor_axis=-1)
            if padded:
                logits = logits[:, :unpadded_vocab_size]
            picked = (
                F.argmax(logits, axis=-1)
                if sampler is None
                else sampler(logits, step)
            )
            previous = picked.reshape([-1])
            tokens.append(previous.reshape([-1, 1]))
        return F.concat(tokens, axis=1)
