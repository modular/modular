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
"""DeepSeek-V4.1 DSpark drafter, which proposes tokens for speculative decoding.

The drafter is a short stack of decoder stages that reads hidden states taken
from the target model's trunk and drafts a block of tokens per request. It
shares the target's token embedding and LM head, and a low-rank Markov head
adds a bigram bias to each draft position's logits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from typing_extensions import Self

from ..deepseekV4.layers import expand_copies
from .deepseekV41 import (
    DeepseekV41,
    DeepseekV41Block,
    DeepseekV41Trunk,
    HCStreams,
    ReplicaBatch,
    VisionMerge,
)
from .layer_resolver import LayerInfo
from .layer_resolver import resolve as resolve_layers
from .layers.attention import CSA2Cache
from .layers.dspark import (
    DeepseekV41DSparkAttention,
    DSparkMarkovHead,
    DSparkWindow,
    dspark_main_proj_quant_config,
)
from .layers.dspark_moe import DeepseekV41DSparkMoE
from .layers.ragged import ReplicaTokens
from .layers.ragged import V41RaggedRows as RaggedRows
from .model_config import DeepseekV41Config, position_capacity
from .weight_adapters import DSPARK_MODULE_PREFIX


def drafter_layer_infos(config: DeepseekV41Config) -> list[LayerInfo]:
    """Returns the :class:`LayerInfo` of each drafter stage.

    Drafter stages are the layers with ``layer_id >= num_hidden_layers``.

    Raises:
        ValueError: If the number of drafter stages is not
            ``config.num_dspark_layers``.
    """
    rows = [row for row in resolve_layers(config) if row.side == "drafter"]
    if len(rows) != config.num_dspark_layers:
        raise ValueError(
            f"config.compress_ratios resolves {len(rows)} DSpark stages, but "
            f"num_dspark_layers is {config.num_dspark_layers}"
        )
    return rows


def _greedy(logits: TensorValue, step: int) -> TensorValue:
    del step
    return ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)


class DeepseekV41DSparkDrafter(Module):
    """DSpark decoder stages plus their input projection and heads.

    A draft step stores the context KV of newly verified tokens
    (:meth:`store_context`), reads each stage's context window
    (:meth:`windows`), runs the draft blocks (:meth:`forward_block`), and
    turns the result into draft tokens (:meth:`logits`, then :meth:`draft`).
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        if not config.dspark_target_layer_ids:
            raise ValueError("DSpark needs dspark_target_layer_ids")
        self.config = config
        self.block_size = config.dspark_block_size
        max_seq_len = (
            position_capacity(config) if max_seq_len is None else max_seq_len
        )
        hidden = config.hidden_size
        n_taps = len(config.dspark_target_layer_ids)

        self.main_proj = Linear(
            in_dim=hidden * n_taps,
            out_dim=hidden,
            dtype=DType.float8_e4m3fn,
            device=device,
            quant_config=dspark_main_proj_quant_config(),
        )
        self.main_norm = RMSNorm(hidden, DType.float32, config.rms_norm_eps)
        self.layer_infos = drafter_layer_infos(config)
        self.stages = LayerList(
            [
                DeepseekV41Block(
                    config,
                    row,
                    device,
                    attn=DeepseekV41DSparkAttention(
                        config, row, device, max_seq_len
                    ),
                    ffn=DeepseekV41DSparkMoE(config, device),
                )
                for row in self.layer_infos
            ]
        )
        self.norm = RMSNorm(hidden, DType.float32, config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            vocab_size=config.vocab_size,
            draft_vocab_size=config.vocab_size,
            markov_rank=config.dspark_markov_rank,
            device=device,
        )

    def _attentions(self) -> list[DeepseekV41DSparkAttention]:
        attentions = []
        for stage in self.stages:
            assert isinstance(stage, DeepseekV41Block)
            assert isinstance(stage.attn, DeepseekV41DSparkAttention)
            attentions.append(stage.attn)
        return attentions

    def combine(self, taps: TensorValue) -> TensorValue:
        """Projects the trunk taps to the hidden size and normalizes them.

        Args:
            taps: ``[..., hidden * n_taps]``, the trunk hidden states the
                drafter reads (one per target layer), concatenated.

        Returns:
            ``[..., hidden]``.
        """
        return self.main_norm(self.main_proj(taps))

    def store_context(
        self,
        taps: TensorValue,
        rows: RaggedRows,
        swa_collection: PagedCacheValues,
    ) -> None:
        """Writes every stage's context KV for a target forward's tokens.

        Each stage writes to its own layer slot of ``swa_collection``.

        Args:
            taps: ``[1, T, hidden * n_taps]``, the trunk taps of those tokens.
            rows: Ragged-batch bookkeeping for those tokens; ``starts`` are
                the cache lengths before the forward.
            swa_collection: The sliding-window KV cache.
        """
        main_x = self.combine(taps)
        for attn in self._attentions():
            attn.store_context(main_x, rows, swa_collection)

    def windows(
        self, starts: TensorValue, swa_collection: PagedCacheValues
    ) -> list[DSparkWindow]:
        """Names every stage's :class:`DSparkWindow` in the KV cache.

        Args:
            starts: ``[b]`` int32 start position of each draft block.
            swa_collection: The sliding-window KV cache.
        """
        return [
            attn.read_window(starts, swa_collection)
            for attn in self._attentions()
        ]

    def shard(self, num_devices: int, *, split_heads: bool = True) -> None:
        """Declares the stages' split across devices.

        Routed experts split, and with ``split_heads`` (tensor parallelism)
        attention heads too; everything else (``main_proj``, norms,
        hyper-connections, shared experts, Markov head) is replicated.
        """
        for stage in self.stages:
            assert isinstance(stage, DeepseekV41Block)
            assert isinstance(stage.attn, DeepseekV41DSparkAttention)
            assert isinstance(stage.ffn, DeepseekV41DSparkMoE)
            if split_heads:
                stage.attn.shard_heads(num_devices)
            stage.ffn.shard_experts(num_devices)

    def keep_local(
        self, num_devices: int, rank: int, *, split_heads: bool = True
    ) -> None:
        """Keeps only this replica's share of the split from :meth:`shard`."""
        for stage in self.stages:
            assert isinstance(stage, DeepseekV41Block)
            assert isinstance(stage.attn, DeepseekV41DSparkAttention)
            assert isinstance(stage.ffn, DeepseekV41DSparkMoE)
            if split_heads:
                stage.attn.keep_local_heads(num_devices)
            stage.ffn.keep_local_experts(num_devices, rank)

    def forward_block(
        self,
        block_ids: TensorValue,
        rows: RaggedRows,
        windows: Sequence[DSparkWindow],
        embed: Callable[[TensorValue], TensorValue],
    ) -> TensorValue:
        """Runs ``b`` draft blocks through the stages.

        Args:
            block_ids: ``[1, b * block_size]`` int32, each request's anchor
                then ``block_size - 1`` ``dspark_noise_token_id``s.
            rows: The blocks' bookkeeping (``block_size`` rows per request,
                positions ``L_r .. L_r + block_size - 1``).
            windows: One :class:`DSparkWindow` per stage, in stage order.
            embed: The target's token embedding.

        Returns:
            The ``[1, b * block_size, hidden]`` hidden state, with the
            hyper-connection copies merged into one stream, before the final
            norm.

        Raises:
            ValueError: If ``windows`` does not have one entry per stage.
        """
        if len(windows) != len(self.stages):
            raise ValueError(
                f"expected {len(self.stages)} stage windows, got {len(windows)}"
            )
        streams = HCStreams(
            expand_copies(embed(block_ids), self.config.hc_mult)
        )
        for stage, window in zip(self.stages, windows, strict=True):
            streams, _ = stage(streams, rows, block_ids, window)
        return streams.collapse()

    def __call__(
        self,
        block_ids: TensorValue,
        rows: RaggedRows,
        windows: Sequence[DSparkWindow],
        embed: Callable[[TensorValue], TensorValue],
    ) -> TensorValue:
        """Runs :meth:`forward_block`."""
        return self.forward_block(block_ids, rows, windows, embed)

    @staticmethod
    def forward_block_tensor_parallel(
        drafters: Sequence[DeepseekV41DSparkDrafter],
        block_ids: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        windows: Sequence[Sequence[DSparkWindow]],
        embeds: Sequence[Callable[[TensorValue], TensorValue]],
        signal_buffers: Sequence[BufferValue],
    ) -> TensorValue:
        """Runs :meth:`forward_block` with one drafter replica per device.

        Each replica holds its share from :meth:`shard` and
        :meth:`keep_local`. Every argument but ``signal_buffers`` is per
        device; ``windows[i]`` is device ``i``'s stage windows. Attention and
        expert partial sums are all-reduced; the hyper-connection mixing runs
        in full on every device.

        Returns:
            Device 0's merged ``[1, b * block_size, hidden]`` hidden state
            (identical on every device).
        """
        lead = drafters[0]
        streams = [
            HCStreams(expand_copies(embed(ids), lead.config.hc_mult))
            for embed, ids in zip(embeds, block_ids, strict=True)
        ]
        for i in range(len(lead.stages)):
            stages = []
            for drafter in drafters:
                stage = drafter.stages[i]
                assert isinstance(stage, DeepseekV41Block)
                stages.append(stage)
            streams, _ = DeepseekV41Block.tensor_parallel(
                stages,
                streams,
                rows,
                block_ids,
                [per_dev[i] for per_dev in windows],
                signal_buffers,
            )
        return streams[0].collapse()

    @staticmethod
    def forward_block_data_parallel(
        drafters: Sequence[DeepseekV41DSparkDrafter],
        block_ids: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        windows: Sequence[Sequence[DSparkWindow]],
        embeds: Sequence[Callable[[TensorValue], TensorValue]],
        signal_buffers: Sequence[BufferValue],
        replica_tokens: ReplicaTokens,
    ) -> list[TensorValue]:
        """Runs :meth:`forward_block` with each device on its own blocks.

        Each replica holds its share from :meth:`shard` and
        :meth:`keep_local` without ``split_heads``. Every argument but
        ``signal_buffers`` and ``replica_tokens`` (where each device's block
        rows sit among all the blocks) is per device and covers only that
        device's blocks. The stages' MoE is expert-parallel over every
        device's blocks (:meth:`DeepseekV41Block.data_parallel`).

        Returns:
            Per device, its merged ``[1, b_i * block_size, hidden]`` hidden
            state.
        """
        lead = drafters[0]
        all_ids = ops.allgather(
            [ops.reshape(ids, [-1]) for ids in block_ids],
            signal_buffers,
            axis=0,
        )
        batch_ids = [
            ops.reshape(ops.rebind(ids, [replica_tokens.total]), [1, -1])
            for ids in all_ids
        ]
        streams = [
            HCStreams(expand_copies(embed(ids), lead.config.hc_mult))
            for embed, ids in zip(embeds, block_ids, strict=True)
        ]
        for i in range(len(lead.stages)):
            stages = []
            for drafter in drafters:
                stage = drafter.stages[i]
                assert isinstance(stage, DeepseekV41Block)
                stages.append(stage)
            streams, _ = DeepseekV41Block.data_parallel(
                stages,
                streams,
                rows,
                batch_ids,
                [per_dev[i] for per_dev in windows],
                signal_buffers,
                replica_tokens,
            )
        return [stream.collapse() for stream in streams]

    def logits(
        self, hidden: TensorValue, head_weight: TensorValue
    ) -> TensorValue:
        """Computes the draft logits before the Markov bias.

        Args:
            hidden: :meth:`forward_block`'s output.
            head_weight: The target's ``[vocab, hidden]`` LM head weight.

        Returns:
            ``[b * block_size, vocab]`` float32, request major.
        """
        normed = self.norm(ops.reshape(hidden, [-1, self.config.hidden_size]))
        if head_weight.dtype == DType.bfloat16:
            # Compute in bfloat16 and widen afterwards, so the draft logits
            # round the same way as the target's bfloat16 LM head.
            return ops.cast(
                ops.matmul(
                    ops.cast(normed, DType.bfloat16),
                    ops.transpose(head_weight, 0, 1),
                ),
                DType.float32,
            )
        return ops.matmul(
            ops.cast(normed, DType.float32),
            ops.transpose(ops.cast(head_weight, DType.float32), 0, 1),
        )

    def draft(
        self,
        logits: TensorValue,
        anchor_ids: TensorValue,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Draws ``block_size`` draft tokens per request, one at a time.

        Step ``i``'s logits get the bigram bias of the token drawn at step
        ``i - 1`` (the anchor at step 0); the loop unrolls over the static
        ``block_size``.

        Args:
            logits: ``[b * block_size, vocab]``, :meth:`logits`' output.
            anchor_ids: ``[b]`` int32, each block's first token.
            sampler: Draws step ``i``'s ``[b]`` int32 tokens from its
                ``[b, vocab]`` biased logits; greedy argmax when ``None``.

        Returns:
            ``([b, block_size]`` int32 draft ids, ``[b, block_size, vocab]``
            float32 biased logits).
        """
        sample = _greedy if sampler is None else sampler
        batch = anchor_ids.shape[0]
        vocab = logits.shape[-1]
        logits = ops.reshape(
            ops.rebind(logits, [batch * self.block_size, vocab]),
            [batch, self.block_size, vocab],
        )
        current = ops.cast(anchor_ids, DType.int32)
        drafted = []
        biased = []
        for i in range(self.block_size):
            step = logits[:, i] + self.markov_head.bias_float32(current)
            biased.append(step)
            current = ops.cast(sample(step, i), DType.int32)
            drafted.append(current)
        return ops.stack(drafted, axis=1), ops.stack(biased, axis=1)


def block_ids_for(
    anchor_ids: TensorValue, block_size: int, noise_token_id: int
) -> TensorValue:
    """Builds the draft block token ids from each request's anchor token.

    Args:
        anchor_ids: ``[b]``, the first token of each request's block.
        block_size: Tokens per draft block.
        noise_token_id: The placeholder token filling the rest of a block.

    Returns:
        ``[1, b * block_size]`` int32: each anchor followed by
        ``block_size - 1`` copies of ``noise_token_id``.
    """
    device = anchor_ids.device
    batch = anchor_ids.shape[0]
    anchors = ops.unsqueeze(ops.cast(anchor_ids, DType.int32), -1)
    noise = ops.broadcast_to(
        ops.constant(np.int32(noise_token_id), DType.int32, device),
        [batch, block_size - 1],
    )
    return ops.reshape(ops.concat([anchors, noise], axis=-1), [1, -1])


class DeepseekV41WithDSpark(DeepseekV41):
    """:class:`~.deepseekV41.DeepseekV41` with the DSpark drafter attached.

    The trunk, norm and head keep their non-speculative names; the drafter
    sits under the ``dspark`` attribute.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef | None = None,
        *,
        engram_host_tables: Mapping[int, tuple[TensorValue, TensorValue]],
        engram_token_map: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            config,
            device,
            engram_host_tables=engram_host_tables,
            engram_token_map=engram_token_map,
        )
        self.dspark = DeepseekV41DSparkDrafter(config, self.device)
        assert DSPARK_MODULE_PREFIX == "dspark"  # this attribute's name

    def tensor_parallel_replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
    ) -> list[Self]:
        """Builds one replica per device, splitting the drafter as well.

        Call it once, after :meth:`load_state_dict`.
        """
        n = len(devices)
        self.dspark.shard(n)
        replicas = super().tensor_parallel_replicas(devices, engram_host_tables)
        for rank, replica in enumerate(replicas):
            replica.dspark.keep_local(n, rank)
        return replicas

    def data_parallel_replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
    ) -> list[Self]:
        """Builds one data-parallel replica per device, drafter included.

        The drafter splits only its routed experts. Call it once, after
        :meth:`load_state_dict`.
        """
        n = len(devices)
        self.dspark.shard(n, split_heads=False)
        replicas = super().data_parallel_replicas(devices, engram_host_tables)
        for rank, replica in enumerate(replicas):
            replica.dspark.keep_local(n, rank, split_heads=False)
        return replicas

    @staticmethod
    def trunk_with_taps_data_parallel(
        replicas: Sequence[DeepseekV41WithDSpark],
        split: ReplicaBatch,
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        visions: Sequence[VisionMerge] | None = None,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Runs :meth:`trunk_with_taps` with each device on its own requests.

        ``split`` is the batch cut per device
        (:meth:`DeepseekV41.split_replicas`), and ``visions`` each device's
        own image rows (:meth:`VisionMerge.for_rows`). Returns each device's
        pre-norm ``[1, t_i, d]`` stream and its taps.
        """
        xs, taps = DeepseekV41Trunk.data_parallel_with_taps(
            [replica.trunk for replica in replicas],
            split.tokens,
            split.batch_tokens,
            split.rows,
            caches,
            signal_buffers,
            split.replica_tokens,
            tap_layer_ids=replicas[0].config.dspark_target_layer_ids,
            engram_operands=DeepseekV41._engram_operands_data_parallel(
                replicas, split
            ),
            visions=visions,
        )
        assert taps is not None
        return xs, taps

    @staticmethod
    def trunk_with_taps_tensor_parallel(
        replicas: Sequence[DeepseekV41WithDSpark],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        engram_lookback_ids: Sequence[TensorValue] | None = None,
    ) -> tuple[TensorValue, list[TensorValue]]:
        """Runs :meth:`trunk_with_taps` with one replica per device.

        Every argument but ``signal_buffers`` is per device. Returns device
        0's pre-norm ``[1, s, d]`` stream and each device's taps.
        """
        x, taps = DeepseekV41Trunk.tensor_parallel_with_taps(
            [replica.trunk for replica in replicas],
            tokens,
            rows,
            caches,
            signal_buffers,
            tap_layer_ids=replicas[0].config.dspark_target_layer_ids,
            engram_operands=DeepseekV41._engram_operands_tensor_parallel(
                replicas, tokens, rows, engram_lookback_ids, signal_buffers
            ),
        )
        assert taps is not None
        return x, taps

    def trunk_with_taps(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        engram_lookback_ids: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Runs the trunk and also returns the hidden states the drafter reads.

        Returns:
            The pre-norm ``[1, s, d]`` stream, and the ``[1, s, d * n_taps]``
            taps: the stream entering each layer in
            ``config.dspark_target_layer_ids``, averaged over its
            hyper-connection copies and concatenated.
        """
        operands = self._engram_operands(tokens, rows, engram_lookback_ids)
        x, taps = self.trunk.forward_with_taps(
            tokens,
            rows,
            cache,
            tap_layer_ids=self.config.dspark_target_layer_ids,
            engram_operands=operands,
        )
        assert taps is not None
        return x, taps
