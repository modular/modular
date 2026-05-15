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
"""Eagle3 MHA draft model for Kimi K2.5 (target uses MLA).

This is the Llama-style MHA counterpart to :class:`Eagle3KimiK25`. The target
remains DeepseekV3 (MLA) and produces per-device hidden states; the draft
swaps in a single MHA decoder block whose KV cache geometry is independent of
the target's. The unified graph wires a separate ``PagedCacheValues`` per
device for this draft.

The draft fuses two or three captured target hidden states via ``fc`` (width
detected from the checkpoint) and concatenates the result with the token
embedding before a single MHA attention block. Layout matches
``EagleLlama3``'s 2-way fusion when the draft checkpoint emits a
[hidden*2, hidden] ``fc.weight``, and matches the existing
``Eagle3KimiK25`` 3-way fusion otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.attention_with_rope import (
    AttentionWithRope,
    DataParallelAttentionWithRope,
    TensorParallelAttentionWithRope,
)
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    extract_hs,
    forward_sharded_layers,
)


@dataclass(kw_only=True)
class Eagle3MHAKimiK25DraftConfig:
    """Minimal config for an Eagle3 MHA draft over a Kimi K2.5 (MLA) target.

    Held separate from ``DeepseekV3Config`` so MLA-specific fields
    (``kv_lora_rank``, ``v_head_dim``, etc.) and validators don't apply to
    the MHA draft.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    devices: list[DeviceRef]
    data_parallel_degree: int
    dtype: DType
    norm_dtype: DType
    kv_params: KVCacheParams
    rope_scaling: dict[str, Any]
    """Yarn rope scaling params (Deepseek-flavored: beta_fast, beta_slow,
    mscale, mscale_all_dim, factor, original_max_position_embeddings)."""

    fc_input_multiplier: int
    """Number of fused target hidden states (2 or 3). Set from the
    ``fc.weight`` shape in the draft checkpoint at load time."""

    sliding_window: int | None = None
    """If set, the draft attention uses a sliding-window causal mask of
    this size (in tokens). ``None`` keeps the default full causal mask."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE


class Eagle3MHAKimiK25(Module):
    """Eagle3 MHA draft over the Kimi K2.5 (MLA) target.

    The ``__call__`` contract mirrors :class:`Eagle3KimiK25` so the unified
    graph can swap drafts without changing the call site:

    .. code-block:: python

        draft(
            tokens,
            fused_target_hs,        # per-device list[TensorValue]
            signal_buffers,
            kv_collections,         # per-device list[PagedCacheValues]
            return_n_logits,
            input_row_offsets,      # per-device list[TensorValue]
            host_input_row_offsets,
            data_parallel_splits,
            batch_context_lengths,
            split_prefix=...,
        )
    """

    def __init__(self, config: Eagle3MHAKimiK25DraftConfig) -> None:
        super().__init__()
        self.config = config
        devices = config.devices
        num_devices = len(devices)
        device0 = devices[0]
        dtype = config.dtype
        norm_dtype = config.norm_dtype

        self.use_tp_ep = config.data_parallel_degree == 1 and num_devices > 1
        self.use_data_parallel_attention = (
            num_devices > 1 and config.data_parallel_degree == num_devices
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            devices=devices,
            quantization_encoding=None,
        )

        # fc: fuses the captured target hidden states.
        # fc_input_multiplier=2  → [seq, 2*hidden] -> [seq, hidden]
        # fc_input_multiplier=3  → [seq, 3*hidden] -> [seq, hidden]
        # The Llama-style EAGLE3 checkpoints we target ship a 2-way fc
        # (Kimi's existing MLA draft ships a 3-way fc); pipeline_model
        # detects which one and passes the multiplier.
        self.fc = Linear(
            config.hidden_size * config.fc_input_multiplier,
            config.hidden_size,
            dtype,
            device0,
            quantization_encoding=None,
            has_bias=False,
        )
        self.fc.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.fc_shards = self.fc.shard(devices)

        # Deepseek-flavored yarn rope (matches the draft HF config's
        # rope_scaling block; the per-head q/k dim is the MHA head_dim).
        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        self.rope = DeepseekYarnRotaryEmbedding(
            config.head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
            interleaved=False,
        )

        wide_hidden_size = config.hidden_size * 2
        attn_kwargs: dict[str, Any] = dict(
            rope=self.rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=wide_hidden_size,
            kv_params=config.kv_params,
            devices=devices,
            dtype=dtype,
            has_bias=False,
            sliding_window=config.sliding_window,
        )
        if self.use_data_parallel_attention:
            self.self_attn: AttentionWithRope = DataParallelAttentionWithRope(
                **attn_kwargs
            )
        elif num_devices > 1:
            self.self_attn = TensorParallelAttentionWithRope(**attn_kwargs)
        else:
            self.self_attn = AttentionWithRope(**attn_kwargs)

        # Replacement o_proj: [n_heads * head_dim] -> [hidden_size]
        # (NOT wide_hidden_size). Same trick as ``Eagle3KimiK25``: the
        # attention reads a wide concat but the residual addition runs on
        # the regular hidden_size, so the output must be narrow.
        q_weight_dim = config.num_attention_heads * config.head_dim
        replacement_o_proj = Linear(
            q_weight_dim,
            config.hidden_size,
            dtype,
            device0,
            quantization_encoding=None,
        )
        if self.use_tp_ep:
            replacement_o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices,
                    config.num_attention_heads,
                    config.head_dim,
                )
            )
        else:
            replacement_o_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        o_proj_shards = replacement_o_proj.shard(devices)
        self.self_attn.o_proj = replacement_o_proj
        shards_seq = getattr(
            self.self_attn,
            "replicated_attentions",
            None,
        )
        if shards_seq is None:
            shards_seq = getattr(self.self_attn, "list_of_attentions", None)
        if shards_seq is not None:
            for shard_idx, attn_shard in enumerate(shards_seq):
                attn_shard.o_proj = o_proj_shards[shard_idx]

        def _replicated_rmsnorm() -> RMSNorm:
            n = RMSNorm(
                config.hidden_size,
                norm_dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            )
            n.sharding_strategy = ShardingStrategy.replicate(num_devices)
            return n

        self.input_layernorm = _replicated_rmsnorm()
        self.input_layernorm_shards = self.input_layernorm.shard(devices)

        self.hidden_norm = _replicated_rmsnorm()
        self.hidden_norm_shards = self.hidden_norm.shard(devices)

        self.post_attention_layernorm = _replicated_rmsnorm()
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        self.mlp = MLP(
            dtype=dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            devices=devices,
            quant_config=None,
        )
        if self.use_tp_ep:
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.mlp_shards = list(self.mlp.shard(devices))

        self.norm = _replicated_rmsnorm()
        self.norm_shards = self.norm.shard(devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype,
            devices=devices,
            quantization_encoding=None,
        )

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        fused_target_hs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        split_prefix: str = "eagle3_mha_draft",
    ) -> tuple[TensorValue, ...]:
        """Forward pass.

        Args mirror :meth:`Eagle3KimiK25.__call__` so the unified module can
        switch implementations without changing the call site.
        """
        config = self.config
        devices = config.devices
        num_devices = len(devices)

        fused_hs: list[TensorValue] = list(fused_target_hs)
        if fused_hs[0].shape[-1] != config.hidden_size:
            fused_hs = forward_sharded_layers(self.fc_shards, fused_hs)

        h_embed = self.embed_tokens(tokens, signal_buffers)

        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        input_row_offsets_ = list(input_row_offsets)

        if self.use_data_parallel_attention:
            host_offsets_i64 = host_input_row_offsets.cast(DType.int64)
            h_embed, input_row_offsets_ = split_batch_replicated(
                list(devices),
                h_embed,
                input_row_offsets_,
                host_offsets_i64,
                data_parallel_splits,
                prefix=split_prefix,
            )
            h_embed = [
                ops.rebind(
                    h_embed[i],
                    [f"{split_prefix}_seq_dev_{i}", config.hidden_size],
                )
                for i in range(num_devices)
            ]
            fused_hs = [
                ops.rebind(
                    fused_hs[i],
                    [f"{split_prefix}_seq_dev_{i}", config.hidden_size],
                )
                for i in range(num_devices)
            ]
        else:
            common_dim = f"{split_prefix}_seq_len"
            h_embed = [
                ops.rebind(h_embed[i], [common_dim, config.hidden_size])
                for i in range(num_devices)
            ]
            fused_hs = [
                ops.rebind(fused_hs[i], [common_dim, config.hidden_size])
                for i in range(num_devices)
            ]

        norm_embed = forward_sharded_layers(
            self.input_layernorm_shards, h_embed
        )
        norm_fused = forward_sharded_layers(self.hidden_norm_shards, fused_hs)
        concat_inputs = [
            ops.concat([norm_embed[i], norm_fused[i]], axis=-1)
            for i in range(num_devices)
        ]

        layer_idx_cpu = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        attn_outs: list[TensorValue]
        if self.use_data_parallel_attention:
            assert isinstance(self.self_attn, DataParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx_cpu,
                concat_inputs,
                kv_collections,
                freqs_cis,
                input_row_offsets_,
            )
        elif self.use_tp_ep:
            assert isinstance(self.self_attn, TensorParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx_cpu,
                concat_inputs,
                signal_buffers,
                kv_collections,
                freqs_cis,
                input_row_offsets_,
            )
        else:
            single_out = self.self_attn(
                layer_idx_cpu,
                concat_inputs[0],
                kv_collections[0],
                freqs_cis[0],
                input_row_offsets_[0],
            )
            attn_outs = [single_out]

        hs = [
            fused + attn_out
            for fused, attn_out in zip(fused_hs, attn_outs, strict=True)
        ]

        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
        if self.use_tp_ep:
            mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        if config.data_parallel_degree > 1:
            last_token_per_dev: list[TensorValue] = []
            for dev_idx in range(num_devices):
                h0 = hs[dev_idx]
                last_token_indices = input_row_offsets_[dev_idx][1:] - 1
                last_token_per_dev.append(
                    ops.gather(h0, last_token_indices, axis=0)
                )
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            last_token_distributed = [
                ops.gather(h_i, offsets_i[1:] - 1, axis=0)
                for h_i, offsets_i in zip(hs, input_row_offsets_, strict=True)
            ]

        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        ret_val: tuple[TensorValue, ...] = (last_logits,)

        if self.return_logits == ReturnLogits.VARIABLE:
            draft_return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="draft_mha_return_n_logits_range",
                dtype=DType.int64,
                device=devices[0],
            )
            if self.use_data_parallel_attention:
                draft_return_n_logits_range_per_dev = ops.distributed_broadcast(
                    draft_return_n_logits_range, signal_buffers
                )
                variable_per_dev: list[TensorValue] = []
                for dev_idx in range(num_devices):
                    dev_offsets = (
                        ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                        - draft_return_n_logits_range_per_dev[dev_idx]
                    )
                    variable_per_dev.append(
                        ops.gather(
                            hs[dev_idx],
                            ops.reshape(dev_offsets, shape=(-1,)),
                            axis=0,
                        )
                    )
                variable_distributed = ops.allgather(
                    variable_per_dev, signal_buffers
                )
                norm_variable = forward_sharded_layers(
                    self.norm_shards, variable_distributed
                )
                variable_logits = ops.cast(
                    self.lm_head(norm_variable, signal_buffers)[0],
                    DType.float32,
                )
            else:
                last_offsets = (
                    ops.unsqueeze(input_row_offsets_[0][1:], -1)
                    - draft_return_n_logits_range
                )
                last_indices = ops.reshape(last_offsets, shape=(-1,))
                variable_logits = ops.gather(
                    ops.cast(
                        self.lm_head(
                            forward_sharded_layers(self.norm_shards, hs),
                            signal_buffers,
                        )[0],
                        DType.float32,
                    ),
                    last_indices,
                    axis=0,
                )
            logit_offsets = ops.range(
                0,
                TensorValue(variable_logits.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="draft_mha_logit_offsets",
                dtype=DType.int64,
                device=devices[0],
            )
            ret_val += (variable_logits, logit_offsets)

        ret_val += extract_hs(
            return_hidden_states=self.return_hidden_states,
            last_token_hs_distributed=last_token_distributed,
            all_hs_distributed=hs,
            normalizer=self.norm_shards,
        )

        return ret_val
