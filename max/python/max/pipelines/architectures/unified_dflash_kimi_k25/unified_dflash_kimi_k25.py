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
"""Unified DFlash Kimi K2.5 nn.Module: MLA target + DFlash MHA/GQA draft."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..dflash_kimi_k25 import DFlashKimiK25
from ..unified_mtp_deepseekV3.unified_mtp_deepseekV3 import (
    compute_host_merged_offsets,
)
from .model_config import UnifiedDflashKimiK25Config

# Magic placeholder value used by graph capture for prefill / dummy-draft
# steps; tokens equal to this in every position of a row mean "no real
# draft prediction to verify".
_MAGIC_DRAFT_TOKEN_ID = 42


class UnifiedDflashKimiK25(Module):
    """Fused: merge -> target (MLA) -> reject -> materialize -> draft block."""

    def __init__(self, config: UnifiedDflashKimiK25Config) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.resolve_block_size()
        self.num_speculative_tokens = self.block_size - 1
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                config.speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
        )

        self.target = DeepseekV3(config.target)
        self.draft = DFlashKimiK25(config.draft)
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
    ) -> tuple[TensorValue, ...]:
        assert draft_kv_collections is not None

        devices = self.config.target.devices
        n_devs = len(devices)
        device0 = devices[0]
        K = self.block_size
        dp_degree = self.config.target.data_parallel_degree
        is_dp = dp_degree > 1

        pre_cache_lengths = [kv.cache_lengths for kv in draft_kv_collections]

        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
        merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
        )

        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
        )
        target_logits = target_outputs[1]
        target_hs_per_dev = list(target_outputs[3 : 3 + n_devs])

        seed_scalar = seed[0]
        num_accepted, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            target_logits,
            seed=seed_scalar,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
        )

        num_steps_scalar_i32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.int32
        )
        num_steps_scalar_u32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.uint32
        )
        zero_u32 = ops.constant(0, DType.uint32, device=device0)
        is_prefill = (num_steps_scalar_u32 == zero_u32).broadcast_to(
            ["batch_size"]
        )
        magic_token = ops.constant(
            _MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device0
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (draft_tokens == magic_token)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        is_dummy_draft = num_magic_tokens == num_steps_scalar_i32.broadcast_to(
            ["batch_size"]
        )
        num_accepted = ops.where(
            is_prefill | is_dummy_draft,
            ops.constant(0, num_accepted.dtype, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )
        prompt_lens = (input_row_offsets[1:] - input_row_offsets[:-1]).rebind(
            ["batch_size"]
        )
        decode_commit = (num_accepted + 1).cast(DType.uint32)
        commit_lengths = ops.where(is_prefill, prompt_lens, decode_commit)
        devices_per_replica = n_devs // dp_degree

        def _replica_sym(i: int) -> str:
            return f"replica_{i // devices_per_replica}_batch_size"

        commit_lengths_per_dev_raw = ops.distributed_broadcast(
            commit_lengths, signal_buffers
        )
        if is_dp:
            commit_lengths_per_dev = []
            for i in range(n_devs):
                start = data_parallel_splits[i]
                end = data_parallel_splits[i + 1]
                local = ops.slice_tensor(
                    commit_lengths_per_dev_raw[i],
                    [(slice(start, end), f"dflash_commit_split_{i}")],
                )
                local = ops.rebind(local, [_replica_sym(i)])
                commit_lengths_per_dev.append(local)
        else:
            commit_lengths_per_dev = [
                ops.rebind(c, [_replica_sym(i)])
                for i, c in enumerate(commit_lengths_per_dev_raw)
            ]

        target_tokens = ops.concat([recovered, bonus], axis=1)
        gather_idx = ops.where(
            is_prefill,
            ops.constant(0, DType.int64, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted.cast(DType.int64),
        )
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(gather_idx, axis=-1),
            batch_dims=1,
        )

        ctx_hidden = self.draft.project_target_hidden(target_hs_per_dev)

        ctx_kv_collections: list[PagedCacheValues] = [
            replace(kv, cache_lengths=pre)
            for kv, pre in zip(
                draft_kv_collections, pre_cache_lengths, strict=True
            )
        ]
        bumped = [
            pre + commit_lengths_per_dev[i]
            for i, pre in enumerate(pre_cache_lengths)
        ]
        block_kv_collections: list[PagedCacheValues] = [
            replace(kv, cache_lengths=b)
            for kv, b in zip(draft_kv_collections, bumped, strict=True)
        ]

        next_tokens_2d = ops.unsqueeze(next_tokens, axis=1)
        mask_const = ops.constant(
            self.mask_token_id, DType.int64, device=device0
        )
        mask_tail = mask_const.broadcast_to(["batch_size", K - 1])
        block_ids = ops.concat([next_tokens_2d, mask_tail], axis=1)
        block_ids_flat = block_ids.reshape((-1,))

        block_embeds_per_dev = self.target.embed_tokens(
            block_ids_flat, signal_buffers
        )

        block_indices = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device0,
            dtype=DType.uint32,
        )
        block_offsets = block_indices * ops.constant(
            K, DType.uint32, device=device0
        )

        block_offsets_per_dev = ops.distributed_broadcast(
            block_offsets, signal_buffers
        )
        if is_dp:
            ctx_input_row_offsets_per_dev: list[TensorValue] = []
            block_offsets_per_dev_list: list[TensorValue] = []
            block_embeds_per_dev_local: list[TensorValue] = []
            for i in range(n_devs):
                start = data_parallel_splits[i]
                end = data_parallel_splits[i + 1]
                start_offset = merged_offsets_per_dev[i][start]
                ctx_offsets_local = (
                    ops.slice_tensor(
                        merged_offsets_per_dev[i],
                        [
                            (
                                slice(start, end + 1),
                                f"dflash_ctx_offset_split_{i}",
                            )
                        ],
                    )
                    - start_offset
                )
                ctx_input_row_offsets_per_dev.append(ctx_offsets_local)

                block_offsets_dev = block_offsets_per_dev[i]
                block_start_offset = block_offsets_dev[start]
                block_offsets_local = (
                    ops.slice_tensor(
                        block_offsets_dev,
                        [
                            (
                                slice(start, end + 1),
                                f"dflash_block_offset_split_{i}",
                            )
                        ],
                    )
                    - block_start_offset
                )
                block_offsets_per_dev_list.append(block_offsets_local)

                token_start = start * K
                token_end = end * K
                block_embeds_local = ops.slice_tensor(
                    block_embeds_per_dev[i],
                    [
                        (
                            slice(token_start, token_end),
                            f"dflash_block_embed_split_{i}",
                        )
                    ],
                )
                block_embeds_per_dev_local.append(block_embeds_local)

            ctx_input_row_offsets_arg = ctx_input_row_offsets_per_dev
            block_input_row_offsets_arg = block_offsets_per_dev_list
            block_embeds_arg = block_embeds_per_dev_local
        else:
            ctx_input_row_offsets_arg = merged_offsets_per_dev
            block_input_row_offsets_arg = block_offsets_per_dev
            block_embeds_arg = block_embeds_per_dev

        block_hs_per_dev = self.draft(
            block_embeds=block_embeds_arg,
            ctx_hidden=ctx_hidden,
            signal_buffers=signal_buffers,
            ctx_kv_collections=ctx_kv_collections,
            block_kv_collections=block_kv_collections,
            ctx_input_row_offsets=ctx_input_row_offsets_arg,
            block_input_row_offsets=block_input_row_offsets_arg,
        )

        hidden_size = self.config.draft.hidden_size
        block_hs_2d_per_dev: list[TensorValue] = []
        for hs in block_hs_per_dev:
            n = hs.shape[0]
            hs_rebound = hs.rebind([(n // K) * K, hidden_size])
            hs_3d = hs_rebound.reshape([n // K, K, hidden_size])
            block_hs_2d_per_dev.append(hs_3d[:, 1:, :])
        if is_dp:
            block_hs_2d_per_dev = ops.allgather(
                block_hs_2d_per_dev, signal_buffers, axis=0
            )
        draft_logits = self.target.lm_head(block_hs_2d_per_dev, signal_buffers)[
            0
        ]
        argmax_out = ops.argmax(draft_logits, axis=-1)
        argmax_out = argmax_out.rebind(["batch_size", K - 1, 1])
        next_draft_tokens = argmax_out.reshape(("batch_size", K - 1))

        num_accepted_out = ops.where(
            is_prefill,
            ops.constant(0, num_accepted.dtype, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        return (num_accepted_out, next_tokens, next_draft_tokens)

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types mirror :class:`Eagle3MHAKimiK25Unified.input_types`.

        Order:
            tokens, device_offsets, host_offsets, return_n_logits,
            data_parallel_splits, signal_buffers, target_kv_cache (flat),
            batch_context_lengths, target_ep_inputs, draft_tokens,
            draft_kv_blocks (one per device), seed, temperature, top_k,
            max_k, top_p, min_top_p.
        """
        devices = self.config.target.devices
        device_ref = devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )
        draft_tokens_type = TensorType(
            DType.int64,
            ["batch_size", "num_steps"],
            device=device_ref,
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=[self.config.target.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        signals = Signals(devices=devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)

        assert isinstance(kv_params, KVCacheParams)
        all_input_types.extend(
            kv_params.get_symbolic_inputs(
                draft_attention_group=draft_kv_params
            ).flatten()
        )

        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.target.ep_manager is not None:
            all_input_types.extend(self.target.ep_manager.input_types())

        all_input_types.append(draft_tokens_type)

        for sym in draft_kv_params.get_symbolic_inputs().inputs:
            assert isinstance(sym, KVCacheInputsPerDevice)
            all_input_types.append(sym.kv_blocks)

        seed_type = TensorType(
            DType.uint64, shape=["batch_size"], device=device_ref
        )
        temperature_type = TensorType(
            DType.float32, shape=["batch_size"], device=device_ref
        )
        top_k_type = TensorType(
            DType.int64, shape=["batch_size"], device=device_ref
        )
        max_k_type = TensorType(DType.int64, shape=[], device=DeviceRef.CPU())
        top_p_type = TensorType(
            DType.float32, shape=["batch_size"], device=device_ref
        )
        min_top_p_type = TensorType(
            DType.float32, shape=[], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [
                seed_type,
                temperature_type,
                top_k_type,
                max_k_type,
                top_p_type,
                min_top_p_type,
            ]
        )

        return tuple(all_input_types)
