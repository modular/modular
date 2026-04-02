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
"""Eagle3 + Kimi K2.5 fused nn.Module: merge + target + rejection + shift."""

from __future__ import annotations

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
from max.kv_cache.paged_kv_cache.increment_cache_lengths import (
    ragged_increment_cache_lengths,
)
from max.nn.comm import Signals
from max.nn.kernels import eagle_prefill_shift_tokens
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.kv_cache.input_types import PagedCacheInputSymbols
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    _reshape_target_logits,
    greedy_acceptance_sampler,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    RaggedTokenMerger,
)

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from .eagle3_kimi_k25 import Eagle3KimiK25


class Eagle3KimiK25Unified(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift.

    The target model returns concatenated hidden states from 3 intermediate
    layers (first, middle, last). The draft model fuses these via ``fc`` and
    generates the next speculative token.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.target = DeepseekV3(config)
        self.merger = RaggedTokenMerger(config.devices[0])

        self.draft: Eagle3KimiK25 | None = None
        if draft_config is not None:
            self.draft = Eagle3KimiK25(draft_config)

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
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        draft_signal_buffers: list[BufferValue] | None = None,
    ) -> tuple[TensorValue, ...]:
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )

        host_merged_offsets = merged_offsets.cast(DType.uint32).to(
            DeviceRef.CPU()
        )

        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            kv_collections,
            return_n_logits,
            merged_offsets,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
        )
        logits = target_outputs[1]
        hidden_states = target_outputs[3]

        first_rejected, recovered, bonus = greedy_acceptance_sampler(
            draft_tokens, logits
        )

        # Compute next_tokens: target argmax at the first rejected position.
        # concat([recovered, bonus]) gives [B, K+1]; gather_nd picks the
        # token at index first_rejected[b] per batch element.
        target_tokens = ops.concat([recovered, bonus], axis=1)
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(first_rejected, axis=-1),
            batch_dims=1,
        )

        # Build corrected merged sequence: replace draft tokens with target
        # argmax so the draft model sees correct tokens at rejected positions.
        corrected_merged, corrected_offsets = self.merger(
            tokens, input_row_offsets, recovered
        )

        # Shift the corrected merged sequence for the draft input.
        draft_input_tokens = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
        )

        _draft_signals = (
            draft_signal_buffers
            if draft_signal_buffers is not None
            else signal_buffers
        )

        assert draft_kv_collections is not None
        assert self.draft is not None

        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            draft_input_tokens,
            hidden_states,
            _draft_signals,
            draft_kv_collections,
            return_n_logits,
            merged_offsets,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
        )
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        new_token = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(first_rejected, axis=-1),
            batch_dims=1,
        ).reshape([-1, 1])

        # Build accepted_offsets for correct cache increment: each sequence
        # grows by original_len + first_rejected (accepted draft count).
        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = input_lengths + first_rejected.cast(DType.int64)
        accepted_offsets = ops.concat(
            [
                ops.constant(
                    0, DType.int64, accepted_lengths.device
                ).broadcast_to([1]),
                ops.cumsum(accepted_lengths, axis=0),
            ],
            axis=0,
        ).cast(DType.uint32)

        self._increment_draft_cache(
            accepted_offsets,
            data_parallel_splits,
            _draft_signals,
            draft_kv_collections,
        )

        return (
            first_rejected,  # num_accepted_draft_tokens [B]
            next_tokens,  # next_tokens [B]
            new_token,  # next_draft_tokens [B, num_steps]
        )

    def _increment_draft_cache(
        self,
        input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        signal_buffers: list[BufferValue],
        draft_kv_collections: list[PagedCacheValues],
    ) -> list[PagedCacheValues]:
        devices = self.config.devices
        use_comm = len(devices) > 1

        updated_lengths = ragged_increment_cache_lengths(
            input_row_offsets,
            data_parallel_splits,
            [kv.cache_lengths for kv in draft_kv_collections],
            signal_buffers if use_comm else None,
        )

        new_kv: list[PagedCacheValues] = []
        for dev_idx, kv in enumerate(draft_kv_collections):
            new_kv.append(
                PagedCacheValues(
                    kv_blocks=kv.kv_blocks,
                    cache_lengths=updated_lengths[dev_idx],
                    lookup_table=kv.lookup_table,
                    max_lengths=kv.max_lengths[1:, :],
                )
            )

        return new_kv

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams | None = None,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the Eagle3 unified graph.

        Order: tokens, device_offsets, host_offsets, draft_tokens,
               return_n_logits, data_parallel_splits, signal_buffers,
               target_kv_cache, draft_kv_blocks_per_device,
               draft_signal_buffers, batch_context_lengths,
               target_ep_inputs.
        """
        devices = self.config.devices
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
            shape=[self.config.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        signals = Signals(devices=devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            draft_tokens_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.get_symbolic_inputs().flatten())

        if draft_kv_params is not None:
            for sym in draft_kv_params.get_symbolic_inputs():
                assert isinstance(sym, PagedCacheInputSymbols)
                all_input_types.append(sym.kv_blocks)

            draft_signals = Signals(devices=devices)
            all_input_types.extend(draft_signals.input_types())

        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.target.ep_manager is not None:
            all_input_types.extend(self.target.ep_manager.input_types())

        return tuple(all_input_types)
