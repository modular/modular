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
"""DeepseekV3 with MTP nn.Module: merge + target forward + rejection + shift."""

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
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    RaggedTokenMerger,
)

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3_nextn.deepseekV3_nextn import DeepseekV3NextN
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig


class UnifiedMTPDeepseekV3(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift.

    Composes RaggedTokenMerger, DeepseekV3 (target), DeepseekV3NextN (draft),
    greedy_acceptance_sampler, and eagle_prefill_shift_tokens into a single
    graph-buildable module.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3NextNConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.target = DeepseekV3(config)
        self.merger = RaggedTokenMerger(config.devices[0])

        assert draft_config is not None
        self.draft = DeepseekV3NextN(draft_config)

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

        num_accepted_draft_tokens, recovered, bonus = greedy_acceptance_sampler(
            draft_tokens, logits
        )

        target_tokens = ops.concat([recovered, bonus], axis=1)
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        )

        corrected_merged, corrected_offsets = self.merger(
            tokens, input_row_offsets, recovered
        )
        corrected_merged = corrected_merged.rebind(["merged_seq_len"])
        corrected_offsets = corrected_offsets.rebind(["input_row_offsets_len"])

        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
        )

        host_merged_offsets = merged_offsets.cast(DType.uint32).to(
            DeviceRef.CPU()
        )

        assert draft_kv_collections is not None
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,  # reuse target signal buffers for draft
            draft_kv_collections,
            return_n_logits,
            merged_offsets,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,  # reuse target ep inputs for draft
        )
        self.draft.return_hidden_states = ReturnHiddenStates.LAST
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_logits = draft_outputs[1]

        draft_logits = _reshape_target_logits(draft_logits)
        draft_token_candidates = ops.squeeze(
            ops.argmax(draft_logits, axis=-1), axis=-1
        )

        next_draft_tokens = ops.gather_nd(
            draft_token_candidates,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        ).reshape([-1, 1])

        return (
            num_accepted_draft_tokens,
            next_tokens,
            next_draft_tokens,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams | None = None,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the with-MTP graph.

        Order: tokens, device_offsets, host_offsets, draft_tokens,
               return_n_logits, data_parallel_splits, signal_buffers,
               target_kv_cache, draft_kv_blocks_per_device,
               batch_context_lengths, ep_inputs.
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

        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.target.ep_manager is not None:
            all_input_types.extend(self.target.ep_manager.input_types())

        return tuple(all_input_types)
