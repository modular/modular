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
    increment_cache_lengths_from_counts,
)
from max.nn.comm import Signals
from max.nn.kernels import (
    compute_mla_dispatch_args_scalar,
    eagle_prefill_shift_tokens,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
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
from ..unified_mtp_deepseekV3.unified_mtp_deepseekV3 import (
    compute_host_merged_offsets,
)
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
        num_draft_steps: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_draft_steps = num_draft_steps
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
    ) -> tuple[TensorValue, ...]:
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
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
        devices = self.config.devices
        n_devs = len(devices)

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
        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
        )

        assert draft_kv_collections is not None
        assert self.draft is not None

        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,
            draft_kv_collections,
            return_n_logits,
            merged_offsets,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
        )
        self.draft.return_hidden_states = ReturnHiddenStates.LAST
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        all_hs = draft_outputs[3]

        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(first_rejected, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        device0 = devices[0]
        hidden_dim = self.draft.config.hidden_size

        # In TP mode each device processes a subset of attention heads.
        use_tp_ep = self.draft.config.data_parallel_degree == 1 and n_devs > 1
        num_heads_per_dev = (
            self.draft.config.num_attention_heads // n_devs
            if use_tp_ep
            else self.draft.config.num_attention_heads
        )

        last_idx = merged_offsets[1:] - 1
        num_draft_sentinel_gpu = (
            ops.shape_to_tensor([draft_tokens.shape[1]])
            .cast(DType.int64)
            .to(device0)
        )
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + first_rejected
        )
        draft_hs = ops.gather(all_hs, last_accepted_idx, axis=0)

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])
        max_cache_length = (
            draft_kv_collections[0]
            .max_lengths[0, 1]
            .cast(DType.uint32)
            .broadcast_to([1])
        ) + 1

        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + first_rejected.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(devices) > 1
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in draft_kv_collections],
            signal_buffers if use_comm else None,
        )

        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        decode_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device0,
            dtype=DType.uint32,
        )
        host_decode_offsets = decode_offsets.to(DeviceRef.CPU())

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        for step in range(1, self.num_draft_steps):
            draft_hs = draft_hs.rebind(["batch_size", hidden_dim])

            step_max_lengths = ops.concat(
                [one, max_cache_length], axis=-1
            ).reshape([1, 2])

            step_kv: list[PagedCacheValues] = []
            for i in range(n_devs):
                orig_metadata = draft_kv_collections[
                    i
                ].attention_dispatch_metadata
                assert orig_metadata is not None
                dev_batch_size = (
                    orig_metadata.tensor[0].reshape([1]).to(DeviceRef.CPU())
                )
                dev_metadata = compute_mla_dispatch_args_scalar(
                    batch_size=dev_batch_size,
                    max_cache_valid_length=max_cache_length.cast(
                        DType.int64
                    ).to(DeviceRef.CPU()),
                    q_max_seq_len=ops.constant(
                        1, DType.int64, DeviceRef.CPU()
                    ).broadcast_to([1]),
                    num_heads=num_heads_per_dev,
                    device=devices[i],
                ).to(devices[i])

                step_kv.append(
                    PagedCacheValues(
                        kv_blocks=draft_kv_collections[i].kv_blocks,
                        cache_lengths=cache_lengths_per_dev[i],
                        lookup_table=draft_kv_collections[i].lookup_table,
                        max_lengths=step_max_lengths,
                        attention_dispatch_metadata=dev_metadata,
                    )
                )

            step_outputs = self.draft(
                next_draft_tokens,
                draft_hs,
                signal_buffers,
                step_kv,
                draft_return_n_logits,
                decode_offsets,
                host_decode_offsets,
                data_parallel_splits,
                batch_context_lengths,
                split_prefix=f"eagle3_draft_step{step}",
            )

            logits = step_outputs[0]
            draft_hs = step_outputs[1]

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            cache_lengths_per_dev = [cl + 1 for cl in cache_lengths_per_dev]
            max_cache_length = max_cache_length + 1
            batch_context_lengths = [bcl + 1 for bcl in batch_context_lengths]

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            first_rejected,  # num_accepted_draft_tokens [B]
            next_tokens,  # next_tokens [B]
            new_token,  # next_draft_tokens [B, num_draft_steps]
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams | None = None,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the Eagle3 unified graph.

        Order: tokens, device_offsets, host_offsets, draft_tokens,
               return_n_logits, data_parallel_splits, signal_buffers,
               target_kv_cache, draft_kv_blocks_per_device,
               batch_context_lengths, target_ep_inputs.
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
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.get_symbolic_inputs().flatten())

        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.target.ep_manager is not None:
            all_input_types.extend(self.target.ep_manager.input_types())

        all_input_types.append(draft_tokens_type)
        if draft_kv_params is not None:
            for sym in draft_kv_params.get_symbolic_inputs().inputs:
                assert isinstance(sym, KVCacheInputsPerDevice)
                all_input_types.append(sym.kv_blocks)

        return tuple(all_input_types)
