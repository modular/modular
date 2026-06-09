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
"""Gemma4 with MTP nn.Module: merge + target forward + rejection + shift."""

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
    ops,
)
from max.nn.comm import Signals
from max.nn.kernels import eagle_prefill_shift_tokens
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)

from ..gemma4.gemma4 import Gemma4TextModel
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.model_config import Gemma4AssistantConfig


def compute_host_merged_offsets(
    host_input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
) -> TensorValue:
    """Compute merged offsets on CPU, avoiding D2H copies.

    merged_offsets[i] = host_input_row_offsets[i] + i * K where K is the
    number of draft tokens per request.
    """
    K = ops.shape_to_tensor([draft_tokens.shape[1]])[0].cast(DType.uint32)
    batch_size_plus_one = ops.shape_to_tensor(
        [host_input_row_offsets.shape[0]]
    )[0]
    indices = ops.range(
        start=0,
        stop=batch_size_plus_one,
        out_dim=host_input_row_offsets.shape[0],
        device=DeviceRef.CPU(),
        dtype=DType.uint32,
    )
    return host_input_row_offsets + indices * K


class UnifiedMTPGemma4(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift.

    Composes RaggedTokenMerger, Gemma4TextModel (target), Gemma4Assistant
    (draft), AcceptanceSampler, and eagle_prefill_shift_tokens into a single
    graph-buildable module.
    """

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        draft_config: Gemma4AssistantConfig,
        speculative_config: SpeculativeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_draft_steps = (
            speculative_config.num_speculative_tokens
            if speculative_config
            else 1
        )
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            speculative_config is not None
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
                if speculative_config
                else None
            ),
            num_draft_steps=self.num_draft_steps,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )
        self.target = Gemma4TextModel(config)
        self.merger = RaggedTokenMerger(config.devices[0])

        self.draft: Any = None
        self._draft_config = draft_config

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        sliding_kv_collections: list[PagedCacheValues],
        global_kv_collections: list[PagedCacheValues],
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
        in_thinking_phase: TensorValue,
    ) -> tuple[TensorValue, ...]:
        # -- 1. Merge tokens + draft tokens --
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
        merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
        )

        # -- 2. Broadcast merged offsets to all devices --
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        devices = self.config.devices
        n_devs = len(devices)
        device0 = devices[0]

        # Create empty image embeddings/indices for text-only MTP
        hidden_size = self.config.text_config.hidden_size
        empty_image_embeddings = [
            ops.constant(0, DType.bfloat16, dev).broadcast_to([0, hidden_size])
            for dev in devices
        ]
        empty_image_indices = [
            ops.constant(0, DType.int32, dev).broadcast_to([0])
            for dev in devices
        ]

        # -- 3. Target forward --
        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            sliding_kv_collections,
            global_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            empty_image_embeddings,
            empty_image_indices,
        )

        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

        # -- 4. Rejection sampling --
        seed_scalar = seed[0]
        num_accepted_draft_tokens, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            logits,
            seed=seed_scalar,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
        )

        target_tokens = ops.concat([recovered, bonus], axis=1)
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        )

        # -- 5. Compute corrected merged sequence and shift --
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

        # Compute q_max_seq_len for cross-attention kernel (uint32 scalar on CPU)
        host_seq_lens = host_merged_offsets[1:] - host_merged_offsets[:-1]
        q_max_seq_len_prefill = (
            ops.max(host_seq_lens, axis=0).cast(DType.uint32).broadcast_to([1])
        )

        # -- 6. Draft step 0 (prefill): pass target hidden states to draft --
        assert self.draft is not None
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            tokens=shifted_corrected,
            hidden_states=hidden_states,
            signal_buffers=signal_buffers,
            target_sliding_kv=sliding_kv_collections,
            target_global_kv=global_kv_collections,
            return_n_logits=return_n_logits,
            input_row_offsets=merged_offsets_per_dev,
            kv_input_row_offsets=merged_offsets_per_dev,
            q_max_seq_len=q_max_seq_len_prefill,
        )

        # Steps 1..K use LAST_PER_DEVICE for hidden states.
        self.draft.return_hidden_states = ReturnHiddenStates.LAST_PER_DEVICE
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        all_hs = list(draft_outputs[3 : 3 + n_devs])

        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        hidden_dim = self._draft_config.backbone_hidden_size

        last_idx = merged_offsets[1:] - 1
        num_draft_sentinel_gpu = _shape_to_scalar(
            draft_tokens.shape[1], device0
        )
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + num_accepted_draft_tokens
        )

        last_accepted_idx_i64 = last_accepted_idx.cast(DType.int64)
        last_accepted_idx_per_dev = ops.distributed_broadcast(
            last_accepted_idx_i64, signal_buffers
        )

        draft_hs: list[TensorValue] = []
        # TP / single-device: each all_hs[i] is a full replica, index
        # directly with the global accepted-idx on each device.
        for i in range(n_devs):
            draft_hs.append(
                ops.gather(all_hs[i], last_accepted_idx_per_dev[i], axis=0)
            )

        # -- 7. Draft steps 1..K (decode) --
        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted_draft_tokens.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(devices) > 1
        # The draft shares the target's KV; its logical cache length equals
        # the target sliding cache's. (Used only for the draft's RoPE
        # position, not for any cache write.)
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in sliding_kv_collections],
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
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, signal_buffers
        )

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        q_max_seq_len_decode = ops.constant(
            1, DType.uint32, DeviceRef.CPU()
        ).broadcast_to([1])

        # Create decode-step KV collections with max_lengths[0,0] = 1.
        # Without this, cross-attention sees max_prompt_length() > 1 (the
        # merged prefill length), is_token_generation stays False, and the
        # depth512 pair-CTA prefill kernel is invoked for a 1-token query —
        # a degenerate case that deadlocks under high KV-cache pressure.
        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])
        decode_sliding_kv = [
            replace(
                kv,
                max_lengths=ops.concat(
                    [one, kv.max_lengths[0, 1].broadcast_to([1])], axis=-1
                ).reshape([1, 2]),
            )
            for kv in sliding_kv_collections
        ]
        decode_global_kv = [
            replace(
                kv,
                max_lengths=ops.concat(
                    [one, kv.max_lengths[0, 1].broadcast_to([1])], axis=-1
                ).reshape([1, 2]),
            )
            for kv in global_kv_collections
        ]

        for step in range(1, self.num_draft_steps):
            draft_hs = [
                draft_hs[i].rebind([f"mtp_step{step}_batch", hidden_dim])
                for i in range(n_devs)
            ]

            step_outputs = self.draft(
                tokens=next_draft_tokens,
                hidden_states=draft_hs,
                signal_buffers=signal_buffers,
                target_sliding_kv=decode_sliding_kv,
                target_global_kv=decode_global_kv,
                return_n_logits=draft_return_n_logits,
                input_row_offsets=decode_offsets_per_dev,
                kv_input_row_offsets=merged_offsets_per_dev,
                q_max_seq_len=q_max_seq_len_decode,
                rope_cache_lengths=cache_lengths_per_dev,
            )

            logits_step = step_outputs[0]
            draft_hs_full = list(step_outputs[1 : 1 + n_devs])
            # TP / single-device: each device already holds a full
            # replica, no per-device slicing needed.
            draft_hs = list(draft_hs_full)

            next_draft_tokens = ops.argmax(logits_step, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            num_accepted_draft_tokens,
            next_tokens,
            new_token,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams | None = None,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified MTP Gemma4 graph.

        Order: tokens, device_offsets, host_offsets, return_n_logits,
               data_parallel_splits, signal_buffers, target_kv_cache,
               batch_context_lengths, draft_tokens,
               draft_kv_blocks_per_device, seed, temperature, top_k,
               max_k, top_p, min_top_p, in_thinking_phase.
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
            shape=[2],  # single-device: [0, batch_size]
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

        all_input_types.append(draft_tokens_type)
        if draft_kv_params is not None:
            for sym in draft_kv_params.get_symbolic_inputs().inputs:
                assert isinstance(sym, KVCacheInputsPerDevice)
                all_input_types.append(sym.kv_blocks)

        # Per-batch device-resident seed.
        seed_type = TensorType(
            DType.uint64, shape=["batch_size"], device=device_ref
        )
        all_input_types.append(seed_type)

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
        in_thinking_phase_type = TensorType(
            DType.bool, shape=["batch_size"], device=device_ref
        )
        all_input_types.extend(
            [
                temperature_type,
                top_k_type,
                max_k_type,
                top_p_type,
                min_top_p_type,
                in_thinking_phase_type,
            ]
        )

        return tuple(all_input_types)
