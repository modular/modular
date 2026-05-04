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
from max.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.nn.comm import Signals
from max.nn.kernels import (
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
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    RaggedTokenMerger,
    shape_to_scalar,
)

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3_nextn.deepseekV3_nextn import DeepseekV3NextN
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig


def compute_host_merged_offsets(
    host_input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
) -> TensorValue:
    """Computes merged offsets on CPU, avoiding D2H copies.

    merged_offsets[i] = host_input_row_offsets[i] + i * K where K is the
    number of draft tokens per request. This mirrors the GPU-side merger
    logic but stays on CPU so CUDA graph capture is not blocked by a
    device-to-host transfer.
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
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
    ) -> tuple[TensorValue, ...]:
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
        merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
        )

        # Broadcast merged_offsets once (uint32, small) and reuse the
        # per-device list for target + draft step 0 + the accept-position
        # gather. Hoisting this eliminates the broadcast that target and
        # each draft step would otherwise run independently.
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
        devices = self.config.devices
        n_devs = len(devices)

        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

        # ``seed`` is the per-batch ``[batch_size]`` uint64 device buffer
        # that feeds ``topk_fused_sampling`` (recovered + bonus tokens) per
        # row. The rejection-decision RNG below is a Bernoulli coin flip —
        # its marginal accept distribution is unchanged whether each row
        # gets its own Philox stream or all rows share one with offset-
        # based diversity, so we collapse to ``seed[0]`` here. The
        # token-sampling path keeps the full tensor.
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

        assert draft_kv_collections is not None
        # Step 0 always uses ALL hidden states (for per-batch-element gather
        # at accepted positions) + VARIABLE logits (for draft argmax).
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,  # reuse target signal buffers for draft
            draft_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,  # reuse target ep inputs for draft
        )
        # Steps 1..K use LAST_PER_DEVICE. The underlying LAST path's
        # internal allgather acts as a collective fence between successive
        # draft invocations — without it, the draft's EP/MoE dispatch
        # leaves pending async work that surfaces `mgp.sync` on the 2nd
        # decoder call during CUDA graph capture. LAST_PER_DEVICE returns
        # the post-allgather per-device tensors directly.
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

        devices = self.config.devices
        device0 = devices[0]
        hidden_dim = self.draft.config.hidden_size

        last_idx = merged_offsets[1:] - 1
        num_draft_sentinel_gpu = shape_to_scalar(draft_tokens.shape[1], device0)
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + num_accepted_draft_tokens
        )
        # Per-device gather at accepted positions. Broadcast indices once,
        # then either slice by DP splits (DP mode, each device holds its
        # local batch shard) or gather directly (TP mode, each device
        # holds a full replica).
        last_accepted_idx_i64 = last_accepted_idx.cast(DType.int64)
        last_accepted_idx_per_dev = ops.distributed_broadcast(
            last_accepted_idx_i64, signal_buffers
        )

        draft_hs: list[TensorValue] = []
        if self.config.data_parallel_degree > 1:
            for i in range(n_devs):
                start = data_parallel_splits[i]
                end = data_parallel_splits[i + 1]
                global_idx_dev_i = ops.slice_tensor(
                    last_accepted_idx_per_dev[i],
                    [(slice(start, end), f"mtp_batch_split_{i}")],
                )
                local_seq_offset_i = merged_offsets_per_dev[i][start].cast(
                    DType.int64
                )
                local_idx_dev_i = global_idx_dev_i - local_seq_offset_i
                draft_hs.append(ops.gather(all_hs[i], local_idx_dev_i, axis=0))
        else:
            # TP / single-device: each all_hs[i] is a full replica, index
            # directly with the global accepted-idx on each device.
            for i in range(n_devs):
                draft_hs.append(
                    ops.gather(all_hs[i], last_accepted_idx_per_dev[i], axis=0)
                )

        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted_draft_tokens.cast(DType.int64)
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
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, signal_buffers
        )
        host_decode_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=DeviceRef.CPU(),
            dtype=DType.uint32,
        )

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])
        new_max_lengths = [
            ops.concat(
                [one, kv.max_lengths[0, 1].broadcast_to([1])], axis=-1
            ).reshape([1, 2])
            for kv in draft_kv_collections
        ]

        draft_kv_collections = [
            replace(
                kv,
                max_lengths=max_lengths,
                attention_dispatch_metadata=kv.draft_attention_dispatch_metadata,
            )
            for kv, max_lengths in zip(
                draft_kv_collections, new_max_lengths, strict=True
            )
        ]

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        for step in range(1, self.num_draft_steps):
            # Per-device shapes differ across DP replicas; use per-device
            # dim names. The draft internally rebinds to
            # `{split_prefix}_seq_len_device_{i}` once inside __call__.
            draft_hs = [
                draft_hs[i].rebind(
                    [f"mtp_step{step}_batch_dev_{i}", hidden_dim]
                )
                for i in range(n_devs)
            ]

            step_kv: list[PagedCacheValues] = [
                replace(kv, cache_lengths=cl)
                for kv, cl in zip(
                    draft_kv_collections, cache_lengths_per_dev, strict=True
                )
            ]

            step_outputs = self.draft(
                next_draft_tokens,
                draft_hs,
                signal_buffers,
                step_kv,
                draft_return_n_logits,
                decode_offsets_per_dev,
                host_decode_offsets,
                data_parallel_splits,
                batch_context_lengths,
                ep_inputs,
                split_prefix=f"mtp_draft_step{step}",
            )

            logits = step_outputs[0]
            draft_hs_full = list(step_outputs[1 : 1 + n_devs])
            next_step = step + 1
            if self.config.data_parallel_degree > 1:
                draft_hs = [
                    ops.slice_tensor(
                        draft_hs_full[i],
                        [
                            (
                                slice(
                                    data_parallel_splits[i],
                                    data_parallel_splits[i + 1],
                                ),
                                f"mtp_step{next_step}_batch_dev_{i}",
                            ),
                        ],
                    )
                    for i in range(n_devs)
                ]
            else:
                # TP / single-device: each device already holds a full
                # replica, no per-device slicing needed.
                draft_hs = list(draft_hs_full)

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            cache_lengths_per_dev = [cl + 1 for cl in cache_lengths_per_dev]
            batch_context_lengths = [bcl + 1 for bcl in batch_context_lengths]

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
        """Input types for the with-MTP graph.

        Order: tokens, device_offsets, host_offsets, return_n_logits,
               data_parallel_splits, signal_buffers, target_kv_cache,
               batch_context_lengths, ep_inputs, draft_tokens,
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

        # Per-batch device-resident seed, derived per request from
        # ``sampling_params.seed + len(tokens)`` by the overlap pipeline.
        # Keeping the seed in device memory (rather than the rank-1 ``[1]``
        # ``ops.random.SeedType``) lets each row carry its own
        # ``sampling_params.seed`` and refreshes per-execute via
        # ``inplace_copy_from`` so CUDA graph replay sees fresh values.
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
