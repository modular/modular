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
"""Eagle3 MHA-draft + Kimi K2.5 (MLA target) fused nn.Module.

Mirrors :class:`Eagle3KimiK25Unified` but:

- carries an MHA draft (:class:`Eagle3MHAKimiK25`) whose KV cache geometry
  is independent of the target's MLA cache;
- declares an independent set of per-device draft KV inputs (kv_blocks,
  cache_lengths, lookup_table, max_lengths, attention_dispatch_metadata)
  instead of borrowing the target's slots.
"""

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
from max.nn.kernels import (
    eagle_prefill_shift_tokens,
    inplace_memcpy,
    wait_host_value_with_dep,
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
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..unified_mtp_deepseekV3.unified_mtp_deepseekV3 import (
    compute_host_merged_offsets,
)
from .eagle3_mha_kimi_k25 import Eagle3MHAKimiK25, Eagle3MHAKimiK25DraftConfig


class Eagle3MHAKimiK25Unified(Module):
    """Fused: merge + target (MLA) forward + rejection + shift + MHA draft.

    The target forwards as a standard distributed DeepseekV3 (MLA). The
    draft consumes its **own** per-device ``PagedCacheValues`` (with MHA
    layout), allowing independent KV geometry and independent dispatch
    metadata.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: Eagle3MHAKimiK25DraftConfig,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        enable_vision: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.draft_config = draft_config
        self.enable_structured_output = enable_structured_output
        # ``enable_vision`` controls whether the unified graph accepts
        # per-device image embeddings + scatter indices and scatters them
        # into the merged token embedding before the target forward.
        # Only Kimi-style targets that carry a vision encoder set this;
        # the bare DeepseekV3 + MHA-draft pipeline leaves it False.
        self.enable_vision = enable_vision
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

        self.draft: Eagle3MHAKimiK25 | None = None
        if draft_config is not None:
            self.draft = Eagle3MHAKimiK25(draft_config)

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
        image_embeddings: list[TensorValue] | None = None,
        image_token_indices: list[TensorValue] | None = None,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
        merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
        )

        assert self.draft is not None
        devices = self.config.devices
        n_devs = len(devices)
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )
        if self.enable_vision:
            # Mirror ``KimiK2_5MoEDecoder.__call__``: embed the merged
            # tokens, scatter image embeddings into the merged sequence at
            # ``image_token_indices``, then run the rest of the target
            # stack on the resulting hidden states.
            #
            # During prefill K=0 (no draft tokens to merge), so the
            # indices remain valid for the merged sequence with no
            # remapping. During decode no new images are introduced so
            # ``image_token_indices`` is empty.
            assert image_embeddings is not None
            assert image_token_indices is not None
            h_per_dev = self.target.embed_tokens(merged_tokens, signal_buffers)
            h_per_dev = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_d,
                    multimodal_embeddings=img_emb_d,
                    image_token_indices=img_idx_d,
                )
                for h_d, img_emb_d, img_idx_d in zip(
                    h_per_dev,
                    image_embeddings,
                    image_token_indices,
                    strict=True,
                )
            ]
            target_outputs = self.target._process_hidden_states(
                h_per_dev,
                signal_buffers,
                kv_collections,
                return_n_logits,
                list(merged_offsets_per_dev),
                host_merged_offsets,
                data_parallel_splits,
                batch_context_lengths,
                ep_inputs,
            )
        else:
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
        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

        # Constrained-decoding overlap: gate the model stream on the
        # async callback's release-store to ``wait_payload``, then
        # in-graph H2D from pinned host memory to
        # ``device_bitmask_scratch``. The sampler reads the scratch.
        # ``device_bitmask_scratch`` is threaded through the wait as
        # a fake mutable operand so the graph compiler / cuGraph
        # capture serialises the memcpy after the wait (both ops
        # mutate the same buffer). The triple is all-or-none.
        if not (
            (pinned_bitmask is None)
            == (wait_payload is None)
            == (device_bitmask_scratch is None)
        ):
            raise ValueError(
                "pinned_bitmask, wait_payload, and device_bitmask_scratch "
                "must be either all None or all non-None; got "
                f"pinned_bitmask={'set' if pinned_bitmask is not None else 'None'}, "
                f"wait_payload={'set' if wait_payload is not None else 'None'}, "
                f"device_bitmask_scratch={'set' if device_bitmask_scratch is not None else 'None'}"
            )
        effective_bitmasks: TensorValue | None = None
        if (
            pinned_bitmask is not None
            and wait_payload is not None
            and device_bitmask_scratch is not None
        ):
            wait_host_value_with_dep(
                wait_payload, device_bitmask_scratch, device=devices[0]
            )
            inplace_memcpy(device_bitmask_scratch, pinned_bitmask)
            # Trim the persistent buffer's worst-case
            # ``num_speculative_tokens + 1`` rows down to
            # ``num_steps + 1`` so the acceptance sampler's rebind
            # to ``num_steps + 1`` lines up. Position ``i`` of the
            # bitmask holds the FSM state with ``i`` drafts
            # consumed, so positions ``0..num_steps`` cover the
            # ``num_steps`` draft-verification slots plus the bonus
            # slot at index ``num_steps``; the target never emits
            # logits for the trailing rows this iter.
            num_steps_plus_one = draft_tokens.shape[1] + 1
            effective_bitmasks = device_bitmask_scratch[
                :, :num_steps_plus_one, :
            ]

        seed_scalar = seed[0]
        first_rejected, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            logits,
            seed=seed_scalar,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
            token_bitmasks=effective_bitmasks,
        )

        target_tokens = ops.concat([recovered, bonus], axis=1)
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(first_rejected, axis=-1),
            batch_dims=1,
        )

        corrected_merged, corrected_offsets = self.merger(
            tokens, input_row_offsets, recovered
        )
        corrected_merged = ops.rebind(corrected_merged, ["merged_seq_len"])
        corrected_offsets = ops.rebind(
            corrected_offsets, ["input_row_offsets_len"]
        )

        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
        )

        assert draft_kv_collections is not None

        draft0_kv_collections = [
            _patch_draft0_kv_cache(kv) for kv in draft_kv_collections
        ]

        # Step 0: ALL hidden states + VARIABLE logits.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,
            draft0_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
        )
        # Steps 1..K: LAST + ALL (per-device hs feeds next step's
        # fused_target_hs directly).
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        all_hs = list(draft_outputs[3 : 3 + n_devs])

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

        last_idx = merged_offsets[1:] - 1
        num_draft_sentinel_gpu = _shape_to_scalar(
            draft_tokens.shape[1], device0
        )
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + first_rejected
        )
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
                    [(slice(start, end), f"eagle3_mha_batch_split_{i}")],
                )
                local_seq_offset_i = merged_offsets_per_dev[i][start].cast(
                    DType.int64
                )
                local_idx_dev_i = global_idx_dev_i - local_seq_offset_i
                draft_hs.append(ops.gather(all_hs[i], local_idx_dev_i, axis=0))
        else:
            for i in range(n_devs):
                draft_hs.append(
                    ops.gather(all_hs[i], last_accepted_idx_per_dev[i], axis=0)
                )

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

        # The pipeline_model already swapped attention_dispatch_metadata to
        # the draft slot at graph-init; carry that through to the step-1+
        # collections along with the updated max_lengths.
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
            draft_hs = [
                draft_hs[i].rebind(
                    [f"draft_mha_step{step}_batch_dev_{i}", hidden_dim]
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
                split_prefix=f"eagle3_mha_draft_step{step}",
            )

            logits = step_outputs[0]
            draft_hs = list(step_outputs[1 : 1 + n_devs])

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
            first_rejected,
            next_tokens,
            new_token,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        draft_kv_params: KVCacheParams,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified MHA-draft graph.

        Order:
            tokens, device_offsets, host_offsets, return_n_logits,
            data_parallel_splits, signal_buffers,
            target_kv_cache (flat), batch_context_lengths, target_ep_inputs,
            draft_tokens, draft_kv_cache (flat per-device — kv_blocks
            + cache_lengths + lookup_table + max_lengths +
            attention_dispatch_metadata + draft_attention_dispatch_metadata),
            seed, temperature, top_k, max_k, top_p, min_top_p,
            in_thinking_phase, [pinned_bitmask, wait_payload,
            device_bitmask_scratch].

        The draft cache contributes its own per-device dispatch metadata
        (one buffer for the q_max_seq_len = ``1 + num_speculative_tokens``
        prefill at step 0 and one for the q_max_seq_len = 1 decode at
        steps 1..K), so the graph-capture branch can populate MHA geometry
        for the draft independently of the target's MLA geometry.
        """
        devices = self.config.devices
        device_ref = devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        image_embeddings_types = [
            TensorType(
                DType.bfloat16,
                shape=["vision_merged_seq_len", self.config.hidden_size],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        ]
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        ]
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
        ]
        if self.enable_vision:
            all_input_types.extend(image_embeddings_types)
            all_input_types.extend(image_token_indices_types)
        all_input_types.extend(
            [
                device_input_row_offsets_type,
                host_input_row_offsets_type,
                return_n_logits_type,
                data_parallel_splits_type,
            ]
        )
        all_input_types.extend(signal_buffer_types)
        # Size the target's ``draft_attention_dispatch_metadata`` slot by
        # the draft's ``is_mla`` (shape [4]/CPU for MHA vs [3]/device for
        # MLA). The colleague's graph-capture branch populates that slot
        # at runtime with the draft resolver's output. The
        # ``draft_attention_group`` kwarg lives on ``KVCacheParams`` (the
        # concrete dataclass), not on the ``KVCacheParamInterface``
        # Protocol — narrow with ``isinstance`` so mypy can see the
        # kwarg.
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

        # Draft KV block storage: one Buffer per device. The rest of the
        # draft cache fields are reused from the target (same logical
        # pages, different physical storage). The draft's MHA dispatch
        # metadata is plumbed via the target KV input's
        # ``draft_attention_dispatch_metadata`` slot, populated by the
        # graph-capture branch.
        for sym in draft_kv_params.get_symbolic_inputs().inputs:
            assert isinstance(sym, KVCacheInputsPerDevice)
            all_input_types.append(sym.kv_blocks)

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

        # Constrained-decoding bitmask triple (pinned host bitmask +
        # wait_payload + device scratch). The pinned input is declared
        # on CPU per the engine's input binding rule.
        # ``num_bitmask_positions = num_speculative_tokens + 1``.
        if self.enable_structured_output:
            pinned_bitmask_type = TensorType(
                DType.bool,
                shape=["batch_size", "num_bitmask_positions", "vocab_size"],
                device=DeviceRef.CPU(),
            )
            wait_payload_type = BufferType(
                DType.int64,
                shape=[2],
                device=DeviceRef.CPU(),
            )
            device_bitmask_scratch_type = BufferType(
                DType.bool,
                shape=["batch_size", "num_bitmask_positions", "vocab_size"],
                device=device_ref,
            )
            all_input_types.extend(
                [
                    pinned_bitmask_type,
                    wait_payload_type,
                    device_bitmask_scratch_type,
                ]
            )

        return tuple(all_input_types)


# TODO(SERVOPT-1437): This is a temporary patch, until we have a proper way
# to pass the draft0 attention dispatch metadata.
def _patch_draft0_kv_cache(kv: PagedCacheValues) -> PagedCacheValues:
    """Returns ``kv`` with ``attention_dispatch_metadata`` re-sized for the
    unified-eagle draft's step-0 prefill.

    The unified-eagle graph calls the draft twice per spec-decode step:
      1. First on a merged prefill sequence whose per-ctx q is larger than 1,
      2. Then on single-token decode steps where q == 1.

    The ``cache_manager`` resolves ``draft_attention_dispatch_metadata`` with
    ``max_prompt_length == 1``, which is the correct width for the
    decode steps, but not for prefill. This function patches the metadata
    to the correct width for the prefill step.
    """
    decode_md = kv.draft_attention_dispatch_metadata
    assert decode_md is not None
    step0_max_prompt = kv.max_lengths[0, 0].cast(DType.int64).reshape([1]) + 1
    step0_md = ops.concat(
        [decode_md[0:1], step0_max_prompt, decode_md[2:4]],
        axis=0,
    )
    return replace(kv, attention_dispatch_metadata=step0_md)
