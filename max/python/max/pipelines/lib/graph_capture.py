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
"""Centralized graph capture runner for overlap serving.

Flow:
- Model worker creates the runner and executes pre-ready warmup.
- Warmup captures hot decode buckets, largest-first.
- Serving path replays by
  ``(batch_token_count, num_partitions, q_max_seq_len, draft_num_partitions)``
  key (``draft_num_partitions`` is 0 without speculative decoding). Only
  ``q_max_seq_len=1 + num_speculative_tokens`` graphs are captured;
  a ``RuntimeError`` is raised for any other value.
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace

import numpy as np
from max._core.driver import _release_buffers_to_borrowed
from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.nn.kv_cache import AttentionDispatchResolver, KVCacheParams
from max.nn.kv_cache import KVCacheInputs as _KVCacheInputs
from max.nn.kv_cache import KVCacheInputsPerDevice as _KVCacheInputsPerDevice
from max.profiler import traced
from tqdm import tqdm

from .interfaces import ModelInputs, ModelOutputs, UnifiedEagleOutputs

KVCacheInputsPerDevice = _KVCacheInputsPerDevice[Buffer, Buffer]
KVCacheInputs = _KVCacheInputs[Buffer, Buffer]
logger = logging.getLogger("max.pipelines")


GraphKey = tuple[int, int, int, int]
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]
WarmupModelInputs = Callable[[int], AbstractContextManager[ModelInputs]]


def _release_graph_capture_outputs_to_borrowed(
    outputs: ModelOutputs,
) -> ModelOutputs:
    """Returns graph-capture warmup outputs as borrowed wrappers.

    The returned buffers continue to point at the same storage, but later
    captures or replays may overwrite that memory.
    """
    buffer_field_names: list[str] = []
    buffers: list[Buffer] = []
    for field_name in (
        "logits",
        "next_token_logits",
        "logit_offsets",
        "hidden_states",
        "num_accepted_draft_tokens",
        "next_tokens",
        "next_draft_tokens",
    ):
        value = getattr(outputs, field_name, None)
        if isinstance(value, Buffer):
            buffer_field_names.append(field_name)
            buffers.append(value)

    if not buffers:
        return outputs

    released_buffers = _release_buffers_to_borrowed(buffers)
    return replace(
        outputs,
        **dict(zip(buffer_field_names, released_buffers, strict=True)),
    )


class AttentionMetadataProbeStrategy(ABC):
    """Determines which num_partitions values to capture during warmup."""

    @abstractmethod
    def probe_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        ...

    def bucket_num_partitions(
        self,
        runtime_np: int,
        captured_nps: Sequence[int],
    ) -> int | None:
        """Buckets runtime np to nearest captured np, or None on miss."""
        return None


class MHAProbeStrategy(AttentionMetadataProbeStrategy):
    """MHA: probe at 256-token granularity, capture all distinct modes."""

    granularity = 256

    def probe_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Probes at ``granularity`` intervals from 1 to ``max_cache_length``."""
        del q_max_seq_len  # unused
        return (
            [1]
            + list(range(self.granularity, max_cache_length, self.granularity))
            + [max_cache_length]
        )


class MLAProbeStrategy(AttentionMetadataProbeStrategy):
    """MLA: probe at 64-token granularity, bucket up to nearest captured np."""

    granularity = 64

    def probe_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Probes at ``granularity`` intervals from 1 to ``max_cache_length``."""
        probe_lengths = (
            [1]
            + list(range(self.granularity, max_cache_length, self.granularity))
            + [max_cache_length]
        )
        if q_max_seq_len > 1:
            # With spec decoding, we need to probe a few more entries to hit all
            # viable (num_partition, draft_num_partition) pairs. This was determined
            # experimentally and is highly fragile. The alternative is to brute
            # force capture all 11 x 11 = 121 pairs but that would increase graph
            # capture time significantly. In practice, I only noticed that we used
            # ~20 of the 121 combos for Kimi.
            additional_probes = list(
                range(self.granularity - 1, max_cache_length, self.granularity)
            )
            probe_lengths.extend(additional_probes)
        return probe_lengths

    def bucket_num_partitions(
        self,
        runtime_np: int,
        captured_nps: Sequence[int],
    ) -> int | None:
        """Buckets to the smallest captured np >= runtime np."""
        candidates = sorted(np for np in captured_nps if np >= runtime_np)
        return candidates[0] if candidates else None


def _ragged_kv_inputs_from_model_inputs(
    model_inputs: ModelInputs,
) -> tuple[KVCacheInputsPerDevice, ...]:
    kv = model_inputs.kv_cache_inputs
    if kv is None:
        raise ValueError(
            "Overlap graph capture requires KVCacheInputs, but "
            "model_inputs.kv_cache_inputs is None."
        )
    seq = kv.inputs
    if not seq:
        raise ValueError("Expected at least one KV cache input for replay.")
    return tuple(seq)


def _pack_model_graph_key(key: GraphKey) -> int:
    """Maps a GraphKey tuple to a uint64 for the C++ capture layer."""
    return hash(key) & 0xFFFFFFFFFFFFFFFF


def _unpack_dispatch_metadata(metadata: Buffer | None) -> tuple[int, int]:
    """Returns ``(num_partitions, q_max_seq_len)`` from packed metadata."""
    if metadata is None:
        return 0, 0
    metadata_np = metadata.to_numpy()
    return int(metadata_np[2]), int(metadata_np[1])


def _unpack_replay_metadata(
    kv: KVCacheInputsPerDevice,
    is_draft: bool = False,
) -> tuple[int, int]:
    """Returns ``(num_partitions, q_max_seq_len)`` from replay metadata."""
    if not is_draft:
        metadata = kv.attention_dispatch_metadata
        if metadata is None:
            raise ValueError(
                "Expected attention_dispatch_metadata in KV inputs."
            )
    else:
        metadata = kv.draft_attention_dispatch_metadata
    return _unpack_dispatch_metadata(metadata)


def _create_model_inputs_with_dispatch_metadata(
    model_inputs: ModelInputs,
    source_ragged: Sequence[KVCacheInputsPerDevice],
    dispatch_metadata: Buffer,
    draft_dispatch_metadata: Buffer | None,
    max_cache_valid_length: int,
    target_is_mla: bool = False,
    draft_is_mla: bool = False,
) -> ModelInputs:
    """Returns a copy of *model_inputs* with capture dispatch metadata."""
    max_cache_u32 = np.uint32(max_cache_valid_length)
    capture_ragged: list[KVCacheInputsPerDevice] = []
    for kv in source_ragged:
        ml = kv.max_lengths.to_numpy().copy()
        ml[:, 1] = max_cache_u32
        metadata = (
            dispatch_metadata.to(kv.kv_blocks.device)
            if target_is_mla
            else dispatch_metadata
        )
        if draft_dispatch_metadata is not None:
            draft_metadata = (
                draft_dispatch_metadata.to(kv.kv_blocks.device)
                if draft_is_mla
                else draft_dispatch_metadata
            )
        else:
            draft_metadata = None
        capture_ragged.append(
            replace(
                kv,
                max_lengths=Buffer.from_numpy(ml),
                attention_dispatch_metadata=metadata,
                draft_attention_dispatch_metadata=draft_metadata,
            )
        )
    result = copy.copy(model_inputs)
    result.kv_cache_inputs = KVCacheInputs(inputs=capture_ragged)
    return result


def _patch_kv_cache_for_capture(
    kv: KVCacheInputs,
    dispatch_metadata: Buffer,
    max_cache_valid_length: int,
) -> KVCacheInputs:
    """Patches max_lengths and dispatch metadata on a KV cache for capture."""
    max_cache_u32 = np.uint32(max_cache_valid_length)
    patched: list[KVCacheInputsPerDevice] = []
    for kv_per_device in kv.inputs:
        ml = kv_per_device.max_lengths.to_numpy().copy()
        ml[:, 1] = max_cache_u32
        patched.append(
            replace(
                kv_per_device,
                max_lengths=Buffer.from_numpy(ml),
                attention_dispatch_metadata=dispatch_metadata,
            )
        )
    return KVCacheInputs(inputs=patched)


class ServeGraphCaptureRunner:
    """Central owner for serve-time graph capture state."""

    def __init__(
        self,
        *,
        model: Model,
        execute_model: Callable[[ModelInputs], ModelOutputs],
        session: InferenceSession,
        kv_params: KVCacheParams,
        warmup_model_inputs: WarmupModelInputs,
        max_cache_length_upper_bound: int,
        max_batch_size: int,
        num_speculative_tokens: int = 0,
        num_kv_caches: int = 1,
        draft_kv_params: KVCacheParams | None = None,
    ) -> None:
        self._model = model
        self._execute_model = execute_model
        self._warmup_model_inputs = warmup_model_inputs
        self._num_speculative_tokens = num_speculative_tokens
        self._num_kv_caches = num_kv_caches
        self._n_devices = len(kv_params.devices)
        if max_cache_length_upper_bound < 1:
            raise ValueError(
                "Decode graph capture requires a positive decode "
                "max-cache length upper bound."
            )
        self._max_cache_length_upper_bound = max_cache_length_upper_bound
        self._target_resolver = AttentionDispatchResolver(
            devices=kv_params.devices,
            is_mla=kv_params.is_mla,
            n_kv_heads_per_device=kv_params.n_kv_heads_per_device,
            num_q_heads_per_device=kv_params.num_q_heads_per_device,
            # TODO(SERVOPT-1094): Replace with quantized_kv_cache once
            # SnapMLA uses a valid scale_dtype.
            is_fp8_kv=kv_params.is_fp8_kv_dtype,
        )
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        self._max_batch_size = max_batch_size
        self._target_probe_strategy: AttentionMetadataProbeStrategy = (
            MLAProbeStrategy() if kv_params.is_mla else MHAProbeStrategy()
        )
        self._target_is_mla = kv_params.is_mla
        self._is_data_parallel = kv_params.data_parallel_degree > 1

        self._draft_resolver: AttentionDispatchResolver
        self._draft_probe_strategy: AttentionMetadataProbeStrategy
        if draft_kv_params is not None:
            self._draft_resolver = AttentionDispatchResolver(
                devices=draft_kv_params.devices,
                is_mla=draft_kv_params.is_mla,
                n_kv_heads_per_device=draft_kv_params.n_kv_heads_per_device,
                num_q_heads_per_device=draft_kv_params.num_q_heads_per_device,
                is_fp8_kv=draft_kv_params.is_fp8_kv_dtype,
            )
            self._draft_probe_strategy = (
                MLAProbeStrategy()
                if draft_kv_params.is_mla
                else MHAProbeStrategy()
            )
            self._draft_is_mla = draft_kv_params.is_mla
        else:
            self._draft_resolver = self._target_resolver
            self._draft_probe_strategy = self._target_probe_strategy
            self._draft_is_mla = self._target_is_mla

        self.graph_entries: dict[GraphKey, GraphEntry] = {}

    def release_graph(self, key: GraphKey) -> None:
        """Releases a single captured graph and its working memory.

        Drops the runner's entry for ``key`` (input + output buffer handles)
        and asks the engine to release the underlying device graph. Safe to
        call when ``key`` is not currently captured: the runner-side ``pop``
        becomes a no-op and the engine-side release is itself a no-op for
        unknown keys.
        """
        self.graph_entries.pop(key, None)
        self._model.release_captured_graph(_pack_model_graph_key(key))

    def dispatch_metadata(
        self, batch_size: int, q_max_seq_len: int
    ) -> list[tuple[int, Buffer, Buffer | None]]:
        """Returns capture metadata selected by the probe strategy.

        Probes at regular cache-length intervals to discover distinct
        ``(num_partitions, draft_num_partitions)`` modes, then delegates to
        the strategy to select which modes to actually capture. Without spec
        dec, ``draft_num_partitions`` is ``0`` and draft buffers are not
        captured.
        """
        target_probe_lengths = self._target_probe_strategy.probe_lengths(
            self._max_cache_length_upper_bound,
            q_max_seq_len,
        )
        if q_max_seq_len > 1:
            draft_probe_lengths = self._draft_probe_strategy.probe_lengths(
                self._max_cache_length_upper_bound,
                1,
            )
        else:
            draft_probe_lengths = []
        probe_lengths = sorted(
            set(target_probe_lengths) | set(draft_probe_lengths)
        )
        metadata_by_dispatch_key: dict[
            tuple[int, int], tuple[int, Buffer, Buffer | None]
        ] = {}
        for length in probe_lengths:
            metadata = self._target_resolver(batch_size, q_max_seq_len, length)
            if q_max_seq_len > 1:
                metadata_draft = self._draft_resolver(batch_size, 1, length)
                draft_np, _ = _unpack_dispatch_metadata(metadata_draft)
            else:
                metadata_draft = None
                draft_np = 0
            num_partitions, _ = _unpack_dispatch_metadata(metadata)
            dispatch_key = (num_partitions, draft_np)
            metadata_by_dispatch_key[dispatch_key] = (
                length,
                metadata,
                metadata_draft,
            )
        return list(metadata_by_dispatch_key.values())

    @traced
    def warmup_pre_ready(self) -> None:
        """Captures decode buckets before the worker becomes ready."""
        logger.info(
            "Pre-capturing overlap device graphs for decode batch sizes [1..%d] "
            "with num_steps=1.",
            self._max_batch_size,
        )
        # Conservative/defensive warmup: capture largest-first so peak
        # allocations happen up front and oversized configs fail fast.
        q_max_seq_len = self._num_speculative_tokens + 1
        batch_sizes = range(self._max_batch_size, 0, -1)
        # Disable progress bar in non-interactive environments (CI) where
        # carriage return doesn't work and each update prints a new line.
        for batch_size in tqdm(
            batch_sizes,
            desc="Capturing device graph shapes",
            disable=not sys.stderr.isatty(),
        ):
            dispatch_entries = sorted(
                self.dispatch_metadata(batch_size, q_max_seq_len),
                key=lambda entry: (
                    _unpack_dispatch_metadata(entry[1])[0],
                    _unpack_dispatch_metadata(entry[2])[0],
                ),
                reverse=True,
            )
            with self._warmup_model_inputs(batch_size) as model_inputs:
                batch_token_count = int(model_inputs.buffers[0].shape[0])
                source_ragged = _ragged_kv_inputs_from_model_inputs(
                    model_inputs
                )
                for (
                    max_cache_valid_length,
                    dispatch_metadata,
                    draft_dispatch_metadata,
                ) in dispatch_entries:
                    num_partitions, _ = _unpack_dispatch_metadata(
                        dispatch_metadata
                    )
                    draft_num_partitions = _unpack_dispatch_metadata(
                        draft_dispatch_metadata
                    )[0]
                    key = (
                        batch_token_count,
                        num_partitions,
                        self._num_speculative_tokens + 1,
                        draft_num_partitions,
                    )
                    assert key not in self.graph_entries, (
                        "unexpected duplicate key"
                    )

                    capture_inputs = (
                        _create_model_inputs_with_dispatch_metadata(
                            model_inputs,
                            source_ragged,
                            dispatch_metadata,
                            draft_dispatch_metadata,
                            max_cache_valid_length,
                            target_is_mla=self._target_is_mla,
                            draft_is_mla=self._draft_is_mla,
                        )
                    )

                    # Patch dispatch metadata for extra caches.
                    # Combined layout: [cache0_dev0..devN, cache1_dev0..devN, ...]
                    if (
                        self._num_kv_caches > 1
                        and capture_inputs.kv_cache_inputs is not None
                    ):
                        all_inputs = list(capture_inputs.kv_cache_inputs.inputs)
                        extra_dispatch = self._draft_resolver(
                            batch_size,
                            self._num_speculative_tokens + 1,
                            max_cache_valid_length,
                        )
                        for cache_idx in range(1, self._num_kv_caches):
                            start = cache_idx * self._n_devices
                            end = start + self._n_devices
                            patched = _patch_kv_cache_for_capture(
                                KVCacheInputs(inputs=all_inputs[start:end]),
                                extra_dispatch,
                                max_cache_valid_length,
                            )
                            all_inputs[start:end] = list(patched.inputs)
                        capture_inputs.kv_cache_inputs = KVCacheInputs(
                            inputs=all_inputs
                        )

                    input_buffers = capture_inputs.buffers
                    packed_key = _pack_model_graph_key(key)
                    output_buffers = self._model.capture(
                        packed_key, *input_buffers
                    )
                    if not self._num_speculative_tokens:
                        outputs = ModelOutputs(*output_buffers)
                    else:
                        assert len(output_buffers) == 3, "Expected 3 outputs"
                        outputs = UnifiedEagleOutputs(
                            num_accepted_draft_tokens=output_buffers[0],
                            next_tokens=output_buffers[1],
                            next_draft_tokens=output_buffers[2],
                        )
                    # Graph-capture warmup keeps many output handles alive.
                    # Drop Python-side ownership so later captures can reuse
                    # the same memory-manager-backed storage.
                    outputs = _release_graph_capture_outputs_to_borrowed(
                        outputs
                    )
                    self.graph_entries[key] = (input_buffers, outputs)

        if hasattr(self._model, "_await_device_graphs"):
            logger.info(
                "Awaiting remaining device graph instantiation threads."
            )
            t0 = time.perf_counter()
            self._model._await_device_graphs()
            logger.info(
                "Device graph instantiation complete in %.3fs.",
                time.perf_counter() - t0,
            )

        logger.info(
            "Overlap device graph pre-capture complete for decode batch sizes "
            "[1..%d] with num_steps=1.",
            self._max_batch_size,
        )

    def _broadcast_num_partitions(
        self,
        ragged_inputs: Sequence[KVCacheInputsPerDevice],
        num_partitions: int,
        is_draft: bool = False,
    ) -> None:
        """Overwrites num_partitions in every shard's packed metadata buffer."""
        cpu_buf: Buffer | None = None
        for kv in ragged_inputs:
            metadata = (
                kv.attention_dispatch_metadata
                if not is_draft
                else kv.draft_attention_dispatch_metadata
            )
            assert metadata is not None
            if cpu_buf is None:
                metadata_np = metadata.to_numpy().copy()
                metadata_np[2] = np.int64(num_partitions)
                cpu_buf = Buffer.from_numpy(metadata_np)
            metadata.inplace_copy_from(cpu_buf.to(metadata.device))

    def _resolve_dp_replay_key(
        self,
        ragged_inputs: Sequence[KVCacheInputsPerDevice],
        batch_token_count: int,
    ) -> GraphKey:
        """Resolves graph key for DP by syncing dispatch metadata across replicas."""
        all_metadata = [_unpack_replay_metadata(kv) for kv in ragged_inputs]
        synced_np = max(num_partitions for num_partitions, _ in all_metadata)
        q_max_seq_len = max(q_max_seq_len for _, q_max_seq_len in all_metadata)

        if q_max_seq_len != 1 + self._num_speculative_tokens:
            raise RuntimeError(
                f"q_max_seq_len={q_max_seq_len} != 1 + "
                f"{self._num_speculative_tokens}; only q_max_seq_len=1 + "
                f"{self._num_speculative_tokens} graphs are captured."
            )

        synced_draft_np = max(
            _unpack_dispatch_metadata(kv.draft_attention_dispatch_metadata)[0]
            for kv in ragged_inputs
        )
        final_np, final_draft_np = self._bucket_num_partitions(
            batch_token_count, synced_np, q_max_seq_len, synced_draft_np
        )
        if any(np != final_np for np, _ in all_metadata):
            self._broadcast_num_partitions(ragged_inputs, final_np)
        if any(
            _unpack_dispatch_metadata(kv.draft_attention_dispatch_metadata)[0]
            != final_draft_np
            for kv in ragged_inputs
        ):
            self._broadcast_num_partitions(
                ragged_inputs, final_draft_np, is_draft=True
            )
        return (batch_token_count, final_np, q_max_seq_len, final_draft_np)

    def _bucket_num_partitions(
        self,
        batch_token_count: int,
        num_partitions: int,
        q_max_seq_len: int,
        draft_num_partitions: int,
    ) -> tuple[int, int]:
        """Buckets ``(num_partitions, draft_num_partitions)`` per-axis.

        MLA axes bucket up to the smallest captured value at or above the
        runtime value; MHA axes require an exact match. The draft axis is
        bucketed first so the target axis considers only captures consistent
        with the chosen draft value.
        """
        key = (
            batch_token_count,
            num_partitions,
            q_max_seq_len,
            draft_num_partitions,
        )
        if key in self.graph_entries:
            return num_partitions, draft_num_partitions

        keys_at_batch = [
            k
            for k in self.graph_entries
            if k[0] == batch_token_count and k[2] == q_max_seq_len
        ]

        if self._draft_is_mla and draft_num_partitions > 0:
            # Fall back to all draft nps at this batch when no captures
            # match the exact target np yet; the target bucketing below will
            # then settle on a consistent target np.
            captured_draft_nps = sorted(
                {k[3] for k in keys_at_batch if k[1] == num_partitions}
                or {k[3] for k in keys_at_batch}
            )
            bucketed_draft = self._draft_probe_strategy.bucket_num_partitions(
                draft_num_partitions, captured_draft_nps
            )
            if bucketed_draft is None:
                raise RuntimeError(
                    f"No captured device graph for {key}. "
                    f"Available draft_num_partitions for batch_token_count="
                    f"{batch_token_count}: {captured_draft_nps}. "
                    f"Available batch_token_counts: "
                    f"{sorted({k[0] for k in self.graph_entries})}."
                )
        else:
            bucketed_draft = draft_num_partitions

        if self._target_is_mla:
            captured_nps = sorted(
                {k[1] for k in keys_at_batch if k[3] == bucketed_draft}
            )
            bucketed_np = self._target_probe_strategy.bucket_num_partitions(
                num_partitions, captured_nps
            )
            if bucketed_np is None:
                raise RuntimeError(
                    f"No captured device graph for {key}. "
                    f"Available num_partitions for batch_token_count="
                    f"{batch_token_count}, draft_num_partitions="
                    f"{bucketed_draft}: {captured_nps}. "
                    f"Available batch_token_counts: "
                    f"{sorted({k[0] for k in self.graph_entries})}."
                )
        else:
            bucketed_np = num_partitions
            if (
                batch_token_count,
                bucketed_np,
                q_max_seq_len,
                bucketed_draft,
            ) not in self.graph_entries:
                raise RuntimeError(
                    f"No captured device graph for {key}. "
                    f"Available batch_token_counts: "
                    f"{sorted({k[0] for k in self.graph_entries})}."
                )
        return bucketed_np, bucketed_draft

    def _resolve_replay_key(self, model_inputs: ModelInputs) -> GraphKey:
        """Resolves the replay graph key for a decode batch."""
        ragged_inputs = _ragged_kv_inputs_from_model_inputs(model_inputs)
        batch_token_count = int(model_inputs.buffers[0].shape[0])

        if self._is_data_parallel and len(ragged_inputs) > 1:
            return self._resolve_dp_replay_key(ragged_inputs, batch_token_count)

        num_partitions, q_max_seq_len = _unpack_replay_metadata(
            ragged_inputs[0]
        )

        if num_partitions < 1:
            raise ValueError(
                "Expected positive decode kernel mode (num_partitions), got "
                f"{num_partitions}."
            )

        if q_max_seq_len != 1 + self._num_speculative_tokens:
            raise RuntimeError(
                f"q_max_seq_len={q_max_seq_len} != 1 + "
                f"{self._num_speculative_tokens}; only q_max_seq_len=1 + "
                f"{self._num_speculative_tokens} graphs are captured."
            )

        draft_np, _ = _unpack_replay_metadata(ragged_inputs[0], is_draft=True)

        final_np, final_draft_np = self._bucket_num_partitions(
            batch_token_count, num_partitions, q_max_seq_len, draft_np
        )
        if final_np != num_partitions:
            self._broadcast_num_partitions(ragged_inputs, final_np)
        if final_draft_np != draft_np:
            self._broadcast_num_partitions(
                ragged_inputs, final_draft_np, is_draft=True
            )
        return (batch_token_count, final_np, q_max_seq_len, final_draft_np)

    @traced
    def replay(
        self,
        *,
        model_inputs: ModelInputs,
        debug_verify_replay: bool = False,
        debug_verify_model_inputs: ModelInputs | None = None,
    ) -> ModelOutputs:
        """Replays a captured graph entry for a replay-safe decode batch.

        ``model_inputs`` are copied into captured replay buffers and used for
        replay. When ``debug_verify_replay`` is enabled, callers may provide
        ``debug_verify_model_inputs`` to verify eager traces against a
        different (but graph-key-equivalent) input shape.
        """
        replay_graph_key = self._resolve_replay_key(model_inputs)

        input_buffers = model_inputs.buffers

        packed_model_graph_key = _pack_model_graph_key(replay_graph_key)
        captured_inputs, outputs = self.graph_entries[replay_graph_key]

        for src_value, dst_value in zip(
            input_buffers, captured_inputs, strict=True
        ):
            dst_value.inplace_copy_from(src_value)

        if debug_verify_replay:
            verify_inputs = debug_verify_model_inputs or model_inputs
            verify_graph_key = self._resolve_replay_key(verify_inputs)
            if verify_graph_key != replay_graph_key:
                raise ValueError(
                    "debug_verify_model_inputs must map to the same graph key "
                    "as replay inputs: "
                    f"{verify_graph_key} != {replay_graph_key}."
                )
            self._model.debug_verify_replay(
                packed_model_graph_key,
                *verify_inputs.buffers,
            )

        self._model.replay(packed_model_graph_key, *captured_inputs)
        return outputs
