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
- Serving path replays by (batch_token_count, num_partitions, q_max_seq_len)
  key. Only ``q_max_seq_len=1`` graphs are captured; a ``RuntimeError`` is
  raised for any other value.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace

import numpy as np
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession, Model
from max.nn.kv_cache import (
    AttentionDispatchMetadataScalars,
    AttentionDispatchResolver,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParams,
)

from .interfaces import ModelInputs, ModelOutputs

logger = logging.getLogger("max.pipelines")


GraphKey = tuple[int, int, int]
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]
WarmupModelInputs = Callable[[int, int], AbstractContextManager[ModelInputs]]


class AttentionMetadataProbeStrategy(ABC):
    """Determines which num_partitions values to capture during warmup."""

    @abstractmethod
    def probe_lengths(self, max_cache_length: int) -> list[int]:
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

    def probe_lengths(self, max_cache_length: int) -> list[int]:
        """Probes at ``granularity`` intervals from 1 to ``max_cache_length``."""
        return (
            [1]
            + list(range(self.granularity, max_cache_length, self.granularity))
            + [max_cache_length]
        )


class MLAProbeStrategy(AttentionMetadataProbeStrategy):
    """MLA: probe at 256-token granularity, bucket up to nearest captured np."""

    granularity = 256

    def probe_lengths(self, max_cache_length: int) -> list[int]:
        """Probes at ``granularity`` intervals from 1 to ``max_cache_length``."""
        return (
            [1]
            + list(range(self.granularity, max_cache_length, self.granularity))
            + [max_cache_length]
        )

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


def _create_model_inputs_with_dispatch_metadata(
    model_inputs: ModelInputs,
    source_ragged: Sequence[KVCacheInputsPerDevice],
    dispatch_metadata: AttentionDispatchMetadataScalars,
) -> ModelInputs:
    """Returns a copy of *model_inputs* with capture dispatch metadata."""
    max_cache_u32 = np.uint32(dispatch_metadata.max_cache_valid_length)
    # MLA: the resolver produced the buffer on the primary GPU.
    # Each TP shard needs metadata on its own device.
    # MHA: a single CPU buffer is shared across all shards.
    gpu_buf = dispatch_metadata.device_buffer
    cpu_buf = None if gpu_buf is not None else dispatch_metadata.to_buffer()
    capture_ragged: list[KVCacheInputsPerDevice] = []
    for kv in source_ragged:
        ml = kv.max_lengths.to_numpy().copy()
        ml[:, 1] = max_cache_u32
        if gpu_buf is not None:
            # Place metadata on the same device as this shard's KV blocks.
            metadata = gpu_buf.to(kv.blocks.device)
        else:
            assert cpu_buf is not None
            metadata = cpu_buf
        capture_ragged.append(
            replace(
                kv,
                max_lengths=Buffer.from_numpy(ml),
                attention_dispatch_metadata=metadata,
            )
        )
    result = copy.copy(model_inputs)
    result.kv_cache_inputs = KVCacheInputs(inputs=capture_ragged)
    return result


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
    ) -> None:
        self._model = model
        self._execute_model = execute_model
        self._warmup_model_inputs = warmup_model_inputs
        if max_cache_length_upper_bound < 1:
            raise ValueError(
                "Decode graph capture requires a positive decode "
                "max-cache length upper bound."
            )
        self._max_cache_length_upper_bound = max_cache_length_upper_bound
        self._resolver = AttentionDispatchResolver(
            session=session,
            device=kv_params.devices[0],
            is_mla=kv_params.is_mla,
            n_kv_heads_per_device=kv_params.n_kv_heads_per_device,
            num_q_heads=kv_params.num_q_heads,
        )
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        self._max_batch_size = max_batch_size
        self._probe_strategy: AttentionMetadataProbeStrategy = (
            MLAProbeStrategy() if kv_params.is_mla else MHAProbeStrategy()
        )
        self._is_data_parallel = kv_params.data_parallel_degree > 1

        self.graph_entries: dict[GraphKey, GraphEntry] = {}

    def dispatch_metadata(
        self, batch_size: int, q_max_seq_len: int
    ) -> list[AttentionDispatchMetadataScalars]:
        """Returns capture metadata selected by the probe strategy.

        Probes at regular cache-length intervals to discover distinct
        num_partitions modes, then delegates to the strategy to select
        which modes to actually capture.
        """
        probe_lengths = self._probe_strategy.probe_lengths(
            self._max_cache_length_upper_bound
        )
        metadata_by_num_partitions = {
            (
                metadata := self._resolver(batch_size, q_max_seq_len, length)
            ).num_partitions: metadata
            for length in probe_lengths
        }
        return list(metadata_by_num_partitions.values())

    def warmup_pre_ready(self) -> None:
        """Captures decode buckets before the worker becomes ready."""
        logger.info(
            "Pre-capturing overlap device graphs for decode batch sizes [1..%d] "
            "with num_steps=1.",
            self._max_batch_size,
        )
        # Conservative/defensive warmup: capture largest-first so peak
        # allocations happen up front and oversized configs fail fast.
        # TODO: Support q_max_seq_len > 1. We currently OOM.
        for batch_size in range(self._max_batch_size, 0, -1):
            with self._warmup_model_inputs(batch_size, 1) as model_inputs:
                batch_token_count = int(model_inputs.buffers[0].shape[0])
                source_ragged = _ragged_kv_inputs_from_model_inputs(
                    model_inputs
                )
                for dispatch_metadata in self.dispatch_metadata(batch_size, 1):
                    key = (
                        batch_token_count,
                        dispatch_metadata.num_partitions,
                        1,
                    )
                    assert key not in self.graph_entries, (
                        "unexpected duplicate key"
                    )

                    capture_inputs = (
                        _create_model_inputs_with_dispatch_metadata(
                            model_inputs,
                            source_ragged,
                            dispatch_metadata,
                        )
                    )
                    # Warmup eager twice for stable kernel/runtime
                    # initialization.
                    self._execute_model(capture_inputs)
                    self._execute_model(capture_inputs)

                    input_buffers = capture_inputs.buffers
                    packed_key = _pack_model_graph_key(key)
                    self.graph_entries[key] = (
                        input_buffers,
                        ModelOutputs(
                            *self._model.capture(packed_key, *input_buffers)
                        ),
                    )
                    for device in self._model.input_devices:
                        Accelerator(id=device.id).synchronize()

        logger.info(
            "Overlap device graph pre-capture complete for decode batch sizes "
            "[1..%d] with num_steps=1.",
            self._max_batch_size,
        )

    def _resolve_replay_key(self, model_inputs: ModelInputs) -> GraphKey:
        """Resolves the replay graph key, handling DP sync.

        1. If DP: syncs max_cache_valid_length to global max across replicas.
        2. Reads the synced num_partitions (max across replicas for DP,
           shard 0 otherwise).
        3. If DP: broadcasts canonical metadata (with synced np) to all
           shards.

        For DP models, we synchronize dispatch metadata across DP replicas so
        all devices agree on num_partitions. The captured CUDA graph bakes
        uniform grid dimensions, so replicas with shorter caches get extra
        CTAs that early-exit.

        Returns the resolved ``GraphKey``.
        """
        ragged_inputs = _ragged_kv_inputs_from_model_inputs(model_inputs)
        batch_token_count = int(model_inputs.buffers[0].shape[0])

        def _np_val(kv: KVCacheInputsPerDevice) -> int:
            meta = kv.attention_dispatch_metadata
            return int(meta.to_numpy()[2]) if meta is not None else 0

        def _q_seq_val(kv: KVCacheInputsPerDevice) -> int:
            meta = kv.attention_dispatch_metadata
            return int(meta.to_numpy()[1]) if meta is not None else 0

        if self._is_data_parallel and len(ragged_inputs) > 1:
            synced_np = max(_np_val(kv) for kv in ragged_inputs)
            q_max_seq_len = max(_q_seq_val(kv) for kv in ragged_inputs)
        else:
            synced_np = _np_val(ragged_inputs[0])
            q_max_seq_len = _q_seq_val(ragged_inputs[0])

        if synced_np < 1:
            raise ValueError(
                "Expected positive decode kernel mode (num_partitions), got "
                f"{synced_np}."
            )

        if q_max_seq_len != 1:
            raise RuntimeError(
                f"q_max_seq_len={q_max_seq_len} != 1; only q_max_seq_len=1 "
                "graphs are captured."
            )

        # Broadcast canonical metadata to all shards for DP.
        if self._is_data_parallel and len(ragged_inputs) > 1:
            primary_meta = ragged_inputs[0].attention_dispatch_metadata
            if primary_meta is not None:
                canonical = primary_meta.to_numpy().copy()
                canonical[1] = q_max_seq_len
                canonical[2] = synced_np
                for kv in ragged_inputs:
                    if kv.attention_dispatch_metadata is not None:
                        kv.attention_dispatch_metadata.inplace_copy_from(
                            Buffer.from_numpy(canonical).to(
                                kv.attention_dispatch_metadata.device
                            )
                        )

        replay_graph_key = (batch_token_count, synced_np, q_max_seq_len)
        if replay_graph_key not in self.graph_entries:
            captured_nps = sorted(
                {
                    k[1]
                    for k in self.graph_entries
                    if k[0] == batch_token_count and k[2] == q_max_seq_len
                }
            )
            bucketed_np = self._probe_strategy.bucket_num_partitions(
                synced_np, captured_nps
            )
            if bucketed_np is None:
                raise RuntimeError(
                    f"No captured device graph for {replay_graph_key}. "
                    f"Available num_partitions for batch_token_count="
                    f"{batch_token_count}: {captured_nps}. "
                    f"Available batch_token_counts: "
                    f"{sorted({k[0] for k in self.graph_entries})}."
                )
            replay_graph_key = (batch_token_count, bucketed_np, q_max_seq_len)

            # Patch num_partitions in each shard's dispatch metadata to match
            # the captured value. This is presumably safe because num_partitions only
            # controls the MLA decode kernel's CTA grid size.
            bucketed_arr = np.int64(bucketed_np)
            for kv in ragged_inputs:
                if kv.attention_dispatch_metadata is not None:
                    meta = kv.attention_dispatch_metadata.to_numpy().copy()
                    meta[2] = bucketed_arr
                    kv.attention_dispatch_metadata.inplace_copy_from(
                        Buffer.from_numpy(meta).to(
                            kv.attention_dispatch_metadata.device
                        )
                    )

        return replay_graph_key

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
