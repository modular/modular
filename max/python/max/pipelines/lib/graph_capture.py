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
- Serving path replays by (batch_token_count, decode_kernel_mode) key,
  where decode_kernel_mode is the MHA decode num_partitions value.
"""

from __future__ import annotations

import copy
import logging
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


GraphKey = tuple[int, int]
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]
WarmupModelInputs = Callable[[int], AbstractContextManager[ModelInputs]]

_GRAPH_KEY_COMPONENT_MAX = 2**32 - 1

# MHA decode kernel mode (num_partitions) changes at multiples of 256 (AMD)
# or 512 (NVIDIA). Probing every 256 tokens catches all mode transitions on
# both architectures.
_PARTITION_PROBE_GRANULARITY = 256


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


def attention_graph_key_from_inputs(model_inputs: ModelInputs) -> GraphKey:
    """Builds a replay key from token count and decode kernel mode."""
    batch_token_count = int(model_inputs.buffers[0].shape[0])
    metadata = _ragged_kv_inputs_from_model_inputs(model_inputs)[
        0
    ].attention_dispatch_metadata
    if metadata is None:
        raise ValueError(
            "Expected attention_dispatch_metadata in "
            "KVCacheInputs for overlap graph capture."
        )
    num_partitions = int(metadata.to_numpy()[2])
    if num_partitions < 1:
        raise ValueError(
            "Expected positive decode kernel mode (num_partitions), got "
            f"{num_partitions}."
        )
    return (batch_token_count, num_partitions)


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


def _pack_model_graph_key(replay_graph_key: GraphKey) -> int:
    batch_token_count, num_partitions = replay_graph_key
    if batch_token_count < 0 or num_partitions < 0:
        raise ValueError(
            f"graph key values must be non-negative, got {replay_graph_key}."
        )
    if (
        batch_token_count > _GRAPH_KEY_COMPONENT_MAX
        or num_partitions > _GRAPH_KEY_COMPONENT_MAX
    ):
        raise ValueError(
            f"graph key values exceed uint32 packing range: {replay_graph_key}."
        )
    return (num_partitions << 32) | batch_token_count


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
            q_max_seq_len=kv_params.q_max_seq_len,
            num_q_heads=kv_params.num_q_heads,
        )
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        self._max_batch_size = max_batch_size

        self.graph_entries: dict[GraphKey, GraphEntry] = {}

    def dispatch_metadata(
        self, batch_size: int
    ) -> list[AttentionDispatchMetadataScalars]:
        """Returns capture metadata for distinct decode kernel modes."""
        # Probe at _PARTITION_PROBE_GRANULARITY intervals to discover every
        # distinct decode kernel mode (num_partitions). Collect the largest
        # representative cache length for each mode (used as the capture upper
        # bound).
        ub = self._max_cache_length_upper_bound
        probe_lengths = (
            [1]
            + list(
                range(
                    _PARTITION_PROBE_GRANULARITY,
                    ub,
                    _PARTITION_PROBE_GRANULARITY,
                )
            )
            + [ub]
        )

        metadata_by_num_partitions = {
            (
                metadata := self._resolver(batch_size, length)
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
        for batch_size in range(self._max_batch_size, 0, -1):
            with self._warmup_model_inputs(batch_size) as model_inputs:
                batch_token_count = int(model_inputs.buffers[0].shape[0])
                source_ragged = _ragged_kv_inputs_from_model_inputs(
                    model_inputs
                )
                # Build the full ModelInputs once per batch size, then derive
                # per-mode variants by only patching KV dispatch metadata and
                # max-cache lengths. This avoids re-running reserve/prepare for
                # every decode kernel mode while still capturing each replay
                # key.
                for dispatch_metadata in self.dispatch_metadata(batch_size):
                    replay_graph_key = (
                        batch_token_count,
                        dispatch_metadata.num_partitions,
                    )
                    assert replay_graph_key not in self.graph_entries, (
                        "unexpected duplicate key"
                    )

                    capture_inputs = (
                        _create_model_inputs_with_dispatch_metadata(
                            model_inputs, source_ragged, dispatch_metadata
                        )
                    )
                    # Warmup eager twice for stable kernel/runtime
                    # initialization.
                    self._execute_model(capture_inputs)
                    self._execute_model(capture_inputs)

                    input_buffers = capture_inputs.buffers
                    packed_model_graph_key = _pack_model_graph_key(
                        replay_graph_key
                    )
                    self.graph_entries[replay_graph_key] = (
                        input_buffers,
                        ModelOutputs(
                            *self._model.capture(
                                packed_model_graph_key, *input_buffers
                            )
                        ),
                    )
                    for device in self._model.input_devices:
                        Accelerator(id=device.id).synchronize()

        logger.info(
            "Overlap device graph pre-capture complete for decode batch sizes "
            "[1..%d] with num_steps=1.",
            self._max_batch_size,
        )

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
        input_buffers = model_inputs.buffers
        replay_graph_key = attention_graph_key_from_inputs(model_inputs)
        packed_model_graph_key = _pack_model_graph_key(replay_graph_key)
        try:
            captured_inputs, outputs = self.graph_entries[replay_graph_key]
        except KeyError as exc:
            batch_token_count, _ = replay_graph_key
            available_decode_kernel_modes = sorted(
                key[1]
                for key in self.graph_entries
                if key[0] == batch_token_count
            )
            available_batch_token_counts = sorted(
                {key[0] for key in self.graph_entries}
            )
            raise RuntimeError(
                "No captured device graph found for key: "
                f"{replay_graph_key}. Available decode kernel modes "
                "(num_partitions) for "
                f"batch_token_count={batch_token_count}: "
                f"{available_decode_kernel_modes}. Available "
                "batch_token_counts: "
                f"{available_batch_token_counts}."
            ) from exc

        for src_value, dst_value in zip(
            input_buffers, captured_inputs, strict=True
        ):
            dst_value.inplace_copy_from(src_value)

        if debug_verify_replay:
            verify_inputs = debug_verify_model_inputs or model_inputs
            verify_graph_key = attention_graph_key_from_inputs(verify_inputs)
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
