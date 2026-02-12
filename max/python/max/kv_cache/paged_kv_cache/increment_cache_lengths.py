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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import KVCacheParams, RaggedKVCacheInputs
from max.nn.legacy.kv_cache.data_parallelism_utils import (
    split_input_row_offsets,
    split_into_groups,
)
from max.profiler import traced


def _build_ragged_increment_cache_lengths_graph(
    params: KVCacheParams,
    devices: list[Device],
    use_broadcast: bool,
) -> Graph:
    input_symbols = params.get_symbolic_inputs()
    cache_lengths_types = [
        input_symbols[i].cache_lengths for i in range(len(devices))
    ]
    dp = params.data_parallel_degree

    devices_per_replica = split_into_groups(devices, dp)

    device0 = devices[0]
    device0_ref = DeviceRef.from_device(device0)
    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["input_row_offsets_len"],
        device=device0_ref,
    )

    data_parallel_splits_type = TensorType(
        DType.int64,
        shape=[dp + 1],
        device=DeviceRef.CPU(),
    )

    # Build input types list
    input_types: list[TensorType | BufferType] = [
        input_row_offsets_type,
        data_parallel_splits_type,
        *cache_lengths_types,
    ]

    # Add signal buffer types when using broadcast
    signal_buffer_types: list[BufferType] = []
    if use_broadcast:
        device_refs = [DeviceRef(d.label, d.id) for d in devices]
        signals = Signals(devices=device_refs)
        signal_buffer_types = signals.input_types()
        input_types.extend(signal_buffer_types)

    with Graph(
        "update_cache_lengths",
        input_types=input_types,
    ) as graph:
        # Unpack inputs
        # row_offsets + splits + cache_lengths
        num_fixed_inputs = 2 + len(devices)
        inp_row_offset, data_parallel_splits, *cache_lengths = [
            inp.tensor for inp in graph.inputs[:num_fixed_inputs]
        ]

        # Unpack signal buffers if using broadcast
        signal_buffers = None
        if use_broadcast:
            signal_buffers = [
                inp.buffer for inp in graph.inputs[num_fixed_inputs:]
            ]

        split_offsets = split_input_row_offsets(
            dp,
            inp_row_offset,
            data_parallel_splits,
        )
        outputs = []
        start_idx = 0
        for replica_idx, replica_devices in enumerate(devices_per_replica):
            # Use broadcast to transfer row_offset to all devices in parallel.
            # Currently only enabled for DP=1 (single replica with all devices).
            if use_broadcast:
                assert signal_buffers is not None
                row_offsets_per_device = ops.distributed_broadcast(
                    split_offsets[replica_idx], signal_buffers
                )
                for i in range(len(replica_devices)):
                    row_offset = row_offsets_per_device[i]
                    cache_length = cache_lengths[start_idx + i]
                    assert isinstance(cache_length, TensorValue)
                    right_slice = row_offset[1:].rebind(cache_length.shape)
                    left_slice = row_offset[: row_offset.shape[0] - 1].rebind(
                        cache_length.shape
                    )
                    increment_amount = right_slice - left_slice
                    outputs.append(cache_length + increment_amount)
            else:
                # Fall back to sequential .to(device) transfers for DP>1.
                # TODO(SERVOPT-970): Replace with scatter+broadcast for DP>1.
                for i, device in enumerate(replica_devices):
                    row_offset = split_offsets[replica_idx].to(
                        DeviceRef.from_device(device)
                    )
                    cache_length = cache_lengths[start_idx + i]
                    assert isinstance(cache_length, TensorValue)
                    right_slice = row_offset[1:].rebind(cache_length.shape)
                    left_slice = row_offset[: row_offset.shape[0] - 1].rebind(
                        cache_length.shape
                    )
                    increment_amount = right_slice - left_slice
                    outputs.append(cache_length + increment_amount)
            start_idx += len(replica_devices)
        graph.output(*outputs)

    return graph


@traced
def _execute_ragged_increment_cache_lengths_graph(
    model: Model,
    params: KVCacheParams,
    devices: list[Device],
    use_broadcast: bool,
    kv_cache_inputs: Sequence[RaggedKVCacheInputs],
    prev_model_inputs: Any,
) -> Sequence[RaggedKVCacheInputs]:
    """Prepares cache inputs for the next token in multistep execution.

    Updates the cache lengths for the next inference step without requiring device
    synchronization or memory copies. This is crucial for maintaining performance
    during multi-token generation.

    Args:
        model: Loaded model executing the increment cache lengths graph.
        params: KVCache parameters (e.g. data parallel degree).
        devices: Devices to run on (one per replica).
        use_broadcast: Whether to use broadcast for row-offset transfers.
        kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, max_lengths).
        prev_model_inputs: Previous model inputs including row offsets.

    Returns:
        Updated cache input tuples with incremented lengths.
    """
    blocks = [kv_cache_inputs[i].blocks for i in range(len(devices))]
    cache_lengths = [
        kv_cache_inputs[i].cache_lengths for i in range(len(devices))
    ]
    lookup_table = [
        kv_cache_inputs[i].lookup_table for i in range(len(devices))
    ]
    devices_per_replica = split_into_groups(
        devices, params.data_parallel_degree
    )

    if params.data_parallel_degree > 1:
        data_parallel_splits = prev_model_inputs.data_parallel_splits
    else:
        batch_size = cache_lengths[0].shape[0]
        data_parallel_splits = Buffer.from_numpy(
            np.array([0, batch_size], dtype=np.int64)
        )

    # Update the cache_lengths of our batch by the previous sequence length.
    # Handle both single tensor and list of tensors for compatibility
    if isinstance(prev_model_inputs.input_row_offsets, list):
        # InternVL case: use the first tensor (row offsets are identical across devices)
        row_offsets = prev_model_inputs.input_row_offsets[0]
    else:
        # Standard case: single tensor
        row_offsets = prev_model_inputs.input_row_offsets
    row_offsets = row_offsets.to(devices[0])

    # Build execution args, including signal buffers when using broadcast
    exec_args: list[Buffer] = [
        row_offsets,
        data_parallel_splits,
        *cache_lengths,
    ]
    if use_broadcast:
        if not hasattr(prev_model_inputs, "signal_buffers"):
            raise ValueError(
                "signal_buffers required in model inputs when broadcast is "
                "enabled (data_parallel_degree=1 with multiple devices)"
            )
        exec_args.extend(prev_model_inputs.signal_buffers)

    updated_cache_lengths = model.execute(*exec_args)

    start_idx = 0
    for replica_devices in devices_per_replica:
        # max_lengths is ho st allocated and the same across each replica.
        max_lengths = kv_cache_inputs[start_idx].max_lengths

        # Advance to the next step of the max_lengths tensor.
        updated_max_lengths = max_lengths[1:, :]

        # Return our updated batch.
        assert isinstance(kv_cache_inputs, list)
        for i in range(len(replica_devices)):
            updated_cache_length = updated_cache_lengths[start_idx + i]
            assert isinstance(updated_cache_length, Buffer)
            kv_cache_inputs[start_idx + i] = RaggedKVCacheInputs(
                blocks=blocks[start_idx + i],
                cache_lengths=updated_cache_length,
                lookup_table=lookup_table[start_idx + i],
                max_lengths=updated_max_lengths,
            )
        start_idx += len(replica_devices)
    return kv_cache_inputs


class IncrementCacheLengthsProcessor:
    def __init__(
        self,
        session: InferenceSession,
        params: KVCacheParams,
        devices: list[Device],
    ) -> None:
        # Enable broadcast for row_offset transfers when DP=1 with multiple devices.
        # - DP=1 check: DP>1 requires scatter semantics (different data per replica),
        #   not yet supported. See SERVOPT-970 for DP>1 broadcast support.
        # - len(devices)>1 check: single-device models don't provide signal_buffers
        #   in their ModelInputs. The broadcast kernel is a no-op for single device,
        #   but we need signal_buffers as graph inputs to call it.
        self._use_broadcast = (
            params.data_parallel_degree == 1 and len(devices) > 1
        )

        graph = _build_ragged_increment_cache_lengths_graph(
            params, devices, self._use_broadcast
        )
        self._model = session.load(graph)
        self._params = params
        self._devices = devices

    def execute(
        self,
        kv_cache_inputs: Sequence[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> Sequence[RaggedKVCacheInputs]:
        """Runs the increment cache lengths graph and returns updated cache inputs."""
        return _execute_ragged_increment_cache_lengths_graph(
            self._model,
            self._params,
            self._devices,
            self._use_broadcast,
            kv_cache_inputs,
            prev_model_inputs,
        )
