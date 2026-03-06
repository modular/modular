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

"""Builds computation graphs for incrementing cache lengths in ragged tensor operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputs, KVCacheInputsPerDevice, KVCacheParams
from max.nn.kv_cache.data_parallelism_utils import (
    split_input_row_offsets,
    split_into_groups,
)
from max.profiler import traced


def _build_ragged_increment_cache_lengths_graph(
    params: KVCacheParams,
    devices: list[Device],
    use_comm_kernel: bool,
) -> Graph:
    input_symbols = params.get_symbolic_inputs()
    cache_lengths_types = [
        input_symbols[i].cache_lengths for i in range(len(devices))
    ]
    dp = params.data_parallel_degree

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

    # Add signal buffer types for comm kernels (broadcast or scatter).
    if use_comm_kernel:
        device_refs = [DeviceRef(d.label, d.id) for d in devices]
        signals = Signals(devices=device_refs)
        input_types.extend(signals.input_types())

    with Graph(
        "update_cache_lengths",
        input_types=input_types,
    ) as graph:
        # Unpack inputs: row_offsets + splits + cache_lengths
        num_fixed_inputs = 2 + len(devices)
        inp_row_offset, data_parallel_splits, *cache_lengths = [
            inp.tensor for inp in graph.inputs[:num_fixed_inputs]
        ]

        split_offsets = split_input_row_offsets(
            dp,
            inp_row_offset,
            data_parallel_splits,
        )

        if use_comm_kernel:
            signal_buffers = [
                inp.buffer for inp in graph.inputs[num_fixed_inputs:]
            ]
            # Use comm kernels for parallel row_offset transfer.
            if dp == 1:
                # DP=1: broadcast same data to all GPUs.
                row_offsets_all = ops.distributed_broadcast(
                    split_offsets[0], signal_buffers
                )
            else:
                # DP>1: scatter different chunks to different replica groups.
                row_offsets_all = ops.distributed_scatter(
                    split_offsets, signal_buffers
                )
        else:
            # Single device: use split_offsets directly.
            row_offsets_all = split_offsets

        outputs = []
        for gpu_idx in range(len(devices)):
            row_offset = row_offsets_all[gpu_idx]
            cache_length = cache_lengths[gpu_idx]
            assert isinstance(cache_length, TensorValue)
            right_slice = row_offset[1:].rebind(cache_length.shape)
            left_slice = row_offset[: row_offset.shape[0] - 1].rebind(
                cache_length.shape
            )
            increment_amount = right_slice - left_slice
            outputs.append(cache_length + increment_amount)

        graph.output(*outputs)

    return graph


@traced
def _execute_ragged_increment_cache_lengths_graph(
    model: Model,
    params: KVCacheParams,
    devices: list[Device],
    use_comm_kernel: bool,
    kv_cache_inputs: KVCacheInputs,
    prev_model_inputs: Any,
) -> KVCacheInputs:
    """Prepares cache inputs for the next token in multistep execution.

    Updates the cache lengths for the next inference step without requiring device
    synchronization or memory copies. This is crucial for maintaining performance
    during multi-token generation.

    Args:
        model: Loaded model executing the increment cache lengths graph.
        params: KVCache parameters (e.g. data parallel degree).
        devices: Devices to run on (one per replica).
        use_comm_kernel: Whether to use comm kernels (broadcast/scatter) for
            row-offset transfers.
        kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, max_lengths).
        prev_model_inputs: Previous model inputs including row offsets.

    Returns:
        Updated cache input tuples with incremented lengths.
    """
    blocks = [kv_cache_inputs.inputs[i].blocks for i in range(len(devices))]
    cache_lengths = [
        kv_cache_inputs.inputs[i].cache_lengths for i in range(len(devices))
    ]
    lookup_table = [
        kv_cache_inputs.inputs[i].lookup_table for i in range(len(devices))
    ]
    kv_scales = [
        kv_cache_inputs.inputs[i].kv_scales for i in range(len(devices))
    ]
    attention_dispatch_metadata = [
        kv_cache_inputs.inputs[i].attention_dispatch_metadata
        for i in range(len(devices))
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

    # Build execution args, including signal buffers for comm kernels.
    exec_args: list[Buffer] = [
        row_offsets,
        data_parallel_splits,
        *cache_lengths,
    ]
    if use_comm_kernel:
        if not hasattr(prev_model_inputs, "signal_buffers"):
            raise ValueError(
                "signal_buffers required in model inputs when using "
                "comm kernels (broadcast/scatter) with multiple devices"
            )
        exec_args.extend(prev_model_inputs.signal_buffers)

    updated_cache_lengths = model.execute(*exec_args)

    start_idx = 0
    for replica_devices in devices_per_replica:
        # max_lengths is host allocated and the same across each replica.
        max_lengths = kv_cache_inputs.inputs[start_idx].max_lengths

        # Advance to the next step of the max_lengths tensor.
        updated_max_lengths = max_lengths[1:, :]

        metadata = attention_dispatch_metadata[start_idx]
        if metadata is None:
            raise ValueError(
                "attention_dispatch_metadata must be present in KV cache inputs"
            )
        metadata_np = metadata.to_numpy().copy()
        if updated_max_lengths.shape[0] > 0:
            # Update dispatch metadata with new max_cache_valid_length.
            updated_max_lengths_np = updated_max_lengths.to_numpy()
            metadata_np[3] = np.int64(updated_max_lengths_np[0, 1])
        updated_metadata_cpu = Buffer.from_numpy(metadata_np)

        # Return our updated batch.
        assert isinstance(kv_cache_inputs.inputs, list)
        for i in range(len(replica_devices)):
            updated_cache_length = updated_cache_lengths[start_idx + i]
            assert isinstance(updated_cache_length, Buffer)
            # Preserve the original device (GPU for MLA, CPU for MHA).
            orig = attention_dispatch_metadata[start_idx + i]
            if orig is not None and not orig.device.is_host:
                dev_metadata = updated_metadata_cpu.to(orig.device)
            else:
                dev_metadata = updated_metadata_cpu
            kv_cache_inputs.inputs[start_idx + i] = KVCacheInputsPerDevice(
                blocks=blocks[start_idx + i],
                cache_lengths=updated_cache_length,
                lookup_table=lookup_table[start_idx + i],
                max_lengths=updated_max_lengths,
                kv_scales=kv_scales[start_idx + i],
                attention_dispatch_metadata=dev_metadata,
            )
        start_idx += len(replica_devices)
    return kv_cache_inputs


class IncrementCacheLengthsProcessor:
    """Processes KV cache length increments after each decoding step."""

    def __init__(
        self,
        session: InferenceSession,
        params: KVCacheParams,
        devices: list[Device],
    ) -> None:
        # Use comm kernels (broadcast for DP=1, scatter for DP>1) when there
        # are multiple GPU devices. CPU-only or single-device models don't
        # provide signal_buffers in their ModelInputs.
        self._use_comm_kernel = len(devices) > 1 and devices[0].label == "gpu"

        graph = _build_ragged_increment_cache_lengths_graph(
            params, devices, self._use_comm_kernel
        )
        self._model = session.load(graph)
        self._params = params
        self._devices = devices

    def execute(
        self,
        kv_cache_inputs: KVCacheInputs,
        prev_model_inputs: Any,
    ) -> KVCacheInputs:
        """Runs the increment cache lengths graph and returns updated cache inputs."""
        return _execute_ragged_increment_cache_lengths_graph(
            self._model,
            self._params,
            self._devices,
            self._use_comm_kernel,
            kv_cache_inputs,
            prev_model_inputs,
        )
