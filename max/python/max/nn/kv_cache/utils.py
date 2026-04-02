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

import numpy as np
from max._kv_cache_ops import (
    mha_decode_num_partitions,
    mla_dispatch_args_scalar,
)
from max.driver import Buffer
from max.graph import DeviceRef


class AttentionDispatchResolver:
    """Resolves packed attention decode metadata via kernel custom ops.

    Supports both MHA (``mo.mha.decode.get_num_partitions``) and MLA
    (``mo.mla.compute_dispatch_args.scalar``) decode kernels.  The mode
    is selected automatically from ``kv_params.is_mla``.

    """

    def __init__(
        self,
        devices: Sequence[DeviceRef],
        is_mla: bool,
        n_kv_heads_per_device: int,
        num_q_heads_per_device: int | None = None,
        is_fp8_kv: bool = False,
    ) -> None:
        if not devices:
            raise ValueError("devices must not be empty")
        devices = list(devices)
        self._is_mla = is_mla
        self._output_devices = [
            None if device.is_cpu() else device.to_device()
            for device in devices
        ]
        self._device = self._output_devices[0]
        self._n_kv_heads_per_device = n_kv_heads_per_device
        self._num_q_heads = num_q_heads_per_device
        self._is_fp8_kv = is_fp8_kv
        self.host_only = False

        if self._is_mla:
            assert num_q_heads_per_device is not None

    def _default_metadata(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> Buffer:
        metadata = Buffer.from_numpy(
            np.array(
                [batch_size, max_prompt_length, 1]
                + ([] if self._is_mla else [max_cache_valid_length]),
                dtype=np.int64,
            )
        )
        if self._is_mla and self._device is not None and not self.host_only:
            return metadata.to(self._device)
        return metadata

    def __call__(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> Buffer:
        """Returns packed decode dispatch metadata for the given shape."""
        if batch_size <= 0 or self._device is None:
            return self._default_metadata(
                batch_size, max_prompt_length, max_cache_valid_length
            )

        if self._is_mla:
            assert self._num_q_heads is not None
            metadata = Buffer.from_numpy(
                np.array(
                    mla_dispatch_args_scalar(
                        batch_size,
                        max_cache_valid_length,
                        max_prompt_length,
                        self._num_q_heads,
                        self._is_fp8_kv,
                        self._device,
                    ),
                    dtype=np.int64,
                )
            )
            if not self.host_only and self._device is not None:
                return metadata.to(self._device)
            return metadata

        num_partitions = mha_decode_num_partitions(
            batch_size,
            max_cache_valid_length,
            self._n_kv_heads_per_device,
            self._device,
        )
        return Buffer.from_numpy(
            np.array(
                [batch_size, 1, num_partitions, max_cache_valid_length],
                dtype=np.int64,
            )
        )

    def resolve_for_replica(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> list[Buffer]:
        """Returns one dispatch-metadata buffer per shard in a replica."""
        metadata = self(
            batch_size,
            max_prompt_length,
            max_cache_valid_length,
        )
        if not self._is_mla or self.host_only or self._device is None:
            return [metadata] * len(self._output_devices)

        return [
            metadata
            if device is None or metadata.device == device
            else metadata.to(device)
            for device in self._output_devices
        ]


def build_max_lengths_tensor(
    num_steps: int, max_seq_length: int, max_cache_length: int
) -> Buffer:
    """Builds a ``[num_steps, 2]`` uint32 buffer of per-step maximum lengths.

    Each row encodes the maximum sequence length and maximum cache length for
    that decode step. The first step uses ``max_seq_length``; subsequent steps
    use 1 (one new token per step). Cache length increases by 1 each step.

    Args:
        num_steps: The number of decode steps to pre-compute lengths for.
        max_seq_length: The maximum sequence length for the first step.
        max_cache_length: The maximum cache length for the first step.

    Returns:
        A :class:`~max.driver.Buffer` of shape ``[num_steps, 2]`` and dtype
        ``uint32`` containing ``(max_seq_length, max_cache_length)`` pairs.
    """
    # Build a tensor of maximum lengths. Each step slices the first row to
    # advance to the values for the next row.
    max_lengths_np = np.empty((num_steps, 2), np.uint32)
    step_max_seq_length = max_seq_length
    step_max_cache_length = max_cache_length
    for step in range(num_steps):
        max_lengths_np[step, 0] = step_max_seq_length
        max_lengths_np[step, 1] = step_max_cache_length
        step_max_seq_length = 1
        step_max_cache_length += 1

    return Buffer.from_numpy(max_lengths_np)
