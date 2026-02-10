# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""SSM state cache for Mamba models.

This module provides cache infrastructure for Mamba's selective scan state (SSM state)
analogous to the KV cache used in transformer attention.

Unlike KV cache which stores key-value pairs for attention, SSM state cache stores:
- conv_state: Convolution state of shape (batch, intermediate_size, conv_kernel)
- ssm_state: Selective scan state of shape (batch, intermediate_size, d_state)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul

from max.driver import Tensor
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
)
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.nn.mamba")


@dataclass
class SSMStateInputSymbols:
    """Input type symbols for SSM state cache in graph building.

    These define the tensor types that will be passed as graph inputs
    for SSM state caching during autoregressive generation.
    """

    # Conv state: (num_layers, batch, intermediate_size, conv_kernel)
    conv_state: BufferType
    # SSM state: (num_layers, batch, intermediate_size, d_state)
    ssm_state: BufferType
    # Sequence offset to distinguish prefill (0) from step (>0)
    seqlen_offset: TensorType

    def __iter__(self):
        """Iterate through input types."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset


@dataclass
class SSMStateValues:
    """SSM state values passed through the graph during execution.

    These hold the actual buffer/tensor values during graph execution.
    """

    conv_state: BufferValue
    ssm_state: BufferValue
    seqlen_offset: TensorValue

    def __iter__(self):
        """Iterate through values."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset


@dataclass
class SSMStateCacheInputs:
    """Runtime inputs for SSM state cache (Tensor form).

    These are the actual Tensors passed to model.execute() at runtime.
    """

    conv_state: Tensor
    ssm_state: Tensor
    seqlen_offset: Tensor

    def __iter__(self):
        """Iterate through tensors."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset

    def __len__(self) -> int:
        return 3


@dataclass
class SSMStateCacheParams:
    """Parameters for SSM state cache configuration.

    Similar to KVCacheParams but for Mamba's SSM state.
    """

    dtype: DType
    """Data type for SSM state tensors."""

    num_layers: int
    """Number of Mamba layers."""

    intermediate_size: int
    """Intermediate dimension (d_inner) of the SSM."""

    d_state: int
    """State dimension of the SSM."""

    conv_kernel: int
    """Convolution kernel size (d_conv)."""

    device: DeviceRef
    """Device for the state tensors."""

    devices: Sequence[DeviceRef] | None = None
    """Multiple devices for distributed caching (optional)."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    page_size: int = 128
    """Number of tokens per page (block) when using paged cache strategy."""

    def __post_init__(self):
        """Validate configuration and compute derived fields after initialization."""
        if self.devices is None:
            self.devices = [self.device]

    @property
    def n_devices(self) -> int:
        """Returns the number of devices.

        Returns:
            The number of devices.
        """
        assert self.devices is not None
        return len(self.devices)

    @property
    def shape_per_block(self) -> list[int]:
        """Returns the shape of each cache block.

        Returns:
            The shape of the cache block.
        """
        return [
            self.num_layers,
            self.page_size,
            self.intermediate_size,
            self.conv_kernel,
        ]

    @property
    def bytes_per_block(self) -> int:
        """Returns the number of bytes per cache block.

        Returns:
            The number of bytes per cache block.
        """
        return (
            reduce(mul, self.shape_per_block, 1)
            * self.dtype.size_in_bytes
            * self.n_devices
        )

    def compute_num_device_blocks(
        self,
        available_cache_memory: int,
        max_batch_size: int | None,
        max_seq_len: int | None,
    ) -> int:
        """Computes the number of blocks that can be allocated based on the available cache memory.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.
            max_batch_size: The maximum batch size, or None.
            max_seq_len: The maximum sequence length, or None.

        Returns:
            The number of blocks that can be allocated for a single replica.
        """
        # Compute upper bound of total number of pages required.
        max_blocks_per_req: int | None = None
        max_total_blocks: int | None = None
        if max_seq_len is not None and max_batch_size is not None:
            max_blocks_per_req = math.ceil(max_seq_len / self.page_size)
            max_total_blocks = max_blocks_per_req * max_batch_size

        # Compute total number of blocks allocatable based on available memory.
        available_cache_memory_per_replica = (
            available_cache_memory // self.n_devices
        )
        num_allocable_blocks = (
            available_cache_memory_per_replica // self.bytes_per_block
        )

        if max_total_blocks is not None:
            num_blocks = min(num_allocable_blocks, max_total_blocks)
        else:
            num_blocks = num_allocable_blocks

        # Check if we are allocating sufficient blocks.
        single_page_size_bytes_str = to_human_readable_bytes(
            self.bytes_per_block
        )
        cache_memory_str = to_human_readable_bytes(
            available_cache_memory_per_replica
        )
        across_x_devices_str = (
            f" across {self.n_devices} devices" if self.n_devices > 1 else ""
        )
        if num_allocable_blocks == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes_str} but only "
                f"{cache_memory_str} are available{across_x_devices_str}."
            )

        if max_batch_size is not None and max_batch_size > num_allocable_blocks:
            memory_needed_str = to_human_readable_bytes(
                max_batch_size * self.bytes_per_block
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} "
                f"requests with one token per request. Need to allocate at least {max_batch_size} "
                f"pages ({memory_needed_str}), but only have enough memory for {num_allocable_blocks} "
                f"pages ({cache_memory_str}{across_x_devices_str})."
            )

        if (
            max_blocks_per_req is not None
            and max_blocks_per_req > num_allocable_blocks
        ):
            memory_needed_str = to_human_readable_bytes(
                max_blocks_per_req * self.bytes_per_block
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request "
                f"at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {max_blocks_per_req} "
                f"pages ({memory_needed_str}), but only have enough memory for "
                f"{num_allocable_blocks} pages ({cache_memory_str}{across_x_devices_str})."
            )

        return num_blocks

    def estimated_memory_size(
        self, available_cache_memory: int, max_batch_size: int, max_seq_len: int
    ) -> int:
        """Computes the estimated memory size of the SSM cache used by all replicas.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.
            max_batch_size: The maximum batch size.
            max_seq_len: The maximum sequence length.

        Returns:
            The estimated memory usage of the SSM cache in bytes.
        """
        num_device_blocks = self.compute_num_device_blocks(
            available_cache_memory=available_cache_memory,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        return num_device_blocks * self.bytes_per_block * self.n_devices

    def compute_max_seq_len_fitting_in_cache(
        self, available_cache_memory: int
    ) -> int:
        """Computes the maximum sequence length that can fit in the available cache memory.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.

        Returns:
            The maximum sequence length that can fit in the available cache memory.
        """
        num_blocks = self.compute_num_device_blocks(
            available_cache_memory=available_cache_memory,
            max_batch_size=1,
            # Do not limit the sequence length.
            max_seq_len=None,
        )
        return num_blocks * self.page_size

    def get_input_symbols(self) -> SSMStateInputSymbols:
        """Get graph input type symbols for SSM state cache."""
        # Conv state: (num_layers, batch, intermediate_size, conv_kernel)
        # Using symbolic batch dimension
        conv_state_type = BufferType(
            self.dtype,
            shape=[
                self.num_layers,
                "batch",
                self.intermediate_size,
                self.conv_kernel,
            ],
            device=self.device,
        )

        # SSM state: (num_layers, batch, intermediate_size, d_state)
        ssm_state_type = BufferType(
            self.dtype,
            shape=[
                self.num_layers,
                "batch",
                self.intermediate_size,
                self.d_state,
            ],
            device=self.device,
        )

        # Seqlen offset: scalar
        seqlen_offset_type = TensorType(
            DType.int64,
            shape=[1],
            device=DeviceRef.CPU(),
        )

        return SSMStateInputSymbols(
            conv_state=conv_state_type,
            ssm_state=ssm_state_type,
            seqlen_offset=seqlen_offset_type,
        )

    def allocate_cache(self, batch_size: int) -> SSMStateCacheInputs:
        """Allocate SSM state cache tensors for a given batch size.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            SSMStateCacheInputs with zero-initialized state tensors.
        """
        import numpy as np

        # Initialize conv_state to zeros
        conv_state_np = np.zeros(
            (
                self.num_layers,
                batch_size,
                self.intermediate_size,
                self.conv_kernel,
            ),
            dtype=self.dtype.to_numpy(),
        )
        conv_state = Tensor.from_numpy(conv_state_np).to(
            self.device.to_device()
        )

        # Initialize ssm_state to zeros
        ssm_state_np = np.zeros(
            (self.num_layers, batch_size, self.intermediate_size, self.d_state),
            dtype=self.dtype.to_numpy(),
        )
        ssm_state = Tensor.from_numpy(ssm_state_np).to(self.device.to_device())

        # Initialize seqlen_offset to 0 (prefill mode)
        seqlen_offset = Tensor.from_numpy(np.array([0], dtype=np.int64))

        return SSMStateCacheInputs(
            conv_state=conv_state,
            ssm_state=ssm_state,
            seqlen_offset=seqlen_offset,
        )


def create_ssm_state_params(
    dtype: DType,
    num_layers: int,
    intermediate_size: int,
    d_state: int,
    conv_kernel: int,
    device: DeviceRef,
    devices: Sequence[DeviceRef] | None = None,
    enable_prefix_caching: bool = False,
    page_size: int = 128,
) -> SSMStateCacheParams:
    """Create SSM state cache parameters.

    Args:
        dtype: Data type for state tensors.
        num_layers: Number of Mamba layers.
        intermediate_size: Intermediate dimension of the SSM.
        d_state: State dimension of the SSM.
        conv_kernel: Convolution kernel size.
        device: Device for state tensors.
        devices: Multiple devices for distributed caching (optional).
        enable_prefix_caching: Whether to enable prefix caching.
        page_size: Number of tokens per page (block).

    Returns:
        SSMStateCacheParams instance.
    """
    return SSMStateCacheParams(
        dtype=dtype,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        d_state=d_state,
        conv_kernel=conv_kernel,
        device=device,
        devices=devices,
        enable_prefix_caching=enable_prefix_caching,
        page_size=page_size,
    )
