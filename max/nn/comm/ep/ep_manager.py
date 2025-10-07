# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""
Expert Parallelism (EP) Communication Manager.

This module provides classes and utilities for managing Expert Parallelism (EP)
communication in distributed inference scenarios.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    TensorValue,
    Value,
)

from .ep_config import NUM_GROUPS, EPConfig
from .ep_kernels import (
    call_ep_combine,
    call_ep_combine_cb,
    call_ep_dispatch,
    call_ep_dispatch_cb,
    call_ep_init,
)

logger = logging.getLogger("max.pipelines")


class EPBatchManager:
    """Batch manager for Expert Parallelism (EP).

    This module manages two groups of SHMEM buffers in the graph. It switches
    between the two groups to avoid racing.
    """

    config: EPConfig
    """Configuration for the Expert Parallelism (EP)."""

    _send_buf_ptrs: list[TensorValue] | None
    """SHMEM send buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references addresses to staging
    buffers for outgoing tokens."""

    _recv_buf_ptrs: list[TensorValue] | None
    """SHMEM receive buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references UInt64 addresses to
    buffers for incoming tokens from remote devices."""

    _recv_count_ptrs: list[TensorValue] | None
    """SHMEM receive count buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references UInt64 addresses to
    buffers for signalling transfer completion."""

    _atomic_counters: list[list[BufferValue]] | None
    """Atomic synchronization counters. Shape: [NUM_GROUPS][n_gpus_per_node]
    of BufferValue. Used for inter-thread-block coordination."""

    _src_info: dict[int, TensorValue | None] = {}
    """Source routing information for combine phase. Each key is a device ID,
    and the value is a TensorValue with shape [max_recv_tokens, 2]. Maps expert
    outputs back to their source positions."""

    _dispatch_dim: dict[int, Dim | None] = {}
    """Dictionary of device ID to dimension for the dispatch input tensor.
    Used to determine the shape of the combined output tensor.
    """

    def __init__(self, config: EPConfig):
        """Initialize the EP batch manager.

        Args:
            config: EP configuration.
        """
        self.config = config

    @property
    def send_buf_ptrs(self) -> list[TensorValue]:
        if self._send_buf_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._send_buf_ptrs

    @property
    def recv_buf_ptrs(self) -> list[TensorValue]:
        if self._recv_buf_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._recv_buf_ptrs

    @property
    def recv_count_ptrs(self) -> list[TensorValue]:
        if self._recv_count_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._recv_count_ptrs

    @property
    def atomic_counters(self) -> list[list[BufferValue]]:
        if self._atomic_counters is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._atomic_counters

    def fetch_buffers(self, _input_vals: Iterable[Value[Any]]) -> None:
        """Extract and organize communication buffers from graph input values.

        Args:
            input_vals: List of input values containing all buffer references.
        """
        input_vals = list(_input_vals)
        start_idx = 0
        # First NUM_GROUPS * self.config.n_gpus_per_node elements are atomic counters
        # These are used for synchronization between different thread blocks
        self._atomic_counters = []
        # Organize atomic counters by groups
        for _ in range(NUM_GROUPS):
            end_idx = start_idx + self.config.n_gpus_per_node
            group_buffers = [
                val.buffer for val in input_vals[start_idx:end_idx]
            ]
            self.atomic_counters.append(group_buffers)
            start_idx = end_idx

        # Next NUM_GROUPS are send buffer pointers
        end_idx = start_idx + NUM_GROUPS
        self._send_buf_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]
        start_idx = end_idx

        # Next NUM_GROUPS are recv buffer pointers
        end_idx = start_idx + NUM_GROUPS
        self._recv_buf_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]
        start_idx = end_idx

        # Next NUM_GROUPS are recv count pointers
        end_idx = start_idx + NUM_GROUPS
        self._recv_count_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]
        start_idx = end_idx

    def ep_dispatch(
        self, input_tokens: TensorValue, topk_ids: TensorValue, device_id: int
    ) -> None:
        """Initiate Expert Parallelism token dispatch phase.

        This function launches the EP dispatch kernel that distributes input
        tokens to expert devices based on top-k routing decisions.

        Args:
            input_tokens: Input tokens for the current device. A TensorValue with
                shape (num_local_tokens, hidden_size).
            topk_ids: Top-k expert IDs for the current device. A TensorValue with
                shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
        """
        DISPATCH_GROUP = 0
        # Store the symbolic token numbers of each device for the combine phase
        self._dispatch_dim[device_id] = input_tokens.shape[0]
        call_ep_dispatch(
            input_tokens,
            topk_ids,
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
        )

    def ep_dispatch_cb(
        self, device_id: int
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        """Complete Expert Parallelism token dispatch phase.

        This function launches the EP dispatch callback kernel that waits for
        all transfers to complete for the current GPU, then organizes the received tokens
        into a format suitable for grouped matmul computation.

        Args:
            device_id: Device ID for the current device.

        Returns:
            A tuple containing:
            - output_tokens: Aggregated tokens ready for grouped matmul computation.
                Shape: (max_recv_tokens, hidden_size).
            - expert_start_indices: Row offsets for grouped matmul computation.
                Shape: (n_local_experts + 1,).
            - expert_ids: Local expert IDs for the grouped computation.
                Shape: (n_local_experts,).
            - expert_usage_stats: Statistics for the grouped matmul computation.
                Shape: (2,).
        """
        DISPATCH_GROUP = 0

        # Collect results from all devices
        results = call_ep_dispatch_cb(
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
        )

        # The first four elements are the input for the grouped matmul
        # operation. The last element is the src_info, we need to store it for
        # the combine phase.
        self._src_info[device_id] = results[4]

        return results[:4]

    def ep_combine(self, input_tokens: TensorValue, device_id: int) -> None:
        """Initiate Expert Parallelism combine phase.

        This method launches the combine phase of Expert Parallelism, sending
        expert outputs back to their original devices based on source routing
        information stored during the dispatch phase.

        Args:
            input_tokens: Expert output tensors from the current device.
                A TensorValue with shape (max_recv_tokens, hidden_size).
            device_id: Device ID for the current device.
        """
        COMBINE_GROUP = 1
        # always use group 0 atomic counters unless we enable
        # two-batch-overlap.

        src_info = self._src_info[device_id]
        assert src_info is not None, (
            "Source info is not set, you should call ep_dispatch_cb() first."
        )

        call_ep_combine(
            input_tokens,
            src_info,
            self.atomic_counters[0][device_id],
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
        )

        # reset src_info to None to avoid reusing it for the next batch
        self._src_info[device_id] = None

    def ep_combine_cb(self, device_id: int) -> TensorValue:
        """Complete Expert Parallelism combine phase.

        This method waits for all expert output transfers to complete, then
        organizes the received tokens back into their original format and
        positions for the current device.

        Args:
            device_id: Device ID for the current device.

        Returns:
            Final output tensor with shape (num_local_tokens, top_k, hidden_size).
        """
        COMBINE_GROUP = 1

        # Collect results from all devices
        # always use group 0 atomic counters unless we enable
        # two-batch-overlap.
        dispatch_dim = self._dispatch_dim[device_id]
        assert dispatch_dim is not None, (
            "Dispatch dimension is not set, you should call ep_dispatch() first."
        )
        results = call_ep_combine_cb(
            self.atomic_counters[0][device_id],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dim,
        )

        return results


class EPCommInitializer:
    """Helper class for initializing buffers for Expert Parallelism (EP).

    This class handles the initialization of the SHMEM communication
    infrastructure required for Expert Parallelism. It creates and manages
    atomic counters, initializes the SHMEM library, and allocates symmetric
    memory buffers.
    """

    config: EPConfig
    """EP configuration."""

    init_model: Model
    """Compiled model that sets up the SHMEM library context for local GPUs and
    allocates the SHMEM memory for the send, receive, and receive count buffers."""

    send_buf_ptrs: list[Tensor]
    """List of device pointers for the send buffer."""

    recv_buf_ptrs: list[Tensor]
    """List of device pointers for the receive buffer."""

    recv_count_ptrs: list[Tensor]
    """List of device pointers for the receive count buffer."""

    atomic_counters: list[Tensor]
    """List of atomic counters used for synchronization."""

    def _estimate_ep_memory_usage(self) -> int:
        """Estimate the memory usage for the EP communication.

        Returns:
            int: Total estimated memory usage in bytes.
        """
        # fmt: off
        d_token_size = self.config.hidden_size * self.config.dispatch_dtype.size_in_bytes
        dispatch_send_buf_size = self.config.max_tokens_per_rank * d_token_size
        dispatch_recv_buf_size = self.config.n_experts * self.config.max_tokens_per_rank * d_token_size

        c_token_size = self.config.hidden_size * self.config.combine_dtype.size_in_bytes
        combine_send_buf_size = self.config.n_experts * self.config.max_tokens_per_rank * c_token_size
        combine_recv_buf_size = self.config.top_k * self.config.max_tokens_per_rank * c_token_size

        return dispatch_send_buf_size + dispatch_recv_buf_size + combine_send_buf_size + combine_recv_buf_size
        # fmt: on

    def __init__(self, config: EPConfig):
        """Initialize the EP communication initializer.

        Args:
            config: EP configuration.
        """
        self.config = config
        # Each expert needs 2 atomic counters
        self.atomic_counter_size = 2 * self.config.n_experts

        # Create atomic counters for each GPU in each buffer group
        self.atomic_counters = [
            Tensor(
                DType.int32,
                [self.atomic_counter_size],
                device=Accelerator(i % self.config.n_gpus_per_node),
            )
            for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
        ]

    def _atomic_counters_input_types(self) -> list[BufferType]:
        """Generate input types for atomic counter buffers.

        Returns:
            list[BufferType]: List of buffer types for atomic counters.
        """
        return [
            BufferType(
                DType.int32,
                [self.atomic_counter_size],
                device=DeviceRef.GPU(i % self.config.n_gpus_per_node),
            )
            for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
        ]

    def _dev_ptrs_input_types(self) -> list[TensorType]:
        """Generate input types for device pointer tensors.

        Returns:
            list[TensorType]: List of tensor types for device pointers.
        """
        return (
            [
                TensorType(
                    DType.uint64,
                    [self.config.n_gpus_per_node],
                    device=DeviceRef.CPU(),
                ),
            ]
            * 3  # 3 buffer types: send, recv, recv_count
            * NUM_GROUPS  # For double buffering
        )

    def _build_ep_init_graph(self) -> Graph:
        """Build the computation graph for EP initialization.

        Creates a graph that initializes SHMEM context and allocates symmetric
        memory buffers on each GPU. The graph takes atomic counter buffers as
        input and returns device pointers to allocated SHMEM buffers.

        Returns:
            Graph: Computation graph for EP initialization.
        """
        with Graph(
            "ep_init",
            input_types=self._atomic_counters_input_types(),
        ) as g:
            dev_ptrs_list: list[TensorValue] = []

            # Initialize SHMEM context and allocate buffers for each GPU
            for i in range(self.config.n_gpus_per_node):
                # Get atomic counter buffers for both groups
                atomic_counter_group_0 = g.inputs[i].buffer
                atomic_counter_group_1 = g.inputs[
                    i + self.config.n_gpus_per_node
                ].buffer

                # Call the custom EP initialization kernel
                dev_ptrs = call_ep_init(
                    atomic_counter_group_0, atomic_counter_group_1, self.config
                )
                dev_ptrs_list.append(dev_ptrs)

            g.output(*dev_ptrs_list)
        return g

    def ep_init(self, session: InferenceSession) -> None:
        """Initialize Expert Parallelism communication infrastructure.

        Args:
            session: Inference session used to compile and execute the graph.
        """
        logger.info("Initializing SHMEM context and allocating SHMEM memory...")
        logger.info(
            f"Estimated EP memory usage: {self._estimate_ep_memory_usage() / 1024 / 1024 / 1024:.2f} GB"
        )

        # Build and compile the initialization graph
        graph = self._build_ep_init_graph()
        self.init_model = session.load(graph)

        # Execute the graph to initialize SHMEM and get device pointers
        all_outputs = self.init_model.execute(*self.atomic_counters)
        all_outputs_np: list[npt.NDArray[Any]] = []
        for dev_ptr in all_outputs:
            assert isinstance(dev_ptr, Tensor)
            all_outputs_np.append(np.from_dlpack(dev_ptr))

        # Process the output device pointers:
        # Each device returns a tensor of shape (NUM_GROUPS, 3) where:
        # - NUM_GROUPS of buffers for EP communication
        # - 3 corresponds to: [send_buffer_ptr, recv_buffer_ptr, recv_count_ptr]
        # We reorganize these pointers by buffer type and group for easy access.

        # Reorganize device pointers by buffer type and group
        send_buf_ptrs_np: list[npt.NDArray[Any]] = []
        recv_buf_ptrs_np: list[npt.NDArray[Any]] = []
        recv_count_ptrs_np: list[npt.NDArray[Any]] = []

        for group_idx in range(NUM_GROUPS):
            # Collect pointers from all devices for this group
            curr_group_list: list[npt.NDArray[Any]] = []
            for device_idx in range(self.config.n_gpus_per_node):
                curr_group_list.append(all_outputs_np[device_idx][group_idx])
            curr_group_ptrs = np.stack(curr_group_list, axis=0)

            # Extract pointers by buffer type (send, recv, recv_count)
            send_buf_ptrs_np.append(curr_group_ptrs[:, 0])
            recv_buf_ptrs_np.append(curr_group_ptrs[:, 1])
            recv_count_ptrs_np.append(curr_group_ptrs[:, 2])

        self.send_buf_ptrs = [
            Tensor.from_numpy(dev_ptr) for dev_ptr in send_buf_ptrs_np
        ]
        self.recv_buf_ptrs = [
            Tensor.from_numpy(dev_ptr) for dev_ptr in recv_buf_ptrs_np
        ]
        self.recv_count_ptrs = [
            Tensor.from_numpy(dev_ptr) for dev_ptr in recv_count_ptrs_np
        ]

    def input_types(self) -> list[TensorType | BufferType]:
        """Get the input types for the MoE graph.

        Returns:
            list[TensorType | BufferType]: List of input types for atomic
                                          counters and device pointers.
        """
        return (
            self._atomic_counters_input_types() + self._dev_ptrs_input_types()
        )

    def model_inputs(self) -> list[Tensor]:
        """Get the model inputs for the MoE model.

        Returns:
            list[Tensor]: List of all tensors needed as model inputs.
        """
        return (
            self.atomic_counters
            + self.send_buf_ptrs
            + self.recv_buf_ptrs
            + self.recv_count_ptrs
        )
