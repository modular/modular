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

"""
Expert Parallelism (EP) Communication Manager.

This module provides classes and utilities for managing Expert Parallelism (EP)
communication in distributed inference scenarios.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Accelerator, Buffer
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
    ops,
)
from max.support.human_readable_formatter import to_human_readable_bytes

from .ep_config import (
    NUM_GROUPS,
    EPConfig,
)
from .ep_kernels import (
    call_ep_combine,
    call_ep_combine_async,
    call_ep_combine_wait,
    call_ep_dispatch,
    call_ep_dispatch_async,
    call_ep_dispatch_wait,
    call_ep_init,
)

logger = logging.getLogger("max.pipelines")


def get_ep_local_sync_counters_size(n_experts: int) -> int:
    """Returns the total size in Int32 elements needed for EP sync counters.

    This must match the EPLocalSyncCounters.total_size() in ep_comm.mojo.

    Memory Layout (all sizes in Int32 elements):
    - dispatch_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - dispatch_wait/combine_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - combine_wait: 2 * n_experts

    Args:
        n_experts: Number of experts in the model.

    Returns:
        Total size in Int32 elements needed for all EP sync counters.
    """

    MAX_GPUS_PER_NODE = 8

    dispatch_async_size = 2 * n_experts + MAX_GPUS_PER_NODE
    dispatch_wait_size = 2 * n_experts + MAX_GPUS_PER_NODE
    combine_wait_size = 2 * n_experts
    return dispatch_async_size + dispatch_wait_size + combine_wait_size


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

    _input_x: dict[int, TensorValue | None] = {}
    """Input tokens for the current device. If shared experts fusion is
    enabled, this will be used to temporarily store the inputs of the MoE
    module, and passed to the ep_dispatch_wait kernel.
    """

    _shared_expert_outputs: dict[int, TensorValue | None] = {}
    """Shared expert outputs for the current device. If shared experts fusion is
    enabled, this will be used to store the outputs of the shared experts from
    the ep_combine kernel."""

    def __init__(self, config: EPConfig):
        """Initialize the EP batch manager.

        Args:
            config: EP configuration.
        """
        self.config = config

    def _common_grouped_matmul_metadata(self) -> TensorValue:
        """Common grouped matmul metadata for all devices. Shape: (2,). Contains
        the max number of tokens per expert and the number of active experts.
        """
        n_ranks = self.config.n_gpus_per_node * self.config.n_nodes
        max_recv_tokens_per_expert = self.config.max_tokens_per_rank * n_ranks
        n_active_experts = self.config.n_experts // n_ranks + (
            1 if self.config.fused_shared_expert else 0
        )
        return ops.constant(
            [
                max_recv_tokens_per_expert,
                n_active_experts,
            ],
            dtype=DType.uint32,
            device=DeviceRef.CPU(),
        )

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

    def _atomic_counters_input_types(self) -> list[BufferType]:
        """Generate input types for atomic counter buffers.

        Returns:
            list[BufferType]: List of buffer types for atomic counters.
        """
        return [
            BufferType(
                DType.int32,
                [get_ep_local_sync_counters_size(self.config.n_experts)],
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

    def input_types(self) -> list[TensorType | BufferType]:
        """Get the input types for the MoE graph.

        Returns:
            list[TensorType | BufferType]: List of input types for atomic
                                          counters and device pointers.
        """
        return (
            self._atomic_counters_input_types() + self._dev_ptrs_input_types()
        )

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

    def ep_dispatch_async(
        self,
        input_tokens: TensorValue,
        topk_ids: TensorValue,
        device_id: int,
        input_scales: TensorValue | None = None,
    ) -> None:
        """Initiate Expert Parallelism token dispatch phase (async).

        This function launches the EP async dispatch kernel that distributes
        input tokens to expert devices based on top-k routing decisions.

        Args:
            input_tokens: Input tokens for the current device. A TensorValue with
                shape (num_local_tokens, hidden_size).
            topk_ids: Top-k expert IDs for the current device. A TensorValue with
                shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
            input_scales: Optional input scales tensor. Required for NVFP4
                dispatch.
        """
        DISPATCH_GROUP = 0
        # Store the symbolic token numbers of each device for the combine phase
        self._dispatch_dim[device_id] = input_tokens.shape[0]
        call_ep_dispatch_async(
            input_tokens,
            topk_ids,
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            input_scales=input_scales,
        )

        if self.config.fused_shared_expert:
            self._input_x[device_id] = input_tokens
        else:
            self._input_x[device_id] = None

    def ep_dispatch_wait(self, device_id: int) -> tuple[TensorValue, ...]:
        """Wait for Expert Parallelism token dispatch phase completion.

        This function launches the EP dispatch wait kernel that waits for all
        transfers to complete for the current GPU, then organizes the received
        tokens into a format suitable for grouped matmul computation.

        Args:
            device_id: Device ID for the current device.

        Returns:
            A tuple containing:
            - output_tokens: Aggregated tokens ready for grouped matmul computation.
                Shape: (max_recv_tokens, hidden_size).
            - output_scales: Aggregated scales ready for grouped matmul computation.
                Only returned for quantized dispatch. Shape depends on format.
            - expert_start_indices: Row offsets for grouped matmul computation.
                Shape: (n_local_experts + 1,).
            - expert_ids: Local expert IDs for the grouped computation.
                Shape: (n_local_experts,).
            - expert_usage_stats: Statistics for the grouped matmul computation.
                Shape: (2,).
        """
        DISPATCH_GROUP = 0

        results = call_ep_dispatch_wait(
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            self._input_x[device_id],
        )

        # The last element is the src_info, we need to store it for the
        # combine phase. Also add the common grouped matmul metadata to the
        # results.
        self._src_info[device_id] = results[-1]

        return (*results[:-1], self._common_grouped_matmul_metadata())

    def ep_combine_async(
        self, input_tokens: TensorValue, device_id: int
    ) -> None:
        """Initiate Expert Parallelism combine phase (async).

        This method launches the async combine phase of Expert Parallelism,
        sending expert outputs back to their original devices based on source
        routing information stored during the dispatch phase.

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
            "Source info is not set, you should call ep_dispatch_wait() first."
        )

        self._shared_expert_outputs[device_id] = call_ep_combine_async(
            input_tokens,
            src_info,
            self.atomic_counters[0][device_id],
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            self._dispatch_dim[device_id],
        )

        # reset src_info to None to avoid reusing it for the next batch
        self._src_info[device_id] = None

    def ep_combine_wait(
        self, router_weight: TensorValue, device_id: int
    ) -> TensorValue:
        """Wait for Expert Parallelism combine phase completion.

        This method waits for all expert output transfers to complete, then
        organizes the received tokens back into their original format and
        positions for the current device.

        Args:
            expert_weights: Router weights for the current device.
                A TensorValue with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.

        Returns:
            Final output tensor with shape (num_local_tokens, hidden_size).
        """
        COMBINE_GROUP = 1

        # Collect results from all devices
        # always use group 0 atomic counters unless we enable
        # two-batch-overlap.
        dispatch_dim = self._dispatch_dim[device_id]
        assert dispatch_dim is not None, (
            "Dispatch dimension is not set, you should call ep_dispatch_async() first."
        )
        results = call_ep_combine_wait(
            self.atomic_counters[0][device_id],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dim,
            router_weight,
        )

        if self.config.fused_shared_expert:
            shared_expert_outputs = self._shared_expert_outputs[device_id]
            assert shared_expert_outputs is not None
            results += shared_expert_outputs

        return results

    # ===-------------------------------------------------------------------===#
    # Fused EP Operations
    # ===-------------------------------------------------------------------===#

    def ep_dispatch(
        self,
        input_tokens: TensorValue,
        topk_ids: TensorValue,
        device_id: int,
        input_scales: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Execute fused Expert Parallelism token dispatch (async + wait).

        This method launches the fused EP dispatch kernel that combines both
        dispatch_async and dispatch_wait functionality in a single kernel
        launch. It distributes input tokens to expert devices, waits for all
        tokens to arrive, and organizes received tokens for grouped matmul.

        For FP8 dispatch, input tokens are quantized to FP8 format during
        dispatch and the output includes both FP8 tokens and their scales.

        Args:
            input_tokens: Input tokens for the current device. A TensorValue
                with shape (num_local_tokens, hidden_size).
            topk_ids: Top-k expert IDs for the current device. A TensorValue
                with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
            input_scales: Optional input scales tensor. Needed for NVFP4
                dispatch.

        Returns:
            A tuple containing:
            - output_tokens: Aggregated tokens ready for grouped matmul.
                Shape: (max_recv_tokens, hidden_size).
            - For FP8: output_scales: Scales for the FP8 tokens.
                Shape: (hidden_size // block_size, max_recv_tokens).
            - expert_start_indices: Row offsets for grouped matmul.
                Shape: (n_local_experts + 1,).
            - expert_ids: Local expert IDs for the grouped operation.
                Shape: (n_local_experts,).
            - expert_usage_stats: Statistics for the grouped matmul.
                Shape: (2,).
        """
        # Use group 0 for both send and recv buffers in fused kernel
        DISPATCH_GROUP = 0

        # Store the symbolic token numbers for the combine phase
        self._dispatch_dim[device_id] = input_tokens.shape[0]

        results = call_ep_dispatch(
            input_tokens,
            topk_ids,
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            input_scales=input_scales,
        )

        # The last element is the src_info, we need to store it for the
        # combine phase. Also add the common grouped matmul metadata to the
        # results.
        self._src_info[device_id] = results[-1]

        return (*results[:-1], self._common_grouped_matmul_metadata())

    def ep_combine(
        self,
        input_tokens: TensorValue,
        router_weight: TensorValue,
        device_id: int,
    ) -> TensorValue:
        """Execute fused Expert Parallelism token combine (async + wait).

        This method launches the fused EP combine kernel that combines both
        combine_async and combine_wait functionality in a single kernel launch.
        It sends expert outputs back to original devices, waits for all
        transfers to complete, and computes the weighted sum of routed expert
        outputs.

        Note: For fused_shared_expert mode with the fused combine kernel, the
        shared expert outputs in input_tokens are automatically added to the
        reduced routed expert outputs.

        Args:
            input_tokens: Expert output tensors from the current device.
                A TensorValue with shape (max_recv_tokens, hidden_size).
                For fused_shared_expert mode, the shared expert outputs are
                stored at the start.
            router_weight: Router weights for the current device.
                A TensorValue with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.

        Returns:
            Final output tensor with shape (num_local_tokens, hidden_size).
        """
        COMBINE_GROUP = 1

        src_info = self._src_info[device_id]
        assert src_info is not None, (
            "Source info is not set, you should call ep_dispatch() or "
            "ep_dispatch_wait() first."
        )

        dispatch_dim = self._dispatch_dim[device_id]
        assert dispatch_dim is not None, (
            "Dispatch dimension is not set, you should call ep_dispatch() or "
            "ep_dispatch_async() first."
        )

        results = call_ep_combine(
            input_tokens,
            src_info,
            self.atomic_counters[0][device_id],
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dim,
            router_weight,
        )

        # Reset src_info to None to avoid reusing it for the next batch
        self._src_info[device_id] = None

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

    send_buf_ptrs: list[Buffer]
    """List of device pointers for the send buffer."""

    recv_buf_ptrs: list[Buffer]
    """List of device pointers for the receive buffer."""

    recv_count_ptrs: list[Buffer]
    """List of device pointers for the receive count buffer."""

    atomic_counters: list[Buffer]
    """List of atomic counters used for synchronization."""

    def __init__(self, config: EPConfig):
        """Initialize the EP communication initializer.

        Args:
            config: EP configuration.
        """
        self.config = config
        # Allocated based on the EPLocalSyncCounters struct in ep_comm.mojo
        self.atomic_counter_size = get_ep_local_sync_counters_size(
            self.config.n_experts
        )

        # Create atomic counters for each GPU in each buffer group
        self.atomic_counters = [
            Buffer(
                DType.int32,
                [self.atomic_counter_size],
                device=Accelerator(i % self.config.n_gpus_per_node),
            )
            for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
        ]

    def _build_ep_init_graph(self) -> Graph:
        """Build the computation graph for EP initialization.

        Creates a graph that initializes SHMEM context and allocates symmetric
        memory buffers on each GPU. The graph takes atomic counter buffers as
        input and returns device pointers to allocated SHMEM buffers.

        Returns:
            Graph: Computation graph for EP initialization.
        """

        atomic_counter_shape = self.atomic_counters[0].shape
        with Graph(
            "ep_init",
            input_types=[
                BufferType(
                    DType.int32,
                    atomic_counter_shape,
                    device=DeviceRef.GPU(i % self.config.n_gpus_per_node),
                )
                for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
            ],
        ) as g:
            dev_ptrs_list: list[TensorValue] = []
            my_rank_list: list[TensorValue] = []

            # Initialize SHMEM context and allocate buffers for each GPU
            for i in range(self.config.n_gpus_per_node):
                # Get atomic counter buffers for both groups
                atomic_counter_group_0 = g.inputs[i].buffer
                atomic_counter_group_1 = g.inputs[
                    i + self.config.n_gpus_per_node
                ].buffer

                # Call the custom EP initialization kernel
                dev_ptrs, my_rank = call_ep_init(
                    atomic_counter_group_0, atomic_counter_group_1, self.config
                )
                # Device pointers cannot be output as CPU tensors since the InferenceSession
                # may not be initialized with CPU; moved to the device as a workaround.
                dev_ptrs_list.append(dev_ptrs.to(atomic_counter_group_0.device))

                my_rank_list.append(my_rank)

            my_ranks = ops.concat(my_rank_list, axis=0)

            g.output(*dev_ptrs_list, my_ranks.to(DeviceRef.GPU(0)))
        return g

    def ep_init(self, session: InferenceSession) -> None:
        """Initialize Expert Parallelism communication infrastructure.

        Args:
            session: Inference session used to compile and execute the graph.
        """
        logger.info("Initializing EP communication infrastructure...")
        logger.info(
            f"Estimated EP memory usage per device: {to_human_readable_bytes(self.config.estimate_memory_usage())}"
        )

        # Skip setting NVSHMEM-specific env vars on single node.
        if self.config.n_nodes > 1 or os.getenv("NVSHMEM_DISABLE_P2P") == "1":
            # Set ENVs for NVSHMEM
            n_gpus = self.config.n_nodes * self.config.n_gpus_per_node
            num_experts_per_gpu = self.config.n_experts // n_gpus
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
            os.environ["NVSHMEM_IBGDA_NIC_HANDLER"] = "gpu"
            os.environ["NVSHMEM_IBGDA_RC_MAP_BY"] = "warp"
            os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = str(num_experts_per_gpu)

            # TODO: Provide a way to let user manually map NICs to different GPU
            os.environ["NVSHMEM_ENABLE_NIC_PE_MAPPING"] = "1"

        # Build and compile the initialization graph
        graph = self._build_ep_init_graph()
        self.init_model = session.load(graph)

        # Execute the graph to initialize SHMEM and get device pointers
        all_outputs = self.init_model.execute(*self.atomic_counters)
        all_outputs_np: list[npt.NDArray[Any]] = []
        for dev_ptr in all_outputs:
            assert isinstance(dev_ptr, Buffer)
            all_outputs_np.append(dev_ptr.to_numpy())

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
            Buffer.from_numpy(dev_ptr) for dev_ptr in send_buf_ptrs_np
        ]
        self.recv_buf_ptrs = [
            Buffer.from_numpy(dev_ptr) for dev_ptr in recv_buf_ptrs_np
        ]
        self.recv_count_ptrs = [
            Buffer.from_numpy(dev_ptr) for dev_ptr in recv_count_ptrs_np
        ]

        # The last element is the my_ranks tensor
        my_ranks_np = all_outputs_np[-1]
        my_node_id = my_ranks_np // self.config.n_gpus_per_node

        # check if all GPUs in the same node have the same node_id
        if not np.all(my_node_id == my_node_id[0]):
            raise ValueError(
                "All GPUs in the same node must have the same node ID."
            )
        self.config.node_id = my_node_id[0]

        logger.info(f"Initialized EP for node {self.config.node_id}")

    def model_inputs(self) -> list[Buffer]:
        """Get the model inputs for the MoE model.

        Returns:
            list[Buffer]: List of all tensors needed as model inputs.
        """
        return (
            self.atomic_counters
            + self.send_buf_ptrs
            + self.recv_buf_ptrs
            + self.recv_count_ptrs
        )
