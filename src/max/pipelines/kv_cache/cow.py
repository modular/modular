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

"""Prefix cache to enable reuse of KV projections during context encoding with PagedAttention."""

from __future__ import annotations

from typing import Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    SymbolicDim,
    TensorType,
    TensorValue,
    ops,
)
from max.profiler import traced


@traced
def construct_cow_strided_memcpy_graph(
    block_shape: list[int | str], dtype: DType, devices: list[Device]
) -> Graph:
    """
    Returns a graph for performing COW operations on the KV cache.
    """
    assert len(block_shape) == 6
    ds = [DeviceRef(device.label, device.id) for device in devices]
    batch_dim = SymbolicDim("batch")
    blocks_ty = [BufferType(dtype, shape=block_shape, device=d) for d in ds]
    block_src_idx_ty = TensorType(DType.uint32, shape=[batch_dim])
    block_dst_idx_ty = TensorType(DType.uint32, shape=[batch_dim])
    num_tokens_ty = TensorType(DType.uint32, shape=[batch_dim])
    max_num_tokens_ty = TensorType(DType.uint32, shape=[])

    with Graph(
        "mo.kv_collection_cow_strided_memcpy.paged",
        input_types=[
            block_dst_idx_ty,
            block_src_idx_ty,
            num_tokens_ty,
            max_num_tokens_ty,
            *blocks_ty,
        ],
        output_types=[],
    ) as graph:
        (
            block_dst_idx_tensor,
            block_src_idx_tensor,
            num_tokens_tensor,
            max_num_tokens_tensor,
            *all_blocks,
        ) = graph.inputs

        assert isinstance(block_dst_idx_tensor, TensorValue)
        assert isinstance(block_src_idx_tensor, TensorValue)
        assert isinstance(num_tokens_tensor, TensorValue)
        assert isinstance(max_num_tokens_tensor, TensorValue)

        for blocks in all_blocks:
            assert isinstance(blocks, BufferValue)
            dev_ref = blocks.device
            assert dev_ref is not None

            ops.inplace_custom(
                "mo.kv_collection_cow_strided_memcpy.paged",
                values=[
                    blocks,
                    block_dst_idx_tensor.to(dev_ref),
                    block_src_idx_tensor.to(dev_ref),
                    num_tokens_tensor.to(dev_ref),
                    max_num_tokens_tensor.to(dev_ref),
                ],
                out_types=[],
            )
        graph.output()

    return graph


class CowExecutor:
    def __init__(
        self,
        session: InferenceSession,
        block_shape: list[int | str],
        dtype: DType,
        devices: list[Device],
        tensors: list[Tensor],
        page_size: int,
        enable_prefix_caching: bool,
    ):
        self.tensors = tensors
        self.cow_enqueued_args: list[tuple[int, int, int]] = []
        self.cow_blocks_copied = 0
        self.cow_strided_memcpy_model: Optional[Model] = None
        if page_size > 1 and enable_prefix_caching:
            # List of (block_dst, block_src, num_tokens)
            self.cow_strided_memcpy_model = session.load(
                construct_cow_strided_memcpy_graph(
                    block_shape,
                    dtype,
                    devices,
                ),
            )

    def enqueue_cow(self, block_dst: int, block_src: int, num_tokens: int):
        assert self.cow_strided_memcpy_model is not None
        self.cow_enqueued_args.append((block_dst, block_src, num_tokens))

    @traced
    def batch_async_execute(self):
        """Execute all of the COW memcpy operations enqueued during `fetch`.

        This launches 1 kernel even if we need N strided memcpys.
        """
        if len(self.cow_enqueued_args) == 0:
            return

        assert self.cow_strided_memcpy_model is not None
        # Convert the list of (block_dst, block_src, num_tokens) to tensors
        args = np.array(self.cow_enqueued_args, dtype=np.uint32)
        self.cow_blocks_copied += len(self.cow_enqueued_args)
        self.cow_enqueued_args = []
        # copy is needed to make the tensors contiguous
        block_dst_idx_tensor = np.ascontiguousarray(args[:, 0])
        block_src_idx_tensor = np.ascontiguousarray(args[:, 1])
        num_tokens_tensor = np.ascontiguousarray(args[:, 2])
        max_num_tokens_scalar = np.max(num_tokens_tensor)

        # Execute the COW operation
        self.cow_strided_memcpy_model.execute(
            block_dst_idx_tensor,
            block_src_idx_tensor,
            num_tokens_tensor,
            max_num_tokens_scalar,
            *self.tensors,
        )

    def reset_cow_blocks_copied(self):
        self.cow_blocks_copied = 0
        self.cow_blocks_copied = 0
