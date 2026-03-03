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

from dataclasses import dataclass

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

from .cache_params import KVCacheParams


class DecodeNumPartitionsResolver:
    """Resolves MHA decode partition counts via the kernel-side custom op.

    Encapsulates the ``mo.mha.decode.get_num_partitions`` graph so that
    callers can query partition counts with a simple
    ``resolver(batch_size, max_cache_valid_length)`` call.

    The graph-based custom-op path keeps decode partitioning logic in Mojo as
    the single source of truth until a direct Python binding is available.
    """

    def __init__(
        self, session: InferenceSession, kv_params: KVCacheParams
    ) -> None:
        self._model: Model | None = None
        if kv_params.devices[0].is_cpu():
            return

        with Graph(
            "get_decode_num_partitions",
            input_types=[
                TensorType(DType.int64, shape=[2], device=DeviceRef.CPU())
            ],
        ) as graph:
            (request,) = graph.inputs
            (num_partitions,) = ops.custom(
                "mo.mha.decode.get_num_partitions",
                values=[request.tensor],
                out_types=[
                    TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())
                ],
                parameters={"n_kv_heads": kv_params.n_kv_heads_per_device},
                device=kv_params.devices[0],
            )
            graph.output(num_partitions.tensor)
        self._model = session.load(graph)

    def __call__(
        self, batch_size: int, max_cache_valid_length: int
    ) -> MHADecodeDispatchMetadataScalars:
        """Returns decode dispatch metadata for the given shape."""
        if self._model is None:
            return MHADecodeDispatchMetadataScalars(
                batch_size=batch_size,
                q_max_seq_len=1,
                num_partitions=1,
                max_cache_valid_length=max_cache_valid_length,
            )

        request = Buffer.from_numpy(
            np.array([batch_size, max_cache_valid_length], dtype=np.int64)
        )
        (output,) = self._model(request)
        return MHADecodeDispatchMetadataScalars(
            batch_size=batch_size,
            q_max_seq_len=1,
            num_partitions=int(output.to_numpy()[0]),
            max_cache_valid_length=max_cache_valid_length,
        )


@dataclass(frozen=True)
class MHADecodeDispatchMetadataScalars:
    """Scalar MHA decode dispatch metadata used by ragged decode kernels."""

    batch_size: int
    q_max_seq_len: int
    num_partitions: int
    max_cache_valid_length: int

    def to_buffer(self) -> Buffer:
        return Buffer.from_numpy(
            np.array(
                [
                    self.batch_size,
                    self.q_max_seq_len,
                    self.num_partitions,
                    self.max_cache_valid_length,
                ],
                dtype=np.int64,
            )
        )


def build_max_lengths_tensor(
    num_steps: int, max_seq_length: int, max_cache_length: int
) -> Buffer:
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
