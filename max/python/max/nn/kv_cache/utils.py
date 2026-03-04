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


class AttentionDispatchResolver:
    """Resolves attention decode partition counts via kernel custom ops.

    Supports both MHA (``mo.mha.decode.get_num_partitions``) and MLA
    (``mo.mla.compute_dispatch_args.scalar``) decode kernels.  The mode
    is selected automatically from ``kv_params.is_mla``.

    Callers query partition counts with a simple
    ``resolver(batch_size, max_cache_valid_length)`` call.

    For MLA, the resolver graph runs entirely on GPU and returns the
    result buffer on the compute device, avoiding a redundant CPU→GPU
    round-trip.  Host-side scalars (needed for graph-key computation)
    are read via an implicit D2H copy inside ``to_numpy()``.
    """

    def __init__(
        self,
        session: InferenceSession,
        device: DeviceRef,
        is_mla: bool,
        n_kv_heads_per_device: int,
        q_max_seq_len: int,
        num_q_heads: int | None = None,
    ) -> None:
        self._model: Model | None = None
        self._is_mla = is_mla
        self._q_max_seq_len = q_max_seq_len if self._is_mla else 1

        if device.is_cpu():
            return

        if self._is_mla:
            assert num_q_heads is not None
            self._model = self._build_mla_model(session, device, num_q_heads)
        else:
            self._model = self._build_mha_model(
                session, device, n_kv_heads_per_device
            )

    @staticmethod
    def _build_mha_model(
        session: InferenceSession, device: DeviceRef, n_kv_heads: int
    ) -> Model:
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
                parameters={"n_kv_heads": n_kv_heads},
                device=device,
            )
            graph.output(num_partitions.tensor)
        return session.load(graph)

    @staticmethod
    def _build_mla_model(
        session: InferenceSession, device: DeviceRef, num_heads: int
    ) -> Model:
        with Graph(
            "mla_dispatch_args",
            input_types=[
                TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
                TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
                TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
            ],
        ) as graph:
            batch_size_val = graph.inputs[0].tensor
            max_cache_val = graph.inputs[1].tensor
            q_max_seq_len_val = graph.inputs[2].tensor
            (gpu_args,) = ops.custom(
                "mo.mla.compute_dispatch_args.scalar",
                device=device,
                values=[batch_size_val, max_cache_val, q_max_seq_len_val],
                out_types=[
                    TensorType(shape=[4], dtype=DType.int64, device=device),
                ],
                parameters={"num_heads": num_heads},
            )
            graph.output(gpu_args.tensor)
        return session.load(graph)

    def __call__(
        self, batch_size: int, max_cache_valid_length: int
    ) -> AttentionDispatchMetadataScalars:
        """Returns decode dispatch metadata for the given shape."""
        if self._model is None or batch_size <= 0:
            return AttentionDispatchMetadataScalars(
                batch_size=batch_size,
                q_max_seq_len=self._q_max_seq_len,
                num_partitions=1,
                max_cache_valid_length=max_cache_valid_length,
            )

        if self._is_mla:
            bs_buf = Buffer.from_numpy(np.array([batch_size], dtype=np.int64))
            mc_buf = Buffer.from_numpy(
                np.array([max_cache_valid_length], dtype=np.int64)
            )
            qs_buf = Buffer.from_numpy(
                np.array([self._q_max_seq_len], dtype=np.int64)
            )
            (output,) = self._model(bs_buf, mc_buf, qs_buf)
            # output lives on GPU.  to_numpy() does an implicit D2H copy
            # so we can read the scalars on the host for graph-key
            # computation, while keeping the device buffer for direct use.
            out_np = output.to_numpy()
            return AttentionDispatchMetadataScalars(
                batch_size=int(out_np[0]),
                q_max_seq_len=int(out_np[1]),
                num_partitions=int(out_np[2]),
                max_cache_valid_length=int(out_np[3]),
                device_buffer=output,
            )

        request = Buffer.from_numpy(
            np.array([batch_size, max_cache_valid_length], dtype=np.int64)
        )
        (output,) = self._model(request)
        return AttentionDispatchMetadataScalars(
            batch_size=batch_size,
            q_max_seq_len=1,
            num_partitions=int(output.to_numpy()[0]),
            max_cache_valid_length=max_cache_valid_length,
        )


@dataclass(frozen=True)
class AttentionDispatchMetadataScalars:
    """Scalar attention dispatch metadata used by ragged decode kernels.

    For MLA the resolver graph runs on GPU and the result buffer is
    kept on device in ``device_buffer``.  Callers that need the
    metadata on GPU should use ``device_buffer`` directly instead of
    round-tripping through ``to_buffer()`` → ``.to(device)``.
    """

    batch_size: int
    q_max_seq_len: int
    num_partitions: int
    max_cache_valid_length: int
    device_buffer: Buffer | None = None

    def to_buffer(self) -> Buffer:
        """Returns a CPU ``[4]`` int64 buffer with the packed metadata."""
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
