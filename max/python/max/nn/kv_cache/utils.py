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

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops


class AttentionDispatchResolver:
    """Resolves packed attention decode metadata via kernel custom ops.

    Supports both MHA (``mo.mha.decode.get_num_partitions``) and MLA
    (``mo.mla.compute_dispatch_args.scalar``) decode kernels.  The mode
    is selected automatically from ``kv_params.is_mla``.

    """

    def __init__(
        self,
        session: InferenceSession,
        device: DeviceRef,
        is_mla: bool,
        n_kv_heads_per_device: int,
        num_q_heads: int | None = None,
        is_fp8_kv: bool = False,
    ) -> None:
        self._model: Model | None = None
        self._is_mla = is_mla
        self._device = device

        if device.is_cpu():
            return

        if self._is_mla:
            assert num_q_heads is not None
            self._model = self._build_mla_model(
                session, device, num_q_heads, is_fp8_kv
            )
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
        session: InferenceSession,
        device: DeviceRef,
        num_heads: int,
        is_fp8_kv: bool = False,
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
            (scalars,) = ops.custom(
                "mo.mla.compute_dispatch_args.scalar",
                device=device,
                values=[batch_size_val, max_cache_val, q_max_seq_len_val],
                out_types=[
                    TensorType(
                        shape=[3], dtype=DType.int64, device=DeviceRef.CPU()
                    ),
                ],
                parameters={
                    "num_heads": num_heads,
                    "is_fp8_kv": is_fp8_kv,
                },
            )
            graph.output(scalars.tensor)
        return session.load(graph)

    def __call__(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> Buffer:
        """Returns packed decode dispatch metadata for the given shape."""
        if self._model is None or batch_size <= 0:
            metadata = Buffer.from_numpy(
                np.array(
                    [batch_size, max_prompt_length, 1]
                    + ([] if self._is_mla else [max_cache_valid_length]),
                    dtype=np.int64,
                )
            )
            return (
                metadata.to(self._device.to_device())
                if self._is_mla and not self._device.is_cpu()
                else metadata
            )

        if self._is_mla:
            bs_buf = Buffer.from_numpy(np.array([batch_size], dtype=np.int64))
            mc_buf = Buffer.from_numpy(
                np.array([max_cache_valid_length], dtype=np.int64)
            )
            qs_buf = Buffer.from_numpy(
                np.array([max_prompt_length], dtype=np.int64)
            )
            (output,) = self._model(bs_buf, mc_buf, qs_buf)
            return output.to(self._device.to_device())

        request = Buffer.from_numpy(
            np.array([batch_size, max_cache_valid_length], dtype=np.int64)
        )
        (output,) = self._model(request)
        return Buffer.from_numpy(
            np.array(
                [
                    batch_size,
                    1,
                    int(output.to_numpy()[0]),
                    max_cache_valid_length,
                ],
                dtype=np.int64,
            )
        )


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
