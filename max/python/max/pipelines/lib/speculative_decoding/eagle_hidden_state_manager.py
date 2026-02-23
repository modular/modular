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
"""Hidden state buffer manager for EAGLE speculative decoding."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.interfaces import RequestID
from max.pipelines.core import TextContext

from .eagle_hidden_state_graphs import (
    build_gather_graph,
    build_gather_scatter_graph,
)

logger = logging.getLogger("max.pipelines")


class EagleHiddenStateManager:
    """Hidden state buffer manager for EAGLE speculative decoding."""

    def __init__(
        self,
        hidden_dim: int,
        dtype: DType,
        devices: Sequence[Device],
        max_batch_size: int,
        num_draft_steps: int,
        session: InferenceSession,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._dtype = dtype
        self._devices: Sequence[Device] = devices

        # max rows per request in TG: num_draft_steps + 2
        # (num_draft_steps draft tokens + 1 input token + 1 bonus)
        tg_slot_size = num_draft_steps + 2
        self._tg_slot_size = tg_slot_size
        max_tg_rows = max_batch_size * tg_slot_size

        self._hs_storage: list[Buffer] = [
            Buffer.zeros((max_tg_rows, hidden_dim), dtype=dtype, device=dev)
            for dev in devices
        ]

        self._request_info: dict[RequestID, tuple[int, int, int]] = {}
        self._free_slots: list[int] = list(range(max_batch_size - 1, -1, -1))

        device_refs = [DeviceRef.from_device(dev) for dev in devices]
        self._gather_scatter_model: Model = session.load(
            build_gather_scatter_graph(
                device_refs, dtype, hidden_dim, max_tg_rows
            )
        )
        self._gather_model: Model = session.load(
            build_gather_graph(device_refs, dtype, hidden_dim)
        )

    def release(self, request_id: RequestID) -> None:
        """Releases the hidden state slot for a completed request."""
        info = self._request_info.pop(request_id, None)
        if info is not None:
            self._free_slots.append(info[0])

    def _get_or_allocate_slot(self, request_id: RequestID) -> int:
        info = self._request_info.get(request_id)
        if info is not None:
            return info[0]
        return self._free_slots.pop()

    def save_prefill_hidden_states(
        self,
        context_batch: list[TextContext],
        draft_hs: list[Buffer],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> None:
        """Saves the last hidden state from prefill into per-request slots."""
        model_args: list[Buffer] = []
        for dev_idx in range(len(self._devices)):
            start_batch = int(data_parallel_splits_np[dev_idx])
            end_batch = int(data_parallel_splits_np[dev_idx + 1])

            src_indices: list[int] = []
            dst_indices: list[int] = []
            for batch_idx in range(start_batch, end_batch):
                ctx = context_batch[batch_idx]
                if not ctx.tokens.actively_chunked:
                    slot = self._get_or_allocate_slot(ctx.request_id)
                    slot_start = slot * self._tg_slot_size
                    row_idx = batch_idx - start_batch
                    src_indices.append(row_idx)
                    dst_indices.append(slot_start)
                    self._request_info[ctx.request_id] = (
                        slot,
                        dev_idx,
                        1,
                    )

            dev = self._devices[dev_idx]
            if src_indices:
                gather_np = np.array(src_indices, dtype=np.int64)
                scatter_np = np.array(dst_indices, dtype=np.int64).reshape(
                    -1, 1
                )
            else:
                gather_np = np.array([], dtype=np.int64)
                scatter_np = np.array([], dtype=np.int64).reshape(0, 1)
            gather_buf = Buffer.from_numpy(gather_np).to(dev)
            scatter_buf = Buffer.from_numpy(scatter_np).to(dev)
            model_args.extend(
                [
                    draft_hs[dev_idx],
                    gather_buf,
                    self._hs_storage[dev_idx],
                    scatter_buf,
                ]
            )

        self._gather_scatter_model(*model_args)

    def save_extracted(
        self,
        context_batch: list[TextContext],
        target_hidden_states: list[Buffer],
        logit_offsets_np: npt.NDArray[np.int64],
        first_rejected_tokens: npt.NDArray[np.integer],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> None:
        """Saves accepted hidden states from target verification."""
        model_args: list[Buffer] = []
        for dev_idx in range(len(self._devices)):
            start_batch = int(data_parallel_splits_np[dev_idx])
            end_batch = int(data_parallel_splits_np[dev_idx + 1])
            local_offset = int(logit_offsets_np[start_batch])

            # Allocate slots and compute gather/scatter indices.
            gather_indices: list[int] = []
            scatter_indices: list[int] = []
            for i in range(start_batch, end_batch):
                ctx = context_batch[i]
                if ctx.is_done:
                    continue
                num_rows = int(first_rejected_tokens[i]) + 1
                slot = self._get_or_allocate_slot(ctx.request_id)
                slot_start = slot * self._tg_slot_size
                src_start = int(logit_offsets_np[i]) - local_offset
                for r in range(num_rows):
                    gather_indices.append(src_start + r)
                    scatter_indices.append(slot_start + r)
                self._request_info[ctx.request_id] = (
                    slot,
                    dev_idx,
                    num_rows,
                )

            dev = self._devices[dev_idx]
            if gather_indices:
                gather_np = np.array(gather_indices, dtype=np.int64)
                scatter_np = np.array(scatter_indices, dtype=np.int64).reshape(
                    -1, 1
                )
            else:
                gather_np = np.array([], dtype=np.int64)
                scatter_np = np.array([], dtype=np.int64).reshape(0, 1)
            gather_buf = Buffer.from_numpy(gather_np).to(dev)
            scatter_buf = Buffer.from_numpy(scatter_np).to(dev)
            model_args.extend(
                [
                    target_hidden_states[dev_idx],
                    gather_buf,
                    self._hs_storage[dev_idx],
                    scatter_buf,
                ]
            )

        self._gather_scatter_model(*model_args)

    def get_draft_input(
        self,
        replica_batches: list[list[TextContext]],
    ) -> list[Buffer]:
        """Gathers stored hidden states as input for the draft model."""
        model_args: list[Buffer] = []
        for dev_idx, replica_batch in enumerate(replica_batches):
            gather_indices: list[int] = []
            for ctx in replica_batch:
                info = self._request_info.get(ctx.request_id)
                if info is None:
                    raise ValueError(
                        f"No hidden state for request {ctx.request_id}"
                    )
                slot_idx, src_dev_idx, num_rows = info
                if src_dev_idx != dev_idx:
                    raise ValueError(
                        f"Request {ctx.request_id} hidden state is on "
                        f"device {src_dev_idx} but batch assigned to "
                        f"device {dev_idx}. Cross-device migration is "
                        f"not supported."
                    )
                slot_start = slot_idx * self._tg_slot_size
                for r in range(num_rows):
                    gather_indices.append(slot_start + r)

            dev = self._devices[dev_idx]
            if gather_indices:
                indices_np = np.array(gather_indices, dtype=np.int64)
            else:
                indices_np = np.array([], dtype=np.int64)
            indices_buf = Buffer.from_numpy(indices_np).to(dev)
            model_args.extend([self._hs_storage[dev_idx], indices_buf])

        outputs = self._gather_model(*model_args)
        # Single-device returns a single Buffer; multi-device returns a tuple.
        if isinstance(outputs, Buffer):
            return [outputs]
        result: list[Buffer] = []
        for out in outputs:
            assert isinstance(out, Buffer)
            result.append(out)
        return result
