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
"""LFM2's per-request short-convolution states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.pipelines.context import TextContext
from max.pipelines.lib.request_state import RequestStateStore
from max.pipelines.modeling.types import RequestID

_BATCH_DIM = "conv_batch"


def _cat_buffers(buffers: list[Buffer], device: Device) -> Buffer:
    """Concatenate buffers along dim 0 via numpy round-trip.

    Used for the ``N > 1`` get-states fast path; conv states are small
    (``hidden * kernel`` per slot) so this round-trip is acceptable.

    numpy has no native bfloat16 support and rejects bfloat16 DLPack
    tensors with ``RuntimeError: Unsupported dtype in DLTensor``, so we
    reinterpret bfloat16 as uint16 (same byte width) for the round-trip
    and view the result back to the source dtype.
    """
    src_dtype = buffers[0].dtype
    if src_dtype == DType.bfloat16:
        viewed = [b.view(DType.uint16) for b in buffers]
        arrays = [v.to_numpy() for v in viewed]
        combined = np.concatenate(arrays, axis=0)
        return Buffer.from_numpy(combined).to(device).view(src_dtype)

    arrays = [b.to_numpy() for b in buffers]
    combined = np.concatenate(arrays, axis=0)
    return Buffer.from_numpy(combined).to(device)


def _split_buffer_dim0(buf: Buffer, device: Device) -> list[Buffer]:
    """Split a buffer into per-row slices along dim 0 via numpy round-trip.

    Mirror of ``_cat_buffers`` for the ``N > 1`` update-states path.
    ``Buffer.__getitem__`` requires one index per tensor dimension, so a
    rank-3 ``[batch, hidden, kernel]`` conv-state buffer cannot be sliced
    with a single slice expression; we round-trip through numpy instead,
    matching the bfloat16 view trick used in ``_cat_buffers``.
    """
    src_dtype = buf.dtype
    if src_dtype == DType.bfloat16:
        arr = buf.view(DType.uint16).to_numpy()
        return [
            Buffer.from_numpy(arr[i : i + 1]).to(device).view(src_dtype)
            for i in range(arr.shape[0])
        ]

    arr = buf.to_numpy()
    return [
        Buffer.from_numpy(arr[i : i + 1]).to(device)
        for i in range(arr.shape[0])
    ]


class ConvStateStore(RequestStateStore):
    """Per-request slots for the convolution states.

    The decoder graph takes each convolution layer's states for the batch,
    ``[batch, hidden, kernel]``, after its KV cache inputs, and returns the
    updated ones after its logits, which the store writes back into each
    request's slot.
    """

    def __init__(
        self,
        num_conv_layers: int,
        hidden_size: int,
        conv_kernel: int,
        dtype: DType,
        max_slots: int,
        device: Device,
    ) -> None:
        self._num_conv_layers = num_conv_layers
        self._hidden_size = hidden_size
        self._conv_kernel = conv_kernel
        self._dtype = dtype
        self._device = device
        self._max_slots = max_slots
        self._slots = [self._make_slot() for _ in range(max_slots)]
        self._free = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

    def _make_slot(self) -> list[Buffer]:
        return [
            Buffer.zeros(
                [1, self._hidden_size, self._conv_kernel],
                self._dtype,
                self._device,
            )
            for _ in range(self._num_conv_layers)
        ]

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Returns one ``[batch, hidden, kernel]`` state per conv layer."""
        device_ref = DeviceRef.from_device(self._device)
        return tuple(
            TensorType(
                self._dtype,
                [_BATCH_DIM, self._hidden_size, self._conv_kernel],
                device=device_ref,
            )
            for _ in range(self._num_conv_layers)
        )

    def max_dims(self) -> Mapping[str, int]:
        """Bounds the batch dim by the slots, one per request in flight."""
        return {_BATCH_DIM: self._max_slots}

    def prepare(self, batch: Sequence[TextContext]) -> tuple[Buffer, ...]:
        """Returns the batch's states, claiming a slot for a new request."""
        request_ids = [ctx.request_id for ctx in batch]
        for request_id in request_ids:
            self.claim(request_id)
        return tuple(self.get_states(request_ids))

    def update(
        self, request_ids: Sequence[RequestID], outputs: Sequence[Buffer]
    ) -> None:
        """Writes the step's updated states back into the requests' slots.

        Raises:
            ValueError: If the outputs aren't one ``[batch, hidden, kernel]``
                state per conv layer for these requests.
        """
        expected = (len(request_ids), self._hidden_size, self._conv_kernel)
        if not request_ids or len(outputs) != self._num_conv_layers:
            raise ValueError(
                f"expected {self._num_conv_layers} conv states for "
                f"{len(request_ids)} requests, got {len(outputs)}"
            )
        for output in outputs:
            if tuple(output.shape) != expected:
                raise ValueError(
                    f"expected conv states of shape {list(expected)}, got "
                    f"{list(output.shape)}"
                )
        self.update_states(list(request_ids), list(outputs))

    def claim(self, request_id: RequestID) -> None:
        """Returns the slot a request writes its convolution state to."""
        if request_id in self._request_to_slot:
            return
        if not self._free:
            raise RuntimeError("No free LFM2 conv-state slots.")
        slot = self._free.pop()
        self._request_to_slot[request_id] = slot
        self._slots[slot] = self._make_slot()

    def release(self, request_id: RequestID) -> None:
        """Frees the slot a request held."""
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self._free.add(slot)

    def get_states(self, request_ids: list[RequestID]) -> list[Buffer]:
        """Return one ``[N, hidden, kernel]`` buffer per conv layer.

        For ``N == 1`` this is zero-copy (returns the slot's buffer
        directly). For ``N > 1`` per-slot buffers are concatenated along
        the leading batch dim via numpy round-trip — the conv state is
        small (``hidden * kernel`` per slot), so this is acceptable.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid!r} has no allocated conv-state slot. "
                    "Call claim() before get_states()."
                )

        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            return list(self._slots[slot])

        slots = [self._request_to_slot[rid] for rid in request_ids]
        result: list[Buffer] = []
        for layer_idx in range(self._num_conv_layers):
            slot_bufs = [self._slots[s][layer_idx] for s in slots]
            result.append(_cat_buffers(slot_bufs, self._device))
        return result

    def update_states(
        self, request_ids: list[RequestID], new_states: list[Buffer]
    ) -> None:
        """Store updated per-layer states back into their request slots.

        For ``N == 1`` the buffer reference is stored directly. For
        ``N > 1`` the leading batch dim is split and each slice is
        copied into the matching slot.
        """
        if len(new_states) != self._num_conv_layers:
            raise ValueError(
                f"Expected {self._num_conv_layers} state tensors, "
                f"got {len(new_states)}"
            )

        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            self._slots[slot] = list(new_states)
            return

        for layer_idx, state_buf in enumerate(new_states):
            pieces = _split_buffer_dim0(state_buf, self._device)
            for batch_idx, rid in enumerate(request_ids):
                slot = self._request_to_slot[rid]
                self._slots[slot][layer_idx] = pieces[batch_idx]
