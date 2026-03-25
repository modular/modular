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
"""Slot-based SSM state cache for Mamba models.

Manages pre-allocated conv and SSM state buffers with per-request slot
assignment, analogous to how PagedKVCacheManager manages KV cache for
transformer models — but much simpler since SSM states are fixed-size
per batch element with no sequence-length scaling.
"""

from __future__ import annotations

import logging

from max.driver import Buffer, Device
from max.dtype import DType
from max.interfaces import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


class SSMStateCache:
    """Fixed-size slot-based cache for Mamba SSM conv and scan states.

    Pre-allocates state buffers for ``max_slots`` concurrent requests.
    Each slot holds one set of per-layer (conv_state, ssm_state) tensors,
    each shaped ``[1, intermediate, dim]``.

    Lifecycle:
        1. ``claim(request_id)`` — assign a free slot, zero its states
        2. ``get_states(request_ids)`` — retrieve states for a batch
        3. ``update_states(request_ids, new_states)`` — store updated states
        4. ``release(request_id)`` — free the slot

    For batch_size=1 (the common serving case), get/update are zero-copy
    and just return/replace the slot's buffer references. For batch>1,
    states are concatenated along the batch dimension for get, and split
    back for update.
    """

    def __init__(
        self,
        num_layers: int,
        intermediate_size: int,
        d_state: int,
        conv_kernel: int,
        dtype: DType,
        max_slots: int,
        device: Device,
    ) -> None:
        self._num_layers = num_layers
        self._intermediate = intermediate_size
        self._d_state = d_state
        self._conv_kernel = conv_kernel
        self._dtype = dtype
        self._max_slots = max_slots
        self._device = device

        # Per-slot state storage: _slots[slot_idx] is a list of
        # 2*num_layers Buffers alternating [conv_state, ssm_state, ...]
        # Each buffer is shaped [1, intermediate, dim].
        self._slots: list[list[Buffer]] = []
        for _ in range(max_slots):
            self._slots.append(self._make_zero_states())

        # Slot tracking
        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}
        # Tracks whether a slot has been written to by model execution
        # (as opposed to being freshly claimed with zero states).
        self._valid_states: set[int] = set()

        total_bytes = (
            max_slots
            * num_layers
            * intermediate_size
            * (conv_kernel + d_state)
            * dtype.size_in_bytes
        )
        logger.info(
            f"SSM cache: {max_slots} slots x {num_layers} layers = "
            f"{to_human_readable_bytes(total_bytes)}"
        )

    def _make_zero_states(self) -> list[Buffer]:
        """Create a fresh set of zero-filled state buffers for one slot."""
        states: list[Buffer] = []
        for _ in range(self._num_layers):
            states.append(
                Buffer(
                    self._dtype,
                    [1, self._intermediate, self._conv_kernel],
                    self._device,
                )
            )
            states.append(
                Buffer(
                    self._dtype,
                    [1, self._intermediate, self._d_state],
                    self._device,
                )
            )
        return states

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def num_active_slots(self) -> int:
        return len(self._request_to_slot)

    @property
    def max_slots(self) -> int:
        return self._max_slots

    def claim(self, request_id: RequestID) -> int:
        """Assign a slot to a request, zeroing its state buffers.

        If the request is already claimed, returns the existing slot.

        Returns:
            The slot index assigned to this request.

        Raises:
            RuntimeError: If no free slots are available.
        """
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free SSM cache slots ({self._max_slots} slots in use). "
                "Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        # Zero the slot — replace with fresh zero buffers.
        self._slots[slot] = self._make_zero_states()
        return slot

    def release(self, request_id: RequestID) -> None:
        """Free a slot, making it available for future requests."""
        if request_id not in self._request_to_slot:
            return
        slot = self._request_to_slot.pop(request_id)
        self._valid_states.discard(slot)
        self._free_slots.add(slot)

    def contains(self, request_id: RequestID) -> bool:
        return request_id in self._request_to_slot

    def has_valid_state(self, request_id: RequestID) -> bool:
        """Check if a request's slot has been written to by model execution."""
        if request_id not in self._request_to_slot:
            return False
        return self._request_to_slot[request_id] in self._valid_states

    def get_states(self, request_ids: list[RequestID]) -> list[Buffer]:
        """Retrieve state buffers for a batch of requests.

        Args:
            request_ids: Ordered list of request IDs forming the batch.

        Returns:
            List of ``2 * num_layers`` Buffers. For batch_size=1, each is
            shaped ``[1, intermediate, dim]``. For batch>1, states are
            concatenated along dim 0 to ``[batch, intermediate, dim]``.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")

        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid} not found in SSM cache. "
                    "Call claim() before get_states()."
                )

        # Fast path: batch=1, return slot buffers directly (zero-copy).
        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            return list(self._slots[slot])

        # batch>1: concatenate per-slot buffers along dim 0.
        num_state_tensors = 2 * self._num_layers
        slots = [self._request_to_slot[rid] for rid in request_ids]
        result: list[Buffer] = []
        for state_idx in range(num_state_tensors):
            slot_bufs = [self._slots[s][state_idx] for s in slots]
            result.append(_cat_buffers(slot_bufs, self._device))
        return result

    def update_states(
        self, request_ids: list[RequestID], new_states: list[Buffer]
    ) -> None:
        """Store updated state buffers back into their slots.

        Args:
            request_ids: Ordered list of request IDs matching the batch dim.
            new_states: ``2 * num_layers`` Buffers from model output.
                For batch=1, each is ``[1, intermediate, dim]``.
                For batch>1, each is ``[batch, intermediate, dim]``.
        """
        if len(new_states) != 2 * self._num_layers:
            raise ValueError(
                f"Expected {2 * self._num_layers} state tensors, "
                f"got {len(new_states)}"
            )

        # Fast path: batch=1, just store buffer references directly.
        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            self._slots[slot] = list(new_states)
            self._valid_states.add(slot)
            return

        # batch>1: split along dim 0 and store per-slot.
        for state_idx, state_buf in enumerate(new_states):
            for batch_idx, rid in enumerate(request_ids):
                slot = self._request_to_slot[rid]
                # Slice out this request's state and make a contiguous copy.
                self._slots[slot][state_idx] = state_buf[
                    batch_idx : batch_idx + 1
                ].contiguous()

        for rid in request_ids:
            self._valid_states.add(self._request_to_slot[rid])


def _cat_buffers(buffers: list[Buffer], device: Device) -> Buffer:
    """Concatenate buffers along dim 0 via numpy round-trip.

    This is acceptable for small SSM states (typically <1MB per buffer).
    For high-throughput batch>1 serving, this could be replaced with a
    compiled concatenation kernel.
    """
    import numpy as np

    arrays = [b.to_numpy() for b in buffers]
    combined = np.concatenate(arrays, axis=0)
    return Buffer.from_numpy(combined).to(device)
