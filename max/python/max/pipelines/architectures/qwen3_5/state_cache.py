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
"""GPU-resident slot pool for Qwen3.5 GatedDeltaNet linear attention states.

Manages per-request conv and recurrent state tensors with slot assignment.
States are stored as pre-allocated GPU `Buffer` objects (one `[max_slots, ...]`
pool per layer) in the model's native dtype (typically `bfloat16`).
At each execute step:

* `get_states()` performs a *gather*: for each active slot `s` at batch
  position `b`, it copies `pool[l][s:s+1]` → `batch[l][b:b+1]` using
  `Buffer.inplace_copy_from`. All copies are GPU-to-GPU (no PCIe transfer).

* `update_states()` performs the inverse *scatter*: for each slot `s` at
  batch position `b`, it copies `output[l][b:b+1]` → `pool[l][s:s+1]`.

This eliminates the ~19 GiB/step PCIe round-trip from the old CPU-numpy
approach (~1-2 s per step) replacing it with GPU-to-GPU memcpy (~3-5 ms per
step at HBM bandwidth).

Memory note: the pool occupies permanent GPU memory. At peak, three sets of
state buffers coexist on the GPU: the persistent pool (`max_batch x per_req`)
plus input and output working copies (`batch x per_req` each).
`estimate_activation_memory()` in `model.py` reserves **3 x max_batch x
per_req** bytes so the KV-cache allocator leaves sufficient room for all three.
"""

from __future__ import annotations

import logging

from max.driver import Buffer, Device
from max.dtype import DType
from max.interfaces import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


class GatedDeltaNetStateCache:
    """GPU-resident slot pool for Qwen3.5 GatedDeltaNet conv and recurrent states.

    Two sets of per-layer `Buffer` objects are pre-allocated on the GPU at
    construction time:

    * `conv_pool[l]`:  `[max_slots, conv_dim, conv_kernel-1]`  float32
    * `rec_pool[l]`:   `[max_slots, num_v_heads, key_dim, val_dim]`  float32

    `get_states()` gathers the active-request rows into dense `[batch, ...]`
    working buffers via GPU-to-GPU `inplace_copy_from` (no PCIe).
    `update_states()` scatters the model's output rows back into their pool
    slots the same way.

    Lifecycle:
        1. `claim(request_id)`  — register a request, zero its pool rows
        2. `get_states(request_ids)` — GPU-gather batch Buffers from pool
        3. `update_states(request_ids, conv_buffers, rec_buffers)` — GPU-scatter back
        4. `release(request_id)` — free the slot
    """

    def __init__(
        self,
        num_layers: int,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        max_slots: int,
        device: Device,
        dtype: DType = DType.float32,
    ) -> None:
        self._num_layers = num_layers
        self._conv_dim = conv_dim
        self._conv_kernel = conv_kernel_size - 1  # K-1 (state window size)
        self._num_v_heads = num_v_heads
        self._key_head_dim = key_head_dim
        self._value_head_dim = value_head_dim
        self._max_slots = max_slots
        self._device = device
        self._dtype = dtype

        # Pre-allocate GPU state pools (zero-initialised).
        # conv_pool[l]: [max_slots, conv_dim, K-1]
        # rec_pool[l]:  [max_slots, num_v_heads, key_head_dim, val_head_dim]
        self._conv_pool: list[Buffer] = [
            Buffer.zeros(
                [max_slots, conv_dim, conv_kernel_size - 1],
                dtype,
                device,
            )
            for _ in range(num_layers)
        ]
        self._rec_pool: list[Buffer] = [
            Buffer.zeros(
                [max_slots, num_v_heads, key_head_dim, value_head_dim],
                dtype,
                device,
            )
            for _ in range(num_layers)
        ]

        # Pre-allocated zero buffers used to wipe a slot on claim().
        self._zero_conv = Buffer.zeros(
            [1, conv_dim, conv_kernel_size - 1], dtype, device
        )
        self._zero_rec = Buffer.zeros(
            [1, num_v_heads, key_head_dim, value_head_dim],
            dtype,
            device,
        )

        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

        elem_bytes = dtype.size_in_bytes
        conv_bytes = (
            num_layers
            * max_slots
            * conv_dim
            * (conv_kernel_size - 1)
            * elem_bytes
        )
        rec_bytes = (
            num_layers
            * max_slots
            * num_v_heads
            * key_head_dim
            * value_head_dim
            * elem_bytes
        )
        logger.info(
            f"GatedDeltaNet state pool: {max_slots} slots x {num_layers} layers"
            f" — conv {to_human_readable_bytes(conv_bytes)}"
            f" + rec {to_human_readable_bytes(rec_bytes)}"
            f" = {to_human_readable_bytes(conv_bytes + rec_bytes)} on {device}"
        )

    @property
    def num_free_slots(self) -> int:
        """Number of available (unclaimed) slots."""
        return len(self._free_slots)

    @property
    def num_active_slots(self) -> int:
        """Number of currently claimed slots."""
        return len(self._request_to_slot)

    @property
    def max_slots(self) -> int:
        """Maximum number of concurrent requests supported."""
        return self._max_slots

    def claim(self, request_id: RequestID) -> int:
        """Assign a slot to a request, zeroing its pool rows on the GPU.

        If the request is already claimed, returns the existing slot
        (idempotent — state is preserved for chunked-prefill continuations).

        Args:
            request_id: The unique identifier for the request.

        Returns:
            The slot index assigned to this request.

        Raises:
            RuntimeError: If no free slots are available.
        """
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free GatedDeltaNet state cache slots"
                f" ({self._max_slots} slots in use). "
                "Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        # Zero the slot rows in both pools via GPU-to-GPU copy.
        for l in range(self._num_layers):
            self._conv_pool[l][slot, :, :].inplace_copy_from(self._zero_conv)
            self._rec_pool[l][slot, :, :, :].inplace_copy_from(self._zero_rec)
        return slot

    def release(self, request_id: RequestID) -> None:
        """Free a slot, making it available for future requests.

        Args:
            request_id: The unique identifier for the request to release.
        """
        if request_id not in self._request_to_slot:
            return
        slot = self._request_to_slot.pop(request_id)
        self._free_slots.add(slot)

    def contains(self, request_id: RequestID) -> bool:
        """Returns True if a slot is claimed for this request."""
        return request_id in self._request_to_slot

    def get_states(
        self, request_ids: list[RequestID]
    ) -> tuple[list[Buffer], list[Buffer]]:
        """GPU-gather state Buffers for a batch of requests.

        For each layer `l` and each active request at batch position `b`
        (pool slot `s`), copies `pool[l][s, ...]` into a fresh
        `[batch, ...]` working buffer via `inplace_copy_from` (GPU-to-GPU,
        no PCIe).

        Args:
            request_ids: Ordered list of request IDs forming the batch.

        Returns:
            Tuple of (conv_buffers, recurrent_buffers), each a list of
            `num_layers` GPU Buffers shaped `[batch, ...]`.

        Raises:
            ValueError: If `request_ids` is empty.
            KeyError: If any request ID has no claimed slot.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")

        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid} not found in GatedDeltaNet state cache. "
                    "Call claim() before get_states()."
                )

        B = len(request_ids)
        slots = [self._request_to_slot[rid] for rid in request_ids]

        conv_bufs: list[Buffer] = []
        rec_bufs: list[Buffer] = []
        for l in range(self._num_layers):
            # Allocate working buffers for this layer's batch.
            batch_conv = Buffer(
                self._dtype,
                [B, self._conv_dim, self._conv_kernel],
                self._device,
            )
            batch_rec = Buffer(
                self._dtype,
                [
                    B,
                    self._num_v_heads,
                    self._key_head_dim,
                    self._value_head_dim,
                ],
                self._device,
            )
            # Gather: copy each slot row into the corresponding batch row.
            for b, s in enumerate(slots):
                batch_conv[b, :, :].inplace_copy_from(
                    self._conv_pool[l][s, :, :]
                )
                batch_rec[b, :, :, :].inplace_copy_from(
                    self._rec_pool[l][s, :, :, :]
                )
            conv_bufs.append(batch_conv)
            rec_bufs.append(batch_rec)

        return conv_bufs, rec_bufs

    def update_states(
        self,
        request_ids: list[RequestID],
        conv_buffers: list[Buffer],
        rec_buffers: list[Buffer],
    ) -> None:
        """GPU-scatter updated state Buffers back into the pool.

        For each layer `l` and each active request at batch position `b`
        (pool slot `s`), copies `output[l][b, ...]` back into
        `pool[l][s, ...]` via `inplace_copy_from` (GPU-to-GPU, no PCIe).

        Args:
            request_ids: Ordered list of request IDs matching the batch dim.
            conv_buffers: `num_layers` conv-state Buffers from model output,
                each shaped `[batch, conv_dim, K-1]`.
            rec_buffers: `num_layers` recurrent-state Buffers from model output,
                each shaped `[batch, num_v_heads, key_head_dim, val_head_dim]`.
        """
        slots = [self._request_to_slot[rid] for rid in request_ids]

        for l in range(self._num_layers):
            for b, s in enumerate(slots):
                self._conv_pool[l][s, :, :].inplace_copy_from(
                    conv_buffers[l][b, :, :]
                )
                self._rec_pool[l][s, :, :, :].inplace_copy_from(
                    rec_buffers[l][b, :, :, :]
                )
