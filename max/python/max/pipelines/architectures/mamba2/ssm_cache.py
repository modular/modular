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
"""Slot-based SSM state cache for the Mamba2 SSD architecture.

Mirrors :mod:`max.pipelines.architectures.mamba.ssm_cache` ; only the
per-slot buffer shapes change to match the Mamba2 mixer layout:

* ``conv_state``: ``[1, conv_dim, d_conv-1]`` where
  ``conv_dim = d_inner + 2 * ngroups * d_state`` (the post-projection
  xBC slot that the depthwise causal conv runs over).
* ``ssm_state``: ``[1, nheads, head_dim, d_state]`` — Mamba2 stores SSM
  state per-head; the existing :func:`selective_scan_update` decode
  kernel expects ``(batch, dim, dstate)`` so callers flatten
  ``(nheads, head_dim) -> dim = d_inner`` at the call site.

Lifecycle, free-list semantics, and the get/update concatenation path
match Phase-1 exactly so the same code review patterns apply.
"""

from __future__ import annotations

import logging

from max.driver import Buffer, Device
from max.dtype import DType
from max.pipelines.modeling.types import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


class Mamba2SSMStateCache:
    """Fixed-size slot-based cache for Mamba2 SSD conv and scan states.

    Pre-allocates state buffers for ``max_slots`` concurrent requests.
    Each slot holds one set of per-layer ``(conv_state, ssm_state)``
    tensors with the Mamba2 shapes documented at module level.

    Lifecycle:
        1. :meth:`claim` — assign a free slot, zero its states.
        2. :meth:`get_states` — retrieve states for a batch.
        3. :meth:`update_states` — store updated states back.
        4. :meth:`release` — free the slot.

    For ``batch_size == 1`` (the common serving case), get/update are
    zero-copy and just return/replace the slot's buffer references. For
    ``batch > 1``, states are concatenated along dim 0 for get, and
    split back for update — exactly mirroring the Mamba1 cache.
    """

    def __init__(
        self,
        num_layers: int,
        conv_dim: int,
        d_conv: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        dtype: DType,
        max_slots: int,
        device: Device,
    ) -> None:
        if d_conv < 2:
            raise ValueError(
                f"d_conv must be >= 2 for a rolling conv state (got {d_conv})"
            )
        self._num_layers = num_layers
        self._conv_dim = conv_dim
        self._d_conv = d_conv
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._dtype = dtype
        self._max_slots = max_slots
        self._device = device

        # Per-slot state storage: ``_slots[slot_idx]`` is a list of
        # ``2 * num_layers`` Buffers alternating
        # ``[conv_state, ssm_state, ...]`` per layer.
        self._slots: list[list[Buffer]] = []
        for _ in range(max_slots):
            self._slots.append(self._make_zero_states())

        # Slot tracking.
        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}
        # Tracks whether a slot has been written to by model execution
        # (as opposed to being freshly claimed with zero states).
        self._valid_states: set[int] = set()

        # Conv state holds ``d_conv - 1`` historical tokens (rolling
        # buffer convention shared with the Mamba1 causal_conv1d_update
        # kernel). SSM state is the full per-head decoder state.
        conv_elems = conv_dim * (d_conv - 1)
        ssm_elems = nheads * head_dim * d_state
        total_bytes = (
            max_slots
            * num_layers
            * (conv_elems + ssm_elems)
            * dtype.size_in_bytes
        )
        logger.info(
            f"Mamba2 SSM cache: {max_slots} slots x {num_layers} layers = "
            f"{to_human_readable_bytes(total_bytes)} "
            f"(conv_dim={conv_dim}, d_conv={d_conv}, nheads={nheads}, "
            f"head_dim={head_dim}, d_state={d_state})"
        )

    def _make_zero_states(self) -> list[Buffer]:
        """Create a fresh set of zero-filled state buffers for one slot."""
        states: list[Buffer] = []
        conv_history = self._d_conv - 1
        for _ in range(self._num_layers):
            states.append(
                Buffer(
                    self._dtype,
                    [1, self._conv_dim, conv_history],
                    self._device,
                )
            )
            states.append(
                Buffer(
                    self._dtype,
                    [1, self._nheads, self._head_dim, self._d_state],
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
                f"No free Mamba2 SSM cache slots ({self._max_slots} in use). "
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
            List of ``2 * num_layers`` Buffers. For ``batch=1`` each is
            shaped exactly as the per-slot buffer (conv:
            ``[1, conv_dim, d_conv-1]``, ssm:
            ``[1, nheads, head_dim, d_state]``). For ``batch>1`` states
            are concatenated along dim 0 to give a leading ``batch``
            axis.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")

        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid} not found in Mamba2 SSM cache. "
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
            new_states: ``2 * num_layers`` Buffers from model output. For
                ``batch=1``, shapes match the per-slot buffers. For
                ``batch>1``, each has a leading ``batch`` axis.
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

    def update_ssm_states(
        self, request_ids: list[RequestID], ssm_states: list[Buffer]
    ) -> None:
        """Store updated SSM-only state buffers back into their slots.

        Used by the prefill path, which surfaces per-layer SSM ``final_state``
        via the ``ssd_chunk_scan_combined`` op but does not (yet) surface a
        rolling conv-state tail. The slot's conv buffer is left untouched —
        if the slot was freshly claimed it stays zero, which is the same
        behavior as before plus a correctly-seeded SSM state. Marks the slot
        as valid so subsequent ``has_valid_state`` checks route to step mode.

        Args:
            request_ids: Ordered list of request IDs matching the batch dim.
            ssm_states: ``num_layers`` buffers, one per layer's SSM state.
                For ``batch=1`` each shape matches the per-slot SSM buffer;
                for ``batch>1`` each carries a leading ``batch`` axis.
        """
        if len(ssm_states) != self._num_layers:
            raise ValueError(
                f"Expected {self._num_layers} SSM state tensors, "
                f"got {len(ssm_states)}"
            )

        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            for layer_idx, ssm_buf in enumerate(ssm_states):
                self._slots[slot][2 * layer_idx + 1] = ssm_buf
            self._valid_states.add(slot)
            return

        for layer_idx, state_buf in enumerate(ssm_states):
            for batch_idx, rid in enumerate(request_ids):
                slot = self._request_to_slot[rid]
                self._slots[slot][2 * layer_idx + 1] = state_buf[
                    batch_idx : batch_idx + 1
                ].contiguous()

        for rid in request_ids:
            self._valid_states.add(self._request_to_slot[rid])


def _cat_buffers(buffers: list[Buffer], device: Device) -> Buffer:
    """Concatenate buffers along dim 0 via numpy round-trip.

    Acceptable for small SSM states (the Mamba2 prefill state is
    typically <2MB per layer per request). For high-throughput
    ``batch>1`` serving this could be replaced with a compiled
    concatenation kernel — same follow-up note as Phase-1.
    """
    import numpy as np

    arrays = [b.to_numpy() for b in buffers]
    combined = np.concatenate(arrays, axis=0)
    return Buffer.from_numpy(combined).to(device)


# ---------------------------------------------------------------------------
# Smoke test — exercises the cache shape arithmetic without any kernel
# load (the SSD kernel is currently blocked by the upstream Tuple-shape
# bug, see ``proposed/approvals/causal_conv1d_ops-tuple-shape-bug.md``).
# Run via:
#   python -m max.pipelines.architectures.mamba2.ssm_cache
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Construct the cache with a synthetic config and assert buffer dims."""
    from max.driver import CPU

    # Small synthetic config — matches the ``mamba2.weight_adapters`` smoke.
    num_layers = 2
    nheads = 8
    head_dim = 32
    d_state = 16
    d_conv = 4
    ngroups = nheads  # mixer enforces ngroups == nheads
    d_inner = nheads * head_dim
    conv_dim = d_inner + 2 * ngroups * d_state

    cache = Mamba2SSMStateCache(
        num_layers=num_layers,
        conv_dim=conv_dim,
        d_conv=d_conv,
        nheads=nheads,
        head_dim=head_dim,
        d_state=d_state,
        dtype=DType.float32,
        max_slots=2,
        device=CPU(),
    )

    assert cache.max_slots == 2
    assert cache.num_free_slots == 2
    assert cache.num_active_slots == 0

    rid = RequestID("smoke-0")

    # Claim a slot and check buffer shapes.
    slot = cache.claim(rid)
    assert slot in (0, 1)
    assert cache.num_active_slots == 1
    assert not cache.has_valid_state(rid)

    states = cache.get_states([rid])
    assert len(states) == 2 * num_layers
    for i in range(num_layers):
        conv = states[2 * i]
        ssm = states[2 * i + 1]
        assert tuple(conv.shape) == (1, conv_dim, d_conv - 1), (
            f"conv_state shape mismatch at layer {i}: {tuple(conv.shape)}"
        )
        assert tuple(ssm.shape) == (1, nheads, head_dim, d_state), (
            f"ssm_state shape mismatch at layer {i}: {tuple(ssm.shape)}"
        )

    # Release frees the slot.
    cache.release(rid)
    assert cache.num_free_slots == 2

    print("mamba2 ssm_cache smoke: OK")


if __name__ == "__main__":
    _smoke()
