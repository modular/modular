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
"""Runtime resources for overlapping constrained-decoding bitmask compute.

This module hosts :class:`StructuredOutputOverlapState`, the
process-lifetime container for the completion flag and pinned/device
buffers that let the constrained-decoding FSM + bitmask write run
concurrently with the model forward pass.

The state is consumed at iteration boundaries:

  * The model-stream trampoline (enqueued via
    :meth:`enqueue_async_callback`) dispatches the bitmask callback
    onto an AsyncRT worker, then returns. The trampoline resets
    ``bitmask_flag`` on entry and the worker release-stores ``1`` on
    exit (signalling even on exception, per
    ``Device.cpp:asyncHostFuncTrampoline``).
  * The captured model graph waits on ``bitmask_flag`` via
    ``mo.wait_host_value_with_dep`` (keyed off :attr:`wait_payload`),
    then ``mo.inplace_memcpy`` copies :attr:`pinned_bitmask` into
    :attr:`device_bitmask_scratch` for the sampler.
  * Cold-start paths (prefill -> decode boundary, capture-time warmup,
    first iter after a new-context join) use :meth:`prime` to pre-fill
    pinned memory and drop the flag to ``1`` so the first replay's wait
    node passes immediately.

The pinned/device bitmask buffers carry packed ``int32`` data (1 bit per
token, 32 tokens per word). The host callback never unpacks; the GPU
acceptance sampler unpacks and applies the mask to logits in one fused pass
(``apply_packed_bitmask``), so no bool tensor is ever materialized and the
in-graph H2D moves 8x less data than a bool representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    CPU,
    Buffer,
    CompletionFlag,
    Device,
    DevicePinnedBuffer,
)
from max.dtype import DType
from max.support.math import ceildiv


class StructuredOutputOverlapState:
    """Owns the runtime resources for the constrained-decoding overlap path.

    Construct one instance per overlap-text-generation pipeline. The
    completion flag, pinned bitmask, and payload buffer are
    process-lifetime; the device-side scratch buffer is the destination
    of the in-graph H2D copy that feeds the sampler.

    The class is intentionally decoupled from :class:`SpecDecodeState`
    and the architecture graph builders.
    """

    def __init__(
        self,
        device: Device,
        cpu: CPU,
        max_batch_size: int,
        num_positions: int,
        vocab_size: int,
    ) -> None:
        """Allocates the flag and pinned/device buffers.

        Args:
            device: The accelerator the model runs on. Must be a CUDA
                device; :class:`CompletionFlag` rejects other backends.
            cpu: The host CPU device whose AsyncRT worker pool the
                async callbacks will dispatch onto.
            max_batch_size: Maximum batch capacity. Matches the
                model's ``total_max_batch``.
            num_positions: Per-batch positions per iteration. For
                Eagle spec-decode this is ``num_speculative_tokens + 1``.
            vocab_size: Tokenizer vocabulary size.

        Raises:
            ValueError: If any dimension is non-positive.
            RuntimeError: If ``device`` is not a CUDA accelerator.
        """
        if max_batch_size <= 0 or num_positions <= 0 or vocab_size <= 0:
            raise ValueError(
                "max_batch_size, num_positions, and vocab_size must all be "
                f"positive; got ({max_batch_size}, {num_positions}, "
                f"{vocab_size})"
            )

        self.device: Device = device
        """The accelerator the model graph runs on."""
        self.cpu: CPU = cpu
        """The host CPU device whose AsyncRT worker pool will execute callbacks dispatched via :meth:`enqueue_async_callback`."""
        self.max_batch_size: int = max_batch_size
        """Configured batch capacity."""
        self.num_positions: int = num_positions
        """Per-batch positions written per iteration (typically ``num_speculative_tokens + 1`` for spec-decode)."""
        self.vocab_size: int = vocab_size
        """Tokenizer vocabulary width."""
        # The bitmask is stored packed: 1 bit per token, 32 tokens per int32
        # word. The GPU acceptance sampler unpacks and applies it in one fused
        # pass (apply_packed_bitmask), so the host callback never unpacks and
        # the in-graph H2D moves 8x less data than a bool representation.
        self.packed_vocab_size: int = ceildiv(vocab_size, 32)
        """Packed bitmask width: ``ceil(vocab_size / 32)`` int32 words."""

        self.bitmask_flag: CompletionFlag = CompletionFlag(device)
        """The 64-bit device-mapped pinned flag the callback signals. Single process-lifetime flag."""

        # Payload consumed by the in-graph `mo.wait_host_value_with_dep`
        # op. Contract: CPU int64[2] = [flag._unsafe_ptr,
        # expected_value]. Allocated as pinned host memory so the
        # engine's input binding sees a pinned -> CPU-graph-input
        # pairing and avoids a per-iteration pageable HtoD copy through
        # this 16-byte buffer. Written once here; the buffer is reused
        # across every iteration's wait.
        self.wait_payload: DevicePinnedBuffer = DevicePinnedBuffer(
            dtype=DType.int64,
            shape=(2,),
            device=device,
        )
        """A CPU-resident ``int64[2]`` buffer holding ``[bitmask_flag._unsafe_ptr, 1]``. This is the payload consumed by the in-graph ``mo.wait_host_value_with_dep`` op. Allocated once; contents written once at construction."""
        payload_np = self.wait_payload.to_numpy()
        payload_np[0] = self.bitmask_flag._unsafe_ptr
        payload_np[1] = 1

        shape = (max_batch_size, num_positions, self.packed_vocab_size)
        self.pinned_bitmask: DevicePinnedBuffer = DevicePinnedBuffer(
            dtype=DType.int32,
            shape=shape,
            device=device,
        )
        """Single persistent packed ``int32[max_batch_size, num_positions, packed_vocab_size]`` pinned buffer (1 bit per token, 32 tokens per word). The AsyncRT worker writes into this buffer's pinned host backing store and the in-graph wait gates the captured H2D on the worker's release-store of the flag, so writer and reader cannot race even though they share storage."""
        self.device_bitmask_scratch: Buffer = Buffer(
            dtype=DType.int32,
            shape=shape,
            device=device,
        )
        """A device-resident packed ``int32`` buffer with the same shape as :attr:`pinned_bitmask`. Destination of the in-graph H2D; the acceptance sampler unpacks and applies it in one fused pass (``apply_packed_bitmask``), so no bool tensor is ever materialized. A single scratch is safe because the model stream is FIFO: sampler reads serialise with the next-iter H2D write."""

        # Cache of (batch_size, num_positions) -> view of pinned_bitmask
        # / device_bitmask_scratch. Populated lazily by
        # :meth:`get_input_views`. The cache exists because
        # ``GraphCaptureRunner.replay`` does a per-input preface
        # ``inplace_copy_from`` that only short-circuits when the
        # buffer object passed at replay is the SAME Python object
        # passed at capture (``Buffer.inplace_copy_from`` has an
        # identity-only ``if self is src: return`` check; aliased
        # views of the same underlying storage still trigger a real
        # memcpy). Reusing one view object per (batch_size,
        # num_positions) tuple across all iterations -- including
        # the original warmup capture -- ensures the preface
        # short-circuits and the captured cuGraph reads the worker's
        # writes through the same underlying memory.
        self._input_view_cache: dict[
            tuple[int, int], tuple[Buffer, Buffer]
        ] = {}

    @property
    def max_bitmask_shape(self) -> tuple[int, int, int]:
        """Returns the persistent buffer's max shape.

        ``(max_batch_size, num_positions, packed_vocab_size)``. Per-iteration
        writes only fill the leading ``(batch_size, num_positions,
        packed_vocab_size)`` rectangle; this property is the storage
        capacity, not the runtime shape.
        """
        return (
            self.max_batch_size,
            self.num_positions,
            self.packed_vocab_size,
        )

    def get_input_views(
        self, batch_size: int, num_positions: int
    ) -> tuple[Buffer, Buffer]:
        """Returns the cached (pinned_view, device_scratch_view) pair.

        Both views are stable :class:`Buffer` objects (returned by
        ``flat[:N].view(...)``) that alias the leading ``batch_size *
        num_positions * vocab_size`` elements of
        :attr:`pinned_bitmask` and :attr:`device_bitmask_scratch`
        respectively. The first call for a given ``(batch_size,
        num_positions)`` pair creates the views and caches them;
        subsequent calls return the same objects.

        Always retrieve the views through this method (never call
        ``_contiguous_prefix_3d(...)`` directly on the underlying
        buffers from the iteration loop) so the per-iteration
        ``model_inputs.pinned_bitmask`` and
        ``model_inputs.device_bitmask_scratch`` bindings are the
        same ``Buffer`` objects at warmup capture and at every
        steady-state replay. This is what makes the engine's
        preface ``inplace_copy_from`` skip via ``self is src``.

        Args:
            batch_size: Runtime batch size for this iteration.
            num_positions: Per-batch positions (``num_speculative_tokens + 1``).
                Must equal :attr:`num_positions` -- both
                ``pinned_bitmask`` and ``device_bitmask_scratch`` are
                laid out as ``(max_batch_size, self.num_positions,
                vocab_size)``, so a contiguous prefix view with a
                smaller second dim would alias an interleaved subset
                of rows instead of the desired ``[:batch_size,
                :num_positions, :]`` rectangle.

        Returns:
            ``(pinned_view, device_scratch_view)`` tuple.
        """
        if num_positions != self.num_positions:
            raise ValueError(
                f"get_input_views num_positions={num_positions} must equal "
                f"state.num_positions={self.num_positions}; the underlying "
                "buffers are laid out as (max_batch_size, num_positions, "
                "vocab_size) so a smaller second dim would alias an "
                "interleaved subset of rows."
            )
        key = (batch_size, num_positions)
        cached = self._input_view_cache.get(key)
        if cached is not None:
            return cached
        num_elements = batch_size * num_positions * self.packed_vocab_size
        pinned_flat = self.pinned_bitmask.view(
            self.pinned_bitmask.dtype, (self.pinned_bitmask.num_elements,)
        )
        pinned_view = pinned_flat[:num_elements].view(
            self.pinned_bitmask.dtype,
            (batch_size, num_positions, self.packed_vocab_size),
        )
        scratch_flat = self.device_bitmask_scratch.view(
            self.device_bitmask_scratch.dtype,
            (self.device_bitmask_scratch.num_elements,),
        )
        scratch_view = scratch_flat[:num_elements].view(
            self.device_bitmask_scratch.dtype,
            (batch_size, num_positions, self.packed_vocab_size),
        )
        self._input_view_cache[key] = (pinned_view, scratch_view)
        return pinned_view, scratch_view

    def prime(self, bitmask: npt.NDArray[Any]) -> None:
        """Pre-fills the pinned bitmask and signals the flag.

        Writes into :attr:`pinned_bitmask` and then release-stores
        ``1`` to the flag so the next replay's ``mo.wait_host_value_with_dep``
        passes immediately.

        Used at cold-start sites where no prior async callback has
        run and the next replay's ``mo.wait_host_value_with_dep`` would
        otherwise block forever:

          * Before the first decode iteration post-prefill.
          * Before capture-time warmup replays.
          * Before re-priming after a mid-stream context join, if the
            pipeline cannot prove the prior callback covered the new
            row layout.

        Args:
            bitmask: A packed ``int32`` numpy array shaped
                ``(batch, self.num_positions, self.packed_vocab_size)`` with
                ``1 <= batch <= max_batch_size``. The trailing
                ``num_positions`` and ``packed_vocab_size`` axes must match
                state exactly so the in-graph wait reads consistent
                rows regardless of which slot was used at warmup
                capture. The graph only reads the leading ``batch``
                rows (via the symbolic ``batch_size`` graph input), so
                rows past ``batch`` are left untouched.

        Raises:
            ValueError: If ``bitmask`` has the wrong rank, an
                out-of-bounds batch dim, a mismatched ``num_positions``
                or ``packed_vocab_size`` dim, or a non-int32 dtype.
        """
        if bitmask.dtype != np.int32:
            raise ValueError(
                f"prime() bitmask dtype {bitmask.dtype} is not int32"
            )
        if bitmask.ndim != 3:
            raise ValueError(
                f"prime() expects a 3D bitmask, got shape {bitmask.shape}"
            )
        batch, num_positions, packed_vocab = bitmask.shape
        if not (1 <= batch <= self.max_batch_size):
            raise ValueError(
                f"prime() batch dim {batch} out of bounds "
                f"[1, {self.max_batch_size}]"
            )
        # Strict equality keeps the contract consistent with
        # ``get_input_views``: every iteration writes (and the captured
        # graph reads) the same ``self.num_positions`` rows, so a
        # shorter prefix would leave a stale tail under the
        # ``num_bitmask_positions`` slot the graph binds to.
        if num_positions != self.num_positions:
            raise ValueError(
                f"prime() num_positions {num_positions} must equal "
                f"state.num_positions={self.num_positions}"
            )
        if packed_vocab != self.packed_vocab_size:
            raise ValueError(
                f"prime() packed vocab dim {packed_vocab} does not match "
                f"state packed_vocab_size {self.packed_vocab_size}"
            )

        pinned_view = self.pinned_bitmask.to_numpy()
        # DevicePinnedBuffer.to_numpy() returns a writable view that
        # aliases the pinned host backing store. Copy in place so the
        # caller's source array can be freed.
        pinned_view[:batch, :num_positions, :] = bitmask
        self.bitmask_flag.signal(1)

    def enqueue_async_callback(self, fn: Any) -> None:
        """Dispatches ``fn`` onto the device default stream's AsyncRT worker.

        The trampoline auto-resets :attr:`bitmask_flag` on entry and
        release-stores ``1`` on exit (signalling even if ``fn``
        raises), so callers must not signal or reset the flag
        manually. ``fn`` is responsible for writing the iteration's
        bitmask data into :attr:`pinned_bitmask` before returning.

        Posts the kickoff host node on the **device default stream**, not
        a side stream. This is intentional: the trampoline's
        ``flag.reset()`` and the next iter's captured-graph
        ``mo.wait_host_value_with_dep`` end up serialised on the same stream, so
        the wait can never observe a stale ``1`` left over from a prior
        iter's prime / worker signal. The trampoline body is small
        (atomic store + heap alloc + dispatch + return -- microseconds),
        and ``fn`` itself runs on a separate AsyncRT worker so the slow
        FSM advance + bitmask compute still overlaps with the target
        forward.

        Args:
            fn: A zero-argument callable. It will run on an AsyncRT
                worker thread, not on the main Python thread.
        """
        # Use getattr to dodge the name-mangling of the dunder method.
        getattr(self.device, "__unsafe_enqueue_async_py_host_func")(
            fn, self.bitmask_flag, 1, self.cpu
        )
