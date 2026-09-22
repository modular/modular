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

"""The DeepSeek-V4 CSA compressor.

Pools every ``compress_ratio`` consecutive tokens into one compressed KV entry
with a learned softmax gate. Reference: ``inference/model.py`` class
``Compressor``.

Two shapes, selected by ratio:

* ``ratio == 4`` -- *overlapping*. ``wkv`` and ``wgate`` are twice as wide and
  the output is split in half: the low ``head_dim`` dims of window ``i`` are
  pooled together with the high ``head_dim`` dims of window ``i-1``, so each
  compressed entry sees ``2 * ratio`` source tokens straddling the boundary.
  Window 0 has no predecessor, so its overlap half is zero for the values and
  ``-inf`` for the gate scores, which makes softmax ignore it exactly.
* ``ratio == 128`` -- *non-overlapping*. Plain pooling over disjoint windows.

Everything up to the norm runs in float32; the reference stores ``wkv`` and
``wgate`` in bf16 but computes the pooling in fp32, and the ``ape`` and gate
scores are fp32 parameters.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from ..model_config import DeepseekV4Config
from .hadamard import hadamard_rotate
from .quantization import fp4_qat_quantize, fp8_qat_quantize
from .rope import apply_rope_tail


class DeepseekV4Compressor(Module):
    """Gated pooling over ``compress_ratio`` consecutive tokens.

    Args:
        config: The model config.
        compress_ratio: 4 (overlapping) or 128 (non-overlapping).
        head_dim: Width of one compressed entry. ``config.head_dim`` for the
            attention compressor, ``config.index_head_dim`` for the indexer's
            own compressor.
        device: Device the parameters live on.
        rotate: The indexer's compressor sets this; it Hadamard-rotates the
            result and simulates FP4 instead of FP8.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        device: DeviceRef,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        self.rotate = rotate
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        # ``coff`` in the reference: the projections are doubled when
        # overlapping so one matmul produces both halves.
        coff = 2 if self.overlap else 1
        self.proj_dim = coff * head_dim

        # Absolute positional embedding over the ``compress_ratio`` slots of a
        # window, added to the gate scores (not to the values).
        self.ape = Weight(
            name="ape",
            dtype=DType.float32,
            shape=(compress_ratio, self.proj_dim),
            device=device,
        )
        self.wkv = Linear(
            config.hidden_size, self.proj_dim, DType.float32, device
        )
        self.wgate = Linear(
            config.hidden_size, self.proj_dim, DType.float32, device
        )
        self.norm = RMSNorm(head_dim, config.dtype, config.rms_norm_eps)

    def projections(self, x32: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Raw ``wkv`` / ``wgate`` projections, ``[b, s, proj_dim]`` float32.

        Exposed for prefill state export: the decode-phase ``kv_state`` /
        ``score_state`` buffers are initialized from these rows (the trailing
        partial window, plus the last full window when overlapping), and the
        host assembles them following ``Compressor.forward``'s ``start_pos ==
        0`` writes. ``ape`` is *not* added here; the host adds the checkpoint's
        ``ape`` rows itself, since the slot each row lands in decides which
        ``ape`` row it gets.
        """
        return self.wkv(x32), self.wgate(x32)

    def decode(
        self,
        x: TensorValue,
        ape_idx: TensorValue,
        kv_state: TensorValue,
        score_state: TensorValue,
        should: TensorValue,
        comp_freqs_row: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """One decode token: update the state buffers, pool unconditionally.

        Args:
            x: ``[b, 1, hidden_size]`` block input for this token.
            ape_idx: ``[1]`` int32, ``start_pos % ratio`` -- the ``ape`` row for
                this token and the write slot within the current window.
            kv_state: ``[b, coff * ratio, proj_dim]`` float32.
            score_state: Same shape; unfilled slots hold ``-inf``.
            should: ``[1]`` bool, ``(start_pos + 1) % ratio == 0``.
            comp_freqs_row: ``[1, rope_head_dim // 2, 2]`` -- the rotary row at
                the compressed entry's position, ``start_pos + 1 - ratio``
                (host clamps to 0 when no window has closed; the entry is
                discarded then).

        Returns:
            The candidate compressed entry ``[b, 1, head_dim]`` (meaningful
            only when ``should``; the caller gates the zone write on it), and
            the updated ``kv_state`` / ``score_state``.

        The pooling runs every step rather than behind a branch: the graph is
        static, and at a non-boundary step the result is simply not written.
        No NaN can escape the wasted pool -- the slot just written is always
        finite, so the softmax denominator never collapses to ``exp(-inf)``
        alone.
        """
        ratio = self.compress_ratio
        d = self.head_dim
        coff = 2 if self.overlap else 1
        device = x.device

        x32 = ops.cast(x, DType.float32)
        kv = self.wkv(x32)
        score = self.wgate(x32) + ops.gather(self.ape, ape_idx, axis=0)

        # Overlap writes into the *current* window half, slots [ratio:].
        slot = ape_idx + ops.constant(
            ratio if self.overlap else 0, DType.int32, device
        )
        rows = ops.range(
            0,
            coff * ratio,
            1,
            out_dim=coff * ratio,
            device=device,
            dtype=DType.int32,
        )
        write_mask = ops.reshape(rows == slot, [1, coff * ratio, 1])
        new_kv_state = ops.where(write_mask, kv, kv_state)
        new_score_state = ops.where(write_mask, score, score_state)

        if self.overlap:
            pool_kv = ops.concat(
                [
                    new_kv_state[:, :ratio, :d],
                    new_kv_state[:, ratio:, d:],
                ],
                axis=1,
            )
            pool_score = ops.concat(
                [
                    new_score_state[:, :ratio, :d],
                    new_score_state[:, ratio:, d:],
                ],
                axis=1,
            )
        else:
            pool_kv = new_kv_state
            pool_score = new_score_state
        pooled = ops.sum(pool_kv * ops.softmax(pool_score, axis=1), axis=1)
        out = self.norm(ops.cast(pooled, self.norm.dtype))
        out = apply_rope_tail(out, comp_freqs_row, self.rope_head_dim)
        if self.rotate:
            entry = fp4_qat_quantize(hadamard_rotate(out))
        else:
            width = int(out.shape[-1])
            entry = ops.concat(
                [
                    fp8_qat_quantize(out[..., : width - self.rope_head_dim]),
                    out[..., width - self.rope_head_dim :],
                ],
                axis=-1,
            )

        if self.overlap:
            # On a closed window the current half becomes the previous half;
            # the current half's stale rows are each overwritten before the
            # next close reads them.
            should_mask = ops.reshape(should, [1, 1, 1])
            shifted_kv = ops.concat(
                [new_kv_state[:, ratio:], new_kv_state[:, ratio:]], axis=1
            )
            shifted_score = ops.concat(
                [new_score_state[:, ratio:], new_score_state[:, ratio:]],
                axis=1,
            )
            new_kv_state = ops.where(should_mask, shifted_kv, new_kv_state)
            new_score_state = ops.where(
                should_mask, shifted_score, new_score_state
            )
        return entry, new_kv_state, new_score_state

    def _overlap_transform(self, x: TensorValue, fill: float) -> TensorValue:
        """``[b, w, ratio, 2*d]`` -> ``[b, w, 2*ratio, d]``.

        Slots ``[ratio:]`` take the high half of the *current* window; slots
        ``[:ratio]`` take the low half of the *previous* window, with window 0
        filled by ``fill`` (0 for values, ``-inf`` for gate scores, so the
        softmax drops the non-existent predecessor).
        """
        d = self.head_dim
        ratio = self.compress_ratio
        current = x[:, :, :, d:]
        previous = x[:, :, :, :d]
        # Shift ``previous`` one window later and fill window 0.
        pad = ops.broadcast_to(
            ops.constant(fill, previous.dtype, previous.device),
            (previous.shape[0], 1, ratio, d),
        )
        shifted = ops.concat([pad, previous[:, :-1]], axis=1)
        return ops.concat([shifted, current], axis=2)

    def __call__(
        self, x: TensorValue, seq_len: int, freqs_cis: TensorValue
    ) -> TensorValue:
        """Compress a full prefill sequence.

        Args:
            x: ``[batch, seq_len, hidden_size]`` block input.
            seq_len: Static sequence length; ``seq_len % compress_ratio``
                trailing tokens are dropped, matching the reference, which keeps
                them in ``kv_state`` for the next step rather than compressing a
                partial window.
            freqs_cis: The layer's full rotary table. The compressor takes the
                ``[:cutoff:ratio]`` stride of it -- compressed entry ``i`` is
                rotated at the position of the *first* token of its window,
                ``i * ratio``, not at the window's end.

        Returns:
            ``[batch, seq_len // compress_ratio, head_dim]`` compressed entries:
            normed, rotated, and activation-quantized, matching what the
            reference appends to the KV latent.
        """
        ratio = self.compress_ratio
        num_windows = seq_len // ratio
        if num_windows == 0:
            raise ValueError(
                f"seq_len={seq_len} is shorter than compress_ratio={ratio}; "
                "the reference buffers such a prefix instead of compressing it"
            )
        cutoff = num_windows * ratio

        x32 = ops.cast(x, DType.float32)[:, :cutoff]
        kv = ops.reshape(
            self.wkv(x32), (x.shape[0], num_windows, ratio, self.proj_dim)
        )
        # ``ape`` is per-slot-within-a-window, so it is added after the reshape
        # and broadcasts over batch and window. Adding it before the reshape
        # would need an ``ops.tile``, which has no GPU kernel and silently
        # round-trips the tensor through the host (GEX-2056).
        score = (
            ops.reshape(
                self.wgate(x32),
                (x.shape[0], num_windows, ratio, self.proj_dim),
            )
            + self.ape
        )
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))

        pooled = ops.sum(kv * ops.softmax(score, axis=2), axis=2)
        pooled = ops.squeeze(pooled, axis=2)
        out = self.norm(ops.cast(pooled, self.norm.dtype))

        # Strided positions: entry i sits at token position i * ratio.
        out = apply_rope_tail(
            out, freqs_cis[: cutoff : self.compress_ratio], self.rope_head_dim
        )
        if self.rotate:
            # The indexer's variant. Note it quantizes the *whole* vector: the
            # Hadamard rotation mixes the RoPE dims into the rest, so they are
            # no longer a separable tail the way they are on the FP8 path.
            return fp4_qat_quantize(hadamard_rotate(out))
        # Positive bounds: see the note in rope.py and ISSUES Issue 34.
        width = int(out.shape[-1])
        return ops.concat(
            [
                fp8_qat_quantize(out[..., : width - self.rope_head_dim]),
                out[..., width - self.rope_head_dim :],
            ],
            axis=-1,
        )
