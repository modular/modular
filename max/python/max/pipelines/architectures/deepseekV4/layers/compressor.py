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

The layer is stateless. The reference accumulates ``kv_state`` /
``score_state`` across decode steps and reduces them when a window closes;
here the raw per-token projections live in a sliding-window cache leaf
(``layers/cache.py``) and :meth:`__call__` re-pools whichever windows the
caller assembles from it, so prefill, chunked prefill and decode are one
computation.

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
        self.coff = 2 if self.overlap else 1
        self.proj_dim = self.coff * head_dim

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

        These are what the state leaf stores per token. ``ape`` is *not* added
        here: it depends on the token's slot within its window, which
        :meth:`__call__` knows and the store does not.
        """
        return self.wkv(x32), self.wgate(x32)

    def __call__(
        self,
        kv_rows: TensorValue,
        score_rows: TensorValue,
        present: TensorValue,
        freqs_cis: TensorValue,
    ) -> TensorValue:
        """Pool candidate windows into compressed entries.

        Args:
            kv_rows: ``[b, n_windows + coff - 1, ratio, proj_dim]`` float32
                raw ``wkv`` rows, one window per axis-1 index, window ``i``'s
                ``ratio`` tokens in position order along axis 2. When
                overlapping, the first window is the predecessor of the first
                window being emitted.
            score_rows: Same shape, raw ``wgate`` rows.
            present: ``[b, n_windows + coff - 1, ratio]`` bool, whether the
                token exists (``False`` before the start of the sequence).
            freqs_cis: ``[b, n_windows, rope_head_dim // 2, 2]`` rotary rows at
                each emitted window's *first* token -- entry ``i`` is rotated
                at position ``i * ratio``, not at the window's end.

        Returns:
            ``[b, n_windows, head_dim]`` compressed entries: normed, rotated,
            and activation-quantized, matching what the reference appends to
            the KV latent. A window that has not closed yet still yields a
            row -- finite, but not meaningful; the caller keeps it out of
            reach.
        """
        d = self.head_dim
        device = kv_rows.device
        mask = ops.unsqueeze(present, -1)
        kv = ops.where(mask, kv_rows, ops.constant(0.0, DType.float32, device))
        # ``ape`` is per-slot-within-a-window and broadcasts over batch and
        # window. Adding it before a reshape would need an ``ops.tile``, which
        # has no GPU kernel and silently round-trips the tensor through the
        # host (GEX-2056).
        score = ops.where(
            mask,
            score_rows + self.ape,
            ops.constant(float("-inf"), DType.float32, device),
        )
        if self.overlap:
            # Slots ``[:ratio]`` take the low half of the previous window,
            # slots ``[ratio:]`` the high half of the current one. Positive
            # bounds throughout (ISSUES Issue 34).
            n = int(kv.shape[1])
            kv = ops.concat([kv[:, : n - 1, :, :d], kv[:, 1:n, :, d:]], axis=2)
            score = ops.concat(
                [score[:, : n - 1, :, :d], score[:, 1:n, :, d:]], axis=2
            )
        pooled = ops.squeeze(
            ops.sum(kv * ops.softmax(score, axis=2), axis=2), axis=2
        )
        out = self.norm(ops.cast(pooled, self.norm.dtype))
        out = apply_rope_tail(out, freqs_cis, self.rope_head_dim)
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
