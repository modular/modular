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
"""Pools token projections into compressed KV latents.

Groups of adjacent tokens are merged into one normalized latent row, so
attention over long contexts reads fewer KV entries.
"""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm


class DeepseekV41Compressor(Module):
    """Pools packed token rows and returns masked pre-RoPE BF16 latents.

    Holds no state across calls: the caller keeps each request's last raw
    row between chunks (the carry) and must reject, before launch, a request
    that starts at an odd position without its exact predecessor row. Active
    rows must be consecutive within each request, with padding only in the
    packed tail.

    Args:
        compress_ratio: One for plain normalization, or two for gated pairs.
        eps: RMSNorm epsilon from the model configuration.
        hidden_size: Model hidden size. Given together with ``device``, the
            module owns the ``wkv``/``wgate`` projections (:meth:`project`);
            omit both to supply ``raw`` directly.
        device: Device of the projection weights; paired with
            ``hidden_size``.
        head_dim: Latent width.

    Raises:
        ValueError: If ``compress_ratio`` is not 1 or 2, or only one of
            ``hidden_size`` and ``device`` is given.
    """

    def __init__(
        self,
        compress_ratio: int,
        eps: float = 1e-20,
        *,
        hidden_size: int | None = None,
        device: DeviceRef | None = None,
        head_dim: int = 512,
    ) -> None:
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError("CSA2 compression ratio must be 1 or 2")
        if (hidden_size is None) != (device is None):
            raise ValueError("hidden_size and device must be given together")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, DType.float32, eps)
        if hidden_size is not None:
            assert device is not None
            # FP32, as ``__call__`` requires; bf16 checkpoint weights must be
            # upcast at load.
            self.wkv = Linear(hidden_size, head_dim, DType.float32, device)
            if compress_ratio == 2:
                self.wgate = Linear(
                    hidden_size, head_dim, DType.float32, device
                )

    def project(self, h: TensorValue) -> TensorValue:
        """Projects hidden states into this compressor's ``raw`` input.

        Requires construction with ``hidden_size`` and ``device``.

        Args:
            h: ``[..., hidden_size]`` hidden states of any float dtype; cast
                to float32 before the projection.

        Returns:
            ``[..., head_dim * compress_ratio]`` float32, the ``raw`` input
            of :meth:`__call__`.
        """
        h = ops.cast(h, DType.float32)
        raw = self.wkv(h)
        if self.compress_ratio == 2:
            raw = ops.concat([raw, self.wgate(h)], axis=-1)
        return raw

    def __call__(
        self,
        raw: TensorValue,
        positions: TensorValue,
        offsets: TensorValue,
        active: TensorValue,
        prior_row: TensorValue | None = None,
        prior_valid: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Normalizes raw rows, pooling complete pairs when the ratio is two.

        Ratio two pools the pair ``(2j, 2j + 1)`` per channel as
        ``sum_i kv_i * softmax_i(score)`` in float32, RMS-normalizes the
        float32 result without first rounding it to bf16, then casts to bf16.
        The pooled row is emitted on the row closing the pair (odd position
        ``2j + 1``); the caller applies RoPE at position ``2j``.

        Args:
            raw: FP32 projections, ``[T, head_dim * compress_ratio]``: the
                ``wkv`` row, then (ratio two) the ``wgate`` score row.
            positions: Absolute int64 token positions, ``[T]``.
            offsets: Packed uint32 request boundaries, ``[B + 1]``.
            active: Boolean real-row mask, ``[T]``.
            prior_row: Ratio-two FP32 predecessors, ``[B, 2 * head_dim]``:
                each request's raw row at position ``start - 1``.
            prior_valid: Boolean exact-predecessor validity, ``[B]``.

        Returns:
            ``[T, head_dim]`` BF16 latent rows (zero where invalid) and their
            ``[T]`` boolean publication mask. A missing carry suppresses
            publication but does not replace the caller's pre-launch check.

        Raises:
            ValueError: On wrong input dtypes/widths or carry arguments that
                do not match the ratio.
        """
        if (
            raw.dtype != DType.float32
            or raw.shape[-1] != self.head_dim * self.compress_ratio
        ):
            raise ValueError(
                "CSA2 raw projections have the wrong dtype or width"
            )
        if positions.dtype != DType.int64 or offsets.dtype != DType.uint32:
            raise ValueError("CSA2 positions/offsets must be int64/uint32")
        if active.dtype != DType.bool:
            raise ValueError("CSA2 active mask must be boolean")
        valid = active
        pooled = raw
        if self.compress_ratio == 1:
            if prior_row is not None or prior_valid is not None:
                raise ValueError("Ratio one has no carry")
        else:
            if prior_row is None or prior_valid is None:
                raise ValueError("Ratio two requires carry rows and validity")
            device = raw.device
            rows = ops.range(0, raw.shape[0], dtype=DType.int64, device=device)
            starts = offsets[:-1].cast(DType.int64)
            # TODO: Replace the O(T*B) request lookup with a GPU search for
            # production-sized packed batches.
            request = ops.squeeze(
                ops.sum(
                    (ops.unsqueeze(rows, 1) >= ops.unsqueeze(starts, 0)).cast(
                        DType.int64
                    ),
                    axis=1,
                ),
                1,
            )
            real_end = offsets[-1].cast(DType.int64)
            request = ops.where(active & (rows < real_end), request, 0)
            width = self.head_dim
            zero_row = ops.constant(
                [[0.0] * (2 * width)], DType.float32, device=device
            )
            zero_index = ops.constant([0], DType.int64, device=device)
            false = ops.constant([False], DType.bool, device=device)
            # Sentinel entries make empty requests and inactive tail rows safe
            # before either gather, including a bucket with B == 0.
            start = ops.gather(ops.concat([zero_index, starts]), request, 0)
            first = rows == start
            carry = ops.gather(ops.concat([zero_row, prior_row]), request, 0)
            carry_valid = ops.gather(
                ops.concat([false, prior_valid]), request, 0
            )
            previous = ops.gather(ops.concat([zero_row, raw]), rows, 0)
            previous = ops.where(ops.unsqueeze(first, 1), carry, previous)
            previous_position = ops.gather(
                ops.concat([zero_index, positions]), rows, 0
            )
            predecessor_valid = ops.where(
                first, carry_valid, previous_position == positions - 1
            )
            valid = (
                active
                & (rows < real_end)
                & (positions % 2 == 1)
                & predecessor_valid
            )
            kv0, score0 = previous[:, :width], previous[:, width:]
            kv1, score1 = raw[:, :width], raw[:, width:]
            peak = ops.max(score0, score1)
            w0, w1 = ops.exp(score0 - peak), ops.exp(score1 - peak)
            pooled = (kv0 * w0 + kv1 * w1) / (w0 + w1)
        pooled = ops.where(ops.unsqueeze(valid, 1), pooled, 0.0)
        latent = self.norm(pooled).cast(DType.bfloat16)
        return ops.where(ops.unsqueeze(valid, 1), latent, 0.0), valid


def compressed_rope_positions(
    positions: TensorValue, compress_ratio: int
) -> TensorValue:
    """Returns each row's compressed RoPE position: its group's first token.

    Group ``j`` rotates at position ``j * ratio``. Identity for ratio one.
    """
    if compress_ratio == 1:
        return positions
    return positions - positions % compress_ratio


def compressed_visibility(
    pos_q: TensorValue, pos_k: TensorValue, compress_ratio: int
) -> TensorValue:
    """Returns whether compressed-cache row ``pos_k`` is visible to ``pos_q``.

    The cache holds one row per raw token, with a group's pooled entry on the
    row closing it (``(pos_k + 1) % ratio == 0``). An entry is visible once
    its whole group is at or before the query (``pos_k <= pos_q``), so a
    query at ``pos`` sees ``(pos + 1) // ratio`` entries. Broadcasts
    ``pos_q`` against ``pos_k``.
    """
    visible = pos_k <= pos_q
    if compress_ratio == 1:
        return visible
    return visible & ((pos_k + 1) % compress_ratio == 0)
