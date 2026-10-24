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
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear, linear
from max.nn.norm.rms_norm import RMSNorm

from .ragged import V41RaggedRows as RaggedRows

_KERNEL_HEAD_DIM = 512
"""The only latent width the pooling ops support."""


def workspace_bytes(compress_ratio: int, head_dim: int = 512) -> int:
    """Returns the pooling op's scratch bytes per token row.

    The group's KV and score rows in float32, the bf16 latent staging, four
    int32 group maps and a presence flag per group slot.
    """
    return (
        2 * compress_ratio * head_dim * 4
        + head_dim * 2
        + 4 * 4
        + (compress_ratio)
    )


def ring_slots(state: PagedCacheValues, rows: RaggedRows) -> TensorValue:
    """Returns each row's ``[T]`` int32 slot in its request's state ring,
    ``block * ring_rows + position % ring_rows``, where the block is column
    zero of the ring leaf's lookup table."""
    ring_rows = int(state.kv_blocks.shape[3])
    bid = ops.cast(rows.bid, DType.int32)
    first_col = ops.constant(0, DType.int32, bid.device).broadcast_to(bid.shape)
    block = ops.gather_nd(
        state.lookup_table, ops.stack([bid, first_col], axis=-1)
    )
    return ops.cast(block, DType.int32) * ring_rows + ops.cast(
        rows.positions % ring_rows, DType.int32
    )


def ring_save_slots(
    state: PagedCacheValues, rows: RaggedRows, slots: TensorValue
) -> TensorValue:
    """Returns ``slots`` with ``-1`` on rows outside their request's last
    ``ring_rows`` rows, so the ring save copies only the rows that survive a
    wrap."""
    ring_rows = int(state.kv_blocks.shape[3])
    assert rows.lengths is not None
    length = ops.gather(rows.lengths, rows.bid, axis=0)
    keep = rows.local >= length - ring_rows
    return ops.where(keep, slots, -1)


class DeepseekV41Compressor(Module):
    """Pools packed token rows and returns pre-RoPE BF16 latents.

    A group of ``compress_ratio`` consecutive tokens of a request pools into
    one latent, written on the row of the token closing the group. At ratio
    two a pair can straddle forwards: a request's last rows are saved to its
    state ring, and a request starting at an odd position pools its first
    pair with the row saved at the position before it. Nothing checks that
    row's origin, so a request may resume at an odd position only when the
    same request's last forward wrote that row.

    Args:
        compress_ratio: One for plain normalization, or two for gated pairs.
        eps: RMSNorm epsilon from the model configuration.
        hidden_size: Model hidden size. Given together with ``device``, the
            module owns the ``wkv``/``wgate`` projections (:meth:`project`);
            omit both to supply ``raw`` directly.
        device: Device of the projection weights; paired with
            ``hidden_size``.
        head_dim: Latent width; the pooling ops support only 512.
        weight_dtype: Dtype of the ``wkv``/``wgate`` weights, bf16 or
            float32; :meth:`project` multiplies the input in it.

    Raises:
        ValueError: If ``compress_ratio`` is not 1 or 2, ``head_dim`` is not
            512, only one of ``hidden_size`` and ``device`` is given, or
            ``weight_dtype`` is neither bf16 nor float32.
    """

    def __init__(
        self,
        compress_ratio: int,
        eps: float = 1e-20,
        *,
        hidden_size: int | None = None,
        device: DeviceRef | None = None,
        head_dim: int = 512,
        weight_dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError("CSA2 compression ratio must be 1 or 2")
        if head_dim != _KERNEL_HEAD_DIM:
            raise ValueError(
                f"CSA2 compressor head_dim must be {_KERNEL_HEAD_DIM}, got"
                f" {head_dim}"
            )
        if (hidden_size is None) != (device is None):
            raise ValueError("hidden_size and device must be given together")
        if weight_dtype not in (DType.bfloat16, DType.float32):
            raise ValueError(
                f"CSA2 compressor weights must be bf16 or float32, got"
                f" {weight_dtype}"
            )
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.eps = eps
        self.norm = RMSNorm(head_dim, DType.float32, eps)
        if hidden_size is not None:
            assert device is not None
            self.wkv = Linear(hidden_size, head_dim, weight_dtype, device)
            if compress_ratio == 2:
                self.wgate = Linear(hidden_size, head_dim, weight_dtype, device)

    def project(self, h: TensorValue) -> TensorValue:
        """Projects hidden states into this compressor's ``raw`` input.

        Requires construction with ``hidden_size`` and ``device``. The input
        is cast to the weights' dtype, and the matmuls accumulate and return
        float32 either way.

        Args:
            h: ``[..., hidden_size]`` hidden states of any float dtype.

        Returns:
            ``[..., head_dim * compress_ratio]`` float32, the ``raw`` input
            of :meth:`__call__`.
        """
        weights = [self.wkv.weight]
        if self.compress_ratio == 2:
            weights.append(self.wgate.weight)
        h = ops.cast(h, weights[0].dtype)
        raws = [
            linear(h, w.to(h.device), out_dtype=DType.float32) for w in weights
        ]
        return raws[0] if len(raws) == 1 else ops.concat(raws, axis=-1)

    def __call__(
        self,
        raw: TensorValue,
        positions: TensorValue,
        offsets: TensorValue,
        state: PagedCacheValues | None = None,
        slots: TensorValue | None = None,
        save_slots: TensorValue | None = None,
        layer_idx: TensorValue | None = None,
    ) -> TensorValue:
        """Normalizes raw rows, pooling complete pairs when the ratio is two.

        Ratio two pools the pair ``(2j, 2j + 1)`` per channel as
        ``sum_i kv_i * softmax_i(score)`` in float32, RMS-normalizes the
        float32 result without first rounding it to bf16, then casts to bf16.
        The pooled row is emitted on the row closing the pair (odd position
        ``2j + 1``); the caller applies RoPE at position ``2j``.

        Args:
            raw: FP32 projections, ``[T, head_dim * compress_ratio]``: the
                ``wkv`` row, then (ratio two) the ``wgate`` score row.
            positions: Absolute token positions, ``[T]``; each request's rows
                are consecutive.
            offsets: Packed uint32 request boundaries, ``[B + 1]``.
            state: Ratio two only: the float32 state ring leaf, one
                ``2 * head_dim`` row per ring slot.
            slots: Ratio two only: ``[T]`` int32 ring slot of each row
                (:func:`ring_slots`).
            save_slots: Ratio two only: ``slots`` with ``-1`` on the rows that
                are not saved to the ring (:func:`ring_save_slots`).
            layer_idx: Ratio two only: host uint32 index of this layer in
                ``state``.

        Returns:
            ``[T, head_dim]`` BF16 latent rows, zero on rows that close no
            group.

        Raises:
            ValueError: On wrong input dtypes/widths or state arguments that
                do not match the ratio.
        """
        if (
            raw.dtype != DType.float32
            or raw.shape[-1] != self.head_dim * self.compress_ratio
        ):
            raise ValueError(
                "CSA2 raw projections have the wrong dtype or width"
            )
        if offsets.dtype != DType.uint32:
            raise ValueError("CSA2 offsets must be uint32")
        given = [x is not None for x in (state, slots, save_slots, layer_idx)]
        if any(given) != (self.compress_ratio == 2) or (
            any(given) and not all(given)
        ):
            raise ValueError(
                "Ratio two requires the state leaf, slots, save slots and"
                " layer index; ratio one takes none"
            )
        device = raw.device
        rows = raw.shape[0]
        out_types = [
            TensorType(DType.bfloat16, [rows, self.head_dim], device),
            TensorType(
                DType.uint8,
                [rows * workspace_bytes(self.compress_ratio, self.head_dim)],
                device,
            ),
        ]
        values = [
            raw,
            ops.cast(positions, DType.int32),
            offsets,
        ]
        weight = self.norm.weight.cast(DType.float32).to(device)
        if self.compress_ratio == 1:
            return ops.custom(
                "mo.csa.compress",
                device=device,
                values=[*values, weight],
                out_types=out_types,
                parameters={"epsilon": self.eps},
            )[0].tensor
        assert state is not None and slots is not None
        assert save_slots is not None and layer_idx is not None
        return ops.inplace_custom(
            "mo.csa.compress.paged",
            device=device,
            values=[
                *values,
                ops.cast(slots, DType.int32),
                ops.cast(save_slots, DType.int32),
                weight,
                state.kv_blocks,
                state.page_stride,
                layer_idx,
            ],
            out_types=out_types,
            parameters={"epsilon": self.eps},
        )[0].tensor


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
