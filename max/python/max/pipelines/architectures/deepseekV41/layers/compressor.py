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
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

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


class DeepseekV41Compressor(Module):
    """Pools packed token rows and returns pre-RoPE BF16 latents.

    A group of ``compress_ratio`` consecutive tokens of a request pools into
    one latent, written on the row of the token closing the group. At ratio
    two a pair can straddle forwards: every token's ``raw`` row is saved to a
    paged state leaf, and a request starting at an odd position pools its
    first pair with the row saved at the position before it. Nothing checks
    that row's origin, so a request may resume at an odd position only when
    the same request wrote that row.

    Args:
        compress_ratio: One for plain normalization, or two for gated pairs.
        eps: RMSNorm epsilon from the model configuration.
        hidden_size: Model hidden size. Given together with ``device``, the
            module owns the ``wkv``/``wgate`` projections (:meth:`project`);
            omit both to supply ``raw`` directly.
        device: Device of the projection weights; paired with
            ``hidden_size``.
        head_dim: Latent width; the pooling ops support only 512.

    Raises:
        ValueError: If ``compress_ratio`` is not 1 or 2, ``head_dim`` is not
            512, or only one of ``hidden_size`` and ``device`` is given.
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
        if head_dim != _KERNEL_HEAD_DIM:
            raise ValueError(
                f"CSA2 compressor head_dim must be {_KERNEL_HEAD_DIM}, got"
                f" {head_dim}"
            )
        if (hidden_size is None) != (device is None):
            raise ValueError("hidden_size and device must be given together")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.eps = eps
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
        state: PagedCacheValues | None = None,
        slots: TensorValue | None = None,
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
            state: Ratio two only: the float32 state leaf, one
                ``2 * head_dim`` row per token slot.
            slots: Ratio two only: ``[T]`` int32 state slot of each row,
                ``block * page_size + position % page_size``.
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
        given = [x is not None for x in (state, slots, layer_idx)]
        if any(given) != (self.compress_ratio == 2) or (
            any(given) and not all(given)
        ):
            raise ValueError(
                "Ratio two requires the state leaf, slots and layer index;"
                " ratio one takes none"
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
        assert layer_idx is not None
        return ops.inplace_custom(
            "mo.csa.compress.paged",
            device=device,
            values=[
                *values,
                ops.cast(slots, DType.int32),
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
