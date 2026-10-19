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
"""Engram n-gram hashing and table gather on the device.

Each token's preceding n-grams are hashed to rows of large embedding tables
kept in pinned host memory, and those rows are gathered directly into the
quantized form the Engram projection consumes, so they are never
dequantized in between.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorType, TensorValue, ops
from max.graph.dim import Dim

from .engram_hash import EngramLayout, compute_hash_multipliers

# The kernels' fixed layout.
_LAYERS = 2
_MAX_NGRAM = 4
_HEADS = 8
_HEAD_DIM = 256
_ROW_SCALES = _HEAD_DIM // 32
# Rows per bucket of the sorted host path, 2^19.
_BUCKET_SHIFT = 19
# Tokens from which the full-table op sorts host reads; the sharded op takes
# the same crossover in local rows.
_SORT_MIN_TOKENS = 1024
# Rows per interleaved block of the MXFP8 scales.
_SF_ROWS = 128


@dataclass(frozen=True)
class EngramOperand:
    """One Engram layer's MXFP8 ``wkv`` input."""

    codes: TensorValue
    """``[T, n_hash_cols * head_dim]`` ``float8_e4m3fn``."""

    scales: TensorValue
    """``[ceil(T / 128), K / 128, 32, 4, 4]`` ``float8_e8m0fnu``."""


@dataclass(frozen=True)
class EngramTokens:
    """One device's per-forward hash inputs."""

    ids: TensorValue
    """``[T]`` int32 raw token ids."""

    positions: TensorValue
    """``[T]`` int64 absolute positions."""

    query_start_loc: TensorValue
    """``[R + 1]`` int32 request boundaries."""

    dead_mask: TensorValue
    """``[T]`` bool, true where a token takes no part in an n-gram."""

    lookback: TensorValue
    """``[R, max_ngram_size - 1]`` int32 tokens before each request's chunk.

    Column ``j`` is the token at ``start - 1 - j``, ``-1`` where unknown.
    """

    lookback_dead: TensorValue
    """``[R, max_ngram_size - 1]`` bool, true where a lookback token is dead."""

    def values(self) -> list[TensorValue]:
        return [
            self.ids,
            self.positions,
            self.query_start_loc,
            self.dead_mask,
            self.lookback,
            self.lookback_dead,
        ]


class EngramLookup:
    """Builds the Engram hash-and-gather ops for one layout and token map.

    The ops' table layout (layer and head counts, n-gram size, row width) is
    fixed when the kernels are compiled, so the constructor rejects a layout
    that differs.
    """

    def __init__(
        self, layout: EngramLayout, token_map: npt.NDArray[np.integer]
    ) -> None:
        if (
            len(layout.layer_ids) != _LAYERS
            or layout.max_ngram_size != _MAX_NGRAM
            or layout.n_heads != _HEADS
            or layout.head_dim != _HEAD_DIM
        ):
            raise ValueError(
                "the Engram ops need 2 layers, max_ngram_size 4, 8 heads and "
                f"head_dim 256, got {len(layout.layer_ids)} layers, "
                f"max_ngram_size {layout.max_ngram_size}, {layout.n_heads} "
                f"heads and head_dim {layout.head_dim}"
            )
        rows = [int(r) for r in layout.primes.sum(axis=1)]
        if list(layout.num_embeddings) != rows:
            raise ValueError(
                f"engram_num_embeddings {list(layout.num_embeddings)} do not "
                f"match the hash layout's table rows {rows}"
            )
        token_map_i32 = np.asarray(token_map, dtype=np.int32)
        if not 0 <= layout.pad_token_id < token_map_i32.shape[0]:
            raise ValueError(
                f"pad_token_id {layout.pad_token_id} is out of range for a "
                f"token map of size {token_map_i32.shape[0]}"
            )
        self.layout = layout
        self.layer_ids = layout.layer_ids
        self.n_hash_cols = layout.n_hash_cols
        self.table_rows = rows
        self.pad_id = int(token_map_i32[layout.pad_token_id])
        self._token_map = token_map_i32
        self._multipliers = compute_hash_multipliers(
            layout.layer_ids,
            layout.max_ngram_size,
            layout.compressed_vocab_size,
        )
        self._buckets = (max(rows) >> _BUCKET_SHIFT) + 1

    @property
    def lookback_depth(self) -> int:
        return self.layout.max_ngram_size - 1

    def heads_per_rank(self, n_ranks: int) -> int:
        if n_ranks < 1 or self.n_hash_cols % n_ranks:
            raise ValueError(
                f"{self.n_hash_cols} Engram hash columns do not split over "
                f"{n_ranks} ranks"
            )
        return self.n_hash_cols // n_ranks

    def shard_rows(
        self, layer_index: int, rank: int, n_ranks: int
    ) -> tuple[int, int]:
        """Rank ``rank``'s first row and row count in a layer's table.

        A rank owns hash columns ``[rank * h, (rank + 1) * h)``; their
        buckets are contiguous, so its rows are too.
        """
        h = self.heads_per_rank(n_ranks)
        cols = slice(rank * h, (rank + 1) * h)
        primes = self.layout.primes[layer_index, cols]
        offsets = self.layout.offsets[layer_index, cols]
        return int(offsets[0]), int(primes.sum())

    def _constants(self, device: DeviceRef) -> list[TensorValue]:
        return [
            ops.constant(self._token_map, DType.int32, device=device),
            ops.constant(self._multipliers, DType.int64, device=device),
            ops.constant(self.layout.primes, DType.int64, device=device),
            ops.constant(self.layout.offsets, DType.int64, device=device),
        ]

    def _work_type(
        self, tokens: Dim, heads: int, device: DeviceRef
    ) -> TensorType:
        return TensorType(
            DType.int32,
            [4 * heads * tokens + 4 * self._buckets],
            device=device,
        )

    def _operand_types(
        self, tokens: Dim, device: DeviceRef
    ) -> list[TensorType]:
        k = self.n_hash_cols * _HEAD_DIM
        return [
            TensorType(DType.float8_e4m3fn, [_LAYERS, tokens, k], device),
            TensorType(
                DType.float8_e8m0fnu,
                [
                    _LAYERS,
                    (tokens + _SF_ROWS - 1) // _SF_ROWS,
                    k // 128,
                    32,
                    4,
                    4,
                ],
                device,
            ),
        ]

    def _split(
        self, codes: TensorValue, scales: TensorValue
    ) -> dict[int, EngramOperand]:
        return {
            layer_id: EngramOperand(codes[i], scales[i])
            for i, layer_id in enumerate(self.layer_ids)
        }

    def full(
        self,
        tokens: EngramTokens,
        tables: Sequence[TensorValue],
        device: DeviceRef,
    ) -> dict[int, EngramOperand]:
        """Both layers' operands from the full tables, on one device.

        Args:
            tokens: This forward's hash inputs, on ``device``.
            tables: Each layer's pinned host codes ``[rows, head_dim]`` and
                scales ``[rows, head_dim / 32]``, layer-major.
            device: The GPU the op runs on.
        """
        codes, scales, _ = ops.custom(
            "mo.engram.hash_gather",
            device=device,
            values=[*tokens.values(), *self._constants(device), *tables],
            out_types=[
                *self._operand_types(tokens.ids.shape[0], device),
                self._work_type(tokens.ids.shape[0], self.n_hash_cols, device),
            ],
            parameters={"pad_id": self.pad_id, "host_tables": True},
        )
        return self._split(codes.tensor, scales.tensor)

    def sharded(
        self,
        tokens: Sequence[EngramTokens],
        tables: Sequence[Sequence[TensorValue]],
        devices: Sequence[DeviceRef],
        signal_buffers: Sequence[BufferValue],
    ) -> list[dict[int, EngramOperand]]:
        """Both layers' operands on every device, heads split over ranks.

        Each rank reads only its own share of the host table, so host-to-
        device table traffic splits across ranks. Each rank gathers its
        contiguous share of the hash columns, the ranks all-gather the packed
        rows, and every rank assembles the full operands.

        Args:
            tokens: Per device, this forward's hash inputs.
            tables: Per device, that rank's table rows
                (:meth:`shard_rows`): codes and scales, layer-major.
            devices: One GPU per rank, in rank order.
            signal_buffers: The all-gather's signal buffers.
        """
        n = len(devices)
        heads = self.heads_per_rank(n)
        packed_codes: list[TensorValue] = []
        packed_scales: list[TensorValue] = []
        for rank, (x, shard, device) in enumerate(
            zip(tokens, tables, devices, strict=True)
        ):
            t = x.ids.shape[0]
            codes, scales, _ = ops.custom(
                "mo.engram.hash_gather_shard",
                device=device,
                values=[*x.values(), *self._constants(device), *shard],
                out_types=[
                    TensorType(
                        DType.uint8, [_LAYERS, t, heads, _HEAD_DIM], device
                    ),
                    TensorType(
                        DType.uint8, [_LAYERS, t, heads, _ROW_SCALES], device
                    ),
                    self._work_type(t, heads, device),
                ],
                parameters={
                    "pad_id": self.pad_id,
                    "head_start": rank * heads,
                    "sort_min_rows": _SORT_MIN_TOKENS * _LAYERS * heads,
                    "host_tables": True,
                },
            )
            packed_codes.append(codes.tensor)
            packed_scales.append(scales.tensor)
        all_codes = ops.allgather(packed_codes, signal_buffers, axis=0)
        all_scales = ops.allgather(packed_scales, signal_buffers, axis=0)
        out = []
        for codes, scales, device in zip(
            all_codes, all_scales, devices, strict=True
        ):
            full_codes, full_scales = ops.custom(
                "mo.engram.assemble_shards",
                device=device,
                values=[codes, scales],
                out_types=self._operand_types(codes.shape[1], device),
            )
            out.append(self._split(full_codes.tensor, full_scales.tensor))
        return out
